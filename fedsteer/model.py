from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import ChatFormatter
from .lora import SteerLoraConfig, inject_steer_lora


def load_model(model_name: str, lora_cfg: SteerLoraConfig, device: Union[str, torch.device] = "cuda",
               dtype: torch.dtype = torch.bfloat16, grad_checkpointing: bool = True,
               attn_implementation: Optional[str] = "sdpa"):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation=attn_implementation
    ).to(device)
    model.config.use_cache = False
    inject_steer_lora(model, lora_cfg)
    if grad_checkpointing:
        # non-reentrant checkpointing does not need input grads, and recomputes the
        # forward inside `control.use_alpha`, which also wraps backward
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tok


@torch.no_grad()
def generate_at_alpha(model, fmt: ChatFormatter, prompts: Sequence[str],
                      alpha: Union[float, Sequence[float]], max_new_tokens: int = 128,
                      batch_size: int = 16, do_sample: bool = False, temperature: float = 0.7,
                      top_p: float = 1.0, num_return_sequences: int = 1, bf16: bool = True) -> list[list[str]]:
    """Generate for every prompt at the given alpha (scalar or one per prompt).

    Returns ``num_return_sequences`` completions per prompt.
    """
    tok = fmt.tok
    model.eval()
    device = next(model.parameters()).device
    alphas = [float(alpha)] * len(prompts) if isinstance(alpha, (int, float)) else [float(a) for a in alpha]
    old_side, tok.padding_side = tok.padding_side, "left"
    old_cache, model.config.use_cache = model.config.use_cache, True
    outs: list[list[str]] = []
    try:
        for i in range(0, len(prompts), batch_size):
            ids = [fmt.prompt_ids(p) for p in prompts[i:i + batch_size]]
            n = max(len(x) for x in ids)
            input_ids = torch.full((len(ids), n), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(input_ids)
            for j, x in enumerate(ids):
                input_ids[j, n - len(x):] = torch.tensor(x)
                mask[j, n - len(x):] = 1
            gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                              num_return_sequences=num_return_sequences, pad_token_id=tok.pad_token_id)
            if do_sample:
                gen_kwargs.update(temperature=temperature, top_p=top_p)
            else:
                gen_kwargs.update(temperature=None, top_p=None, top_k=None)
            with model.steer_control.use_alpha(torch.tensor(alphas[i:i + batch_size])), \
                    torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                out = model.generate(input_ids=input_ids.to(device), attention_mask=mask.to(device), **gen_kwargs)
            texts = tok.batch_decode(out[:, n:], skip_special_tokens=True)
            k = num_return_sequences
            outs.extend([texts[j * k:(j + 1) * k] for j in range(len(ids))])
    finally:
        tok.padding_side = old_side
        model.config.use_cache = old_cache
    return outs
