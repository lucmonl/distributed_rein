from __future__ import annotations

import os
from typing import Optional, Sequence, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import ChatFormatter
from .lora import SteerLoraConfig, inject_steer_lora


def load_model(model_name: str, lora_cfg: SteerLoraConfig, device: Union[str, torch.device] = "cuda",
               dtype: torch.dtype = torch.bfloat16, grad_checkpointing: bool = True,
               attn_implementation: Optional[str] = "sdpa"):
    cap_gb = os.environ.get("FEDSTEER_GPU_MEM_GB")
    if cap_gb and torch.cuda.is_available():
        # emulate a smaller GPU (e.g. 40 GB A100s on a GH200): exceeding the cap raises a real OOM
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap_gb) * 1024 ** 3 / total))
        print(f"GPU memory capped at {float(cap_gb):.1f} GiB of {total / 1024 ** 3:.1f} GiB", flush=True)
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
    # Generation runs no backward pass, so it can use SDPA even when training uses eager attention
    # (the math config trains with eager because SDPA's *backward* gave NaN on DeltaAI). Eager prefill
    # materializes batch x heads x L^2 fp32 scores, which OOMs a 40 GB GPU on ~3k-token few-shot
    # prompts (math log MATH-11). Override with FEDSTEER_GEN_ATTN=eager to generate with eager.
    old_attn = model.config._attn_implementation
    gen_attn = os.environ.get("FEDSTEER_GEN_ATTN", "sdpa")
    if old_attn == "eager" and gen_attn != "eager":
        model.config._attn_implementation = gen_attn
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
        model.config._attn_implementation = old_attn
    return outs


def report_peak_memory(tag: str) -> None:
    """One line with the peak GPU memory allocated by this process (for sizing runs)."""
    if torch.cuda.is_available():
        print(f"PEAK_GPU_MEM {tag}: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB allocated, "
              f"{torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GiB reserved", flush=True)
