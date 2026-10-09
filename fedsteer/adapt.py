"""Small training routines for the evaluation settings E2 (held-out clients) and E3
(drift).  Each trains exactly one component of a client's model and freezes the rest:

* ``train_adapter_sft``     the private adapter P (A_p, B_p) by plain SFT.  By default the
                            steering module is switched off during training
                            (``direction_off``): ordinary fine-tuning that knows nothing
                            about the knob (E2 new clients, E3 drift).
* ``fit_calibration``       the calibration g(alpha) = o + s h(alpha) (gain u, offset o,
                            warp) on k labelled examples, with P and D frozen.
* ``train_local_direction`` a client's own direction B_d (+ calibration) from zero on k
                            labelled examples, with P frozen.
* ``train_steered``         alpha-conditioned training (the participants' objective) of a
                            chosen set of components with everything else frozen: E2's
                            new client joining with the frozen direction (P [+ private
                            calibration]) or training its own (P + D + calibration).

Components are the keys of ``trainable_parameter_groups`` (private / shared / gain / warp) plus
two opt-in finer keys that split ``gain`` (NR-68): ``u`` (the gain parameter alone) and
``offset`` (the offset o alone), so a client can fit o_i with the gain fixed at s = 1.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Optional

import torch

from .data import ChatFormatter, ClientStream
from .lora import trainable_parameter_groups


@contextmanager
def direction_off(model):
    """Temporarily zero the shared direction B_d (restored afterwards)."""
    saved = {n: p.detach().clone() for n, p in model.named_parameters() if n.endswith("lora_B_d")}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in saved:
                p.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in saved:
                    p.copy_(saved[n])


@contextmanager
def only_trainable(model, params):
    """requires_grad only for ``params`` inside the block (restored afterwards)."""
    ids = {id(p) for p in params}
    prev = {n: p.requires_grad for n, p in model.named_parameters()}
    for _, p in model.named_parameters():
        p.requires_grad_(id(p) in ids)
    try:
        yield
    finally:
        for n, p in model.named_parameters():
            p.requires_grad_(prev[n])


def _train(model, fmt: ChatFormatter, examples: list[dict], params, steps: int, lr: float,
           batch_size: int = 8, fixed_alpha: Optional[float] = None, seed: int = 0, bf16: bool = True,
           max_grad_norm: float = 1.0, warmup_steps: int = 0) -> list[float]:
    """Generic loop: AdamW on ``params`` only (a list of tensors, or a list of param-group
    dicts with their own "lr"); alpha per example from ``example['alpha']`` unless
    ``fixed_alpha`` is given (label-free training).  Linear warmup over ``warmup_steps``."""
    if not examples or steps <= 0:
        return []
    device = next(model.parameters()).device
    exs = [dict(e, alpha=fixed_alpha if fixed_alpha is not None else e["alpha"]) for e in examples]
    stream = ClientStream(exs, fmt, min(batch_size, len(exs)), seed=seed)
    groups = params if params and isinstance(params[0], dict) else [{"params": list(params)}]
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / warmup_steps) if warmup_steps else 1.0)
    losses = []
    model.train()
    with only_trainable(model, params):
        for _ in range(steps):
            batch = {k: v.to(device) for k, v in stream.next_batch().items()}
            with model.steer_control.use_alpha(batch.pop("alpha")), \
                    torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                loss = model(**batch).loss
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
    model.eval()
    return losses


def train_adapter_sft(model, fmt, examples, steps, lr=2e-4, batch_size=8, steering_off: bool = True,
                      fixed_alpha: float = 0.0, seed: int = 0, bf16: bool = True) -> list[float]:
    """Plain, label-free SFT of the private adapter.  With ``steering_off`` the direction is
    zeroed during training (and restored), otherwise it stays attached at ``fixed_alpha``."""
    params = trainable_parameter_groups(model)["private"]
    if steering_off:
        with direction_off(model):
            return _train(model, fmt, examples, params, steps, lr, batch_size, fixed_alpha, seed, bf16)
    return _train(model, fmt, examples, params, steps, lr, batch_size, fixed_alpha, seed, bf16)


def component_groups(model) -> dict[str, list]:
    """trainable_parameter_groups plus the opt-in split of the "gain" group into "u" (gain
    parameter) and "offset" (o; empty without lora.offset)."""
    g = trainable_parameter_groups(model)
    ctl = model.steer_control
    g["u"] = [p for p in g["gain"] if p is ctl.u]
    g["offset"] = [p for p in g["gain"] if ctl.o is not None and p is ctl.o]
    return g


def fit_calibration(model, fmt, labelled, steps=100, lr=1e-2, batch_size=8, seed: int = 0,
                    bf16: bool = True, components=("gain", "warp")) -> list[float]:
    """Fit the calibration on labelled examples (each with 'alpha'); P and D frozen.  By
    default gain/offset/warp; ``components=("offset",)`` fits only o (NR-68)."""
    g = component_groups(model)
    params = [p for c in components for p in g[c]]
    return _train(model, fmt, labelled, params, steps, lr, batch_size, None, seed, bf16)


def train_local_direction(model, fmt, labelled, steps=100, lr=2e-4, lr_calibration=1e-2, batch_size=8,
                          seed: int = 0, bf16: bool = True) -> list[float]:
    """A client's own direction from zero (+ its calibration) on labelled examples, P frozen."""
    g = trainable_parameter_groups(model)
    with torch.no_grad():
        for p in g["shared"]:
            p.zero_()
    groups = [{"params": g["shared"], "lr": lr}, {"params": g["gain"] + g["warp"], "lr": lr_calibration}]
    return _train(model, fmt, labelled, groups, steps, lr, batch_size, None, seed, bf16)


def train_steered(model, fmt, examples, steps, components=("private",), lrs=None, batch_size=8,
                  warmup_steps: int = 0, seed: int = 0, bf16: bool = True) -> list[float]:
    """Alpha-conditioned training (alpha per example, steering on) of ``components`` (keys of
    trainable_parameter_groups: private / shared / gain / warp, or the finer u / offset), each at
    ``lrs[component]``; u and offset fall back to ``lrs["gain"]``."""
    g = component_groups(model)
    groups = [{"params": g[c], "lr": lrs[c] if c in lrs else lrs["gain"]} for c in components if g[c]]
    return _train(model, fmt, examples, groups, steps, groups[0]["lr"], batch_size, None, seed, bf16,
                  warmup_steps=warmup_steps)


def sample_k(examples: list[dict], k: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return rng.sample(examples, min(k, len(examples)))
