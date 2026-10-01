"""Small training routines for the evaluation settings E2 (held-out clients) and E3
(drift).  Each trains exactly one component of a client's model and freezes the rest:

* ``train_adapter_sft``     the private adapter P (A_p, B_p) by plain SFT.  By default the
                            steering module is switched off during training
                            (``direction_off``): ordinary fine-tuning that knows nothing
                            about the knob (E2 new clients, E3 drift).
* ``fit_calibration``       the calibration g(alpha) = o + s h(alpha) (gain u, offset o,
                            warp) on k labelled examples, with P and D frozen.
* ``train_local_direction`` a client's own direction B_d (+ calibration) from zero on k
                            labelled examples, with P frozen (E2 baseline: "a local
                            direction trained on the same k examples").
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
           max_grad_norm: float = 1.0) -> list[float]:
    """Generic loop: AdamW on ``params`` only (a list of tensors, or a list of param-group
    dicts with their own "lr"); alpha per example from ``example['alpha']`` unless
    ``fixed_alpha`` is given (label-free training)."""
    if not examples or steps <= 0:
        return []
    device = next(model.parameters()).device
    exs = [dict(e, alpha=fixed_alpha if fixed_alpha is not None else e["alpha"]) for e in examples]
    stream = ClientStream(exs, fmt, min(batch_size, len(exs)), seed=seed)
    groups = params if params and isinstance(params[0], dict) else [{"params": list(params)}]
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.0)
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


def fit_calibration(model, fmt, labelled, steps=100, lr=1e-2, batch_size=8, seed: int = 0,
                    bf16: bool = True) -> list[float]:
    """Fit gain/offset/warp on labelled examples (each with 'alpha'); P and D frozen."""
    g = trainable_parameter_groups(model)
    return _train(model, fmt, labelled, g["gain"] + g["warp"], steps, lr, batch_size, None, seed, bf16)


def train_local_direction(model, fmt, labelled, steps=100, lr=2e-4, lr_calibration=1e-2, batch_size=8,
                          seed: int = 0, bf16: bool = True) -> list[float]:
    """A client's own direction from zero (+ its calibration) on labelled examples, P frozen."""
    g = trainable_parameter_groups(model)
    with torch.no_grad():
        for p in g["shared"]:
            p.zero_()
    groups = [{"params": g["shared"], "lr": lr}, {"params": g["gain"] + g["warp"], "lr": lr_calibration}]
    return _train(model, fmt, labelled, groups, steps, lr, batch_size, None, seed, bf16)


def sample_k(examples: list[dict], k: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return rng.sample(examples, min(k, len(examples)))
