"""Held-out monitoring: dev loss and a cheap steering check.

Training loss only measures fit to data the model trains on (and, over several
passes, memorization).  The monitor measures, per client, on the held-out split:

* ``loss``      mean token loss with the client's own alpha labels
* steering      greedy generations at a few alphas on a handful of articles, scored
                with the task scorer, summarized with fedsteer.metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from .data import ChatFormatter, ClientQuantiles, collate
from .metrics import SCORERS, constant_output_pct_err, metrics_for_client, summarize
from .model import generate_at_alpha


@dataclass
class MonitorConfig:
    split: str = "dev"
    loss_every: int = 1            # rounds; 0 disables
    loss_max_examples: int = 100
    steer_every: int = 5           # rounds; 0 disables
    steer_prompts: int = 20
    steer_alphas: Sequence[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    scorer: str = "density"
    max_new_tokens: int = 96
    batch_size: int = 16


def label_split(records: list[dict], clients: Sequence[str], quantiles: dict[str, ClientQuantiles],
                split: str) -> dict[str, list[dict]]:
    """Records of ``split`` per client, with alpha from the client's *train* CDF."""
    out = {c: [] for c in clients}
    for r in records:
        if r.get("split") == split and r["client"] in out:
            out[r["client"]].append(dict(r, alpha=quantiles[r["client"]].cdf(r["score"])))
    return out


@torch.no_grad()
def mean_loss(model, fmt: ChatFormatter, examples: list[dict], batch_size: int = 16, bf16: bool = True) -> float:
    """Token-weighted mean loss of ``examples`` (each with prompt/target/alpha)."""
    model.eval()
    device = next(model.parameters()).device
    tot_loss, tot_tok = 0.0, 0
    feats = [dict(fmt.encode(e["prompt"], e["target"]), alpha=e["alpha"]) for e in examples]
    for i in range(0, len(feats), batch_size):
        b = {k: v.to(device) for k, v in collate(feats[i:i + batch_size], fmt.tok.pad_token_id).items()}
        alpha = b.pop("alpha")
        labels = b.pop("labels")
        with model.steer_control.use_alpha(alpha), \
                torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            logits = model(**b).logits
        shift_logits, shift_labels = logits[:, :-1].float(), labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]),
                                                 shift_labels.reshape(-1), ignore_index=-100, reduction="sum")
        tot_loss += loss.item()
        tot_tok += int((shift_labels != -100).sum())
    return tot_loss / max(tot_tok, 1)


def steering_check(model, fmt: ChatFormatter, examples: list[dict], q: ClientQuantiles, alphas: Sequence[float],
                   scorer: str, max_new_tokens: int, batch_size: int, bf16: bool = True) -> dict:
    score = SCORERS[scorer]
    prompts = [e["prompt"] for e in examples]
    grid = np.zeros((len(prompts), len(alphas)))
    for j, a in enumerate(alphas):
        outs = generate_at_alpha(model, fmt, prompts, a, max_new_tokens=max_new_tokens,
                                 batch_size=batch_size, bf16=bf16)
        grid[:, j] = [score(o[0], e) for o, e in zip(outs, examples)]
    return metrics_for_client(grid, list(alphas), q)


def make_monitor(records: list[dict], clients: Sequence[str], quantiles: dict[str, ClientQuantiles],
                 fmt: ChatFormatter, cfg: MonitorConfig):
    """Returns ``eval_fn(trainer, round)`` for FedSteerTrainer.

    ``round`` is the number of completed rounds.  Each client is evaluated with its
    own private state and the current direction (``trainer.load_client``).
    """
    held = label_split(records, clients, quantiles, cfg.split)
    empty = [c for c, v in held.items() if not v]
    if empty:
        raise ValueError(f"no '{cfg.split}' examples for clients {empty}")
    steer_ex = {c: v[: cfg.steer_prompts] for c, v in held.items()}
    loss_ex = {c: v[: cfg.loss_max_examples] for c, v in held.items()}

    def eval_fn(trainer, rnd: int) -> dict:
        do_loss = cfg.loss_every > 0 and rnd % cfg.loss_every == 0
        do_steer = cfg.steer_every > 0 and (rnd % cfg.steer_every == 0 or rnd == trainer.cfg.rounds)
        if not (do_loss or do_steer):
            return {}
        out: dict = {"split": cfg.split, "clients": {}}
        for c in clients:
            trainer.load_client(c)
            res = {}
            if do_loss:
                res["loss"] = mean_loss(trainer.model, fmt, loss_ex[c], cfg.batch_size, trainer.cfg.bf16)
            if do_steer:
                res["steer"] = steering_check(trainer.model, fmt, steer_ex[c], quantiles[c], cfg.steer_alphas,
                                              cfg.scorer, cfg.max_new_tokens, cfg.batch_size, trainer.cfg.bf16)
            out["clients"][c] = res
        if do_loss:
            out["loss_mean"] = float(np.mean([r["loss"] for r in out["clients"].values()]))
        if do_steer:
            out["steer_summary"] = summarize({c: r["steer"] for c, r in out["clients"].items()})
            out["steer_summary"]["constant_output_pct_err"] = constant_output_pct_err(cfg.steer_alphas)
        return out

    return eval_fn


def format_monitor(ev: dict) -> str:
    """One-line summary for the training log."""
    if not ev:
        return ""
    parts = []
    if "loss_mean" in ev:
        parts.append(f"{ev['split']} loss {ev['loss_mean']:.4f}")
    if "steer_summary" in ev:
        s = ev["steer_summary"]
        parts.append(f"{ev['split']} spearman {s['spearman']['mean']:.3f} (worst {s['spearman']['worst']:.3f}) "
                     f"pct err {s['pct_calib_err']['mean']:.3f} (const {s['constant_output_pct_err']:.3f})")
    return " | " + " | ".join(parts)
