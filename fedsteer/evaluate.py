"""Per-client direction evaluation shared by eval_direction.py (from snapshots) and the
in-training monitor (from the live trainer), so both produce identical files.

A result file (``<run>/evals/eval_<what>__<stamp>.json``) holds, per client, the
direction-quality metrics (fedsteer.metrics), the per-article score grid, every
generated text (``outputs[article][alpha]``) with the source record ids (URLs), so
quality metrics can be computed afterwards (scripts/score_quality.py), and the
client's private control state (gain, offset, warp).
"""

from __future__ import annotations

import json
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from .calibrate import fit_remap
from .metrics import (SCORERS, constant_output_pct_err, decoration_sensitivity_metrics,
                      metrics_for_client, summarize, text_tie_metrics)
from .model import generate_at_alpha
from .monitor import mean_loss


def score_grid(model, fmt, recs, alphas, score, max_new_tokens=128, batch_size=32):
    """Generate for every record at every alpha; return scores [n, n_alpha] and texts."""
    prompts = [r["prompt"] for r in recs]
    grid = np.zeros((len(prompts), len(alphas)))
    texts = [[None] * len(alphas) for _ in prompts]
    for j, a in enumerate(alphas):
        outs = generate_at_alpha(model, fmt, prompts, a, max_new_tokens=max_new_tokens, batch_size=batch_size)
        for i, (o, rec) in enumerate(zip(outs, recs)):
            texts[i][j] = o[0]
            grid[i, j] = score(o[0], rec)
    return grid, texts


def control_state(model) -> dict:
    ctl = model.steer_control
    out = {"gain": float(ctl.gain().item())}
    if ctl.o is not None:
        out["offset"] = float(ctl.offset().item())
    if ctl.warp.kind != "none":
        out["warp"] = ctl.warp.describe()
    return out


def evaluate_loaded_client(model, fmt, recs: list[dict], alphas: Sequence[float], score: Callable,
                           ref_q, support=None, max_new_tokens: int = 128, batch_size: int = 32,
                           dev_loss: bool = False, remap_recs: Optional[list[dict]] = None,
                           remap_grid: int = 11) -> dict:
    """Evaluate the client currently loaded in ``model``.  ``ref_q`` is the CDF that defines
    alpha; ``support`` is (lo, hi) in global mode; ``remap_recs`` enables option F."""
    alphas = list(alphas)
    gen_alphas, remap = alphas, None
    if remap_recs is not None:
        cal_alphas = np.linspace(0, 1, remap_grid).tolist()
        cal_scores, _ = score_grid(model, fmt, remap_recs, cal_alphas, score, max_new_tokens, batch_size)
        achieved = np.vectorize(ref_q.cdf)(cal_scores).mean(axis=0)
        remap = fit_remap(cal_alphas, achieved, alphas)
        gen_alphas = [remap["mapped_alpha"][str(a)] for a in alphas]
    grid, texts = score_grid(model, fmt, recs, gen_alphas, score, max_new_tokens, batch_size)
    res = metrics_for_client(grid, alphas, ref_q, support=support)   # scored against the *target* alphas
    res.update(text_tie_metrics(texts, grid, alphas))
    if score is SCORERS["clogp_residual_deco"]:
        res.update(decoration_sensitivity_metrics(texts, recs, alphas, ref_q, support))
    res.update(control_state(model))
    res["grid"] = grid.round(4).tolist()
    res["outputs"] = texts
    res["record_ids"] = [r.get("url", str(i)) for i, r in enumerate(recs)]
    if remap is not None:
        res["remap"] = remap
        res["generated_at"] = gen_alphas
    if dev_loss:
        labeled = [dict(r, alpha=ref_q.cdf(r["score"])) for r in recs]
        res["loss"] = mean_loss(model, fmt, labeled, batch_size=min(batch_size, 16))
    return res


def assemble(results: dict[str, dict], alphas, split: str, snapshot: str, rnd: int, shared=None) -> dict:
    summary = summarize(results)
    summary["constant_output_pct_err"] = constant_output_pct_err(alphas)
    losses = [r["loss"] for r in results.values() if "loss" in r]
    if losses:
        summary["loss"] = {"mean": float(np.mean(losses)), "worst": float(max(losses))}
    samples = {c: [{"alpha": a, "output": r["outputs"][0][j]} for j, a in enumerate(alphas)]
               for c, r in results.items() if r["outputs"]}
    return {"snapshot": snapshot, "round": rnd, "shared": shared, "split": split, "alphas": list(alphas),
            "clients": results, "summary": summary, "samples": samples}


def brief(res: dict) -> str:
    return json.dumps({k: v for k, v in res.items() if k not in ("grid", "outputs", "record_ids")})
