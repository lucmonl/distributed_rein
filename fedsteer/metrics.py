"""Attribute scorers and direction-quality metrics (plan §4), shared by the
training monitor and eval_direction.py so both report identical numbers."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import spearmanr

from .data import ClientQuantiles
from .extractive import fragment_stats

# scorer(generated_text, source_record) -> attribute value on the same scale as the
# training ``score`` field
SCORERS = {
    "words": lambda text, rec: float(len(text.split())),
    "density": lambda text, rec: fragment_stats(text, rec["article"])["density"],
}

LOWER_IS_BETTER = {"calib_mae_iqr", "pct_calib_err", "pct_err_in_support", "pct_err_out_support",
                   "no_effect_rate", "adjacent_tie_rate", "adjacent_decrease_rate"}
SUMMARY_KEYS = ["pct_err_in_support", "pct_err_out_support", "reach_rate", "concordance", "spearman",
                "endpoint_increase_rate", "adjacent_increase_rate", "adjacent_tie_rate", "adjacent_decrease_rate",
                "no_effect_rate", "pct_calib_err", "pct_range", "order_rate", "norm_range", "calib_mae_iqr"]


def metrics_for_client(scores: np.ndarray, alphas: list[float], q: ClientQuantiles,
                       support: Optional[tuple[float, float]] = None) -> dict:
    """scores: [n_prompts, n_alphas]

    ``q`` is the reference CDF that defines alpha (the client's own CDF, or the shared
    mixture in global mode).  Percentile metrics map each output's score through it,
    the same transform used for the training labels, so they are bounded and robust
    to heavy-tailed attributes such as density.  ``support`` (lo, hi) splits the alpha
    grid into the part covered by the client's own data and the part it can only
    reach through the shared direction.
    """
    scores = np.asarray(scores, dtype=float)
    order = np.all(np.diff(scores, axis=1) > 0, axis=1).mean()
    # pairwise concordance: fraction of (alpha_i < alpha_j) pairs with score_i < score_j (ties = 1/2)
    i, j = np.triu_indices(len(alphas), 1)
    diff = scores[:, j] - scores[:, i]
    concord = ((diff > 0) + 0.5 * (diff == 0)).mean()
    pct = np.vectorize(q.cdf)(scores)
    pct_err = np.abs(pct - np.asarray(alphas)[None, :])
    # Spearman per article; an article whose outputs do not change with alpha counts as 0
    constant = np.ptp(scores, axis=1) == 0
    rhos = [0.0 if c else spearmanr(alphas, s).correlation for s, c in zip(scores, constant)]
    adj = np.diff(scores, axis=1)
    rng = (scores[:, -1] - scores[:, 0]).mean() / max(q.iqr(), 1e-9)
    targets = np.array([q.quantile(a) for a in alphas])
    mae = np.abs(scores - targets[None, :]).mean() / max(q.iqr(), 1e-9)
    out = {}
    if support is not None:
        # in-support: alphas inside the part of the axis the client's own data covers;
        # out-of-support: alphas the client can only reach through the shared direction
        lo, hi = support
        a = np.asarray(alphas)
        ins = (a >= lo) & (a <= hi)
        out["support"] = [round(float(lo), 4), round(float(hi), 4)]
        out["pct_err_in_support"] = float(pct_err[:, ins].mean()) if ins.any() else None
        out["pct_err_out_support"] = float(pct_err[:, ~ins].mean()) if (~ins).any() else None
        # reach: at out-of-support alphas, did the output leave the client's own range in
        # the requested direction?
        below, above = a < lo, a > hi
        hits = [pct[:, below] < lo] if below.any() else []
        hits += [pct[:, above] > hi] if above.any() else []
        out["reach_rate"] = float(np.concatenate([h.ravel() for h in hits]).mean()) if hits else None
    return {
        **out,
        "concordance": float(concord),
        "pct_calib_err": float(pct_err.mean()),
        "pct_range": float((pct[:, -1] - pct[:, 0]).mean()),
        "mean_pct_by_alpha": pct.mean(axis=0).round(3).tolist(),
        "order_rate": float(order),
        "no_effect_rate": float(constant.mean()),              # all alphas give the same score
        "adjacent_increase_rate": float((adj > 0).mean()),     # P(score goes up | next alpha step)
        "adjacent_tie_rate": float((adj == 0).mean()),
        "adjacent_decrease_rate": float((adj < 0).mean()),
        "endpoint_increase_rate": float((scores[:, -1] > scores[:, 0]).mean()),  # highest alpha above lowest
        "spearman": float(np.mean(rhos)),
        "norm_range": float(rng),
        "calib_mae_iqr": float(mae),
        "mean_score_by_alpha": scores.mean(axis=0).round(3).tolist(),
        "target_by_alpha": targets.round(3).tolist(),
    }


def constant_output_pct_err(alphas: list[float]) -> float:
    """Percentile calibration error of a model that always outputs the client median."""
    return float(np.mean(np.abs(np.asarray(alphas) - 0.5)))


def summarize(results: dict[str, dict], keys=SUMMARY_KEYS) -> dict:
    out = {}
    for k in keys:
        vals = [r[k] for r in results.values() if r.get(k) is not None]
        if vals:
            worst = max(vals) if k in LOWER_IS_BETTER else min(vals)
            out[k] = {"mean": float(np.mean(vals)), "worst": float(worst)}
    return out
