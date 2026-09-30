"""Attribute scorers and direction-quality metrics (plan §4), shared by the
training monitor and eval_direction.py so both report identical numbers."""

from __future__ import annotations

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

LOWER_IS_BETTER = {"calib_mae_iqr", "pct_calib_err", "no_effect_rate", "adjacent_tie_rate",
                   "adjacent_decrease_rate"}
SUMMARY_KEYS = ["concordance", "spearman", "endpoint_increase_rate", "adjacent_increase_rate",
                "adjacent_tie_rate", "adjacent_decrease_rate", "no_effect_rate", "pct_calib_err", "pct_range",
                "order_rate", "norm_range", "calib_mae_iqr"]


def metrics_for_client(scores: np.ndarray, alphas: list[float], q: ClientQuantiles) -> dict:
    """scores: [n_prompts, n_alphas]

    Percentile metrics map each output's score through the client's own CDF, the
    same transform that defines alpha in training, so they are bounded and robust
    to heavy-tailed attributes such as density.
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
    return {
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
        vals = [r[k] for r in results.values() if k in r]
        if vals:
            worst = max(vals) if k in LOWER_IS_BETTER else min(vals)
            out[k] = {"mean": float(np.mean(vals)), "worst": float(worst)}
    return out
