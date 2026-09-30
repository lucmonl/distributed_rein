"""Post-hoc monotone remap of alpha (warp option F, an ablation).

After training, measure on a calibration split which output percentile each alpha
actually produces, fit an increasing curve (isotonic regression), and invert it:
to ask for target percentile t, feed the alpha that achieved t.  This relabels the
knob; unlike a trained warp it cannot create behavior the model does not have
(targets outside the achieved range are clipped to the nearest end).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def isotonic_increasing(y: Sequence[float], w: Optional[Sequence[float]] = None) -> np.ndarray:
    """Pool-adjacent-violators: least-squares non-decreasing fit to y."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    blocks = []  # [mean, weight, count]
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, c2 = blocks.pop()
            m1, w1, c1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    return np.concatenate([[m] * c for m, _, c in blocks])


def invert_monotone(grid: Sequence[float], fitted: Sequence[float], target: float) -> float:
    """Smallest-interpolated alpha on ``grid`` whose fitted response reaches ``target``."""
    grid, fitted = np.asarray(grid, float), np.asarray(fitted, float)
    if target <= fitted[0]:
        return float(grid[0])
    if target >= fitted[-1]:
        return float(grid[-1])
    k = int(np.argmax(fitted >= target))          # first grid point at or above target
    lo, hi = fitted[k - 1], fitted[k]
    if hi == lo:
        return float(grid[k])
    return float(grid[k - 1] + (target - lo) / (hi - lo) * (grid[k] - grid[k - 1]))


def fit_remap(grid: Sequence[float], achieved_pct: Sequence[float], targets: Sequence[float]) -> dict:
    fitted = isotonic_increasing(achieved_pct)
    return {"grid": [float(g) for g in grid], "achieved": [float(a) for a in achieved_pct],
            "fitted": fitted.round(4).tolist(),
            "mapped_alpha": {str(t): invert_monotone(grid, fitted, t) for t in targets}}
