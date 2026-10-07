"""Coverage-aware borrowing for private nonlinear calibrations (``fed.calibration: coverage``).

Plan section 2.1.  Each client keeps its own warp h_i (gain fixed at 1, no offset), so its
coefficient on the shared direction is h_i(alpha).  The server never averages warp
parameters; it pools *function values* on a common grid a_k, weighted by how much nearby
data each client has, and broadcasts a monotone target table z.  Clients are pulled toward
z only where their own evidence is weak and other clients' evidence is strong:

  c_ik       = sum_j max(0, 1 - |alpha_ij - a_k| / b)                 smoothed local count
  R_ik       = sum_{l != i} c_lk                                      peer evidence
  lambda_ik  = lambda_max * tau_local / (tau_local + c_ik) * R_ik / (tau_peer + R_ik)
  m_k        = sum_i c_ik h_i(a_k) / sum_i c_ik                       (where sum_i c_ik > 0)
  z          = weighted isotonic projection of m, z_1 = 0, z_K = 1, uncovered z_k kept

The local objective adds  (1/K) sum_k lambda_ik (h_i(a_k) - stopgrad(z_k))^2.
Everything here is plain numpy on K numbers; the trainer (fedsteer/fed.py) owns the state.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .calibrate import isotonic_increasing


def make_grid(k: int) -> np.ndarray:
    """K equally spaced evaluation points including 0 and 1."""
    if k < 2:
        raise ValueError(f"coverage grid needs at least 2 points, got {k}")
    return np.linspace(0.0, 1.0, k)


def evidence_counts(alphas: Sequence[float], grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """c_k: triangular-kernel count of the alphas near each grid point (each example once)."""
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    a = np.asarray(alphas, dtype=float).reshape(-1, 1)
    if a.size == 0:
        return np.zeros(len(grid))
    return np.clip(1.0 - np.abs(a - grid[None, :]) / bandwidth, 0.0, None).sum(axis=0)


def borrow_weights(counts: dict[str, np.ndarray], lambda_max: float, tau_local: float,
                   tau_peer: float) -> dict[str, np.ndarray]:
    """lambda_ik: strong only where the client's own evidence is weak and its peers' is strong."""
    if tau_local <= 0 or tau_peer <= 0:
        raise ValueError("tau_local and tau_peer must be positive")
    total = sum(counts.values())
    out = {}
    for c, ci in counts.items():
        peer = total - ci
        out[c] = lambda_max * tau_local / (tau_local + ci) * peer / (tau_peer + peer)
    return out


def project_monotone(m: np.ndarray, weights: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """Weighted isotonic projection of m with z_1 = 0, z_K = 1 and z_k = prev_k wherever the
    weight is zero (no evidence: keep the prior).  The fixed values are monotone (prev is a
    projected table), so each run of free points between two fixed points is an isotonic fit
    clipped to the fixed values on either side, which is the exact box-constrained solution."""
    m, weights, prev = (np.asarray(x, dtype=float) for x in (m, weights, prev))
    k = len(m)
    fixed = weights <= 0
    fixed[0] = fixed[-1] = True
    z = np.where(fixed, prev, m)
    z[0], z[-1] = 0.0, 1.0
    idx = np.flatnonzero(fixed)
    for left, right in zip(idx[:-1], idx[1:]):
        if right - left < 2:
            continue
        free = slice(left + 1, right)
        z[free] = np.clip(isotonic_increasing(m[free], weights[free]), z[left], z[right])
    assert len(z) == k
    return z


def pooled_target(values: dict[str, np.ndarray], counts: dict[str, np.ndarray],
                  prev: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Count-weighted pooling of the uploaded curve values, then the monotone projection.

    Returns (raw m, with prev where nobody has evidence; projected z; diagnostics)."""
    clients = sorted(values)
    total = sum(counts[c] for c in clients)
    num = sum(counts[c] * values[c] for c in clients)
    covered = total > 0
    m = np.where(covered, num / np.where(covered, total, 1.0), prev)
    z = project_monotone(m, np.where(covered, total, 0.0), prev)
    interior = covered.copy()
    interior[0] = interior[-1] = False
    adj = np.abs(z - m)[interior]
    diag = {
        "proj_adjust_max": float(adj.max()) if adj.size else 0.0,
        "proj_adjust_wrms": float(np.sqrt((total[interior] * adj ** 2).sum() / total[interior].sum()))
        if adj.size else 0.0,
        "uncovered": [int(k) for k in np.flatnonzero(~covered)],
    }
    return m, z, diag


# --------------------------------------------------------------------------- consensus mode
# ``fed.calibration: consensus`` (NR-56): the reverse of borrowing.  Each client is tied to the
# server table only WHERE IT HAS DATA, nothing outside its support; inference uses the table
# (g-bar) for every client, so a client's out-of-support warp values are never used.

def saturating_weights(counts: dict[str, np.ndarray], tau: float, scale: float = 1.0
                       ) -> dict[str, np.ndarray]:
    """scale * c / (tau + c): ~1 wherever a client has much nearby data, 0 where it has none,
    so one dense client cannot dominate a pooled value (equal-client spirit)."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    return {c: scale * v / (tau + v) for c, v in counts.items()}
