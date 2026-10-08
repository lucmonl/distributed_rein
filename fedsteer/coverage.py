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


# ----------------------------------------------------------------------------- aligned mode
# ``fed.calibration: aligned`` (NR-58): train AND infer with the same pooled curve g-bar.
# In client i's forward pass g-bar is built from its own live warp values (with gradient) and the
# other clients' values frozen from the last round; the identical operator, applied to everyone's
# final values, is the inference map.  One function below is used on both sides.

def isotonic_blocks(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Block label per point of the weighted pool-adjacent-violators solution (points with
    the same label share one fitted value: their weighted mean)."""
    blocks = []                                   # [mean, weight, first, last]
    for k, (yk, wk) in enumerate(zip(y, w)):
        blocks.append([yk, wk, k, k])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, _, l2 = blocks.pop()
            m1, w1, f1, _ = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, f1, l2])
    lab = np.empty(len(y), dtype=np.int64)
    for b, (_, _, f, l) in enumerate(blocks):
        lab[f:l + 1] = b
    return lab


def pool_and_project(num: "torch.Tensor", den: "torch.Tensor") -> tuple["torch.Tensor", dict]:
    """g-bar on the grid from pooled sums: num [L, K] = sum_j w_jk g_j(a_k), den [K] = sum_j w_jk.

    m = num / den, then a weighted isotonic projection per layer (weights den) that passes
    gradients: block membership is found without gradient, each block's value is the
    weighted mean of its members WITH gradient, so where nothing violates monotonicity the
    projection is the identity.  Grid points nobody covers (den = 0) are linearly interpolated
    from their covered neighbours; the endpoints are exactly 0 and 1 (every warp has them)."""
    import torch
    L, K = num.shape
    covered = den > 0
    if not bool(covered.any()):
        raise ValueError("no client covers any grid point")
    m = num / den.clamp_min(1e-12)
    w = den.detach().double().cpu().numpy()
    md = m.detach().double().cpu().numpy()
    cov_idx = np.flatnonzero(w > 0)
    labels = np.stack([isotonic_blocks(md[l, cov_idx], w[cov_idx]) for l in range(L)])     # [L, Kc]
    flat = torch.as_tensor(labels + np.arange(L)[:, None] * K, device=num.device).reshape(-1)
    mc, wc = m[:, cov_idx], den[cov_idx].expand(L, -1)
    s_w = torch.zeros(L * K, device=num.device, dtype=m.dtype).index_add_(0, flat, wc.reshape(-1))
    s_wm = torch.zeros(L * K, device=num.device, dtype=m.dtype).index_add_(0, flat, (wc * mc).reshape(-1))
    zc = (s_wm / s_w.clamp_min(1e-12))[flat].reshape(L, -1)                               # [L, Kc]
    if len(cov_idx) == K:
        z = zc
    else:                                         # fill uncovered points by linear interpolation
        cols = []
        for k in range(K):
            if covered[k]:
                cols.append(zc[:, int(np.flatnonzero(cov_idx == k)[0])])
                continue
            lo = cov_idx[cov_idx < k]
            hi = cov_idx[cov_idx > k]
            if len(lo) == 0 or len(hi) == 0:      # uncovered end: the fixed endpoint value
                cols.append(torch.full((L,), 0.0 if len(lo) == 0 else 1.0, device=num.device, dtype=m.dtype))
                continue
            a, b = lo[-1], hi[0]
            ia, ib = int(np.flatnonzero(cov_idx == a)[0]), int(np.flatnonzero(cov_idx == b)[0])
            t = (k - a) / (b - a)
            cols.append(zc[:, ia] * (1 - t) + zc[:, ib] * t)
        z = torch.stack(cols, dim=1)
    # exact endpoints (pooled values are 0 and 1 up to rounding)
    z = torch.cat([torch.zeros(L, 1, device=z.device, dtype=z.dtype), z[:, 1:-1],
                   torch.ones(L, 1, device=z.device, dtype=z.dtype)], dim=1)
    adj = (z - m).detach().abs()[:, covered]
    diag = {"proj_adjust_max": float(adj.max()) if adj.numel() else 0.0,
            "proj_layers": int((adj.max(dim=1).values > 1e-6).sum()) if adj.numel() else 0,
            "uncovered": [int(k) for k in np.flatnonzero(w <= 0)]}
    return z, diag
