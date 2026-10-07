"""Private monotone reparameterizations of the control coordinate.

The coefficient on the shared direction for client i is  g_i(alpha) = s_i * h_i(alpha),
where s_i is the private gain and h_i: [0,1] -> [0,1] is increasing with h(0) = 0 and
h(1) = 1.  s_i says how far along D a client's range reaches; h_i says how it moves
along it (e.g. where rewriting switches to copying).  Every warp starts as the
identity, so ``warp: none`` is the special case of the original linear model.

  none              h(a) = a
  kumaraswamy (A)   h(a) = 1 - (1 - a^p)^q                          params p, q > 0
  kumaraswamy_mix (B, recommended)
                    h(a) = (1 - w) a + w [1 - (1 - a^p)^q]           params w in (0,1), p, q
  step (C)          h(a) = (1 - w) a + w S((a - c) / tau)            params w in [0,1], c in (0,1), tau
                    with S a logistic normalized to S(0) = 0, S(1) = 1

(The Kumaraswamy shape parameters are called p, q here to avoid clashing with alpha.)
"""

from __future__ import annotations

import math

import torch
from torch import nn

WARP_KINDS = ("none", "kumaraswamy", "kumaraswamy_mix", "step")


def _safe_pow(base: torch.Tensor, expo: torch.Tensor) -> torch.Tensor:
    """base**expo for base in [0, 1] with finite gradients at base = 0
    (d/d expo of 0**expo would be 0 * log 0 = nan)."""
    pos = base > 0
    safe = torch.where(pos, base, torch.ones_like(base))
    return torch.where(pos, safe ** expo, torch.zeros_like(base))


class AlphaWarp(nn.Module):
    kind = "none"
    grid = torch.linspace(0, 1, 33)

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        return alpha

    def penalty(self) -> torch.Tensor:
        """Mean squared deviation from the identity on a grid of alphas."""
        g = self.grid.to(self._device())
        return ((self(g) - g) ** 2).mean()

    def params_dict(self) -> dict:
        return {}

    @torch.no_grad()
    def describe(self, points=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
        x = torch.tensor(points, device=self._device())
        return {"kind": self.kind, **{k: round(float(v), 4) for k, v in self.params_dict().items()},
                "curve": {str(p): round(float(v), 4) for p, v in zip(points, self(x))}}

    def _device(self):
        ps = list(self.parameters())
        return ps[0].device if ps else torch.device("cpu")


class KumaraswamyWarp(AlphaWarp):
    """A: CDF of the Kumaraswamy distribution.  Identity at p = q = 1."""
    kind = "kumaraswamy"

    def __init__(self, shape_min: float = 0.2, shape_max: float = 20.0):
        super().__init__()
        self.log_p = nn.Parameter(torch.zeros(()))
        self.log_q = nn.Parameter(torch.zeros(()))
        self.lo, self.hi = math.log(shape_min), math.log(shape_max)

    def shapes(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.exp(self.log_p.clamp(self.lo, self.hi)), torch.exp(self.log_q.clamp(self.lo, self.hi)))

    def kumaraswamy(self, alpha: torch.Tensor) -> torch.Tensor:
        p, q = self.shapes()
        x = alpha.float().clamp(0.0, 1.0)
        return 1.0 - _safe_pow(1.0 - _safe_pow(x, p), q)

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        return self.kumaraswamy(alpha)

    def params_dict(self) -> dict:
        p, q = self.shapes()
        return {"p": p, "q": q}


class KumaraswamyMixWarp(KumaraswamyWarp):
    """B: linear part plus a Kumaraswamy bend.  Identity at p = q = 1 for any w, so w
    starts at 0.5 (logit 0) with a live gradient path through p and q."""
    kind = "kumaraswamy_mix"

    def __init__(self, shape_min: float = 0.2, shape_max: float = 20.0, w_init: float = 0.5):
        super().__init__(shape_min, shape_max)
        self.w_logit = nn.Parameter(torch.tensor(math.log(w_init / (1 - w_init))))

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        w = torch.sigmoid(self.w_logit)
        x = alpha.float().clamp(0.0, 1.0)
        return (1 - w) * x + w * self.kumaraswamy(x)

    def params_dict(self) -> dict:
        return {**super().params_dict(), "w": torch.sigmoid(self.w_logit)}


class StepWarp(AlphaWarp):
    """C: linear part plus one normalized logistic step at c with width tau.
    w is a hard-clamped weight starting at exactly 0 (identity)."""
    kind = "step"

    def __init__(self, c_init: float = 0.5, tau_init: float = 0.2, tau_min: float = 0.02, tau_max: float = 1.0):
        super().__init__()
        self.w_raw = nn.Parameter(torch.zeros(()))
        self.c_logit = nn.Parameter(torch.tensor(math.log(c_init / (1 - c_init))))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        self.tlo, self.thi = math.log(tau_min), math.log(tau_max)

    def _parts(self):
        w = self.w_raw.clamp(0.0, 1.0)
        c = torch.sigmoid(self.c_logit)
        tau = torch.exp(self.log_tau.clamp(self.tlo, self.thi))
        return w, c, tau

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        w, c, tau = self._parts()
        x = alpha.float().clamp(0.0, 1.0)
        s0, s1 = torch.sigmoid(-c / tau), torch.sigmoid((1 - c) / tau)
        step = (torch.sigmoid((x - c) / tau) - s0) / (s1 - s0)
        return (1 - w) * x + w * step

    def params_dict(self) -> dict:
        w, c, tau = self._parts()
        return {"w": w, "c": c, "tau": tau}


def make_warp(kind: str, **kw) -> AlphaWarp:
    if kind == "none":
        return AlphaWarp()
    if kind == "kumaraswamy":
        return KumaraswamyWarp(kw.get("shape_min", 0.2), kw.get("shape_max", 20.0))
    if kind == "kumaraswamy_mix":
        return KumaraswamyMixWarp(kw.get("shape_min", 0.2), kw.get("shape_max", 20.0), kw.get("mix_w_init", 0.5))
    if kind == "step":
        return StepWarp(kw.get("step_c_init", 0.5), kw.get("step_tau_init", 0.2),
                        kw.get("step_tau_min", 0.02), kw.get("step_tau_max", 1.0))
    raise ValueError(f"unknown warp {kind!r}; choose from {WARP_KINDS}")


class WarpBank(nn.Module):
    """Independent warps for different adapted layers (``SteerLoraConfig.warp_scope``).

    Layer l of client i then uses g_{i,l}(alpha) = s_i * h_{i,l}(alpha): the shared direction
    is applied with a layer-specific coefficient.  Every member is an ordinary warp of the
    same kind with its own parameters (identity at init), stored as ``bank.<l>.<param>``."""

    def __init__(self, warps: list[AlphaWarp], scope: str):
        super().__init__()
        if not warps:
            raise ValueError("WarpBank needs at least one warp")
        self.bank = nn.ModuleList(warps)
        self.kind = warps[0].kind
        self.scope = scope

    def __len__(self) -> int:
        return len(self.bank)

    def forward(self, alpha: torch.Tensor, idx: int) -> torch.Tensor:
        return self.bank[idx](alpha)

    def all_on(self, x: torch.Tensor) -> torch.Tensor:
        """Every member evaluated at x: shape [n_warps, len(x)]."""
        return torch.stack([w(x) for w in self.bank])

    def penalty(self) -> torch.Tensor:
        return torch.stack([w.penalty() for w in self.bank]).mean()

    @torch.no_grad()
    def describe(self, points=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
        x = torch.tensor(points, device=self._device())
        v = self.all_on(x)
        r = lambda t: {str(p): round(float(y), 4) for p, y in zip(points, t)}
        return {"kind": self.kind, "scope": self.scope, "n_warps": len(self.bank),
                "curve": r(v.mean(0)), "curve_min": r(v.min(0).values), "curve_max": r(v.max(0).values)}

    def _device(self):
        return self.bank[0]._device()
