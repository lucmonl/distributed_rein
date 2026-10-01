"""Per-component regularizers (config section ``reg:``; every weight defaults to 0).

component            term             what it does
-------------------  ---------------  -------------------------------------------------------
private adapter P_i  private_wd       decoupled weight decay on A_p and B_p.  L2 on both LoRA
                                      factors equals a nuclear-norm (low-rank) penalty on
                                      the product B_p A_p: it simplifies P_i, not just scales it
                     decorr           mean over layers of cos^2(P_i, D) between the update
                                      matrices, so P_i cannot carry the attribute along D
shared direction D   fedprox_mu       (mu / 2) ||B_d - B_d^server||^2 during local steps
                                      (FedProx, Li et al. 2020): limits client drift per round
                     shared_wd        decoupled weight decay on B_d.  A_d is frozen and random,
                                      so this approximates a norm penalty on D and moves scale
                                      into the gains
gain s_i = exp(u_i)  gain_l2          lambda * u_i^2   (prior toward s_i = 1)
offset o_i           offset_l2        lambda * o_i^2   (prior toward o_i = 0)
warp h_i             fed.warp_reg     (existing) penalty toward the identity

LoRA dropout is ``lora.dropout`` (applied to the layer input of both branches).
All products are evaluated through r x r matrices, never as full d x d updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .lora import SteerLinear


@dataclass
class RegConfig:
    private_wd: float = 0.0
    shared_wd: float = 0.0
    decorr: float = 0.0
    fedprox_mu: float = 0.0
    gain_l2: float = 0.0
    offset_l2: float = 0.0

    def any_penalty(self) -> bool:
        return any(v > 0 for v in (self.decorr, self.fedprox_mu, self.gain_l2, self.offset_l2))


def _prod_inner(b1, a1, b2, a2) -> torch.Tensor:
    """<B1 A1, B2 A2>_F = tr((B1^T B2)(A2 A1^T)) using only r x r products."""
    return torch.trace((b1.T @ b2) @ (a2 @ a1.T))


def decorrelation(layers: list[SteerLinear], eps: float = 1e-12) -> torch.Tensor:
    """Mean over layers of cos^2 between the private update P = B_p A_p and D = B_d A_d."""
    vals = []
    for m in layers:
        bp, ap = m.lora_B_p.float(), m.lora_A_p.float()
        bd, ad = m.lora_B_d.float(), m.lora_A_d.float()
        inner = _prod_inner(bp, ap, bd, ad)
        n_p = _prod_inner(bp, ap, bp, ap)
        n_d = _prod_inner(bd, ad, bd, ad)
        vals.append(inner ** 2 / (n_p * n_d + eps))
    return torch.stack(vals).mean()


def fedprox(layers: list[SteerLinear], anchors: list[torch.Tensor]) -> torch.Tensor:
    """(1/2) sum ||B_d - anchor||^2 (multiply by mu outside)."""
    return 0.5 * sum(((m.lora_B_d.float() - a) ** 2).sum() for m, a in zip(layers, anchors))


def penalty_terms(reg: RegConfig, layers: list[SteerLinear], control, anchors: Optional[list[torch.Tensor]],
                  gain_active: bool, offset_active: bool) -> dict[str, torch.Tensor]:
    """Unweighted penalty values for the active terms (weights applied by the caller)."""
    out = {}
    if reg.decorr > 0:
        out["decorr"] = decorrelation(layers)
    if reg.fedprox_mu > 0 and anchors is not None:
        out["fedprox"] = fedprox(layers, anchors)
    if reg.gain_l2 > 0 and gain_active:
        out["gain_l2"] = control.u.float() ** 2
    if reg.offset_l2 > 0 and offset_active and control.o is not None:
        out["offset_l2"] = control.o.float() ** 2
    return out


WEIGHT = {"decorr": "decorr", "fedprox": "fedprox_mu", "gain_l2": "gain_l2", "offset_l2": "offset_l2"}
