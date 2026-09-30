"""Steering LoRA layers.

Every targeted ``nn.Linear`` becomes

    y = W0 x + scale_p * B_p A_p x + s * alpha * scale_d * B_d A_d x

* ``A_p, B_p``  private adapter of the current client (trainable).
* ``A_d``       shared direction factor, generated from a common seed and frozen,
                so FedAvg on ``B_d`` averages the full products exactly.
* ``B_d``       shared direction factor (trainable, aggregated by the server).
* ``s``         private positive gain of the current client, ``exp(u)`` clamped.
* ``alpha``     per-example control value in [0, 1], read from ``SteerControl``.

With a warp (``SteerLoraConfig.warp``, see fedsteer/warp.py) the coefficient
``s * alpha`` becomes ``s * h(alpha)`` with a private monotone h, h(0)=0, h(1)=1.
With ``SteerLoraConfig.offset`` it becomes ``o + s * h(alpha)``: a private offset o
sets where along the shared direction the client's alpha = 0 starts.

The layers read ``alpha`` and ``s`` from a single ``SteerControl`` module that is
registered once on the model, so no model forward signature has to change.
"""

from __future__ import annotations

import math
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence, Union

import torch
from torch import nn
import torch.nn.functional as F

from .warp import AlphaWarp, make_warp

PRIVATE_KEYS = ("lora_A_p", "lora_B_p")
SHARED_KEYS = ("lora_B_d",)
GAIN_KEY = "steer_control.u"
OFFSET_KEY = "steer_control.o"
CONTROL_PREFIX = "steer_control."          # gain u and warp parameters: private per client
WARP_PREFIX = "steer_control.warp."


@dataclass
class SteerLoraConfig:
    target_modules: Sequence[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    rank_private: int = 16
    rank_shared: int = 16
    lora_alpha_private: float = 16.0
    lora_alpha_shared: float = 16.0
    dropout: float = 0.0
    shared_seed: int = 1234  # every client must use the same seed so A_d is identical
    gain_min: float = 0.25
    gain_max: float = 4.0
    # private monotone reparameterization of alpha (fedsteer/warp.py):
    # none | kumaraswamy (A) | kumaraswamy_mix (B) | step (C)
    warp: str = "none"
    offset: bool = False             # private offset o_i: coefficient = o_i + s_i * h_i(alpha)
    offset_max: float = 2.0          # |o_i| is clamped to this (coefficient units)
    warp_shape_min: float = 0.2      # Kumaraswamy p, q range
    warp_shape_max: float = 20.0
    warp_mix_w_init: float = 0.5     # B: initial mixture weight (identity holds for any w at init)
    warp_step_c_init: float = 0.5    # C: initial switch location
    warp_step_tau_init: float = 0.2  # C: initial step width
    warp_step_tau_min: float = 0.02
    warp_step_tau_max: float = 1.0


class SteerControl(nn.Module):
    """Holds the client's private control parameters (gain u, warp h) and the alpha
    of the current batch."""

    def __init__(self, gain_min: float, gain_max: float, warp: Optional[AlphaWarp] = None,
                 offset: bool = False, offset_max: float = 2.0):
        super().__init__()
        self.u = nn.Parameter(torch.zeros(()))
        self.o = nn.Parameter(torch.zeros(())) if offset else None
        self.offset_max = offset_max
        self.log_min = math.log(gain_min)
        self.log_max = math.log(gain_max)
        self.warp = warp if warp is not None else AlphaWarp()
        self._alpha: Optional[torch.Tensor] = None

    def gain(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.u, self.log_min, self.log_max))

    @contextmanager
    def use_alpha(self, alpha: Union[float, torch.Tensor]) -> Iterator[None]:
        """Set alpha for everything run inside the block (forward *and* backward,
        which matters for non-reentrant gradient checkpointing)."""
        prev = self._alpha
        self._alpha = torch.as_tensor(alpha, dtype=torch.float32).reshape(-1)
        try:
            yield
        finally:
            self._alpha = prev

    def offset(self) -> torch.Tensor:
        if self.o is None:
            return torch.zeros((), device=self.u.device)
        return self.o.clamp(-self.offset_max, self.offset_max)

    def coef_for(self, y: torch.Tensor) -> torch.Tensor:
        """Per-example coefficient o + s * h(alpha), broadcastable against ``y``."""
        coef = self.alpha_for(y, warped=True) * self.gain()
        return coef + self.offset() if self.o is not None else coef

    def alpha_for(self, y: torch.Tensor, warped: bool = False) -> torch.Tensor:
        if self._alpha is None:
            raise RuntimeError("alpha is not set; wrap the forward in `control.use_alpha(...)`")
        a = self._alpha.to(device=y.device)
        if warped:
            a = self.warp(a)
        if a.numel() == 1:
            return a.reshape(())
        batch = y.shape[0]
        if a.numel() != batch:
            if batch % a.numel() != 0:
                raise ValueError(f"alpha has {a.numel()} values but batch is {batch}")
            # generate(num_return_sequences>1) repeats each prompt contiguously
            a = a.repeat_interleave(batch // a.numel())
        return a.reshape(batch, *([1] * (y.dim() - 1)))


def _shared_init(out_rank: int, in_features: int, seed: int, name: str) -> torch.Tensor:
    """Deterministic A_d from (seed, module name): identical on every client."""
    gen = torch.Generator().manual_seed(seed * 1_000_003 + zlib.crc32(name.encode()))
    bound = 1.0 / math.sqrt(in_features)  # same range as kaiming_uniform_(a=sqrt(5))
    return (torch.rand(out_rank, in_features, generator=gen) * 2 - 1) * bound


class SteerLinear(nn.Module):
    def __init__(self, base: nn.Linear, name: str, cfg: SteerLoraConfig, control: SteerControl):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        object.__setattr__(self, "_control", control)  # not a submodule: registered once on the model
        in_f, out_f = base.in_features, base.out_features
        dev = base.weight.device

        self.lora_A_p = nn.Parameter(torch.empty(cfg.rank_private, in_f, device=dev))
        self.lora_B_p = nn.Parameter(torch.zeros(out_f, cfg.rank_private, device=dev))
        nn.init.kaiming_uniform_(self.lora_A_p, a=math.sqrt(5))
        self.scale_p = cfg.lora_alpha_private / cfg.rank_private

        self.register_buffer("lora_A_d", _shared_init(cfg.rank_shared, in_f, cfg.shared_seed, name).to(dev))
        self.lora_B_d = nn.Parameter(torch.zeros(out_f, cfg.rank_shared, device=dev))
        self.scale_d = cfg.lora_alpha_shared / cfg.rank_shared

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        h = self.dropout(x)
        dt = self.lora_A_p.dtype if not torch.is_autocast_enabled() else h.dtype
        h = h.to(dt)
        priv = F.linear(F.linear(h, self.lora_A_p.to(dt)), self.lora_B_p.to(dt)) * self.scale_p
        shared = F.linear(F.linear(h, self.lora_A_d.to(dt)), self.lora_B_d.to(dt)) * self.scale_d
        coef = self._control.coef_for(shared)
        return y + (priv + coef.to(shared.dtype) * shared).to(y.dtype)

    def extra_repr(self) -> str:
        return (f"in={self.base.in_features}, out={self.base.out_features}, "
                f"r_p={self.lora_A_p.shape[0]}, r_d={self.lora_A_d.shape[0]}")


def inject_steer_lora(model: nn.Module, cfg: SteerLoraConfig) -> SteerControl:
    """Freeze the model and replace targeted Linear layers with SteerLinear."""
    for p in model.parameters():
        p.requires_grad_(False)
    warp = make_warp(cfg.warp, shape_min=cfg.warp_shape_min, shape_max=cfg.warp_shape_max,
                     mix_w_init=cfg.warp_mix_w_init, step_c_init=cfg.warp_step_c_init,
                     step_tau_init=cfg.warp_step_tau_init, step_tau_min=cfg.warp_step_tau_min,
                     step_tau_max=cfg.warp_step_tau_max)
    control = SteerControl(cfg.gain_min, cfg.gain_max, warp, offset=cfg.offset,
                           offset_max=cfg.offset_max).to(next(model.parameters()).device)
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in cfg.target_modules:
            targets.append(name)
    if not targets:
        raise ValueError(f"no Linear modules matched {cfg.target_modules}")
    for name in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, SteerLinear(getattr(parent, child), name, cfg, control))
    model.steer_control = control
    return control


def steer_layers(model: nn.Module) -> Iterator[tuple[str, SteerLinear]]:
    for name, m in model.named_modules():
        if isinstance(m, SteerLinear):
            yield name, m


# ---------------------------------------------------------------------------
# State partitioning.  Everything is returned as detached CPU float32 tensors.
# ---------------------------------------------------------------------------

def _collect(model: nn.Module, keys: Sequence[str]) -> dict[str, torch.Tensor]:
    return {n: p.detach().float().cpu().clone()
            for n, p in model.named_parameters() if n.split(".")[-1] in keys}


def get_shared_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return _collect(model, SHARED_KEYS)


def get_private_adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return _collect(model, PRIVATE_KEYS)


def get_gain_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """All private control parameters: the gain ``u`` and, if configured, the warp."""
    return {n: p.detach().float().cpu().clone()
            for n, p in model.named_parameters() if n.startswith(CONTROL_PREFIX)}


@torch.no_grad()
def load_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    for n, v in state.items():
        if n not in params:
            raise KeyError(f"unknown parameter {n}")
        params[n].copy_(v.to(device=params[n].device, dtype=params[n].dtype))


def average_states(states: Sequence[dict[str, torch.Tensor]],
                   weights: Optional[Sequence[float]] = None) -> dict[str, torch.Tensor]:
    """Weighted average of state dicts.  With A_d frozen and shared, averaging B_d
    is exactly averaging the direction products B_d A_d."""
    if weights is None:
        weights = [1.0] * len(states)
    tot = float(sum(weights))
    return {k: sum(w * s[k] for w, s in zip(weights, states)) / tot for k in states[0]}


def direction_products(model: nn.Module, shared_state: Optional[dict[str, torch.Tensor]] = None
                       ) -> dict[str, torch.Tensor]:
    """Effective per-layer direction scale_d * B_d A_d (for diagnostics / normalization)."""
    out = {}
    for name, m in steer_layers(model):
        b = m.lora_B_d.detach().float().cpu() if shared_state is None else shared_state[f"{name}.lora_B_d"]
        out[name] = m.scale_d * b @ m.lora_A_d.float().cpu()
    return out


def trainable_parameter_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    groups: dict[str, list[nn.Parameter]] = {"private": [], "shared": [], "gain": [], "warp": []}
    for n, p in model.named_parameters():
        key = n.split(".")[-1]
        if key in PRIVATE_KEYS:
            groups["private"].append(p)
        elif key in SHARED_KEYS:
            groups["shared"].append(p)
        elif n in (GAIN_KEY, OFFSET_KEY):
            groups["gain"].append(p)
        elif n.startswith(WARP_PREFIX):
            groups["warp"].append(p)
    return groups
