"""Federated training of a shared steering direction.

One simulated federation runs on one device: a single model instance holds the
frozen base, and each client's private state (adapter, gain, optimizer state,
data-stream position) is swapped in before its local steps and saved after.

Modes
-----
* ``fedavg``  the proposed method: clients train (P_i, u_i, B_d) locally for
              ``local_steps``; the server averages B_d.
* ``local``   baseline B2: identical local training, no aggregation; each client
              keeps its own B_d.  The one-shot merged direction (baseline B3) is
              the average of these, written at the end of training.

Private alpha warp (``lora.warp``, fedsteer/warp.py): trained like the gain, with
its own learning rate, a warm-up and a penalty toward the identity (``warp_reg``).

Adapter mode (``adapter``): ``private`` (default; P_i stays on the client), ``shared``
(P is aggregated like the direction; ablation A2, formerly ``share_private``) or
``none`` (no task adapter; only the base model, the direction and private scalars).

Calibration mode (``calibration``): the coefficient g(alpha) = o + s * h(alpha) on the
shared direction (gain s, optional offset o, warp h) is either ``private`` (one per
client, fitted on its own data) or ``shared`` (one for all clients, averaged by the
server every round like the direction).  Under the global alpha scale, ``shared``
keeps everything about the attribute shared and only house style private (P_i).
``shared`` is undefined in ``local`` mode (nothing is aggregated there).
``coverage`` (plan section 2.1, fedsteer/coverage.py): gain fixed at 1, no offset, a private
nonlinear warp per client, plus a borrowing term that pulls each warp toward a server table
of count-weighted, monotone-projected warp values on a common grid, wherever the client has
little nearby data and its peers have much.  Warp parameters are never averaged.

Ablation (orthogonal to mode): ``fix_gain`` (A1: s_i = 1).
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import torch

from . import coverage as cov
from .data import ClientStream
from .monitor import format_monitor
from .regularize import WEIGHT, RegConfig, penalty_terms
from .lora import (
    GAIN_KEY,
    steer_layers,
    average_states,
    get_gain_state,
    get_private_adapter_state,
    get_shared_state,
    load_state,
    trainable_parameter_groups,
)


@dataclass
class FedConfig:
    mode: str = "fedavg"             # fedavg | local
    rounds: int = 50
    local_steps: int = 20            # optimizer steps per client per round
    clients_per_round: int = 0       # 0 = full participation
    batch_size: int = 4
    grad_accum: int = 1
    lr_private: float = 2e-4
    lr_shared: float = 2e-4
    lr_gain: float = 1e-2
    weight_decay: float = 0.0
    warmup_steps: int = 20           # per-client local optimizer steps
    lr_schedule: str = "constant"    # constant | cosine (over rounds * local_steps)
    max_grad_norm: float = 1.0
    gain_warmup_rounds: int = 2      # keep u_i = 0 while D is still ~0
    fix_gain: bool = False           # ablation A1
    lr_warp: float = 1e-2            # private alpha warp (only if lora.warp != none)
    warp_warmup_rounds: int = 5      # keep h_i = identity until D carries signal
    warp_reg: float = 1e-2           # weight of mean (h(a) - a)^2 penalty toward the identity
    adapter: str = "private"         # private | shared (A2) | none
    calibration: str = "private"     # private | shared | coverage (plan 2.1: private warps + borrowing)
    cov_grid: int = 11               # coverage: K common grid points a_k, including 0 and 1
    cov_bandwidth: float = 0.2       # coverage: triangular-kernel half-width b of the counts c_ik
    cov_lambda_max: float = 1.0      # coverage: maximum weight of the borrowing term
    cov_tau_local: float = 100.0     # coverage: local-need factor is 1/2 at c_ik = tau_local
    cov_tau_peer: float = 100.0      # coverage: peer-evidence factor is 1/2 at R_ik = tau_peer
    share_private: bool = False      # deprecated alias for adapter: shared
    aggregation: str = "uniform"     # uniform | size
    server_lr: float = 1.0           # new = old + server_lr * (avg - old)
    reset_shared_opt_state: bool = True
    bf16: bool = True
    seed: int = 0
    save_every: int = 5
    keep_opt_state_on_disk: bool = True


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return copy.deepcopy(obj)


def _flat(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([state[k].reshape(-1) for k in sorted(state)])


class FedSteerTrainer:
    def __init__(self, model, formatter, client_examples: dict[str, list[dict]],
                 cfg: FedConfig, out_dir: str,
                 eval_fn: Optional[Callable[["FedSteerTrainer", int], dict]] = None,
                 reg: Optional[RegConfig] = None):
        if cfg.mode not in ("fedavg", "local"):
            raise ValueError(f"unknown mode {cfg.mode}")
        self.model = model
        self.cfg = cfg
        self.reg = reg or RegConfig()
        self.skipped_steps: dict[str, int] = {}   # NaN guard: steps skipped per client
        self.layers = [m for _, m in steer_layers(model)]
        self.out_dir = out_dir
        self.eval_fn = eval_fn
        self.device = next(model.parameters()).device
        self.control = model.steer_control
        self.client_ids = sorted(client_examples)
        self.n_examples = {c: len(client_examples[c]) for c in self.client_ids}
        self.rng = random.Random(cfg.seed)
        os.makedirs(out_dir, exist_ok=True)

        self.adapter = "shared" if cfg.share_private else cfg.adapter
        if self.adapter not in ("private", "shared", "none"):
            raise ValueError(f"unknown adapter mode {cfg.adapter}")
        self.calibration = cfg.calibration
        if self.calibration not in ("private", "shared", "coverage"):
            raise ValueError(f"unknown calibration mode {cfg.calibration}")
        if self.calibration == "shared" and cfg.mode == "local":
            raise ValueError("calibration=shared is undefined in local mode (nothing is aggregated); "
                             "local training always has per-client calibration: use calibration=private")
        groups = trainable_parameter_groups(model)
        if self.adapter == "none":
            with torch.no_grad():
                for p in groups["private"]:
                    p.requires_grad_(False)
                for name, p in model.named_parameters():
                    if name.endswith("lora_B_p"):
                        p.zero_()                      # B_p = 0: the adapter contributes nothing
        self.shared_params = groups["shared"] + (groups["private"] if self.adapter == "shared" else []) \
            + (groups["gain"] + groups["warp"] if self.calibration == "shared" else [])
        self.opt = torch.optim.AdamW(
            [
                {"params": groups["private"], "lr": cfg.lr_private,
                 "weight_decay": cfg.weight_decay + self.reg.private_wd, "name": "private"},
                {"params": groups["shared"], "lr": cfg.lr_shared,
                 "weight_decay": cfg.weight_decay + self.reg.shared_wd, "name": "shared"},
                {"params": groups["gain"], "lr": cfg.lr_gain, "weight_decay": 0.0, "name": "gain"},
            ] + ([{"params": groups["warp"], "lr": cfg.lr_warp, "weight_decay": 0.0, "name": "warp"}]
                 if groups["warp"] else [])
        )
        self.warp_params = groups["warp"]
        self._base_lrs = [g["lr"] for g in self.opt.param_groups]

        self.streams = {
            c: ClientStream(client_examples[c], formatter, cfg.batch_size, seed=cfg.seed * 7919 + i)
            for i, c in enumerate(self.client_ids)
        }
        # Server state.  All clients start from the same private initialization so
        # that the only initial difference between clients is their data.
        self.server = self._server_state_from_model()
        init_private = get_private_adapter_state(model) if self.adapter == "private" else {}
        self.clients: dict[str, dict] = {
            c: {"private": copy.deepcopy(init_private), "gain": get_gain_state(model),
                "shared_local": None, "opt": None, "steps": 0}
            for c in self.client_ids
        }
        self.round = 0
        self.history: list[dict] = []
        self.cov: Optional[dict] = None
        if self.calibration == "coverage":
            self._check_coverage_config()
            self.cov = self._init_coverage(client_examples)
            with open(os.path.join(out_dir, "coverage.json"), "w") as f:
                json.dump(self.cov, f, indent=1)

    # ---------------------------------------------------------------- coverage
    def _check_coverage_config(self) -> None:
        cfg = self.cfg
        problems = []
        if cfg.mode != "fedavg":
            problems.append("mode must be fedavg (local-only has no peers; use calibration=private)")
        if not cfg.fix_gain:
            problems.append("fed.fix_gain must be true (gain fixed at 1)")
        if self.control.o is not None:
            problems.append("lora.offset must be false")
        if self.control.warp.kind == "none":
            problems.append("lora.warp must be a nonlinear warp (e.g. kumaraswamy_mix)")
        if 0 < cfg.clients_per_round < len(self.client_ids):
            problems.append("full participation only (clients_per_round = 0)")
        if float(self.control.u.detach()) != 0.0:
            problems.append("gain parameter u must be 0 (s = 1) at initialization")
        if problems:
            raise ValueError("calibration=coverage: " + "; ".join(problems))

    def _init_coverage(self, client_examples: dict[str, list[dict]]) -> dict:
        """Grid, evidence counts c_ik from each client's actual training subset (once), the
        borrowing weights lambda_ik, and the identity as the initial server table z."""
        cfg = self.cfg
        grid = cov.make_grid(cfg.cov_grid)
        counts = {c: cov.evidence_counts([e["alpha"] for e in client_examples[c]], grid, cfg.cov_bandwidth)
                  for c in self.client_ids}
        lam = cov.borrow_weights(counts, cfg.cov_lambda_max, cfg.cov_tau_local, cfg.cov_tau_peer)
        n_warps = int(self.control.warp_on(torch.tensor(grid, dtype=torch.float32,
                                                        device=self.control.warp._device())).shape[0])
        return {
            "grid": grid.tolist(), "bandwidth": cfg.cov_bandwidth, "lambda_max": cfg.cov_lambda_max,
            "tau_local": cfg.cov_tau_local, "tau_peer": cfg.cov_tau_peer,
            "counts": {c: v.tolist() for c, v in counts.items()},
            "lambda": {c: v.tolist() for c, v in lam.items()},
            "n_warps": n_warps,          # one table row per warp (per layer with lora.warp_scope)
            "z": [grid.tolist()] * n_warps,   # server table [n_warps][K], starts at the identity
            "ready": False,              # True once a round with trainable warps has completed
            "updates": 0,                # rounds that updated z
        }

    def _warp_on_grid(self) -> torch.Tensor:
        """This client's warps on the grid: [n_warps, K]."""
        g = torch.tensor(self.cov["grid"], dtype=torch.float32, device=self.control.warp._device())
        return self.control.warp_on(g)

    # ------------------------------------------------------------------ state
    def _server_state_from_model(self) -> dict[str, torch.Tensor]:
        st = get_shared_state(self.model)
        if self.adapter == "shared":
            st.update(get_private_adapter_state(self.model))
        if self.calibration == "shared":
            st.update(get_gain_state(self.model))      # gain u, offset o, warp parameters
        return st

    def load_client(self, cid: str, shared: Optional[dict] = None) -> None:
        """Put client ``cid`` into the model (for training or evaluation)."""
        c = self.clients[cid]
        if self.adapter == "private":
            load_state(self.model, c["private"])
        if self.calibration != "shared":
            load_state(self.model, c["gain"])          # shared calibration comes with the server state
        if shared is None:
            shared = c["shared_local"] if (self.cfg.mode == "local" and c["shared_local"] is not None) else self.server
        load_state(self.model, shared)

    def _save_client(self, cid: str) -> None:
        c = self.clients[cid]
        if self.adapter == "private":
            c["private"] = get_private_adapter_state(self.model)
        if self.calibration != "shared":
            c["gain"] = get_gain_state(self.model)
        if self.cfg.mode == "local":
            c["shared_local"] = self._server_state_from_model()
        c["opt"] = _to_cpu(self.opt.state_dict())

    def _load_opt(self, cid: str) -> None:
        saved = self.clients[cid]["opt"]
        self.opt.state.clear()
        if saved is not None:
            self.opt.load_state_dict(saved)
            for g, lr in zip(self.opt.param_groups, self._base_lrs):
                g["lr"] = lr
        if self.cfg.mode == "fedavg" and self.cfg.reset_shared_opt_state:
            for p in self.shared_params:
                self.opt.state.pop(p, None)

    # ---------------------------------------------------------------- training
    def _lr_factor(self, step: int) -> float:
        cfg = self.cfg
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        if cfg.lr_schedule == "cosine":
            total = max(cfg.rounds * cfg.local_steps, 1)
            prog = min((step - cfg.warmup_steps) / max(total - cfg.warmup_steps, 1), 1.0)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))
        return 1.0

    def _report_nonfinite(self, cid: str, what: str, batch: dict, alpha: torch.Tensor) -> None:
        """Print what is known about a step whose loss or gradient is non-finite (NaN guard)."""
        lens = batch["attention_mask"].sum(dim=1).tolist()
        n_tgt = (batch["labels"] != -100).sum(dim=1).tolist()
        msg = (f"[nan-guard] round {self.round} client {cid} step {self.clients[cid]['steps']}: "
               f"non-finite {what}; seq lens {lens}, target tokens {n_tgt}, alpha "
               f"{[round(float(a), 3) for a in alpha]}")
        if what == "grad":
            bad = [n for n, p in self.model.named_parameters()
                   if p.grad is not None and not torch.isfinite(p.grad).all()]
            msg += f"; {len(bad)} params with non-finite grads, e.g. {bad[:4]}"
        print(msg, flush=True)

    def _local_train(self, cid: str) -> dict:
        cfg, model, c = self.cfg, self.model, self.clients[cid]
        model.train()
        gain_trainable = not cfg.fix_gain and self.round >= cfg.gain_warmup_rounds
        self.control.u.requires_grad_(gain_trainable)
        if self.control.o is not None:                  # offset follows the gain warm-up, not fix_gain
            self.control.o.requires_grad_(self.round >= cfg.gain_warmup_rounds)
        warp_trainable = bool(self.warp_params) and self.round >= cfg.warp_warmup_rounds
        for p in self.warp_params:
            p.requires_grad_(warp_trainable)
        losses, penalties = [], []
        reg_vals: dict[str, list[float]] = {}
        # FedProx anchor: the direction as broadcast by the server at the start of this round
        anchors = ([m.lora_B_d.detach().float().clone() for m in self.layers]
                   if self.reg.fedprox_mu > 0 and cfg.mode == "fedavg" else None)
        offset_active = self.control.o is not None and self.round >= cfg.gain_warmup_rounds
        borrow_on = self.cov is not None and self.cov["ready"] and warp_trainable
        if borrow_on:
            dev = self.control.warp._device()
            z_target = torch.tensor(self.cov["z"], dtype=torch.float32, device=dev)   # detached, fixed this round
            lam = torch.tensor(self.cov["lambda"][cid], dtype=torch.float32, device=dev)
        borrows = []
        for _ in range(cfg.local_steps):
            f = self._lr_factor(c["steps"])
            for g, lr in zip(self.opt.param_groups, self._base_lrs):
                g["lr"] = lr * f
            step_loss = 0.0
            bad_loss = False
            for _ in range(cfg.grad_accum):
                batch = {k: v.to(self.device) for k, v in self.streams[cid].next_batch().items()}
                alpha = batch.pop("alpha")
                with self.control.use_alpha(alpha), \
                        torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=cfg.bf16):
                    loss = model(**batch).loss / cfg.grad_accum
                    if not torch.isfinite(loss):
                        # non-finite forward: report the batch and skip this step entirely
                        bad_loss = True
                        self._report_nonfinite(cid, "loss", batch, alpha)
                        break
                    total = loss
                    if warp_trainable and cfg.warp_reg > 0:
                        pen = self.control.warp.penalty()
                        total = total + cfg.warp_reg * pen / cfg.grad_accum
                        penalties.append(pen.item())
                    if borrow_on:
                        # (1/K) sum_k lambda_ik (h_i(a_k) - z_k)^2, averaged over the client's warps
                        # (one per layer with lora.warp_scope); reaches only the warp parameters;
                        # / grad_accum so it counts once per optimizer step
                        borrow = (lam * (self._warp_on_grid() - z_target) ** 2).mean()
                        total = total + borrow / cfg.grad_accum
                        borrows.append(borrow.item())
                    if self.reg.any_penalty():
                        terms = penalty_terms(self.reg, self.layers, self.control, anchors,
                                              gain_trainable, offset_active)
                        for name, val in terms.items():
                            total = total + getattr(self.reg, WEIGHT[name]) * val / cfg.grad_accum
                            reg_vals.setdefault(name, []).append(float(val.item()))
                    total.backward()
                step_loss += loss.item()
            params = [p for g in self.opt.param_groups for p in g["params"] if p.grad is not None]
            gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm) if not bad_loss else None
            if bad_loss or not torch.isfinite(gnorm):
                # NaN guard: a non-finite loss or gradient would be written into the weights by the
                # optimizer (clip_grad_norm_ turns an inf norm into NaN); skip the step instead
                if not bad_loss:
                    self._report_nonfinite(cid, "grad", batch, alpha)
                self.opt.zero_grad(set_to_none=True)
                c["steps"] += 1
                self.skipped_steps[cid] = self.skipped_steps.get(cid, 0) + 1
                continue
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            c["steps"] += 1
            losses.append(step_loss)
        out = {"loss": sum(losses) / len(losses) if losses else float("nan"),
               "gain": float(self.control.gain().item()),
               "skipped_steps": self.skipped_steps.get(cid, 0)}
        if self.control.o is not None:
            out["offset"] = float(self.control.offset().item())
        if self.warp_params:
            out["warp"] = self.control.warp.describe()
            if penalties:
                out["warp_penalty"] = sum(penalties) / len(penalties)
        if borrows:
            out["borrow_loss"] = sum(borrows) / len(borrows)          # weighted term, as optimized
        if reg_vals:
            out["reg"] = {k: sum(v) / len(v) for k, v in reg_vals.items()}   # unweighted term values
        return out

    def _select(self) -> list[str]:
        k = self.cfg.clients_per_round
        if k <= 0 or k >= len(self.client_ids):
            return list(self.client_ids)
        return sorted(self.rng.sample(self.client_ids, k))

    def train_round(self) -> dict:
        t0 = time.time()
        selected = self._select()
        start = copy.deepcopy(self.server)
        uploads, weights, stats, deltas = [], [], {}, []
        curves: dict[str, np.ndarray] = {}
        for cid in selected:
            self.load_client(cid)
            before = self._server_state_from_model()   # server state, or own direction in local mode
            self._load_opt(cid)
            stats[cid] = self._local_train(cid)
            if self.cov is not None:
                with torch.no_grad():
                    curves[cid] = self._warp_on_grid().double().cpu().numpy()
            uploads.append(self._server_state_from_model())
            deltas.append(_flat(uploads[-1]) - _flat(before))
            weights.append(1.0 if self.cfg.aggregation == "uniform" else float(self.n_examples[cid]))
            self._save_client(cid)

        log = {"round": self.round, "clients": stats, "time_s": time.time() - t0}

        def mean_pairwise_cos(vecs):
            d = torch.stack(vecs)
            d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
            iu = torch.triu_indices(len(vecs), len(vecs), 1)
            return float((d @ d.T)[iu[0], iu[1]].mean())

        if len(deltas) > 1:
            # agreement of this round's client updates (B_d space)
            log["client_delta_cos_mean"] = mean_pairwise_cos(deltas)
            if self.cfg.mode == "local":
                # agreement of the independently learned directions themselves
                log["client_direction_cos_mean"] = mean_pairwise_cos([_flat(u) for u in uploads])
        if self.cfg.mode == "fedavg":
            avg = average_states(uploads, weights)
            lr = self.cfg.server_lr
            self.server = {k: start[k] + lr * (avg[k] - start[k]) for k in start}
            log["server_update_norm"] = float((_flat(self.server) - _flat(start)).norm())
        if self.cov is not None:
            log["coverage"] = self._update_coverage(curves)
        log["direction_norm"] = float(_flat({k: v for k, v in self.server.items() if k.endswith("lora_B_d")}).norm()) \
            if self.cfg.mode == "fedavg" else None
        self.round += 1
        return log

    def _update_coverage(self, curves: dict[str, np.ndarray]) -> dict:
        """Server side of plan 2.1: pool the uploaded grid values (count-weighted), project to a
        monotone table, broadcast it next round.  Only rounds with trainable warps update it."""
        rnd = lambda v: [round(float(x), 5) for x in v]
        multi = self.cov["n_warps"] > 1

        def summary(key, rows):                         # log the mean over layers (+ spread)
            rows = np.asarray(rows)
            res = {key: rnd(rows.mean(0))}
            if multi:
                res.update({f"{key}_min": rnd(rows.min(0)), f"{key}_max": rnd(rows.max(0))})
            return res

        out = {"borrow_active": self.cov["ready"] and self.round >= self.cfg.warp_warmup_rounds,
               "values": {c: rnd(v.mean(0)) for c, v in curves.items()}}
        if multi:
            out["values_min"] = {c: rnd(v.min(0)) for c, v in curves.items()}
            out["values_max"] = {c: rnd(v.max(0)) for c, v in curves.items()}
        if self.round < self.cfg.warp_warmup_rounds:
            return out                                  # warps were frozen: no trained teacher yet
        counts = {c: np.asarray(self.cov["counts"][c]) for c in curves}
        prev = np.asarray(self.cov["z"])
        ms, zs, adj_max, adj_sq = [], [], 0.0, []
        for l in range(prev.shape[0]):                  # each layer's warp is pooled on its own
            m, z, diag = cov.pooled_target({c: v[l] for c, v in curves.items()}, counts, prev[l])
            ms.append(m)
            zs.append(z)
            adj_max = max(adj_max, diag["proj_adjust_max"])
            adj_sq.append(diag["proj_adjust_wrms"] ** 2)
        self.cov["z"] = [z.tolist() for z in zs]
        self.cov["ready"] = True
        self.cov["updates"] += 1
        out.update(**summary("raw", ms), **summary("z", zs), proj_adjust_max=adj_max,
                   proj_adjust_wrms=float(np.sqrt(np.mean(adj_sq))), uncovered=diag["uncovered"])
        return out

    def fit(self) -> None:
        self.maybe_resume()
        log_path = os.path.join(self.out_dir, "train_log.jsonl")
        while self.round < self.cfg.rounds:
            log = self.train_round()
            if self.eval_fn is not None:
                log["eval"] = self.eval_fn(self, self.round)
            self.history.append(log)
            with open(log_path, "a") as f:
                f.write(json.dumps(log) + "\n")
            losses = [v["loss"] for v in log["clients"].values()]
            extra = (f" | direction cos {log['client_direction_cos_mean']:.3f}"
                     if "client_direction_cos_mean" in log else "")
            print(f"[round {log['round']:4d}] mean loss {sum(losses) / len(losses):.4f} "
                  f"| delta cos {log.get('client_delta_cos_mean', float('nan')):.3f}{extra} "
                  f"| {log['time_s']:.1f}s{format_monitor(log.get('eval', {}))}", flush=True)
            if self.round % self.cfg.save_every == 0 or self.round == self.cfg.rounds:
                self.save()
        if self.cfg.mode == "local":
            merged = average_states([self.clients[c]["shared_local"] for c in self.client_ids])
            torch.save(merged, os.path.join(self.out_dir, "merged_shared.pt"))

    # ----------------------------------------------------------- checkpointing
    def export(self) -> dict:
        """Lightweight snapshot: everything needed for evaluation, no optimizer state."""
        return {
            "round": self.round,
            "server": self.server,
            "clients": {c: {k: v for k, v in st.items() if k != "opt"} for c, st in self.clients.items()},
            "fed_config": asdict(self.cfg),
            "coverage": copy.deepcopy(self.cov),
        }

    def save(self) -> None:
        snap_dir = os.path.join(self.out_dir, "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        torch.save(self.export(), os.path.join(snap_dir, f"round_{self.round:04d}.pt"))
        full = self.export()
        full["opt"] = {c: st["opt"] for c, st in self.clients.items()} if self.cfg.keep_opt_state_on_disk else None
        full["streams"] = {c: s.state() for c, s in self.streams.items()}
        full["rng"] = self.rng.getstate()
        tmp = os.path.join(self.out_dir, "state.pt.tmp")
        torch.save(full, tmp)
        os.replace(tmp, os.path.join(self.out_dir, "state.pt"))

    def maybe_resume(self) -> bool:
        path = os.path.join(self.out_dir, "state.pt")
        if not os.path.exists(path):
            return False
        st = torch.load(path, map_location="cpu", weights_only=False)
        self.round = st["round"]
        self.server = st["server"]
        for c, cs in st["clients"].items():
            self.clients[c].update(cs)
            if st.get("opt"):
                self.clients[c]["opt"] = st["opt"][c]
        for c, s in st["streams"].items():
            self.streams[c].load(s)
        r = st["rng"]
        self.rng.setstate((r[0], tuple(r[1]), r[2]))
        if self.cov is not None:
            self._resume_coverage(st)
        print(f"resumed from {path} at round {self.round}", flush=True)
        return True


    def _resume_coverage(self, st: dict) -> None:
        saved = st.get("coverage")
        if not saved:
            raise ValueError("calibration=coverage, but the saved state has no coverage table "
                             "(it was trained with another calibration mode)")
        if np.asarray(saved["z"]).ndim == 1:            # runs before per-layer warps: one table
            saved = dict(saved, z=[saved["z"]], n_warps=1)
        if saved["n_warps"] != self.cov["n_warps"]:
            raise ValueError(f"saved run has {saved['n_warps']} warps per client, this config "
                             f"{self.cov['n_warps']} (lora.warp_scope differs)")
        for key in ("grid", "bandwidth", "lambda_max", "tau_local", "tau_peer"):
            if not np.allclose(saved[key], self.cov[key]):
                raise ValueError(f"coverage {key} differs from the saved run: {saved[key]} vs {self.cov[key]}")
        for c in self.client_ids:
            if not np.allclose(saved["counts"][c], self.cov["counts"][c]):
                raise ValueError(f"coverage counts of {c} differ from the saved run (dataset or alpha "
                                 "reference changed); start a new run")
            u = float(self.clients[c]["gain"][GAIN_KEY])
            if u != 0.0:
                raise ValueError(f"client {c} has gain parameter u = {u} (s != 1); coverage needs s = 1")
        self.cov = saved


def load_snapshot_into(model, snapshot: dict, client: str, shared: Optional[dict] = None) -> None:
    """Configure ``model`` as client ``client`` of a saved run (evaluation helper).

    ``shared`` overrides the direction, e.g. a merged direction or a direction from
    another run attached to this client's private model (portability tests).
    """
    cs = snapshot["clients"][client]
    fc = snapshot["fed_config"]
    adapter = "shared" if fc.get("share_private") else fc.get("adapter", "private")
    if adapter == "shared":
        load_state(model, {k: v for k, v in snapshot["server"].items() if not k.endswith("lora_B_d")})
    elif adapter == "private":
        load_state(model, cs["private"])
    else:                                              # none: the adapter must contribute nothing
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith("lora_B_p"):
                    p.zero_()
    if fc.get("calibration", "private") == "shared":
        load_state(model, {k: v for k, v in snapshot["server"].items() if k.startswith("steer_control.")})
    else:
        load_state(model, cs["gain"])
    if shared is None:
        shared = cs["shared_local"] if fc["mode"] == "local" else snapshot["server"]
        shared = {k: v for k, v in shared.items() if k.endswith("lora_B_d")}
    load_state(model, shared)


__all__ = ["FedConfig", "FedSteerTrainer", "load_snapshot_into", "GAIN_KEY"]
