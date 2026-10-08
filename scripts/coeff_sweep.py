"""Raw-coefficient sweep at frozen P_i / D: can the direction reach the endpoints at all?

Diagnostic asked for by `chembl-experiment-plan.md` (2026-10-06, "What coverage calibration can
diagnose or repair"): "freeze each selected P_i/D (and historical gain), sweep the coefficient
directly ... A separately labelled wider coefficient sweep tests magnitude/offset limitations;
improved reach must retain validity."

Why it needs its own script. The coefficient on the shared direction is
``o + s * h(alpha)`` (`SteerControl.coef_for`), and every nonlinear warp **hard-clamps** its input
to [0, 1] (`x = alpha.float().clamp(0.0, 1.0)` in `fedsteer/warp.py`). So with gain fixed at 1 and
no offset, the coefficient cannot leave [0, 1] no matter what alpha is requested: at alpha 0 the
output is the private adapter alone (W_0 + P_i) and at alpha 1 it is W_0 + P_i + D. Passing
`--alphas 1.5` to `eval_direction.py` therefore reproduces alpha 1 exactly.

This script replaces the loaded warp with the identity (which does **not** clamp) and sets gain 1
and offset 0, so the requested value *is* the raw coefficient and values outside [0, 1] become
reachable. It reports, per client and coefficient, the achieved attribute percentile and the
fraction of outputs the scorer can score.

**Separately labelled, as the plan requires:** these are not calibration results. The requested
coefficient is not an alpha target, so no percentile error is reported — a coefficient of 1.25 has
no "correct" percentile. Read it as a reachability-and-validity curve, which is what decides
whether freeing the offset (to move the alpha-0 endpoint) or the gain (to move alpha 1) has any
headroom before a training run is spent on it. Replacing the warp changes the model's calibration
by construction; that is the point, not a side effect.

    python scripts/coeff_sweep.py --run runs/exp46_cov_private_... \
        --scorer clogp_residual_deco_strict --split test --max_prompts 60 \
        --coeffs -0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

# Run as `python <repo>/scripts/coeff_sweep.py` (the job scripts run a snapshot copy, e.g.
# runs/_code/<stamp>/scripts/), where sys.path[0] is THIS file's directory, not the repo root, so
# `import fedsteer` would fail. Every other script in scripts/ does the same insert.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml

from fedsteer.data import (ChatFormatter, ClientQuantiles, alpha_reference_from_json,
                           client_support, read_jsonl)
from fedsteer.evaluate import score_grid
from fedsteer.fed import load_snapshot_into
from fedsteer.lora import SteerLoraConfig
from fedsteer.metrics import SCORERS
from fedsteer.model import load_model
from fedsteer.runinfo import make_stamp, provenance
from fedsteer.warp import AlphaWarp


def latest_snapshot(run: str) -> str:
    d = os.path.join(run, "snapshots")
    return os.path.join(d, sorted(os.listdir(d))[-1])


def make_coefficient_raw(model, identity: AlphaWarp) -> dict:
    """Make the requested value the raw coefficient: identity warp (no clamp), gain 1, offset 0.

    Returns what was replaced, so the change is recorded in the output rather than implied.
    ``identity`` is passed in (not constructed here) because the caller must put the *original*
    warp module back before loading the next client -- see the loop in main().
    """
    ctl = model.steer_control
    before = {"warp_kind": ctl.warp.kind, "gain": float(ctl.gain().item()),
              "offset": float(ctl.offset().item())}
    ctl.set_table(None)                     # ignore a consensus server table if one was loaded
    ctl.warp = identity                     # kind "none": forward returns alpha unclamped
    with torch.no_grad():
        ctl.u.zero_()                        # gain = exp(0) = 1
        if ctl.o is not None:
            ctl.o.zero_()                    # offset = 0
    return before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--snapshot", default=None, help="default: the latest snapshot")
    ap.add_argument("--scorer", default="clogp_residual_deco_strict", choices=sorted(SCORERS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--coeffs", default="-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5")
    ap.add_argument("--max_prompts", type=int, default=60)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(os.path.join(args.run, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(args.run, "client_quantiles.json")) as f:
        local_q = {c: ClientQuantiles(v) for c, v in json.load(f).items()}
    ref_path = os.path.join(args.run, "alpha_reference.json")
    ref = alpha_reference_from_json(json.load(open(ref_path))) if os.path.exists(ref_path) else None
    quantiles = {c: (ref if ref is not None else local_q[c]) for c in local_q}
    supports = {c: client_support(local_q[c], ref) for c in local_q} if ref is not None else {}

    coeffs = [float(x) for x in args.coeffs.replace(":", ",").split(",") if x != ""]
    # A one-point "sweep" is almost always a quoting accident, not a request: SLURM's --export is
    # itself comma-separated, so a comma-separated --coeffs passed that way is silently truncated to
    # its first element. Refuse rather than produce a plausible-looking single column. Colons are
    # accepted as a separator for exactly that situation.
    if len(coeffs) < 2:
        raise SystemExit(f"--coeffs needs at least 2 values, got {coeffs}; if this went through "
                         f"sbatch --export, the commas were eaten -- use colons or a command file")
    snap_path = args.snapshot or latest_snapshot(args.run)
    score = SCORERS[args.scorer]

    by_client = defaultdict(list)
    for r in read_jsonl(cfg["data_path"]):
        if r["client"] in cfg["clients"] and r.get("split") == args.split:
            by_client[r["client"]].append(r)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(cfg["model_name"], SteerLoraConfig(**cfg["lora"]), device=device,
                            grad_checkpointing=False,
                            attn_implementation=cfg.get("attn_implementation", "sdpa"))
    fmt = ChatFormatter(tok, system_prompt=cfg.get("system_prompt"),
                        template_kwargs=cfg.get("chat_template_kwargs"),
                        max_prompt_tokens=cfg.get("max_prompt_tokens", 1024),
                        max_target_tokens=cfg.get("max_target_tokens", 256))

    snap = torch.load(snap_path, map_location="cpu", weights_only=False)
    print(f"== {snap_path} ({args.split}); raw coefficients {coeffs}", flush=True)
    # The snapshot stores each client's warp parameters (steer_control.warp.*), and load_state is
    # strict about names, so the ORIGINAL warp module has to be back in place before every load --
    # otherwise the second client dies with "unknown parameter steer_control.warp.log_p" because
    # the identity warp has no parameters at all.
    pristine_warp = model.steer_control.warp
    identity = AlphaWarp().to(model.steer_control.u.device)
    out_clients = {}
    for c in cfg["clients"]:
        recs = by_client[c][: args.max_prompts]
        if not recs:
            continue
        model.steer_control.warp = pristine_warp
        load_snapshot_into(model, snap, c)
        replaced = make_coefficient_raw(model, identity)
        grid, texts = score_grid(model, fmt, recs, coeffs, score,
                                 max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        q = quantiles[c]
        finite = np.isfinite(grid)
        pct = np.full(grid.shape, np.nan)
        pct[finite] = np.vectorize(q.cdf, otypes=[float])(grid[finite])
        with np.errstate(invalid="ignore"):
            mean_pct = [float(np.nanmean(pct[:, j])) if finite[:, j].any() else None
                        for j in range(len(coeffs))]
        out_clients[c] = {
            "n_prompts": len(recs),
            "support": supports.get(c),
            "replaced_control": replaced,
            "mean_pct_by_coeff": mean_pct,
            "valid_rate_by_coeff": [float(finite[:, j].mean()) for j in range(len(coeffs))],
            "mean_score_by_coeff": [float(np.nanmean(grid[:, j])) if finite[:, j].any() else None
                                    for j in range(len(coeffs))],
            "grid": np.round(grid, 4).tolist(),
        }
        print(f"  {c:12s} pct " + " ".join("  n/a" if v is None else f"{v:5.3f}" for v in mean_pct), flush=True)
        print(f"  {c:12s} val " + " ".join(f"{v:5.3f}" for v in out_clients[c]["valid_rate_by_coeff"]), flush=True)

    valid = [v for v in out_clients.values()]
    summary = {
        "mean_pct_by_coeff": [
            float(np.mean([v["mean_pct_by_coeff"][j] for v in valid
                           if v["mean_pct_by_coeff"][j] is not None])) for j in range(len(coeffs))],
        "valid_rate_by_coeff": [
            float(np.mean([v["valid_rate_by_coeff"][j] for v in valid])) for j in range(len(coeffs))],
    }
    res = {"snapshot": snap_path, "split": args.split, "coeffs": coeffs, "scorer": args.scorer,
           "note": ("raw coefficient o + s*h(alpha) with the warp replaced by the identity, gain 1, "
                    "offset 0; the requested value is the coefficient, NOT an alpha target, so no "
                    "calibration error is reported"),
           "clients": out_clients, "summary": summary, "provenance": provenance(args=vars(args))}

    stamp = make_stamp()
    os.makedirs(os.path.join(args.run, "evals"), exist_ok=True)
    tag = os.path.basename(snap_path).removesuffix(".pt")
    out = args.out or os.path.join(args.run, "evals", f"coeffsweep_{tag}_{args.split}__{stamp}.json")
    if os.path.exists(out):
        raise FileExistsError(out)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("\nSUMMARY pct  " + " ".join(f"{v:5.3f}" for v in summary["mean_pct_by_coeff"]), flush=True)
    print("SUMMARY val  " + " ".join(f"{v:5.3f}" for v in summary["valid_rate_by_coeff"]), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
