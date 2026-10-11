"""Restricted client disagreement for an aligned/consensus/coverage run (CAL-19, 2026-10-09).

    python scripts/disagreement_report.py --run runs/mol25-fed-aligned-off_... [--round 100]

WHY THIS EXISTS. `fed.py` logs a single `disagreement` scalar: the weight-weighted sd of the
clients' own curves around the pooled mean, meaned over ALL layers and grid points,

    dis = sqrt( sum_c W_c (V_c - m)^2 / sum_c W_c ).mean(),   m = sum_c W_c V_c / sum_c W_c

At a grid point where only one client carries non-trivial weight, m collapses onto that client's
own curve, so (V_c - m) -> 0 for it and the others contribute ~0 through W_c: the sd is
structurally ~0 there. Such points CANNOT exhibit disagreement, yet they are averaged in, so on a
skewed dataset the scalar is diluted toward 0 and a low value must NOT be read as "the clients
agree". This script reports the same quantity restricted to grid points where at least
`--min_clients` clients carry at least `--weight_floor` of that point's total weight, next to the
unrestricted number, so the dilution is visible rather than silently corrected.

The weight floor is RELATIVE to the point's total weight, not absolute, so the restriction does not
track client size (ChEMBL's clients range from 643 to 3535 training examples).

LAYOUT NOTE. In the snapshot's `coverage` entry, `values[c]` is (n_warps, n_grid) but
`pool_weights[c]` is (n_grid,) -- a client's weight depends on the grid point (where it has data),
not on the layer, and broadcasts across layers. So which points qualify is a property of the GRID
alone and is identical for every layer.

The script first reproduces the unrestricted scalar and compares it with the value `fed.py` logged
for that round in train_log.jsonl; a mismatch means the formula here has drifted from the trainer
and the restricted number should not be trusted.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def load_cov(run: str, rnd):
    snaps = {int(f[6:10]): f for f in os.listdir(os.path.join(run, "snapshots"))
             if f.startswith("round_") and f.endswith(".pt")}
    if not snaps:
        sys.exit(f"{run}: no snapshots")
    r = max(snaps) if rnd is None else int(rnd)
    if r not in snaps:
        sys.exit(f"{run}: no snapshot for round {r} (have {sorted(snaps)})")
    st = torch.load(os.path.join(run, "snapshots", snaps[r]), map_location="cpu", weights_only=False)
    cov = st.get("coverage")
    if not isinstance(cov, dict) or not cov.get("values"):
        sys.exit(f"{run}: snapshot {r} has no populated coverage entry "
                 f"(not an aligned/consensus/coverage run, or the table was never built)")
    return r, cov


def logged_disagreement(run: str, rnd: int):
    """The value fed.py wrote for this round, for cross-checking. None if not logged.

    OFF-BY-ONE: train_log.jsonl is 0-indexed (rounds 0..N-1) while snapshots are named for the
    number of rounds completed, so snapshot round_0100.pt corresponds to log round 99. We look up
    rnd - 1 first and fall back to rnd, and return which one matched so the caller can say so.
    """
    path = os.path.join(run, "train_log.jsonl")
    if not os.path.exists(path):
        return None, None
    found = {}
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            r = rec.get("round")
            if r not in (rnd - 1, rnd):
                continue
            for v in (rec, rec.get("cov") or {}, rec.get("coverage") or {}):
                if isinstance(v, dict) and "disagreement" in v:
                    found[r] = v["disagreement"]
    for r in (rnd - 1, rnd):
        if r in found:
            return found[r], r
    return None, None


def report(run, rnd, min_clients, floor, quiet=False):
    r, cov = load_cov(run, rnd)
    clients = sorted(cov["values"])
    V = np.stack([np.asarray(cov["values"][c], dtype=np.float64) for c in clients])      # (C, L, K)
    W = np.stack([np.asarray(cov["pool_weights"][c], dtype=np.float64) for c in clients])  # (C, K)
    grid = np.asarray(cov["grid"], dtype=np.float64)
    C, L, K = V.shape
    Wb = W[:, None, :]                                     # broadcast over layers, as fed.py does
    den = Wb.sum(0)
    m = (Wb * V).sum(0) / np.clip(den, 1e-12, None)
    sd = np.sqrt((Wb * (V - m) ** 2).sum(0) / np.clip(den, 1e-12, None))   # (L, K)

    unrestricted = float(sd.mean())
    share = W / np.clip(W.sum(0, keepdims=True), 1e-12, None)              # (C, K) relative weight
    n_eff = (share >= floor).sum(0)                                        # (K,) clients per point
    keep = n_eff >= min_clients
    restricted = float(sd[:, keep].mean()) if keep.any() else float("nan")

    if not quiet:
        print(f"== {os.path.basename(run)}  round {r}  mode={cov.get('mode')} pool={cov.get('pool')} "
              f"tau_pool={cov.get('tau_pool')} lambda_max={cov.get('lambda_max')}")
        print(f"   clients={C}  layers={L}  grid points={K}")
        lg, lg_round = logged_disagreement(run, r)
        if lg is None:
            print(f"   unrestricted disagreement {unrestricted:.5f}  (no logged value to check against)")
        else:
            ok = abs(unrestricted - float(lg)) <= 5e-5
            print(f"   unrestricted disagreement {unrestricted:.5f}  logged {float(lg):.5f} "
                  f"(log round {lg_round})  "
                  f"{'MATCH' if ok else '*** MISMATCH -- formula drifted from fed.py ***'}")
        print(f"   RESTRICTED disagreement   {restricted:.5f}   "
              f"({keep.sum()} of {K} grid points keep >= {min_clients} clients at >= {floor:.0%} weight)")
        if keep.sum():
            ratio = restricted / unrestricted if unrestricted else float('nan')
            print(f"   dilution factor           {ratio:.2f}x  (restricted / unrestricted)")
        print("   clients per grid point (relative weight >= floor):")
        print("     alpha " + " ".join(f"{a:5.2f}" for a in grid))
        print("     nclnt " + " ".join(f"{int(n):5d}" for n in n_eff))
        print("     kept  " + " ".join(f"{'  yes' if k else '   no':>5s}" for k in keep))
        zt = np.asarray(cov["z"], dtype=np.float64)
        print("   per-client support_gap (weighted |own curve - pooled table|) and kept-point gap:")
        for i, c in enumerate(clients):
            w = W[i][None, :]
            gap = float((np.abs(V[i] - zt) * w).sum(1).mean() / max(w.sum(), 1e-12))
            wk = W[i][None, keep]
            gk = (float((np.abs(V[i][:, keep] - zt[:, keep]) * wk).sum(1).mean() / max(wk.sum(), 1e-12))
                  if keep.any() and wk.sum() > 0 else float("nan"))
            print(f"     {c:22s} all {gap:.5f}   kept-points {gk:.5f}   "
                  f"points at >= floor: {int((share[i] >= floor).sum())}/{K}")
    return {"run": run, "round": r, "unrestricted": unrestricted, "restricted": restricted,
            "kept_points": int(keep.sum()), "grid_points": K, "clients": C}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", required=True, help="run directory/ies")
    ap.add_argument("--round", default=None, help="snapshot round (default: the latest)")
    ap.add_argument("--min_clients", type=int, default=2,
                    help="a grid point counts only if at least this many clients are above the floor")
    ap.add_argument("--weight_floor", type=float, default=0.05,
                    help="client's share of that point's total weight to count (default 0.05)")
    ap.add_argument("--json", action="store_true", help="emit one JSON object per run instead")
    args = ap.parse_args()
    out = [report(r, args.round, args.min_clients, args.weight_floor, quiet=args.json) for r in args.run]
    if args.json:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
