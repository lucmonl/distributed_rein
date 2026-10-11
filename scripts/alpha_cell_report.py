"""Per-alpha-cell error decomposition for a steering run (CAL-19 / MOL-20 definition).

    python scripts/alpha_cell_report.py --run NAME=runs/<run> [NAME2=runs/<run2> ...]

Error is computed PER PROMPT x ALPHA CELL as |q.cdf(score) - alpha|, exactly as
``fedsteer.metrics.metrics_for_client`` does, and the reference CDF ``q`` is rebuilt with the run's
own ``alpha_reference.json`` through ``fedsteer.data.alpha_reference_from_json`` -- the same objects
the evaluation used, so nothing about the mapping is reimplemented here.

MOL-20 established that the decomposition must come from the cells and NOT from the per-alpha curve
means: a mean score at one alpha measures BIAS, which understates the interior and overstates the
endpoints. alpha = 0 and alpha = 1 are reported separately because on ChEMBL alpha = 0 has been the
dominant error cell in every arm so far and no arm had moved it.

Two columns per arm:
  pct err      cells with an unscorable output dropped (strict)
  penalized    unscorable cells counted as error 1 (the dev-selection metric)
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedsteer.data import alpha_reference_from_json

# Test-eval selection, shared rule (CAL-21). The pattern must be FOUR digits followed immediately
# by `__`: a looser `eval_round_[0-9]*__*.json` also matches E2 and baseline evals such as
# `eval_round_0100_e2_plugin_nall__<stamp>.json`, and taking the alphabetically last match then
# silently reads the held-out clients' plugin numbers instead of the arm's own. That produced a
# spurious rho = -1.000 on NR-62 (n = 4) before it was caught.
TEST_GLOB = "eval_round_[0-9][0-9][0-9][0-9]__*.json"


def test_eval(run: str, rnd=None) -> str:
    """The run's own test evaluation. Refuses to guess when several rounds are present."""
    files = sorted(glob.glob(os.path.join(run, "evals", TEST_GLOB)))
    files = [f for f in files if "_dev__" not in os.path.basename(f)]
    if not files:
        return ""
    by_round = {}
    for f in files:
        by_round.setdefault(int(os.path.basename(f)[11:15]), f)
    if rnd is not None:
        if int(rnd) not in by_round:
            sys.exit(f"{run}: no test eval for round {rnd} (have {sorted(by_round)})")
        return by_round[int(rnd)]
    if len(by_round) > 1:
        sys.exit(f"{run}: {len(by_round)} test rounds present {sorted(by_round)}; pass --round")
    return next(iter(by_round.values()))



def cells(run: str):
    ref_path = os.path.join(run, "alpha_reference.json")
    if not os.path.exists(ref_path):
        sys.exit(f"{run}: no alpha_reference.json")
    q = alpha_reference_from_json(json.load(open(ref_path)))
    if q is None:
        sys.exit(f"{run}: alpha_reference.json says local mode; this report assumes global alpha")
    ev = test_eval(run)
    if not ev:
        sys.exit(f"{run}: no test eval")
    o = json.load(open(ev))
    alphas = np.asarray(o["alphas"], dtype=float)
    out = {}
    for c, m in o["clients"].items():
        s = np.asarray(m["grid"], dtype=float)                      # [n_prompts, n_alphas]
        pct = np.vectorize(q.cdf)(np.where(np.isfinite(s), s, 0.0))
        err = np.abs(pct - alphas[None, :])
        err[~np.isfinite(s)] = np.nan                               # unscorable cells
        pen = np.where(np.isfinite(err), err, 1.0)                  # penalty 1
        out[c] = (err, pen, np.asarray(m["support"], dtype=float))
    return o["round"], alphas, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="NAME=path")
    ap.add_argument("--per_client", action="store_true", help="also print the per-client table")
    args = ap.parse_args()
    runs = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        runs[name] = cells(path)

    names = list(runs)
    alphas = runs[names[0]][1]
    hdr = "  ".join(f"a={a:g}" for a in alphas)
    print(f"== Per-alpha-cell error (test, equal weight over clients). cells, not curve means (MOL-20)")
    print(f"{'arm':10s} {'metric':10s} " + " ".join(f"{f'a={a:g}':>8s}" for a in alphas)
          + f" {'endpoints':>10s} {'interior':>9s} {'all':>7s}")
    for n in names:
        rnd, _, per = runs[n]
        for label, idx in (("pct err", 0), ("penalized", 1)):
            # equal weight over clients: mean per client first, then over clients
            per_a = np.stack([np.nanmean(per[c][idx], axis=0) for c in per])   # [C, A]
            col = np.nanmean(per_a, axis=0)
            ends = float(np.nanmean(col[[0, -1]]))
            inter = float(np.nanmean(col[1:-1]))
            allc = float(np.nanmean(col))
            print(f"{n:10s} {label:10s} " + " ".join(f"{v:8.3f}" for v in col)
                  + f" {ends:10.3f} {inter:9.3f} {allc:7.3f}")
    if args.per_client:
        for ai, a in enumerate(alphas):
            print(f"\n-- alpha = {a:g}: per-client pct err (penalized in brackets)")
            print(f"{'client':12s} {'support':>14s} " + " ".join(f"{n:>16s}" for n in names))
            for c in sorted(runs[names[0]][2]):
                sup = runs[names[0]][2][c][2]
                row = []
                for n in names:
                    e, p, _ = runs[n][2][c]
                    row.append(f"{np.nanmean(e[:, ai]):.3f} ({np.nanmean(p[:, ai]):.3f})")
                print(f"{c:12s} [{sup[0]:.2f},{sup[1]:.2f}] " + " ".join(f"{x:>16s}" for x in row))


if __name__ == "__main__":
    main()
