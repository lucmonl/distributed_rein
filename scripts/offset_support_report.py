"""Does the learned offset o_i track a client's position on the global percentile scale? (CAL-21)

    python scripts/offset_support_report.py --run NAME=runs/<run> [NAME2=runs/<run2> ...]

MOL-25b found that under a SHARED or POOLED shape the offset is near-monotone in the client's
support floor (spearman -0.905 on ChEMBL `fed-aligned-off` and `fed-shared-off`), while under a
PRIVATE shape it is uncorrelated (+0.190 on `local-off`). The reading is that the shape and the
offset compete for the same degree of freedom, so o_i is only identified when the shape is tied
across clients -- which is also why alpha = 0 moves in the fed arms and not the local one.

Both numbers are printed because they disagree usefully: a large |pearson| with a weak spearman
means one extreme client is carrying the linear fit (Newsroom's nypost.com does this) rather than
the ordering being monotone.

`support` and `offset` are read from the run's own test evaluation, so this reports exactly what the
evaluation recorded and reimplements nothing.
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedsteer.data import ClientQuantiles, alpha_reference_from_json

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


def median_alpha(run: str) -> dict:
    """Each client's MEDIAN training target on the reference percentile scale.

    Built exactly as ``fedsteer.data.client_support`` builds the support bounds, with q=0.5 in
    place of 0.05/0.95: ref.cdf(local_q.quantile(0.5)). On Newsroom the floors bunch at 0.03-0.11
    while the medians spread out (reuters: floor 0.10, median 0.72), so the median separates
    clients the floor cannot.
    """
    qp, rp = os.path.join(run, "client_quantiles.json"), os.path.join(run, "alpha_reference.json")
    if not (os.path.exists(qp) and os.path.exists(rp)):
        return {}
    ref = alpha_reference_from_json(json.load(open(rp)))
    if ref is None:
        return {}
    local = {c: ClientQuantiles(v) for c, v in json.load(open(qp)).items()}
    return {c: float(ref.cdf(q.quantile(0.5))) for c, q in local.items()}


def read(run: str, rnd=None):
    ev = test_eval(run, rnd)
    if not ev:
        cand = sorted(glob.glob(os.path.join(run, "evals", "*_dev*.json")))
        if not cand:
            sys.exit(f"{run}: no evaluation found")
        print(f"   note: {os.path.basename(run)} has no test eval; using the latest dev eval")
        ev = cand[-1]
    o = json.load(open(ev))
    med = median_alpha(run)
    rows = {}
    for c, m in o["clients"].items():
        if m.get("offset") is None:
            continue
        rows[c] = (float(m["offset"]), tuple(float(x) for x in m["support"]), med.get(c))
    if not rows:
        sys.exit(f"{run}: no client reports an offset (run without lora.offset=true?)")
    return o.get("round"), rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="NAME=path")
    ap.add_argument("--round", default=None, help="test-eval round, when a run has several")
    ap.add_argument("--per_client", action="store_true")
    args = ap.parse_args()

    print("== o_i vs two descriptors of where the client sits on the global percentile scale")
    print("   (negative = o_i goes more negative as the descriptor rises)")
    print(f"{'arm':26s} {'round':>5s} {'n':>3s} | {'floor rho':>9s} {'floor r':>8s} "
          f"| {'med rho':>8s} {'med r':>7s} | {'mean o_i':>9s} {'min':>8s} {'max':>8s}")
    per_arm = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        rnd, rows = read(path, args.round)
        per_arm[name] = rows
        off = np.array([v[0] for v in rows.values()])
        lo = np.array([v[1][0] for v in rows.values()])
        med = [v[2] for v in rows.values()]
        def cc(x):
            if len(rows) <= 2 or any(v is None for v in x):
                return float("nan"), float("nan")
            x = np.asarray(x, dtype=float)
            return spearmanr(x, off).correlation, pearsonr(x, off)[0]
        fs, fp = cc(lo)
        ms, mp = cc(med)
        print(f"{name:26s} {rnd:5d} {len(rows):3d} | {fs:+9.3f} {fp:+8.3f} "
              f"| {ms:+8.3f} {mp:+7.3f} | {off.mean():+9.4f} {off.min():+8.4f} {off.max():+8.4f}")

    if args.per_client:
        for name, rows in per_arm.items():
            print(f"\n-- {name}, sorted by support floor")
            for c, (o, sup, md) in sorted(rows.items(), key=lambda kv: kv[1][1][0]):
                m = f"{md:.3f}" if md is not None else "  n/a"
                print(f"   {c:22s} support [{sup[0]:.2f},{sup[1]:.2f}]  median a {m}  o_i {o:+.4f}")


if __name__ == "__main__":
    main()
