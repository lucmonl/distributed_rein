"""Summarize E2 (held-out clients) eval files of a run: one row per setting and data size,
then per-client error, then per-client paired bootstrap of frozen_D - local_D at each n.

    python scripts/e2_report.py --run runs/X [--window w0-0.4]

Reads eval_round_*_e2_{setting}_n{n}[_w..]__*.json with their quality_ / judge_ files (the
newest file per setting and n).  Negative differences = frozen_D better.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedsteer.data import alpha_reference_from_json  # noqa: E402
from fedsteer.metrics import text_tie_metrics  # noqa: E402

PAT = re.compile(r"eval_round_\d+_e2_(?P<setting>[A-Za-z_]+?)_n(?P<n>\d+|all)(?P<w>_w[^_]+)?__")
ORDER = ("frozen_D", "local_D", "plugin", "prompt")


def nkey(n):
    return float("inf") if n == "all" else int(n)


def boot(d, rng, B=4000):
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return f"{d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if hi < 0 or lo > 0 else ' '}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--window", default="", help="e.g. w0-0.4; default: no window")
    args = ap.parse_args()
    ref = alpha_reference_from_json(json.load(open(os.path.join(args.run, "alpha_reference.json"))))
    found = {}
    for f in sorted(glob.glob(os.path.join(args.run, "evals", "eval_round_*_e2_*.json")), key=os.path.getmtime):
        m = PAT.search(os.path.basename(f))
        if m and (m.group("w") or "") == (f"_{args.window}" if args.window else ""):
            found[(m.group("setting"), m.group("n"))] = f          # newest wins
    if not found:
        sys.exit("no E2 files")
    keys = sorted(found, key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 9, nkey(k[1])))
    evs = {k: json.load(open(found[k])) for k in keys}

    def side(k, prefix):
        p = os.path.join(os.path.dirname(found[k]), prefix + os.path.basename(found[k]))
        return json.load(open(p)) if os.path.exists(p) else None

    print(f"{'setting':9s} {'n':>5s} {'pct':>6s} {'in':>6s} {'out':>6s} {'reach':>6s} {'spear':>6s} {'nearT':>6s} {'endNT':>6s} |"
          f" {'gapAlgIn':>8s} {'gapAlgOut':>9s} {'bertIn':>6s} {'gapLenOut':>9s} | {'faithIn':>7s} {'faithOut':>8s} {'gRelOut':>7s}")
    for k in keys:
        ev, s = evs[k], evs[k]["summary"]
        ties = [text_tie_metrics(v["outputs"], v["grid"], ev["alphas"]) for v in ev["clients"].values()]
        q = (side(k, "quality_") or {}).get("summary", {})
        j = side(k, "judge_")
        jm = (lambda c: np.nanmean([v["summary"][c] if v["summary"].get(c) is not None else np.nan
                                    for v in j["clients"].values()])) if j else (lambda c: np.nan)
        g = lambda c: q.get(c) if q.get(c) is not None else np.nan
        print(f"{k[0]:9s} {k[1]:>5s} {s['pct_calib_err']['mean']:6.3f} {s['pct_err_in_support']['mean']:6.3f} "
              f"{s['pct_err_out_support']['mean']:6.3f} {s['reach_rate']['mean']:6.3f} {s['spearman']['mean']:6.3f} "
              f"{np.mean([t['near_tie_rate'] for t in ties]):6.3f} {np.mean([t['endpoint_near_tie_rate'] for t in ties]):6.3f} |"
              f" {g('gap_align_in'):+8.3f} {g('gap_align_out'):+9.3f} {g('bert_f1_in'):6.3f} {g('gap_length_out'):+9.1f} |"
              f" {jm('faithful_in'):7.3f} {jm('faithful_out'):8.3f} {jm('gap_relevance_out'):+7.3f}")

    clients = list(next(iter(evs.values()))["clients"])
    print("\nper-client percentile error")
    print(f"{'setting':9s} {'n':>5s} " + " ".join(f"{c[:12]:>12s}" for c in clients))
    for k in keys:
        print(f"{k[0]:9s} {k[1]:>5s} " + " ".join(f"{evs[k]['clients'][c]['pct_calib_err']:12.3f}" for c in clients))

    rng = np.random.default_rng(0)

    def per_article(ev, c):
        a = np.array(ev["alphas"])
        v = ev["clients"][c]
        err = np.abs(np.vectorize(ref.cdf)(np.array(v["grid"])) - a[None])
        lo, hi = v["support"]
        out = (a < lo) | (a > hi)
        return err.mean(1), (err[:, out].mean(1) if out.any() else None)

    for n in sorted({k[1] for k in keys}, key=nkey):
        if ("frozen_D", n) not in evs or ("local_D", n) not in evs:
            continue
        print(f"\nfrozen_D - local_D, n={n} (paired bootstrap over articles; * = 95% CI excludes 0)")
        for c in clients:
            ea, oa = per_article(evs[("frozen_D", n)], c)
            eb, ob = per_article(evs[("local_D", n)], c)
            sup = evs[("frozen_D", n)]["clients"][c]["support"]
            out = boot(oa - ob, rng) if oa is not None and ob is not None else "n/a"
            print(f"  {c:16s} support [{sup[0]:.2f},{sup[1]:.2f}]  overall {boot(ea - eb, rng)}  out-of-support {out}")


if __name__ == "__main__":
    main()
