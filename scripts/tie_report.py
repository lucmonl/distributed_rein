"""Exact vs. near-tie report computed from saved eval files (read-only).

    python scripts/tie_report.py --run runs/X                 # per round, mean over clients
    python scripts/tie_report.py --run runs/X --round 100     # per client at one round
    python scripts/tie_report.py --run runs/X --split test    # the test evaluation

Near-identical = same text after normalization (quotes/dashes/whitespace; a trailing
ellipsis and the possibly cut-off last word removed) up to a small token edit
(difflib ratio >= --threshold).  See fedsteer.metrics.text_tie_metrics.  Works on
any eval file that saved generated texts (outputs), including ones written before
the metric existed.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedsteer.metrics import text_tie_metrics  # noqa: E402

KEYS = ["text_tie_rate", "near_tie_rate", "near_no_effect_rate", "endpoint_near_tie_rate",
        "adjacent_increase_rate", "adjacent_increase_rate_nt", "concordance", "concordance_nt"]
HEAD = ["exactTie", "nearTie", "nearNoEff", "endNearTie", "adjInc", "adjInc_nt", "concord", "concord_nt"]


def rows_for(path, threshold):
    d = json.load(open(path))
    out = {}
    for c, r in d["clients"].items():
        if "outputs" not in r:
            return d, None
        m = text_tie_metrics(r["outputs"], r["grid"], d["alphas"], threshold)
        m["adjacent_increase_rate"] = r["adjacent_increase_rate"]
        m["concordance"] = r["concordance"]
        out[c] = m
    return d, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="dev", help="dev (in-training evals) or test")
    ap.add_argument("--round", type=int, default=None, help="per-client table for this round")
    ap.add_argument("--threshold", type=float, default=0.95)
    args = ap.parse_args()
    evals = sorted(glob.glob(os.path.join(args.run, "evals", "eval_round_*.json")))
    evals = [f for f in evals if ("_dev__" in f) == (args.split == "dev") and "_dev_" not in f.replace("_dev__", "")]
    if not evals:
        sys.exit("no eval files")
    by_round = {int(re.search(r"round_(\d+)", os.path.basename(f)).group(1)): f for f in evals}
    if args.round is None:
        print("round  " + "  ".join(f"{h:>10s}" for h in HEAD))
        for rnd in sorted(by_round):
            _, rows = rows_for(by_round[rnd], args.threshold)
            if rows is None:
                print(f"{rnd:5d}  (no saved outputs)")
                continue
            print(f"{rnd:5d}  " + "  ".join(f"{np.mean([v[k] for v in rows.values()]):10.3f}" for k in KEYS))
    else:
        _, rows = rows_for(by_round[args.round], args.threshold)
        print(f"{'client':17s}  " + "  ".join(f"{h:>10s}" for h in HEAD))
        for c, v in rows.items():
            print(f"{c:17s}  " + "  ".join(f"{v[k]:10.3f}" for k in KEYS))


if __name__ == "__main__":
    main()
