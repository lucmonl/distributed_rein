"""Table of checkpoint-sweep results (from eval_direction.py --split dev) and
the selected checkpoint.

    python scripts/summarize_sweep.py --run runs/nr_pilot_llama1b [--split dev] [--select pct_calib_err]
    python scripts/summarize_sweep.py --run runs/X --print_best      # prints only the best snapshot path

Selection uses the dev split only; evaluate the chosen snapshot on test afterwards.
"""

import argparse
import glob
import json
import os
import re
import sys

LOWER = {"pct_calib_err", "loss", "adjacent_tie_rate", "no_effect_rate"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--select", default="pct_calib_err",
                    help="summary metric (mean over clients) used to pick the checkpoint")
    ap.add_argument("--print_best", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.run, f"eval_round_[0-9][0-9][0-9][0-9]_{args.split}.json")))
    if not files:
        sys.exit(f"no eval_round_*_{args.split}.json files in {args.run}")
    rows = []
    for f in files:
        d = json.load(open(f))
        rnd = int(re.search(r"round_(\d+)", f).group(1))
        s = d["summary"]
        rows.append({"round": rnd, "snapshot": d["snapshot"],
                     **{k: s[k]["mean"] for k in ("loss", "spearman", "concordance", "endpoint_increase_rate",
                                                  "adjacent_tie_rate", "pct_calib_err", "pct_range") if k in s},
                     "spearman_worst": s["spearman"]["worst"], "pct_err_worst": s["pct_calib_err"]["worst"],
                     "const_pct_err": s.get("constant_output_pct_err")})

    train_loss = {}
    log = os.path.join(args.run, "train_log.jsonl")
    if os.path.exists(log):
        for line in open(log):
            h = json.loads(line)
            ls = [v["loss"] for v in h["clients"].values()]
            train_loss[h["round"] + 1] = sum(ls) / len(ls)   # log round r = r+1 completed rounds

    sign = 1 if args.select in LOWER else -1
    best = min(rows, key=lambda r: sign * r[args.select])
    if args.print_best:
        print(best["snapshot"])
        return

    cols = ["round", "train_loss", "loss", "spearman", "spearman_worst", "concordance", "endpoint_increase_rate",
            "adjacent_tie_rate", "pct_calib_err", "pct_err_worst", "pct_range"]
    heads = ["round", "trainLoss", f"{args.split}Loss", "spearman", "spWorst", "concord", "endpt+", "ties",
             "pctErr", "pctWorst", "pctRange"]
    print("  ".join(f"{h:>9s}" for h in heads))
    for r in rows:
        r["train_loss"] = train_loss.get(r["round"], float("nan"))
        mark = "  <- best" if r is best else ""
        print("  ".join(f"{r[c]:9d}" if c == "round" else f"{r.get(c, float('nan')):9.3f}" for c in cols) + mark)
    print(f"\nselected by mean {args.select} on {args.split}: round {best['round']} ({best['snapshot']})")
    if rows[0].get("const_pct_err") is not None:
        print(f"reference: constant-output pct err = {rows[0]['const_pct_err']:.3f}")


if __name__ == "__main__":
    main()
