"""Table of checkpoint-sweep results (from eval_direction.py --split dev) and
the selected checkpoint.

    python scripts/summarize_sweep.py --run runs/nr_pilot_llama1b [--split dev] [--select pct_calib_err]
    python scripts/summarize_sweep.py --run runs/X --print_best      # prints only the best snapshot path
    python scripts/summarize_sweep.py --run runs/X --match j11006740  # only files from one job

Reads <run>/evals/eval_round_XXXX_<split>__<stamp>.json (and legacy files in <run>/);
if a round was evaluated several times, the newest matching file is used.

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
    ap.add_argument("--match", default="", help="only use eval files whose name contains this (e.g. a job id)")
    args = ap.parse_args()

    pat = f"eval_round_[0-9][0-9][0-9][0-9]_{args.split}"
    cands = glob.glob(os.path.join(args.run, "evals", pat + "__*.json")) + \
        glob.glob(os.path.join(args.run, pat + ".json"))
    cands = [f for f in cands if args.match in os.path.basename(f)]
    latest = {}
    for f in cands:                                   # newest file per round
        rnd = int(re.search(r"round_(\d+)", os.path.basename(f)).group(1))
        if rnd not in latest or os.path.getmtime(f) > os.path.getmtime(latest[rnd]):
            latest[rnd] = f
    files = [latest[r] for r in sorted(latest)]
    if not files:
        sys.exit(f"no eval_round_*_{args.split} files in {args.run} matching '{args.match}'")
    rows = []
    for f in files:
        d = json.load(open(f))
        rnd = int(re.search(r"round_(\d+)", os.path.basename(f)).group(1))
        s = d["summary"]
        snap = d["snapshot"]
        if not os.path.exists(snap):   # path stored before a run folder was renamed
            snap = os.path.join(args.run, "snapshots", f"round_{rnd:04d}.pt")
        rows.append({"round": rnd, "snapshot": snap,
                     **{k: s[k]["mean"] for k in ("loss", "spearman", "concordance", "endpoint_increase_rate",
                                                  "adjacent_tie_rate", "pct_calib_err", "pct_range", args.select) if k in s},
                     # absent when no client could be scored at all (e.g. every generated
                     # molecule was unparseable) -- keep the row so the failure is visible
                     "spearman_worst": s.get("spearman", {}).get("worst"),
                     "pct_err_worst": s.get("pct_calib_err", {}).get("worst"),
                     "const_pct_err": s.get("constant_output_pct_err")})

    train_loss = {}
    log = os.path.join(args.run, "train_log.jsonl")
    if os.path.exists(log):
        for line in open(log):
            h = json.loads(line)
            ls = [v["loss"] for v in h["clients"].values()]
            train_loss[h["round"] + 1] = sum(ls) / len(ls)   # log round r = r+1 completed rounds

    sign = 1 if args.select in LOWER or args.select.startswith(("pct_calib_err", "pct_err_", "unscorable_")) else -1
    eligible = [r for r in rows if r.get(args.select) is not None]
    if not eligible:
        sys.exit(f"no checkpoints have a value for {args.select}")
    best = min(eligible, key=lambda r: sign * r[args.select])
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
