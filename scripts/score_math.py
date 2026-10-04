"""Math-CoT quality metrics for an eval file (the math task's counterpart of
scripts/score_molecules.py).  No regeneration: works on the saved generations.

    python scripts/score_math.py --run runs/X --eval runs/X/evals/eval_*.json

Reports, per client and per alpha:
  accuracy    final \\boxed{} answer equals the gold answer (Math-Verify)   <- utility
  boxed       share of generations with a \\boxed{} answer                  <- gate G0
  truncated   share that hit the generation cap (length >= max_new_tokens - 1)
  loop        share with a repeated 20-word n-gram (>= 3 times)            <- gate G0
  tokens      mean length in ruler tokens (the attribute)
  gzip        mean compressed/raw ratio (low = repetitive)
and splits accuracy / format by whether alpha lies in the client's own support.

Writes <eval dir>/math_<eval name> next to the eval file.
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fedsteer.data import read_jsonl
from fedsteer.mathcot import cot_tokens, extract_boxed, gzip_ratio, is_correct, repetition_loop
from score_molecules import load_supports  # noqa: E402  (same support logic for every task)

KEYS = ("accuracy", "boxed", "truncated", "loop", "tokens", "gzip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", nargs="+", required=True, help="eval json file(s) or globs")
    ap.add_argument("--data", default="data/math_fed/data.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=None,
                    help="generation cap (default: read from the eval file's provenance)")
    args = ap.parse_args()

    by_id = {r["url"]: r for r in read_jsonl(args.data)}
    supports = load_supports(args.run) if os.path.exists(os.path.join(args.run, "client_quantiles.json")) else {}
    files = [f for pat in args.eval for f in sorted(glob.glob(pat))]
    if not files:
        raise SystemExit("no eval files matched")

    for path in files:
        d = json.load(open(path))
        alphas = d["alphas"]
        cap = args.max_new_tokens or (d.get("provenance", {}).get("args", {}) or {}).get("max_new_tokens")
        out = {}
        for c, res in d["clients"].items():
            texts, ids = res["outputs"], res["record_ids"]
            rows = []
            for j, a in enumerate(alphas):
                gens = [texts[i][j] for i in range(len(texts))]
                golds = [by_id[rid]["answer"] for rid in ids]
                toks = [cot_tokens(g) for g in gens]
                row = {"alpha": a, "n": len(gens),
                       "accuracy": float(np.mean([is_correct(g, y) for g, y in zip(gens, golds)])),
                       "boxed": float(np.mean([extract_boxed(g) is not None for g in gens])),
                       "loop": float(np.mean([repetition_loop(g) for g in gens])),
                       "tokens": float(np.mean(toks)),
                       "gzip": float(np.mean([gzip_ratio(g) for g in gens]))}
                if cap:
                    row["truncated"] = float(np.mean([t >= cap - 1 for t in toks]))
                if c in supports:
                    lo, hi = supports[c]
                    row["in_support"] = bool(lo <= a <= hi)
                rows.append(row)
            out[c] = rows

        agg = {}
        for k in KEYS:
            vals = [r[k] for rows in out.values() for r in rows if r.get(k) is not None]
            if not vals:
                continue
            worst = max(vals) if k in ("truncated", "loop") else min(vals)
            agg[k] = {"mean": float(np.mean(vals)), "worst": float(worst)}
            for tag, want in (("in_support", True), ("out_support", False)):
                v = [r[k] for rows in out.values() for r in rows
                     if r.get(k) is not None and r.get("in_support") is want]
                if v:
                    agg[k][tag] = float(np.mean(v))
        dst = os.path.join(os.path.dirname(path), "math_" + os.path.basename(path))
        json.dump({"eval": os.path.basename(path), "max_new_tokens": cap, "per_client": out, "summary": agg},
                  open(dst, "w"), indent=1)

        print(f"\n=== {os.path.basename(path)} (cap {cap})")
        for c, rows in out.items():
            print(f"{c}")
            for r in rows:
                print(f"   a={r['alpha']:<5g} acc {r['accuracy']:.2f}  boxed {r['boxed']:.2f}  "
                      f"trunc {r.get('truncated', float('nan')):.2f}  loop {r['loop']:.2f}  "
                      f"tokens {r['tokens']:6.0f}" + ("" if "in_support" not in r else
                                                      f"  {'in' if r['in_support'] else 'OUT'}"))
        print("summary:", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in agg.items()}))
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
