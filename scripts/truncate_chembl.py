"""Controlled coverage experiment: amplify each client's skew by cutting part of
its TRAINING alpha range, leaving dev/test untouched so the removed region can
still be tested against real molecules.

    python scripts/truncate_chembl.py --out_dir data/chembl_deco_trunc

Design (complementary, all participants -- not a subset):
  * clients whose alpha median is below the pooled median keep only alpha <= hi
    (they lose the TOP of the range);
  * clients above it keep only alpha >= lo (they lose the BOTTOM).
With lo=0.4, hi=0.6 every client loses ~40% of the range, the union of the
retained supports still covers [0, 1], and the overlap band is [0.4, 0.6].

Truncating every client in the SAME direction would be vacuous: no client would
hold the missing region, so there would be nothing for a shared direction to
carry. Complementary truncation is what makes the transfer question well posed,
and it gives 8 clients with real gaps instead of 3.

alpha for the filtering decision is computed under the FULL (untruncated)
equal-weight mixture of participant CDFs, i.e. the natural meaning of "the top
40% of the range". The training pipeline then recomputes its own reference from
whatever data it is given, as the federated protocol requires.
"""
import argparse, collections, json, os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/chembl_deco/data.jsonl")
    ap.add_argument("--clients_file", default="data/chembl_deco/clients.json")
    ap.add_argument("--out_dir", default="data/chembl_deco_trunc")
    ap.add_argument("--rotation", type=int, default=0)
    ap.add_argument("--lo", type=float, default=0.4)
    ap.add_argument("--hi", type=float, default=0.6)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    recs = [json.loads(l) for l in open(args.data)]
    spec = json.load(open(args.clients_file))
    parts = spec["rotations"][args.rotation]["participants"]

    train = collections.defaultdict(list)
    for r in recs:
        if r.get("split") == "train" and r["client"] in parts:
            train[r["client"]].append(r)

    # full-data equal-weight mixture CDF over participants
    sorted_scores = {c: np.sort([r["score"] for r in v]) for c, v in train.items()}

    def alpha(x):
        return float(np.mean([np.searchsorted(s, x, side="right") / len(s)
                              for s in sorted_scores.values()]))

    med = {c: alpha(float(np.median(sorted_scores[c]))) for c in parts}
    pooled = float(np.median(list(med.values())))
    side = {c: ("lose_top" if med[c] < pooled else "lose_bottom") for c in parts}

    out, summary = [], {}
    for r in recs:
        c = r["client"]
        if r.get("split") != "train" or c not in parts:
            out.append(r)                      # dev/test and non-participants untouched
            continue
        a = alpha(r["score"])
        keep = a <= args.hi if side[c] == "lose_top" else a >= args.lo
        if keep:
            out.append(dict(r, alpha_full=round(a, 4)))

    print(f"{'client':14s} {'side':13s} {'alpha med':>9s} {'train':>7s} {'kept':>7s} "
          f"{'kept %':>7s} {'retained alpha':>16s}")
    for c in parts:
        kept = [r for r in out if r["client"] == c and r.get("split") == "train"]
        a = [r["alpha_full"] for r in kept]
        summary[c] = {"side": side[c], "n_train_full": len(train[c]), "n_train_kept": len(kept),
                      "alpha_median_full": med[c],
                      "retained_alpha": [round(min(a), 3), round(max(a), 3)] if a else None,
                      "removed_region": [args.hi, 1.0] if side[c] == "lose_top" else [0.0, args.lo]}
        print(f"  {c:12s} {side[c]:13s} {med[c]:9.2f} {len(train[c]):7,} {len(kept):7,} "
              f"{100*len(kept)/len(train[c]):6.1f}% "
              f"{'[%.2f, %.2f]' % (min(a), max(a)) if a else '-':>16s}")

    with open(os.path.join(args.out_dir, "data.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    spec2 = dict(spec, truncation={"lo": args.lo, "hi": args.hi, "rotation": args.rotation,
                                   "per_client": summary})
    json.dump(spec2, open(os.path.join(args.out_dir, "clients.json"), "w"), indent=1)
    n_lose_top = sum(1 for c in parts if side[c] == "lose_top")
    print(f"\nlose_top {n_lose_top} clients / lose_bottom {len(parts)-n_lose_top}; "
          f"union of retained supports still covers [0, 1]")
    print(f"wrote {args.out_dir}/data.jsonl ({sum(1 for r in out if r.get('split')=='train' and r['client'] in parts):,} participant train rows)")


if __name__ == "__main__":
    main()
