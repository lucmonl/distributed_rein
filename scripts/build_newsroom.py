"""Build the federated Newsroom dataset (plan §3.1).

    python scripts/build_newsroom.py --stats data/newsroom_stats --out data/newsroom_fed

Steps
  1. Filter candidate pairs per publication (summary/article length, templated summaries).
  2. Select clients: publications with enough usable pairs, spread over the range of
     median density (evenly spaced in that ranking).
  3. Per client, by date:  the latest ``n_drift`` pairs form the drift stream; train /
     dev pools come from earlier official-train pairs; the test set comes from the
     official test split, restricted to the same (pre-drift) period.
  4. Truncate articles, recompute density/coverage/compression on the truncated
     article with fedsteer.extractive (the same scorer used at evaluation), and drop
     pairs whose summary relies on the truncated-away part.

Outputs  <out>/data.jsonl    records {client, split, prompt, target, score, article, ...}
         <out>/clients.json  selection, per-client stats, held-out rotations
"""

import argparse
import gzip
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedsteer.extractive import fragment_stats  # noqa: E402

SPLITS = ("train", "dev", "test")
PROMPT = "Write a short summary of the following news article.\n\nArticle:\n{article}"


def truncate_words(text: str, max_words: int) -> str:
    out, n = [], 0
    for para in text.split("\n"):
        words = para.split()
        if not words:
            continue
        if n + len(words) > max_words:
            out.append(" ".join(words[: max_words - n]))
            break
        out.append(" ".join(words))
        n += len(words)
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/newsroom/release")
    ap.add_argument("--stats", default="data/newsroom_stats")
    ap.add_argument("--out", default="data/newsroom_fed")
    ap.add_argument("--n_clients", type=int, default=12)
    ap.add_argument("--clients", nargs="*", default=None, help="explicit publication list (overrides selection)")
    ap.add_argument("--min_usable", type=int, default=5000)
    ap.add_argument("--n_train_pool", type=int, default=5000)
    ap.add_argument("--n_dev", type=int, default=100)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--n_drift", type=int, default=600, help="latest pairs per client (3 drift stages of 200)")
    ap.add_argument("--min_summary_words", type=int, default=8)
    ap.add_argument("--max_summary_words", type=int, default=80)
    ap.add_argument("--min_article_words", type=int, default=150)
    ap.add_argument("--max_article_words", type=int, default=400)
    ap.add_argument("--max_coverage_loss", type=float, default=0.1,
                    help="drop pairs whose coverage drops by more than this after truncation")
    ap.add_argument("--min_pre_drift", type=int, default=3000,
                    help="min usable pre-drift train pairs for a year-level temporal split")
    ap.add_argument("--n_rotations", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    z = np.load(os.path.join(args.stats, "records.npz"))
    names = z["names"]
    pub, split, line, year = z["pub"], z["split"], z["line"], z["year"]

    # ---- 1. filters (on precomputed stats)
    ok = (z["summary_words"] >= args.min_summary_words) & (z["summary_words"] <= args.max_summary_words) \
        & (z["text_words"] >= args.min_article_words) & (year > 0)
    # templated summaries: the same summary string used >= 2 times within a publication
    key = pub.astype(np.int64) * (1 << 32) + z["summary_hash"].astype(np.int64)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    ok &= cnt[inv] == 1

    # ---- 2. client selection
    usable = {}
    for pid, name in enumerate(names):
        m = ok & (pub == pid)
        n_tr, n_te = int((m & (split == 0)).sum()), int((m & (split == 2)).sum())
        if n_tr >= args.min_usable and n_te >= args.n_test:
            usable[str(name)] = float(np.median(z["density"][m & (split == 0)]))
    if args.clients:
        missing = [c for c in args.clients if c not in usable]
        if missing:
            raise ValueError(f"not enough usable data for {missing}")
        chosen = list(args.clients)
    else:
        ranked = sorted(usable, key=usable.get)
        if len(ranked) < args.n_clients:
            raise ValueError(f"only {len(ranked)} usable publications: {ranked}")
        idx = np.linspace(0, len(ranked) - 1, args.n_clients).round().astype(int)
        chosen = [ranked[i] for i in idx]
    chosen = sorted(chosen, key=usable.get)  # ordered by median density
    print("usable publications (median ref density):",
          json.dumps({k: round(v, 2) for k, v in sorted(usable.items(), key=lambda kv: kv[1])}))
    print("chosen:", chosen)

    # ---- 3. per-client splits by date
    wanted: dict[tuple[int, int], tuple[str, str]] = {}   # (split_id, line) -> (client, our split)
    name_to_pid = {str(n): i for i, n in enumerate(names)}
    plan = {}
    for c in chosen:
        pid = name_to_pid[c]
        tr_idx = np.where(ok & (pub == pid) & (split == 0))[0]
        # sort by year, random order within a year (the stats file keeps only the year;
        # drift records are re-sorted by full date after reading)
        tr_idx = tr_idx[np.random.default_rng(args.seed + pid).permutation(len(tr_idx))]
        tr_idx = tr_idx[np.argsort(year[tr_idx], kind="stable")]
        # oversample 1.3x below to leave room for the truncation filter
        n_drift_pool = int(args.n_drift * 1.3)
        n_pool = int((args.n_train_pool + args.n_dev) * 1.3)
        te_all = np.where(ok & (pub == pid) & (split == 2))[0]
        # drift = the latest whole year(s) holding >= n_drift_pool pairs; train/dev/test
        # strictly earlier.  Fall back to "latest pairs" if too little earlier data.
        years_desc = sorted(set(year[tr_idx].tolist()), reverse=True)
        cutoff_year, temporal = None, False
        for y in years_desc:
            if (year[tr_idx] >= y).sum() >= n_drift_pool:
                cutoff_year = y
                break
        if cutoff_year is not None:
            early = tr_idx[year[tr_idx] < cutoff_year]
            te_early = te_all[year[te_all] < cutoff_year]
            temporal = len(early) >= args.min_pre_drift and len(te_early) >= int(args.n_test * 1.3)
        if temporal:
            late = tr_idx[year[tr_idx] >= cutoff_year]
            drift_idx = late[-n_drift_pool:]   # random within the latest year (see shuffle above)
            early_idx, te_idx = list(early), te_early.tolist()
        else:
            drift_idx, early_idx = tr_idx[-n_drift_pool:], list(tr_idx[:-n_drift_pool])
            cutoff_year = int(year[drift_idx].min())
            te_idx = te_all[year[te_all] <= cutoff_year].tolist()
        rng.shuffle(early_idx)
        rng.shuffle(te_idx)
        te_idx = te_idx[: int(args.n_test * 1.3)]
        plan[c] = {"cutoff_year": int(cutoff_year), "temporal": bool(temporal)}
        for i in early_idx[:n_pool]:
            wanted[(int(split[i]), int(line[i]))] = (c, "train_pool")
        for i in drift_idx:
            wanted[(int(split[i]), int(line[i]))] = (c, "drift")
        for i in te_idx:
            wanted[(int(split[i]), int(line[i]))] = (c, "test")

    # ---- 4. read selected records, truncate, rescore
    collected = defaultdict(lambda: defaultdict(list))
    dropped = defaultdict(int)
    for sid, sname in enumerate(SPLITS):
        lines = {ln for (s, ln) in wanted if s == sid}
        if not lines:
            continue
        with gzip.open(os.path.join(args.src, f"{sname}.jsonl.gz"), "rt") as f:
            for n, raw in enumerate(f):
                if n not in lines:
                    continue
                c, part = wanted[(sid, n)]
                r = json.loads(raw)
                art = truncate_words(r["text"], args.max_article_words)
                full = fragment_stats(r["summary"], r["text"])
                tr = fragment_stats(r["summary"], art)
                if tr["coverage"] < full["coverage"] - args.max_coverage_loss:
                    dropped[c] += 1
                    continue
                summary = " ".join(r["summary"].split())
                collected[c][part].append({
                    "client": c, "prompt": PROMPT.format(article=art), "target": summary,
                    "score": tr["density"], "coverage": tr["coverage"], "compression": tr["compression"],
                    "ref_density": float(r["density"]), "article": art, "title": r.get("title", ""),
                    "date": r["date"], "year": int(r["date"][:4]), "url": r["url"], "source_split": sname,
                })
        print(f"read {sname}", flush=True)

    # ---- assemble final splits
    out_rows, stats = [], {}
    for c in chosen:
        parts = collected[c]
        pool = parts["train_pool"]
        rng.shuffle(pool)
        dev, train = pool[: args.n_dev], pool[args.n_dev: args.n_dev + args.n_train_pool]
        test = parts["test"][: args.n_test]
        drift = sorted(parts["drift"], key=lambda r: r["date"])[-args.n_drift:]
        stage = max(len(drift) // 3, 1)
        for k, r in enumerate(drift):
            r["drift_stage"] = min(k // stage, 2)
        for name, rows in (("train", train), ("dev", dev), ("test", test), ("drift", drift)):
            for r in rows:
                r["split"] = name
            out_rows.extend(rows)
        d = np.array([r["score"] for r in train])
        stats[c] = {
            "n": {s: len(v) for s, v in (("train", train), ("dev", dev), ("test", test), ("drift", drift))},
            "dropped_truncation": dropped[c],
            "cutoff_year": plan[c]["cutoff_year"],
            "temporal_split": plan[c]["temporal"],
            "train_years": [min(r["year"] for r in train), max(r["year"] for r in train)],
            "density_quantiles_train": {q: round(float(np.quantile(d, q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "drift_years": [drift[0]["year"], drift[-1]["year"]] if drift else None,
        }
        short = [s for s, v in stats[c]["n"].items() if v < {"train": 1000, "dev": args.n_dev,
                                                               "test": args.n_test, "drift": 300}[s]]
        if short:
            print(f"WARNING {c}: small splits {short}: {stats[c]['n']}")

    # held-out rotations: stratified over the density ranking (chosen is sorted by density)
    rotations = []
    for r in range(args.n_rotations):
        held = [chosen[i] for i in range(r, len(chosen), args.n_rotations)]
        rotations.append({"held_out": held, "participants": [c for c in chosen if c not in held]})

    with open(os.path.join(args.out, "data.jsonl"), "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out, "clients.json"), "w") as f:
        json.dump({"clients": chosen, "median_ref_density": {c: usable[c] for c in chosen},
                   "rotations": rotations, "stats": stats, "args": vars(args), "prompt": PROMPT}, f, indent=1)
    print(json.dumps(stats, indent=1))
    print(f"wrote {len(out_rows)} records to {args.out}/data.jsonl")


if __name__ == "__main__":
    main()
