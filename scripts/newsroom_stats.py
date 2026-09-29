"""Week-1 check (plan gate G0): per-publication statistics of Newsroom, and
validation of our fragment scorer against the dataset's precomputed values.

    python scripts/newsroom_stats.py --src data/newsroom/release --out data/newsroom_stats

Writes
  per_publication.csv     counts per split, year range, density/coverage quantiles, lengths
  scorer_validation.json  agreement of fedsteer.extractive with Newsroom's density/coverage
  records.parquet-like    records.npz with compact per-record stats (for the build script)
"""

import argparse
import csv
import gzip
import json
import os
import sys
import time
import zlib
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedsteer.extractive import fragment_stats, publication  # noqa: E402

SPLITS = ("train", "dev", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/newsroom/release")
    ap.add_argument("--out", default="data/newsroom_stats")
    ap.add_argument("--validate_n", type=int, default=3000, help="records (from dev) to rescore")
    ap.add_argument("--limit", type=int, default=0, help="debug: records per split")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pubs: dict[str, int] = {}
    cols = defaultdict(list)
    val = {"ours_density": [], "ref_density": [], "ours_coverage": [], "ref_coverage": [], "ours_comp": [], "ref_comp": []}
    for split_id, split in enumerate(SPLITS):
        t0 = time.time()
        path = os.path.join(args.src, f"{split}.jsonl.gz")
        with gzip.open(path, "rt") as f:
            for n, line in enumerate(f):
                if args.limit and n >= args.limit:
                    break
                r = json.loads(line)
                pub = publication(r["url"])
                pid = pubs.setdefault(pub, len(pubs))
                summ = r["summary"] or ""
                text = r["text"] or ""
                cols["pub"].append(pid)
                cols["split"].append(split_id)
                cols["line"].append(n)
                cols["year"].append(int(r["date"][:4]) if r.get("date") else -1)
                cols["density"].append(float(r["density"]))
                cols["coverage"].append(float(r["coverage"]))
                cols["compression"].append(float(r["compression"]))
                cols["summary_words"].append(len(summ.split()))
                cols["text_words"].append(len(text.split()))
                cols["summary_hash"].append(zlib.crc32(summ.strip().lower().encode()))
                if split == "dev" and len(val["ref_density"]) < args.validate_n:
                    st = fragment_stats(summ, text)
                    val["ours_density"].append(st["density"])
                    val["ref_density"].append(float(r["density"]))
                    val["ours_coverage"].append(st["coverage"])
                    val["ref_coverage"].append(float(r["coverage"]))
                    val["ours_comp"].append(st["compression"])
                    val["ref_comp"].append(float(r["compression"]))
                if n % 200000 == 0:
                    print(f"{split}: {n} records, {time.time() - t0:.0f}s", flush=True)
        print(f"{split}: done, {n + 1} records, {time.time() - t0:.0f}s", flush=True)

    arr = {k: np.asarray(v) for k, v in cols.items()}
    names = np.array(sorted(pubs, key=pubs.get))
    np.savez_compressed(os.path.join(args.out, "records.npz"), names=names, **arr)

    # ---- scorer validation
    def agree(a, b):
        a, b = np.asarray(a), np.asarray(b)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-9)
        return {"n": int(ok.sum()), "pearson": float(pearsonr(a, b)[0]), "spearman": float(spearmanr(a, b)[0]),
                "frac_within_5pct": float((rel <= 0.05).mean()), "frac_within_10pct": float((rel <= 0.10).mean()),
                "median_abs_err": float(np.median(np.abs(a - b)))}
    validation = {
        "density": agree(val["ours_density"], val["ref_density"]),
        "coverage": agree(val["ours_coverage"], val["ref_coverage"]),
        "compression": agree(val["ours_comp"], val["ref_comp"]),
    }
    with open(os.path.join(args.out, "scorer_validation.json"), "w") as f:
        json.dump(validation, f, indent=1)
    print("scorer validation:", json.dumps(validation, indent=1))

    # ---- per-publication table
    rows = []
    for pid, name in enumerate(names):
        m = arr["pub"] == pid
        tr = m & (arr["split"] == 0)
        d = arr["density"][tr]
        if tr.sum() == 0:
            continue
        h = arr["summary_hash"][tr]
        _, counts = np.unique(h, return_counts=True)
        q = np.quantile(d, [0.1, 0.25, 0.5, 0.75, 0.9])
        yrs = arr["year"][m]
        rows.append({
            "publication": name,
            "n_train": int(tr.sum()), "n_dev": int((m & (arr["split"] == 1)).sum()),
            "n_test": int((m & (arr["split"] == 2)).sum()),
            "year_min": int(yrs.min()), "year_p50": int(np.median(yrs)), "year_max": int(yrs.max()),
            "density_p10": q[0], "density_p25": q[1], "density_p50": q[2], "density_p75": q[3], "density_p90": q[4],
            "density_iqr": q[3] - q[1],
            "coverage_p50": float(np.median(arr["coverage"][tr])),
            "summary_words_p50": float(np.median(arr["summary_words"][tr])),
            "text_words_p50": float(np.median(arr["text_words"][tr])),
            "dup_summary_frac": float(1 - len(counts) / tr.sum()),
        })
    rows.sort(key=lambda r: -r["n_train"])
    with open(os.path.join(args.out, "per_publication.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"{len(rows)} publications written to {args.out}/per_publication.csv")

    # split date ranges (is the official split temporal?)
    for s, split in enumerate(SPLITS):
        y = arr["year"][arr["split"] == s]
        print(f"{split}: years {y.min()}-{y.max()}, median {np.median(y)}")


if __name__ == "__main__":
    main()
