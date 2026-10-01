"""Score the quality of generated summaries saved in eval files.

    python scripts/score_quality.py --run runs/X --eval runs/X/evals/eval_round_0060__<stamp>.json

For every generated summary: AlignScore (consistency with the article), BERTScore F1
against the reference summary, length, repeated-trigram rate, empty flag
(fedsteer/quality.py).  Aggregated per client and target alpha, and split into
in-support / out-of-support alphas.

Reference baseline (matched extractiveness, Ladhak et al. 2022): the *real* summaries
of all participating clients on the same split, binned by their global alpha, give
the quality a summary at that alpha normally has.  ``gap_*`` = generated - reference
at the same alpha; for out-of-support cells this says whether the generated summaries
are as good as real ones at that extractiveness level.

Writes <run>/evals/quality_<eval file name>.json; reference scores are cached in
<run>/evals/quality_refs_<split>.json.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedsteer.data import ClientQuantiles, alpha_reference_from_json, read_jsonl  # noqa: E402
from fedsteer.quality import AlignScorer, BERTScorer, surface  # noqa: E402

METRICS = ("align", "bert_f1", "length", "rep3", "empty")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--no_refs", action="store_true", help="skip the reference baseline")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(args.run, "config.yaml")))
    local_q = {c: ClientQuantiles(v) for c, v in json.load(open(os.path.join(args.run, "client_quantiles.json"))).items()}
    ref_path = os.path.join(args.run, "alpha_reference.json")
    gref = alpha_reference_from_json(json.load(open(ref_path))) if os.path.exists(ref_path) else None
    records = read_jsonl(cfg["data_path"])
    by_url = {r["url"]: r for r in records}   # all clients: E2 evaluates held-out (non-participant) clients

    align = AlignScorer(device=args.device, batch_size=args.batch_size)
    bert = BERTScorer(device=args.device)

    for eval_path in args.eval:
        ev = json.load(open(eval_path))
        split, alphas = ev["split"], ev["alphas"]

        # ---- reference baseline for this split (cached)
        ref_bins = None
        if not args.no_refs:
            cache = os.path.join(args.run, "evals", f"quality_refs_{split}.json")
            refs = json.load(open(cache)) if os.path.exists(cache) else {}
            todo = [r for r in records if r.get("split") == split and r["client"] in cfg["clients"]
                    and r["url"] not in refs]
            if todo:
                al = align.score([r["article"] for r in todo], [r["target"] for r in todo])
                for r, a in zip(todo, al):
                    q = gref if gref is not None else local_q[r["client"]]
                    refs[r["url"]] = {"client": r["client"], "alpha": q.cdf(r["score"]), "align": a,
                                      **surface(r["target"])}
                json.dump(refs, open(cache, "w"))
            # bin reference summaries to the nearest alpha of the grid
            ref_bins = defaultdict(lambda: defaultdict(list))
            for v in refs.values():
                j = int(np.argmin([abs(v["alpha"] - a) for a in alphas]))
                for m in ("align", "length", "rep3"):
                    ref_bins[j][m].append(v[m])
            ref_bins = {j: {m: mean(v) for m, v in d.items()} | {"n": len(d["align"])} for j, d in ref_bins.items()}

        # ---- generated summaries
        out = {"eval": eval_path, "split": split, "alphas": alphas, "reference_by_alpha": ref_bins, "clients": {}}
        for c, res in ev["clients"].items():
            if "outputs" not in res:
                sys.exit(f"{eval_path} has no saved outputs (produced before outputs were saved)")
            texts = res["outputs"]
            recs = [by_url[u] for u in res["record_ids"]]
            flat = [(i, j) for i in range(len(texts)) for j in range(len(alphas))]
            cands = [texts[i][j] for i, j in flat]
            al = align.score([recs[i]["article"] for i, _ in flat], cands)
            bs = bert.score(cands, [recs[i]["target"] for i, _ in flat])
            cell = defaultdict(lambda: defaultdict(list))
            for (i, j), a, b, t in zip(flat, al, bs, cands):
                for m, v in {"align": a, "bert_f1": b["F1"], **surface(t)}.items():
                    cell[j][m].append(v)
            per_alpha = [{m: mean(cell[j][m]) for m in METRICS} for j in range(len(alphas))]
            lo, hi = res.get("support", [0.0, 1.0])
            ins = [lo <= a <= hi for a in alphas]
            summary = {}
            for tag, sel in (("in", ins), ("out", [not x for x in ins])):
                js = [j for j, s in enumerate(sel) if s]
                for m in METRICS:
                    summary[f"{m}_{tag}"] = mean([per_alpha[j][m] for j in js]) if js else None
                if ref_bins is not None and js:
                    for m in ("align", "length"):
                        gaps = [per_alpha[j][m] - ref_bins[j][m] for j in js if j in ref_bins]
                        summary[f"gap_{m}_{tag}"] = mean(gaps) if gaps else None
            out["clients"][c] = {"support": [lo, hi], "per_alpha": per_alpha, "summary": summary}
            print(c, json.dumps(summary), flush=True)

        tab = {k: mean([v["summary"].get(k) for v in out["clients"].values()])
               for k in next(iter(out["clients"].values()))["summary"]}
        out["summary"] = tab
        print("QUALITY SUMMARY", json.dumps(tab), flush=True)
        path = os.path.join(os.path.dirname(eval_path), "quality_" + os.path.basename(eval_path))
        json.dump(out, open(path, "w"), indent=1)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
