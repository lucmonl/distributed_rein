"""Recompute direction-quality metrics from an eval file's saved generations with a
different scorer -- no regeneration.

    python scripts/rescore_eval.py --run runs/X --eval runs/X/evals/eval_*.json \
        --scorer clogp_residual_strict

Used for the ChEMBL robustness check: the primary attribute scores a generation
against the core given in the prompt, and ``clogp_residual_strict`` additionally
requires that core to survive, so comparing the two says how much of the steering
result depends on generations that abandoned the core.

Writes <eval dir>/rescored_<scorer>_<eval name>.
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fedsteer.data import ClientQuantiles, alpha_reference_from_json, client_support, read_jsonl
from fedsteer.metrics import SCORERS, summarize, text_tie_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--scorer", required=True, choices=sorted(SCORERS))
    ap.add_argument("--data", default=None, help="default: the run's data_path")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(os.path.join(args.run, "config.yaml")))
    recs = read_jsonl(args.data or cfg["data_path"])
    by_id = {r.get("url", str(i)): r for i, r in enumerate(recs)}

    local_q = {c: ClientQuantiles(v) for c, v in
               json.load(open(os.path.join(args.run, "client_quantiles.json"))).items()}
    ref_path = os.path.join(args.run, "alpha_reference.json")
    gref = alpha_reference_from_json(json.load(open(ref_path))) if os.path.exists(ref_path) else None
    refs = {c: (gref if gref is not None else local_q[c]) for c in local_q}
    supports = {c: client_support(local_q[c], gref) for c in local_q} if gref is not None else {}

    # import here so metrics_for_client is the same code the eval used
    from fedsteer.metrics import metrics_for_client
    score = SCORERS[args.scorer]

    for path in [f for pat in args.eval for f in sorted(glob.glob(pat))]:
        d = json.load(open(path))
        alphas = d["alphas"]
        out = {}
        for c, res in d["clients"].items():
            texts, ids = res["outputs"], res["record_ids"]
            grid = np.array([[score(texts[i][j], by_id[ids[i]]) for j in range(len(alphas))]
                             for i in range(len(texts))], dtype=float)
            r = metrics_for_client(grid, alphas, refs[c], support=supports.get(c))
            r.update(text_tie_metrics(texts, grid, alphas))
            out[c] = r
        res_all = {"eval": os.path.basename(path), "scorer": args.scorer, "alphas": alphas,
                   "clients": out, "summary": summarize(out)}
        dst = os.path.join(os.path.dirname(path),
                           f"rescored_{args.scorer}_{os.path.basename(path)}")
        json.dump(res_all, open(dst, "w"), indent=1)

        orig = d["summary"]
        print(f"\n=== {os.path.basename(path)}  scorer={args.scorer}")
        print(f"{'metric':24s} {'original':>10s} {'rescored':>10s}")
        for k in ("pct_calib_err", "pct_err_in_support", "pct_err_out_support", "spearman",
                  "concordance", "unscorable_row_rate"):
            o = orig.get(k, {}).get("mean")
            n = res_all["summary"].get(k, {}).get("mean")
            if o is not None or n is not None:
                print(f"{k:24s} {('%.4f' % o) if o is not None else '-':>10s} "
                      f"{('%.4f' % n) if n is not None else '-':>10s}")
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
