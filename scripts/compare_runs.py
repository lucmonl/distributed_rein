"""Compare runs on test: steering, near-ties, quality (AlignScore/BERTScore/length vs the
same-alpha reference) and LLM judge, plus per-client paired bootstrap for chosen pairs.

    python scripts/compare_runs.py \\
        --run F_priv=runs/exp11_cap4k_base_... --run L_priv=runs/exp16_local_cap4k_... \\
        --pair F_priv:L_priv

Uses each run's test evaluation (the non-dev eval file; with several, the newest unless
--eval_glob narrows it) and its quality_/judge_ files.  Paired bootstrap resamples the
articles (both runs use the same test articles, in the same order, and the same global
alpha scale); negative difference = first run better (lower percentile error).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedsteer.data import ClientQuantiles, alpha_reference_from_json  # noqa: E402
from fedsteer.metrics import text_tie_metrics  # noqa: E402


def load_run(path, eval_glob):
    cands = [f for f in glob.glob(os.path.join(path, "evals", eval_glob))
             if "_dev" not in os.path.basename(f) and not os.path.basename(f).startswith(("quality_", "judge_"))]
    if not cands:
        sys.exit(f"no test eval in {path}")
    ev_path = max(cands, key=os.path.getmtime)
    ev = json.load(open(ev_path))
    name = os.path.basename(ev_path)
    q = os.path.join(path, "evals", "quality_" + name)
    j = os.path.join(path, "evals", "judge_" + name)
    ref_path = os.path.join(path, "alpha_reference.json")
    ref = alpha_reference_from_json(json.load(open(ref_path))) if os.path.exists(ref_path) else None
    if ref is None:
        lq = json.load(open(os.path.join(path, "client_quantiles.json")))
        ref = {c: ClientQuantiles(v) for c, v in lq.items()}
    return {"eval": ev, "eval_path": ev_path, "quality": json.load(open(q)) if os.path.exists(q) else None,
            "judge": json.load(open(j)) if os.path.exists(j) else None, "ref": ref}


def per_article(run, c):
    ev = run["eval"]
    alphas = np.array(ev["alphas"])
    ref = run["ref"][c] if isinstance(run["ref"], dict) else run["ref"]
    g = np.array(ev["clients"][c]["grid"])
    err = np.abs(np.vectorize(ref.cdf)(g) - alphas[None, :])
    lo, hi = ev["clients"][c].get("support", [0, 1])
    out = (alphas < lo) | (alphas > hi)
    return err.mean(1), (err[:, out].mean(1) if out.any() else None)


def boot(d, rng, B=4000):
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return d.mean(), lo, hi


def fmt_ci(m, lo, hi):
    star = "*" if hi < 0 or lo > 0 else " "
    return f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]{star}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=path")
    ap.add_argument("--pair", action="append", default=[], help="a:b  (per-client paired comparison a - b)")
    ap.add_argument("--eval_glob", default="eval_round_*__*.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    runs = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        runs[name] = load_run(path, args.eval_glob)
    rng = np.random.default_rng(args.seed)
    clients = list(next(iter(runs.values()))["eval"]["clients"])

    print("== Steering (test; mean over clients; worst client in brackets)")
    cols = ["pct_calib_err", "pct_err_in_support", "pct_err_out_support", "reach_rate", "spearman"]
    print(f"{'run':16s} {'round':>5s} " + " ".join(f"{c[:14]:>16s}" for c in cols)
          + f" {'concord_nt':>11s} {'nearTie':>8s} {'endNearTie':>10s}")
    for n, r in runs.items():
        s = r["eval"]["summary"]
        ties = [text_tie_metrics(v["outputs"], v["grid"], r["eval"]["alphas"]) for v in r["eval"]["clients"].values()]
        cells = []
        for c in cols:
            worst = f"({s[c]['worst']:.3f})" if c != "reach_rate" else ""
            cells.append(f"{s[c]['mean']:.3f}{worst:>8s}")
        print(f"{n:16s} {r['eval']['round']:5d} " + " ".join(f"{x:>16s}" for x in cells)
              + f" {np.mean([t['concordance_nt'] for t in ties]):11.3f} {np.mean([t['near_tie_rate'] for t in ties]):8.3f}"
              f" {np.mean([t['endpoint_near_tie_rate'] for t in ties]):10.3f}")

    print("\n== Quality (test; mean over clients). gap = generated - real summaries at the same alpha")
    qcols = ["align_in", "align_out", "gap_align_in", "gap_align_out", "bert_f1_in", "bert_f1_out",
             "length_in", "length_out", "gap_length_in", "gap_length_out"]
    print(f"{'run':16s} " + " ".join(f"{c:>14s}" for c in qcols))
    for n, r in runs.items():
        if r["quality"]:
            print(f"{n:16s} " + " ".join(f"{r['quality']['summary'].get(c, float('nan')):14.3f}" for c in qcols))
    print("\n== LLM judge (subsample; mean over clients)")
    jcols = ["faithful_in", "faithful_out", "gap_faithful_in", "gap_faithful_out", "relevance_in", "relevance_out",
             "gap_relevance_in", "gap_relevance_out", "coherence_in", "coherence_out"]
    print(f"{'run':16s} " + " ".join(f"{c:>17s}" for c in jcols))
    for n, r in runs.items():
        if r["judge"]:
            vals = [np.mean([v["summary"][c] for v in r["judge"]["clients"].values() if v["summary"].get(c) is not None])
                    for c in jcols]
            print(f"{n:16s} " + " ".join(f"{x:17.3f}" for x in vals))

    print("\n== Per-client percentile error (test)")
    print(f"{'client':17s} {'support':>12s} " + " ".join(f"{n:>14s}" for n in runs))
    for c in clients:
        sup = next(iter(runs.values()))["eval"]["clients"][c].get("support", [0, 1])
        print(f"{c:17s} [{sup[0]:.2f},{sup[1]:.2f}] " + " ".join(
            f"{runs[n]['eval']['clients'][c]['pct_calib_err']:14.3f}" for n in runs))

    for pair in args.pair:
        a, b = pair.split(":")
        print(f"\n== {a} - {b}: per-client paired bootstrap (negative = {a} better; * = 95% CI excludes 0)")
        print(f"{'client':17s} {'overall':>26s} {'out-of-support':>26s}")
        better = worse = 0
        for c in clients:
            ea, oa = per_article(runs[a], c)
            eb, ob = per_article(runs[b], c)
            m, lo, hi = boot(ea - eb, rng)
            better += hi < 0
            worse += lo > 0
            out = fmt_ci(*boot(oa - ob, rng)) if oa is not None and ob is not None else "n/a"
            print(f"{c:17s} {fmt_ci(m, lo, hi):>26s} {out:>26s}")
        print(f"{a} significantly better on {better}/{len(clients)} clients, worse on {worse}/{len(clients)}")


if __name__ == "__main__":
    main()
