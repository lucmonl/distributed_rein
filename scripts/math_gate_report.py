"""Gate report for the math-CoT task (math-cot-experiment-plan.md §8): one client, from the saved
eval files only (no generation).

    python scripts/math_gate_report.py --base runs/<E0a run> --sft runs/<G0 run> --steer runs/<G2 run> \
        --client math

Prints, for the client:
  * per alpha: target length F^-1(alpha), generated length (median, IQR), percentile error,
    accuracy, boxed, truncated, loop -- for the steered run's test eval;
  * accuracy / length / format of base (E0a), plain SFT (G0) and steering (G2), on the SAME test
    problems, with paired differences (McNemar-style counts) against the base and against SFT;
  * the dev curve of each training run (pct err, Spearman, dev loss, dev accuracy if scored);
  * the gate verdicts G0 and G2 with the plan's thresholds.
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fedsteer.data import ClientQuantiles, alpha_reference_from_json, read_jsonl
from fedsteer.mathcot import cot_tokens, extract_boxed, is_correct, repetition_loop


def newest(pattern):
    fs = sorted(glob.glob(pattern), key=os.path.getmtime)
    return fs[-1] if fs else None


def test_eval(run):
    """The run's test eval of the selected checkpoint (newest non-dev, non-baseline eval)."""
    fs = [f for f in glob.glob(os.path.join(run, "evals", "eval_round_*.json"))
          if "_dev" not in os.path.basename(f) and "b1_" not in f and "b4_" not in f]
    return max(fs, key=os.path.getmtime) if fs else None


def per_output(texts, golds, cap):
    toks = np.array([cot_tokens(t) for t in texts])
    return {"correct": np.array([is_correct(t, g) for t, g in zip(texts, golds)]),
            "boxed": np.array([extract_boxed(t) is not None for t in texts]),
            "trunc": toks >= cap - 1, "loop": np.array([repetition_loop(t) for t in texts]), "tokens": toks}


def paired(a, b):
    """counts of (a right, b wrong) and (a wrong, b right)"""
    return int(np.sum(a & ~b)), int(np.sum(~a & b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--sft", required=True)
    ap.add_argument("--steer", required=True)
    ap.add_argument("--client", default="math")
    ap.add_argument("--data", default="data/math_fed/data.jsonl")
    ap.add_argument("--cap", type=int, default=1280)
    ap.add_argument("--base_cap", type=int, default=None, help="default: from the base eval's provenance")
    args = ap.parse_args()
    c = args.client
    gold = {r["url"]: r["answer"] for r in read_jsonl(args.data)}

    # --- steered run, per alpha
    ev = json.load(open(test_eval(args.steer)))
    alphas, res = ev["alphas"], ev["clients"][c]
    ids = res["record_ids"]
    ref = alpha_reference_from_json(json.load(open(os.path.join(args.steer, "alpha_reference.json"))))
    local = ClientQuantiles(json.load(open(os.path.join(args.steer, "client_quantiles.json")))[c])
    print(f"== {c}: steered run {os.path.basename(args.steer)}, test eval round {ev['round']}, n={len(ids)}")
    print(f"   summary pct err {res['pct_calib_err']:.3f} | in-support {res.get('pct_err_in_support')} | "
          f"out-of-support {res.get('pct_err_out_support')} | Spearman {res['spearman']:.3f} | "
          f"concordance {res['concordance']:.3f} | adjacent decrease {res.get('adjacent_decrease_rate'):.3f} | "
          f"gain {res.get('gain')}")
    lo, hi = ref.cdf(local.quantile(0.05)), ref.cdf(local.quantile(0.95))
    print(f"   client support on its own scale [{lo:.2f}, {hi:.2f}]")
    print(f"   {'alpha':>5} {'target':>7} {'gen p25':>8} {'median':>7} {'p75':>6} {'pct err':>8} "
          f"{'acc':>5} {'boxed':>6} {'trunc':>6} {'loop':>5}")
    steer_out = {}
    for j, a in enumerate(alphas):
        texts = [res["outputs"][i][j] for i in range(len(ids))]
        o = per_output(texts, [gold[i] for i in ids], args.cap)
        steer_out[a] = o
        pe = np.mean([abs(ref.cdf(t) - a) for t in o["tokens"]])
        q = np.percentile(o["tokens"], [25, 50, 75])
        print(f"   {a:5.2f} {ref.quantile(a):7.0f} {q[0]:8.0f} {q[1]:7.0f} {q[2]:6.0f} {pe:8.3f} "
              f"{o['correct'].mean():5.2f} {o['boxed'].mean():6.2f} {o['trunc'].mean():6.2f} {o['loop'].mean():5.2f}")

    # --- base and SFT on the same problems
    bev_path = newest(os.path.join(args.base, "evals", "eval_base_*.json"))
    bev = json.load(open(bev_path))
    bcap = args.base_cap or bev.get("provenance", {}).get("args", {}).get("max_new_tokens", args.cap)
    bres = bev["clients"][c]
    bmap = dict(zip(bres["record_ids"], [o[0] for o in bres["outputs"]]))
    sev = json.load(open(test_eval(args.sft)))
    sres = sev["clients"][c]
    smap = dict(zip(sres["record_ids"], [o[0] for o in sres["outputs"]]))   # alpha-independent (D = 0)
    common = [i for i in ids if i in bmap and i in smap]
    B = per_output([bmap[i] for i in common], [gold[i] for i in common], bcap)
    S = per_output([smap[i] for i in common], [gold[i] for i in common], args.cap)
    k = [ids.index(i) for i in common]
    print(f"\n== same {len(common)} test problems: base (cap {bcap}) vs plain SFT (G0, round {sev['round']}) vs steering (G2)")
    print(f"   {'model':22s} {'acc':>5} {'boxed':>6} {'trunc':>6} {'loop':>5} {'tokens med':>11}")
    for name, o in (("base", B), ("SFT (G0)", S)):
        print(f"   {name:22s} {o['correct'].mean():5.2f} {o['boxed'].mean():6.2f} {o['trunc'].mean():6.2f} "
              f"{o['loop'].mean():5.2f} {np.median(o['tokens']):11.0f}")
    for a, o in steer_out.items():
        oo = {kk: v[k] for kk, v in o.items()}
        print(f"   {'G2 alpha=' + format(a, 'g'):22s} {oo['correct'].mean():5.2f} {oo['boxed'].mean():6.2f} "
              f"{oo['trunc'].mean():6.2f} {oo['loop'].mean():5.2f} {np.median(oo['tokens']):11.0f}"
              f"   vs SFT +{paired(oo['correct'], S['correct'])[0]}/-{paired(oo['correct'], S['correct'])[1]}")
    sb = paired(S["correct"], B["correct"])
    print(f"   SFT vs base: SFT right & base wrong {sb[0]}, SFT wrong & base right {sb[1]}")
    untr = ~B["trunc"]
    print(f"   on problems where base finished (n={untr.sum()}): base acc {B['correct'][untr].mean():.2f}, "
          f"SFT acc {S['correct'][untr].mean():.2f}")

    # --- dev curves
    for name, run in (("G0", args.sft), ("G2", args.steer)):
        print(f"\n== dev curve {name}: {os.path.basename(run)}")
        rows = {}
        for f in glob.glob(os.path.join(run, "evals", "eval_round_*_dev__*.json")):
            d = json.load(open(f))
            r = d["clients"][c]
            m = os.path.join(run, "evals", "math_" + os.path.basename(f))
            acc = None
            if os.path.exists(m):
                pc = json.load(open(m))["per_client"][c]
                acc = [round(x["accuracy"], 2) for x in pc]
            rows[d["round"]] = (r["pct_calib_err"], r["spearman"], r.get("loss"), acc)
        for rnd in sorted(rows):
            pe, sp, loss, acc = rows[rnd]
            print(f"   round {rnd:3d}  pct err {pe:.3f}  Spearman {sp:.3f}  dev loss {loss:.4f}  dev acc by alpha {acc}")

    # --- gates
    print("\n== gates")
    g0 = dict(boxed=S["boxed"].mean() >= 0.95, loop=S["loop"].mean() <= 0.05, trunc=S["trunc"].mean() <= 0.05,
              acc=S["correct"].mean() >= B["correct"].mean() - 0.05)
    print(f"   G0 (SFT vs base cap {bcap}): {g0}  -> {'PASS' if all(g0.values()) else 'FAIL'}"
          + ("   [base reference is cap-limited; rerun at 4096 pending]" if bcap < 4096 else ""))
    in_acc = np.mean([steer_out[a]["correct"].mean() for a in alphas if lo <= a <= hi])
    g2 = dict(pct_err=res["pct_calib_err"] < 0.20, spearman=res["spearman"] >= 0.7,
              acc_in_support=in_acc >= S["correct"].mean() - 0.05)
    print(f"   G2: pct err {res['pct_calib_err']:.3f} (<0.20), Spearman {res['spearman']:.3f} (>=0.7), "
          f"in-support acc {in_acc:.2f} vs SFT {S['correct'].mean():.2f} (-0.05): {g2} "
          f"-> {'PASS' if all(g2.values()) else 'FAIL'}")


if __name__ == "__main__":
    main()
