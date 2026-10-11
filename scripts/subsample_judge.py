"""Std of the LLM-judge quality metrics by test-set subsampling (needs a GPU; Qwen2.5-7B judge).

    python scripts/subsample_judge.py --arm ARM=runs/<run>/evals/eval_round_0100__<stamp>.json --arm ... \\
        [--seeds 0,1,2] [--frac 0.75] [--json out.json]

Companion of scripts/subsample_std.py / subsample_quality_style.py (same subsets). The judge prompt,
model and parser are scripts/judge_quality.py's own (imported). Differences from judge_quality.py,
needed for subsampling:
  * EVERY test article of every client is judged (judge_quality.py judges the first 25 per client),
    each summary once, cached per eval file in runs/_subsample_items/ (resumable per arm);
  * the reference baseline (gap_*) is the real summaries of the SAME subset articles of all clients,
    binned to the nearest alpha (judge_quality.py: 30 random real summaries per bin);
  * ceiling: the real summaries themselves, per client, on the same subsets.
Aggregation otherwise follows judge_quality.judge_file: per client and alpha (parse failures dropped),
in-/out-of-support means over alphas, then the mean over clients.
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from fedsteer.data import alpha_reference_from_json, read_jsonl  # noqa: E402
from subsample_std import draw_subsets  # noqa: E402

KS = ("faithful", "relevance", "coherence")


def agg(rows):
    ok = [r for r in rows if r is not None]
    return {k: (float(np.mean([r[k] for r in ok if r[k] is not None])) if any(r[k] is not None for r in ok) else None)
            for k in KS}


def judged(cache_path, items, run_judge):
    done = json.load(open(cache_path)) if os.path.exists(cache_path) else None
    if done is not None and len(done) == len(items):
        return done
    out = run_judge(items)
    json.dump(out, open(cache_path, "w"))
    return out


def aggregate(evs_clients, gen, real, ralpha, idx, alphas):
    """gen[c][i][j], real[c][i]: judge dicts (or None); idx: client -> row indices."""
    rb = defaultdict(list)
    for c, ix in idx.items():
        for i in ix:
            rb[int(np.argmin([abs(ralpha[c][i] - a) for a in alphas]))].append(real[c][i])
    rb = {j: agg(v) for j, v in rb.items()}
    cl, ceil = {}, {}
    for c, ix in idx.items():
        lo, hi = evs_clients[c]["support"]
        per_alpha = [agg([gen[c][i][j] for i in ix]) for j in range(len(alphas))]
        s = {}
        for tag, js in (("in", [j for j, a in enumerate(alphas) if lo <= a <= hi]),
                        ("out", [j for j, a in enumerate(alphas) if not lo <= a <= hi])):
            for k in KS:
                vals = [per_alpha[j][k] for j in js if per_alpha[j][k] is not None]
                s[f"{k}_{tag}"] = float(np.mean(vals)) if vals else None
                gaps = [per_alpha[j][k] - rb[j][k] for j in js
                        if j in rb and per_alpha[j][k] is not None and rb[j][k] is not None]
                s[f"gap_{k}_{tag}"] = float(np.mean(gaps)) if gaps else None
        cl[c] = s
        ceil[c] = agg([real[c][i] for i in ix])
    mean = lambda xs: float(np.mean([x for x in xs if x is not None])) if any(x is not None for x in xs) else None
    out = {k: mean([v[k] for v in cl.values()]) for k in next(iter(cl.values()))}
    out.update({f"real_{k}": mean([v[k] for v in ceil.values()]) for k in KS})
    out["per_client"] = {c: {"judge": cl[c], "real": ceil[c]} for c in idx}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="NAME=path/to/eval.json")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--frac", type=float, default=0.75)
    ap.add_argument("--judge", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--cache_dir", default=os.path.join("runs", "_subsample_items"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.cache_dir, exist_ok=True)

    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms[name] = (path, json.load(open(path)))
    first_path, base = next(iter(arms.values()))
    first_run = os.path.dirname(os.path.dirname(os.path.abspath(first_path)))
    cfg = yaml.safe_load(open(os.path.join(first_run, "config.yaml")))
    ref_json = json.load(open(os.path.join(first_run, "alpha_reference.json")))
    ref = alpha_reference_from_json(ref_json)
    for name, (path, ev) in arms.items():
        run = os.path.dirname(os.path.dirname(os.path.abspath(path)))
        if json.load(open(os.path.join(run, "alpha_reference.json"))) != ref_json:
            sys.exit(f"REFUSING: {name} has a different alpha reference")
        for c in base["clients"]:
            if ev["clients"].get(c, {}).get("record_ids") != base["clients"][c]["record_ids"]:
                sys.exit(f"REFUSING: {name}/{c}: test articles differ")
    by_url = {r["url"]: r for r in read_jsonl(cfg["data_path"])}
    alphas = base["alphas"]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from judge_quality import judge
    tok = AutoTokenizer.from_pretrained(args.judge)
    model = AutoModelForCausalLM.from_pretrained(args.judge, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    run_judge = lambda items: judge(model, tok, items, args.batch_size)

    clients = list(base["clients"])
    # real summaries of the test articles (reference and ceiling), judged once
    rflat = [(c, i) for c in clients for i in range(len(base["clients"][c]["record_ids"]))]
    ritems = [(by_url[base["clients"][c]["record_ids"][i]]["article"], by_url[base["clients"][c]["record_ids"][i]]["target"])
              for c, i in rflat]
    rs = judged(os.path.join(args.cache_dir, "judge_real__" + os.path.basename(first_run) + ".json"), ritems, run_judge)
    real, ralpha = defaultdict(list), defaultdict(list)
    for (c, i), s in zip(rflat, rs):
        real[c].append(s)
        ralpha[c].append(ref.cdf(by_url[base["clients"][c]["record_ids"][i]]["score"]))
    print("judged real summaries", flush=True)

    gens = {}
    for name, (path, ev) in arms.items():
        flat = [(c, i, j) for c in clients for i in range(len(ev["clients"][c]["outputs"])) for j in range(len(alphas))]
        items = [(by_url[ev["clients"][c]["record_ids"][i]]["article"], ev["clients"][c]["outputs"][i][j]) for c, i, j in flat]
        run = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path))))
        sc = judged(os.path.join(args.cache_dir, f"judge__{run}__{os.path.basename(path)}"), items, run_judge)
        g = {c: [[None] * len(alphas) for _ in ev["clients"][c]["outputs"]] for c in clients}
        for (c, i, j), s in zip(flat, sc):
            g[c][i][j] = s
        gens[name] = g
        print(f"judged {name}: {len(items)} summaries", flush=True)

    n = {c: len(base["clients"][c]["record_ids"]) for c in clients}
    subsets = draw_subsets(n, seeds, args.frac)
    full = {c: np.arange(n[c]) for c in clients}
    res = {name: {"full": aggregate(arms[name][1]["clients"], gens[name], real, ralpha, full, alphas),
                  "subsets": [aggregate(arms[name][1]["clients"], gens[name], real, ralpha, subsets[s], alphas)
                              for s in seeds]} for name in arms}
    label = f"{len(seeds)} subsets x {args.frac:.0%} of test articles"
    print(f"\n== LLM judge ({args.judge}), all test articles; mean +- std over {label} (full-test value in brackets)")
    for k in [k for k in next(iter(res.values()))["full"] if k != "per_client"]:
        print(f"\n-- {k}")
        for name in (arms if not k.startswith("real_") else [next(iter(arms))]):
            v = np.array([r[k] for r in res[name]["subsets"]], dtype=float)
            tag = "real summaries (ceiling)" if k.startswith("real_") else name
            print(f"   {tag:28s} {v.mean():.3f} +- {v.std(ddof=1):.3f}   [{res[name]['full'][k]:.3f}]")
    if args.json:
        json.dump({"label": label, "judge": args.judge, "files": {k: v[0] for k, v in arms.items()}, "results": res},
                  open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
