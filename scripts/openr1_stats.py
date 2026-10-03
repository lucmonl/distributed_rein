"""Feasibility statistics for the math CoT task (OpenR1-Math-220k).

Flattens the dataset to one row per reasoning trace and reports what
`math-cot-experiment-plan.md` needs before anything is built:

  * per-source / per-(source, problem_type) counts of *usable* traces
    (reasoning complete AND Math-Verify correct), i.e. which client partitions
    clear the 4k-pairs-per-client budget;
  * trace length on the global percentile scale (equal-weight mixture CDF over
    the candidate clients, the plan's alpha_mode=global): per-client median and
    support [p5, p95];
  * the R6 check -- how much of the length variance is *within* a problem
    (different traces of the same problem) rather than between problems;
  * accuracy and completion rate by length decile (does long mean wrong?).

Length is measured in characters for every trace and converted to Qwen3 tokens
with a ratio fitted on a random subsample (also reports the rank correlation of
chars vs tokens, so the percentile scale is known to be the same either way).

Writes <data_dir>/traces.parquet and <data_dir>/stats.json.
"""
import argparse, glob, json, os, random, re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

THINK_END = "</think>"


def flatten(data_dir: str) -> pd.DataFrame:
    rows = []
    for split, pat in (("default", "data/*.parquet"), ("extended", "extended/*.parquet")):
        for f in sorted(glob.glob(os.path.join(data_dir, pat))):
            cols = ["uuid", "source", "problem_type", "question_type", "problem", "answer",
                    "generations", "is_reasoning_complete", "correctness_math_verify",
                    "correctness_llama", "finish_reasons"]
            t = pq.read_table(f, columns=cols).to_pandas()
            for r in t.itertuples(index=False):
                gens = list(r.generations)
                comp = list(r.is_reasoning_complete) if r.is_reasoning_complete is not None else [None] * len(gens)
                mv = list(r.correctness_math_verify) if r.correctness_math_verify is not None else [None] * len(gens)
                ll = list(r.correctness_llama) if r.correctness_llama is not None else [None] * len(gens)
                fr = list(r.finish_reasons) if r.finish_reasons is not None else [None] * len(gens)
                for k, g in enumerate(gens):
                    think, sep, ans = g.partition(THINK_END)
                    rows.append(dict(
                        uuid=r.uuid, split=split, source=r.source, problem_type=r.problem_type,
                        question_type=r.question_type, k=k, n_traces=len(gens),
                        problem_chars=len(r.problem), chars=len(g),
                        think_chars=len(think) if sep else len(g), answer_chars=len(ans) if sep else 0,
                        has_think_end=bool(sep),
                        # paragraph count inside <think>: a deterministic "step" proxy
                        steps=len([p for p in think.split("\n\n") if p.strip()]),
                        complete=comp[k] if k < len(comp) else None,
                        mv=mv[k] if k < len(mv) else None,
                        llama=ll[k] if k < len(ll) else None,
                        finish=fr[k] if k < len(fr) else None,
                    ))
    return pd.DataFrame(rows)


def token_ratio(data_dir, df, model, n=3000, seed=0):
    from transformers import AutoTokenizer
    from scipy.stats import spearmanr
    tok = AutoTokenizer.from_pretrained(model)
    idx = df.sample(n=min(n, len(df)), random_state=seed).index
    want = set(zip(df.loc[idx, "uuid"], df.loc[idx, "k"]))
    texts = {}
    for pat in ("data/*.parquet", "extended/*.parquet"):
        for f in sorted(glob.glob(os.path.join(data_dir, pat))):
            t = pq.read_table(f, columns=["uuid", "generations"]).to_pandas()
            for r in t.itertuples(index=False):
                for k, g in enumerate(r.generations):
                    if (r.uuid, k) in want:
                        texts[(r.uuid, k)] = g
    keys = list(texts)
    ntok = np.array([len(tok(texts[k], add_special_tokens=False).input_ids) for k in keys])
    nch = np.array([len(texts[k]) for k in keys])
    ratio = float(np.median(nch / ntok))
    rho = float(spearmanr(nch, ntok).correlation)
    return ratio, rho, int(len(keys))


def mixture_cdf(client_scores: dict):
    sorted_ = {c: np.sort(v) for c, v in client_scores.items()}

    def F(x):
        x = np.asarray(x, dtype=float)
        return np.mean([np.searchsorted(s, x, side="right") / len(s) for s in sorted_.values()], axis=0)
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/openr1_math")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--min_traces", type=int, default=4000, help="usable traces for a client")
    ap.add_argument("--max_tokens", type=int, default=8192, help="length cap considered for training")
    args = ap.parse_args()

    cache = os.path.join(args.data_dir, "traces.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
    else:
        df = flatten(args.data_dir)
        df.to_parquet(cache)
    out = {}
    ratio, rho, n_tok = token_ratio(args.data_dir, df, args.tokenizer)
    df["tokens"] = df["chars"] / ratio
    out["chars_per_token"] = dict(median=ratio, spearman_chars_tokens=rho, n=n_tok)
    print(f"chars/token {ratio:.2f}  spearman(chars,tokens) {rho:.4f}  (n={n_tok})")

    out["overall"] = dict(
        problems=int(df.uuid.nunique()), traces=len(df),
        traces_per_problem=df.groupby("uuid").size().value_counts().sort_index().to_dict(),
        complete=float(df.complete.mean()), mv_correct=float(df.mv.mean()),
        tokens_pct={p: float(np.percentile(df.tokens, p)) for p in (5, 25, 50, 75, 95, 99)},
    )
    print(json.dumps(out["overall"], indent=1, default=str))

    use = df[(df.complete == True) & (df.mv == True)].copy()
    out["usable"] = dict(traces=len(use), problems=int(use.uuid.nunique()),
                         tokens_pct={p: float(np.percentile(use.tokens, p)) for p in (5, 25, 50, 75, 95, 99)},
                         share_le_cap={c: float((use.tokens <= c).mean()) for c in (2048, 4096, 8192, 12288)})
    print("usable", json.dumps(out["usable"], indent=1))

    # accuracy / completion by length decile, all traces
    df["dec"] = pd.qcut(df.tokens, 10, labels=False)
    out["by_length_decile"] = df.groupby("dec").agg(
        tok_med=("tokens", "median"), complete=("complete", "mean"), mv=("mv", "mean")).round(3).to_dict("index")
    print(pd.DataFrame(out["by_length_decile"]).T)

    # client candidates: source, and source x problem_type
    for name, keys in (("source", ["source"]), ("source_type", ["source", "problem_type"])):
        g = use.groupby(keys).agg(traces=("tokens", "size"), problems=("uuid", "nunique"),
                                  tok_med=("tokens", "median"),
                                  tok_p5=("tokens", lambda x: np.percentile(x, 5)),
                                  tok_p95=("tokens", lambda x: np.percentile(x, 95)),
                                  steps_med=("steps", "median")).sort_values("tok_med")
        out[f"clients_{name}"] = {" / ".join(map(str, k)) if isinstance(k, tuple) else k: v
                                  for k, v in g.round(1).to_dict("index").items()}
        print(f"\n== candidate clients by {name} (usable traces) ==")
        print(g.round(0).to_string())

    # global percentile scale over eligible source clients (problems, not traces, as the unit)
    elig = [s for s, v in out["clients_source"].items() if v["traces"] >= args.min_traces]
    sc = {s: use.loc[use.source == s, "tokens"].values for s in elig}
    F = mixture_cdf(sc)
    tab = {}
    for s in elig:
        a = F(sc[s])
        tab[s] = dict(n=len(a), alpha_med=float(np.median(a)), support=[float(np.percentile(a, 5)), float(np.percentile(a, 95))])
    out["global_scale_source"] = tab
    meds = [v["alpha_med"] for v in tab.values()]
    print(f"\n== global scale over {len(elig)} sources with >= {args.min_traces} usable traces ==")
    for s, v in sorted(tab.items(), key=lambda kv: kv[1]["alpha_med"]):
        print(f"{s:20s} n={v['n']:6d}  alpha(med)={v['alpha_med']:.2f}  support=[{v['support'][0]:.2f}, {v['support'][1]:.2f}]")
    grid = np.linspace(0.05, 0.95, 19)
    cover = [float(np.mean([(v["support"][0] <= x <= v["support"][1]) for v in tab.values()])) for x in grid]
    out["coverage_by_alpha"] = dict(zip([round(x, 2) for x in grid], cover))
    out["uncovered_share_mean"] = float(np.mean([1 - (v["support"][1] - v["support"][0]) for v in tab.values()]))
    print("client-median spread", round(max(meds) - min(meds), 2), " mean uncovered share", round(out["uncovered_share_mean"], 2))

    # R6: within-problem spread among usable traces of the same problem
    use["alpha"] = F(use.tokens.values)
    use["logt"] = np.log(use.tokens)
    multi = use.groupby("uuid").filter(lambda x: len(x) >= 2)
    within = multi.groupby("uuid").logt.var(ddof=0).mean()
    total = multi.logt.var(ddof=0)
    spread = multi.groupby("uuid").alpha.agg(lambda x: x.max() - x.min())
    ratio_ml = multi.groupby("uuid").tokens.agg(lambda x: x.max() / x.min())
    out["within_problem"] = dict(
        problems_with_ge2_usable=int(multi.uuid.nunique()),
        share_of_logvar_within=float(within / total),
        alpha_range_median=float(spread.median()), alpha_range_p25=float(spread.quantile(.25)),
        alpha_range_p75=float(spread.quantile(.75)),
        max_over_min_len_median=float(ratio_ml.median()),
    )
    print("\nwithin-problem:", json.dumps(out["within_problem"], indent=1))

    # same thing with traces capped at max_tokens (the training regime if long traces are dropped)
    cap = use[use.tokens <= args.max_tokens]
    out["under_cap"] = dict(cap=args.max_tokens, traces=len(cap), problems=int(cap.uuid.nunique()),
                            per_source={s: int((cap.source == s).sum()) for s in elig})
    print("under cap", json.dumps(out["under_cap"], indent=1))

    with open(os.path.join(args.data_dir, "stats.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
