"""Feasibility statistics for a *short*-CoT math task (AI-MO/NuminaMath-CoT).

Alternative to OpenR1-Math (`openr1_stats.py`), whose R1 thinking traces have a
median of ~2.4k tokens.  NuminaMath-CoT has one GPT-4o step-by-step solution per
problem and a `source` field.  Reports, for candidate clients (source, and
source x a coarse topic when a source is too broad):

  * CoT length in backbone tokens (chars converted with a ratio fitted on a sample);
  * per-client median and support [p5, p95] on the global percentile scale
    (equal-weight mixture CDF over the candidate clients);
  * how much of the length is predictable from the problem text (TF-IDF ridge,
    problems held out), overall and within a client -- the learnability check;
  * share of solutions with a \\boxed{} answer (needed for accuracy).

Writes <data_dir>/stats.json.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/numinamath_cot")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--min_rows", type=int, default=5000)
    ap.add_argument("--sample_per_client", type=int, default=6000, help="for the predictability fit")
    args = ap.parse_args()

    df = pd.concat([pd.read_parquet(f, columns=["source", "problem", "solution"])
                    for f in sorted(glob.glob(os.path.join(args.data_dir, "data/train-*.parquet")))])
    df = df.reset_index(drop=True)
    out = {"rows": len(df)}

    from transformers import AutoTokenizer
    from scipy.stats import spearmanr
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    smp = df.sample(n=3000, random_state=0)
    nt = np.array([len(tok(s, add_special_tokens=False).input_ids) for s in smp.solution])
    nc = smp.solution.str.len().values
    ratio = float(np.median(nc / nt))
    out["chars_per_token"] = dict(median=ratio, spearman=float(spearmanr(nc, nt).correlation))
    df["tok"] = df.solution.str.len() / ratio
    df["boxed"] = df.solution.str.contains(r"\\boxed", regex=True)
    print(f"rows {len(df)}  chars/token {ratio:.2f}  spearman {out['chars_per_token']['spearman']:.3f}")
    out["tokens_pct"] = {p: float(np.percentile(df.tok, p)) for p in (5, 25, 50, 75, 95, 99)}
    print("solution tokens pct", {k: round(v) for k, v in out["tokens_pct"].items()})

    g = df.groupby("source").agg(rows=("tok", "size"), tok_med=("tok", "median"),
                                 tok_p5=("tok", lambda x: np.percentile(x, 5)),
                                 tok_p95=("tok", lambda x: np.percentile(x, 95)),
                                 boxed=("boxed", "mean")).sort_values("tok_med")
    print(g.round(2).to_string())
    out["sources"] = g.round(3).to_dict("index")

    clients = list(g.index[g.rows >= args.min_rows])
    S = {c: np.sort(df.tok[df.source == c].values) for c in clients}
    F = lambda v: np.mean([np.searchsorted(S[c], v, side="right") / len(S[c]) for c in clients], axis=0)
    tab = {}
    print(f"\n== global scale over {len(clients)} sources with >= {args.min_rows} rows ==")
    for c in clients:
        a = F(S[c])
        tab[c] = dict(n=len(a), alpha_med=float(np.median(a)),
                      support=[float(np.percentile(a, 5)), float(np.percentile(a, 95))])
        print(f"{c:16s} n={len(a):7d} med={tab[c]['alpha_med']:.2f} "
              f"support=[{tab[c]['support'][0]:.2f}, {tab[c]['support'][1]:.2f}]")
    out["global_scale"] = tab
    meds = [v["alpha_med"] for v in tab.values()]
    widths = [v["support"][1] - v["support"][0] for v in tab.values()]
    out["median_spread"], out["mean_width"] = float(max(meds) - min(meds)), float(np.mean(widths))
    print(f"median spread {out['median_spread']:.2f}  mean width {out['mean_width']:.2f}")

    # learnability: is the length predictable from the problem text?
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from scipy.sparse import hstack, csr_matrix
    d = df[df.source.isin(clients)].groupby("source", group_keys=False).apply(
        lambda x: x.sample(min(len(x), args.sample_per_client), random_state=0)).reset_index(drop=True)
    d["ll"] = np.log(d.tok.clip(lower=5))
    rng = np.random.RandomState(0)
    te_mask = rng.rand(len(d)) < 0.2
    tr, te = d[~te_mask], d[te_mask]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True)
    oh = pd.get_dummies(d.source).astype(float)
    feats = lambda x, X: hstack([X, csr_matrix(oh.loc[x.index].values),
                                 csr_matrix(np.log(x.problem.str.len().values + 1)[:, None])]).tocsr()
    Xtr, Xte = vec.fit_transform(tr.problem), vec.transform(te.problem)
    pr = Ridge(alpha=3.0).fit(feats(tr, Xtr), tr.ll).predict(feats(te, Xte))
    base_tot = te.ll.mean()
    base_cli = te.source.map(tr.groupby("source").ll.mean())
    sse = ((te.ll - pr) ** 2).sum()
    out["predictability"] = dict(
        client_share=float(1 - d.groupby("source").ll.transform(lambda x: x - x.mean()).var(ddof=0) / d.ll.var(ddof=0)),
        r2_total=float(1 - sse / ((te.ll - base_tot) ** 2).sum()),
        r2_within_client=float(1 - sse / ((te.ll - base_cli) ** 2).sum()))
    print("predictability", json.dumps({k: round(v, 3) for k, v in out["predictability"].items()}))
    with open(os.path.join(args.data_dir, "stats.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
