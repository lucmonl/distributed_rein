"""Step 2 of the math CoT feasibility probe: the per-problem *target ladder*.

`openr1_stats.py` showed that natural R1 traces of the same problem barely differ
in length (6% of log-length variance is within-problem), so length is fixed by
the input and alpha cannot vary within a problem (criterion R6).  This script
measures the fix: give every problem several real or derived targets at
different lengths, all ending in the same verified answer --

  solution  the dataset's reference solution (no thinking)
  answer    R1's post-</think> write-up (no thinking)
  cut25/50/75  R1's <think> cut at that fraction of its paragraphs, closed with
            </think>, followed by R1's write-up (derived)
  full      the complete R1 trace(s)

and reports, for candidate clients (source x problem_type with enough problems):
the global percentile scale over the ladder, per-client median and support, the
within-problem alpha range, and the gap between rungs.

Reads data/openr1_math/traces.parquet (from openr1_stats.py) and the raw
parquet for the solution lengths.  Writes data/openr1_math/ladder_stats.json.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RUNGS = ["solution", "answer", "cut25", "cut50", "cut75", "full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/openr1_math")
    ap.add_argument("--cap", type=int, default=8192, help="max target tokens")
    ap.add_argument("--min_problems", type=int, default=1000)
    ap.add_argument("--n_clients", type=int, default=12)
    args = ap.parse_args()
    st = json.load(open(os.path.join(args.data_dir, "stats.json")))
    R = st["chars_per_token"]["median"]

    df = pd.read_parquet(os.path.join(args.data_dir, "traces.parquet"))
    use = df[(df.complete == True) & (df.mv == True) & df.has_think_end & (df.source != "olympiads_ref")].copy()
    sol = pd.concat([pq.read_table(f, columns=["uuid", "solution"]).to_pandas()
                     for f in sorted(glob.glob(os.path.join(args.data_dir, "*/*.parquet")))]).drop_duplicates("uuid")
    sol = sol[sol.uuid.isin(use.uuid)]
    meta = use.drop_duplicates("uuid")[["uuid", "source", "problem_type"]]

    first = use.sort_values("k").drop_duplicates("uuid")  # one trace per problem for the derived rungs
    rows = [pd.DataFrame({"uuid": sol.uuid, "rung": "solution", "tok": sol.solution.str.len() / R}),
            pd.DataFrame({"uuid": first.uuid, "rung": "answer", "tok": first.answer_chars / R})]
    for f in (25, 50, 75):
        rows.append(pd.DataFrame({"uuid": first.uuid, "rung": f"cut{f}",
                                  "tok": (first.think_chars * f / 100 + first.answer_chars) / R}))
    rows.append(pd.DataFrame({"uuid": use.uuid, "rung": "full", "tok": use.chars / R}))
    lad = pd.concat(rows).merge(meta, on="uuid")
    lad = lad[(lad.tok >= 20) & (lad.tok <= args.cap)]
    lad["client"] = lad.source + "/" + lad.problem_type
    out = {"cap": args.cap, "chars_per_token": R}

    # candidate clients: enough problems, then spread over the median of the full traces
    g = lad.groupby("client").agg(problems=("uuid", "nunique"))
    g["full_med"] = lad[lad.rung == "full"].groupby("client").tok.median()
    g = g[g.problems >= args.min_problems].sort_values("full_med")
    out["candidates"] = g.round(0).to_dict("index")
    print(g.round(0).to_string())
    idx = np.unique(np.round(np.linspace(0, len(g) - 1, args.n_clients)).astype(int))
    clients = list(g.index[idx])
    out["selected"] = clients

    def table(d, name):
        S = {c: np.sort(d.tok[d.client == c].values) for c in clients}
        F = lambda v: np.mean([np.searchsorted(S[c], v, side="right") / len(S[c]) for c in clients], axis=0)
        d = d[d.client.isin(clients)].copy()
        d["a"] = F(d.tok.values)
        res = {}
        print(f"\n== {name}: global scale over {len(clients)} clients ==")
        for c in clients:
            a = d.a[d.client == c]
            res[c] = dict(n=int(len(a)), problems=int(d.uuid[d.client == c].nunique()),
                          alpha_med=float(a.median()), support=[float(a.quantile(.05)), float(a.quantile(.95))])
            print(f"{c:32s} problems={res[c]['problems']:6d} med={res[c]['alpha_med']:.2f} "
                  f"support=[{res[c]['support'][0]:.2f}, {res[c]['support'][1]:.2f}]")
        rungs = {r: [float(x) for x in np.percentile(d.a[d.rung == r], [5, 50, 95])] for r in RUNGS if (d.rung == r).any()}
        for r, v in rungs.items():
            print(f"  rung {r:8s} alpha p5/p50/p95 = {v[0]:.2f} / {v[1]:.2f} / {v[2]:.2f}")
        multi = d.groupby("uuid").filter(lambda x: len(x) >= 2)
        lt = np.log(multi.tok)
        within = float(multi.assign(lt=lt).groupby("uuid").lt.var(ddof=0).mean() / lt.var(ddof=0))
        rng = multi.groupby("uuid").a.agg(lambda x: x.max() - x.min())
        # largest alpha gap between consecutive targets of one problem
        gap = multi.groupby("uuid").a.agg(lambda x: np.max(np.diff(np.sort(x.values))))
        meds = [v["alpha_med"] for v in res.values()]
        summ = dict(within_share=within, alpha_range_median=float(rng.median()),
                    max_gap_median=float(gap.median()), median_spread=float(max(meds) - min(meds)),
                    uncovered_mean=float(np.mean([1 - (v["support"][1] - v["support"][0]) for v in res.values()])))
        print("  ", json.dumps({k: round(v, 3) for k, v in summ.items()}))
        return dict(clients=res, rungs=rungs, **summ)

    out["natural_full_only"] = table(lad[lad.rung == "full"], "full R1 traces only")
    out["ladder_natural"] = table(lad[lad.rung.isin(["solution", "answer", "full"])], "solution + answer + full")
    out["ladder_with_cuts"] = table(lad, "all rungs incl. truncated thinking")
    with open(os.path.join(args.data_dir, "ladder_stats.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
