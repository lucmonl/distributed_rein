"""Build the federated math-CoT splits from NuminaMath-CoT (math-cot-experiment-plan.md §3).

    python scripts/build_math_fed.py --out_dir data/math_fed

Writes, in the same layout as data/newsroom_fed/ and data/chembl_fed/:
  data.jsonl    {client, split, prompt, target, score, answer, url, source, problem_type}
  clients.json  {clients, median_tokens, support, rotations, attribute, ruler}

* target  = the GPT-4o step-by-step solution (one per problem), ending in \\boxed{}.
* score   = its length in tokens under a FIXED ruler tokenizer (Qwen3-4B-Instruct-2507),
            the same function the `cot_tokens` scorer applies to generations.
* answer  = the content of the solution's last \\boxed{} (gold answer for accuracy).
* clients = NuminaMath sources; the three large, metadata-joinable sources (cn_k12,
            orca_math, synthetic_math) are split by NuminaMath-1.5's problem_type.
* splits are by problem; exact-duplicate problems (after whitespace normalisation) are
  kept once, preferring the original benchmark sources (gsm8k, math, ...).
"""
import argparse, glob, json, os, random, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedsteer.mathcot import RULER, extract_boxed, ruler  # noqa: E402

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
def make_prompt(problem: str) -> str:
    return f"{problem}\n\n{INSTRUCTION}"

# 12 clients, spread over the median solution length (scripts/numina_cot_stats.py and the
# 10-03 candidate table in EXPERIMENT_LOG entry 37); <= 3 per source family
CLIENTS = [
    "orca_math/Logic and Puzzles", "cn_k12/Logic and Puzzles", "synthetic_math/Algebra",
    "gsm8k", "cn_k12/Inequalities", "synthetic_math/Geometry", "orca_math/Algebra",
    "cn_k12/Geometry", "synthetic_amc", "math", "olympiads", "aops_forum",
]
SPLIT_SOURCES = {"cn_k12", "orca_math", "synthetic_math"}
# duplicate problems are kept under the first of these sources
SOURCE_PRIORITY = ["gsm8k", "math", "amc_aime", "aops_forum", "olympiads", "synthetic_amc",
                   "cn_k12", "synthetic_math", "orca_math"]


def load(cot_dir: str, n15_dir: str) -> pd.DataFrame:
    c = pd.concat([pd.read_parquet(f, columns=["source", "problem", "solution"])
                   for f in sorted(glob.glob(os.path.join(cot_dir, "data/train-*.parquet")))])
    a = pd.concat([pd.read_parquet(f, columns=["problem", "problem_type"])
                   for f in sorted(glob.glob(os.path.join(n15_dir, "data/*.parquet")))])
    norm = lambda s: s.str.strip().str.replace(r"\s+", " ", regex=True)
    c["key"], a["key"] = norm(c.problem), norm(a.problem)
    pri = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
    c["pri"] = c.source.map(pri).fillna(len(pri))
    n0 = len(c)
    c = c.sort_values(["pri"], kind="stable").drop_duplicates("key", keep="first")
    print(f"dropped {n0 - len(c):,} duplicate problems")
    c = c.merge(a.drop_duplicates("key")[["key", "problem_type"]], on="key", how="left")
    c["client"] = np.where(c.source.isin(SPLIT_SOURCES), c.source + "/" + c.problem_type.fillna("NA"), c.source)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot_dir", default="data/numinamath_cot")
    ap.add_argument("--n15_dir", default="data/numinamath_15")
    ap.add_argument("--out_dir", default="data/math_fed")
    ap.add_argument("--max_train", type=int, default=4000)
    ap.add_argument("--n_dev", type=int, default=50)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--max_target_tokens", type=int, default=1024)
    ap.add_argument("--max_prompt_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    tok = ruler()

    df = load(args.cot_dir, args.n15_dir)
    df = df[df.client.isin(CLIENTS)]
    df["answer"] = df.solution.map(extract_boxed)
    n0 = len(df)
    df = df[df.answer.notna() & (df.answer.str.strip() != "")]
    print(f"dropped {n0 - len(df):,} solutions without a \\boxed{{}} answer")

    out, summary = [], {}
    for c in CLIENTS:
        rows = df[df.client == c].sample(frac=1.0, random_state=rng.randrange(1 << 30)).to_dict("records")
        need = args.n_dev + args.n_test + args.max_train
        kept, i = [], 0
        # tokenize in chunks until enough rows pass the length filters
        while len(kept) < need and i < len(rows):
            chunk = rows[i:i + 4096]
            i += len(chunk)
            t_len = [len(x) for x in tok([r["solution"] for r in chunk], add_special_tokens=False).input_ids]
            p_len = [len(x) for x in tok([make_prompt(r["problem"]) for r in chunk],
                                         add_special_tokens=False).input_ids]
            for r, tl, pl in zip(chunk, t_len, p_len):
                if tl <= args.max_target_tokens and pl <= args.max_prompt_tokens - 32:
                    kept.append(dict(r, score=float(tl)))
        if len(kept) < args.n_dev + args.n_test + 1000:
            raise SystemExit(f"{c}: only {len(kept)} usable problems")
        splits = (["dev"] * args.n_dev + ["test"] * args.n_test
                  + ["train"] * min(args.max_train, len(kept) - args.n_dev - args.n_test))
        for k, (r, sp) in enumerate(zip(kept, splits)):
            out.append({"client": c, "split": sp, "prompt": make_prompt(r["problem"]),
                        "target": r["solution"], "score": r["score"], "answer": r["answer"],
                        "problem": r["problem"], "source": r["source"],
                        "problem_type": r["problem_type"] if isinstance(r["problem_type"], str) else None,
                        # generic record id slot (evaluate.record_ids, baselines.pick_shots)
                        "url": f"{c}/{k}"})
        tr = np.array([r["score"] for r, sp in zip(kept, splits) if sp == "train"])
        summary[c] = {"n_train": int(len(tr)), "n_dev": args.n_dev, "n_test": args.n_test,
                      "median_tokens": float(np.median(tr)), "p5": float(np.percentile(tr, 5)),
                      "p95": float(np.percentile(tr, 95))}
        print(f"  {c:30s} train {len(tr):5,}  median {summary[c]['median_tokens']:5.0f}  "
              f"p5-p95 {summary[c]['p5']:4.0f}-{summary[c]['p95']:4.0f} tokens")

    # supports on the global (equal-weight mixture) scale over all 12 clients, for the record;
    # each run recomputes its own reference from its participants
    S = {c: np.sort([r["score"] for r in out if r["client"] == c and r["split"] == "train"]) for c in CLIENTS}
    F = lambda v: float(np.mean([np.searchsorted(S[c], v, side="right") / len(S[c]) for c in CLIENTS]))
    support = {c: [F(np.percentile(S[c], 5)), F(np.percentile(S[c], 95))] for c in CLIENTS}
    order = sorted(CLIENTS, key=lambda c: summary[c]["median_tokens"])
    rotations = [{"held_out": order[i::3][:4],
                  "participants": [c for c in order if c not in set(order[i::3][:4])]}
                 for i in range(3)]
    with open(os.path.join(args.out_dir, "data.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    json.dump({"clients": order, "median_tokens": {c: summary[c]["median_tokens"] for c in order},
               "support_all12": support, "summary": summary, "rotations": rotations,
               "attribute": "cot_tokens", "ruler": RULER, "instruction": INSTRUCTION},
              open(os.path.join(args.out_dir, "clients.json"), "w"), indent=1)
    print(f"\nwrote {len(out):,} records to {args.out_dir}/data.jsonl")
    for c in order:
        print(f"  {c:30s} support on the 12-client scale [{support[c][0]:.2f}, {support[c][1]:.2f}]")
    print(f"rotation 0 held out: {rotations[0]['held_out']}")


if __name__ == "__main__":
    main()
