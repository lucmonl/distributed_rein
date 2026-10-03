"""E0a (math-cot-experiment-plan.md §7): the plain backbone on each client's test problems.

    python scripts/eval_base_math.py --model Qwen/Qwen3-4B-Instruct-2507 --out_dir runs/exp37_math_e0a_base

No adapter, no alpha: one greedy generation per problem with the task prompt.  The
steering LoRA is injected with both B matrices at zero, which is exactly the base model,
so generation goes through the same code path (and chat template) as every other eval.
Writes <out_dir>_<stamp>/evals/eval_base_<split>__<stamp>.json in the standard eval format
(alphas = [0.0]), so scripts/score_math.py applies unchanged.
"""
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from fedsteer.data import ChatFormatter, read_jsonl  # noqa: E402
from fedsteer.lora import SteerLoraConfig  # noqa: E402
from fedsteer.metrics import SCORERS  # noqa: E402
from fedsteer.model import generate_at_alpha, load_model  # noqa: E402
from fedsteer.runinfo import make_stamp, provenance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data", default="data/math_fed/data.jsonl")
    ap.add_argument("--clients_file", default="data/math_fed/clients.json")
    ap.add_argument("--clients", default=None, help="'+'-separated subset (default: all in clients_file)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_prompts", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=1280)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--out_dir", required=True, help="prefix; a timestamped directory is created")
    args = ap.parse_args()

    stamp = make_stamp()
    run = f"{args.out_dir.rstrip('/')}_{stamp}"
    os.makedirs(os.path.join(run, "evals"), exist_ok=True)
    clients = json.load(open(args.clients_file))["clients"]
    if args.clients:
        want = args.clients.split("+")
        clients = [c for c in clients if c in want]
        if len(clients) != len(want):
            raise SystemExit(f"unknown clients in {want}")
    by_client = defaultdict(list)
    for r in read_jsonl(args.data):
        if r["split"] == args.split and r["client"] in clients:
            by_client[r["client"]].append(r)

    model, tok = load_model(args.model, SteerLoraConfig(), grad_checkpointing=False)
    fmt = ChatFormatter(tok, template_kwargs={"enable_thinking": args.enable_thinking},
                        max_prompt_tokens=512, max_target_tokens=1024)
    score = SCORERS["cot_tokens"]
    results = {}
    for c in clients:
        recs = by_client[c][: args.max_prompts]
        outs = generate_at_alpha(model, fmt, [r["prompt"] for r in recs], 0.0,
                                 max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        texts = [[o[0]] for o in outs]
        grid = [[score(t[0], r)] for t, r in zip(texts, recs)]
        results[c] = {"outputs": texts, "grid": grid, "record_ids": [r["url"] for r in recs]}
        mean_len = sum(g[0] for g in grid) / max(len(grid), 1)
        print(f"{c:30s} n={len(recs)} mean tokens {mean_len:.0f}", flush=True)
    res = {"snapshot": None, "round": 0, "split": args.split, "alphas": [0.0], "clients": results,
           "model": args.model, "provenance": provenance(args=vars(args))}
    out = os.path.join(run, "evals", f"eval_base_{args.split}__{stamp}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"RUN={run}\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
