"""Backbone screen for the math-CoT task (EXPERIMENT_LOG entry 39): which backbone can be
fine-tuned on the NuminaMath-CoT (GPT-4o) solutions without losing accuracy?

    python scripts/backbone_screen.py --model meta-llama/Llama-3.1-8B-Instruct --out_dir runs/exp39_screen_llama31_8b

Two measurements, no training:
  familiarity  mean target-token NLL (and perplexity) of the reference solutions under the
               model, per client, with the exact chat formatting used in training.  Low
               perplexity = the data is in the model's own style (Ren et al., EMNLP 2024);
               G0 showed that Qwen3-4B-Instruct-2507 loses accuracy when fine-tuned on it
               (entry 38).
  headroom     greedy base accuracy / boxed / truncation / length on test problems from a
               few clients, at a long cap so the base model is not truncated (E0a lesson).
               If the base model already beats the data, SFT cannot help.

Writes <out_dir>_<stamp>/screen.json and evals/eval_base_test__<stamp>.json (standard eval
format, alphas = [0.0]) so scripts/score_math.py applies.
"""
import argparse, json, os, random, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fedsteer.data import ChatFormatter, read_jsonl  # noqa: E402
from fedsteer.lora import SteerLoraConfig  # noqa: E402
from fedsteer.mathcot import cot_tokens, extract_boxed, is_correct, repetition_loop  # noqa: E402
from fedsteer.model import generate_at_alpha, load_model  # noqa: E402
from fedsteer.monitor import mean_loss  # noqa: E402
from fedsteer.runinfo import make_stamp, provenance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/math_fed/data.jsonl")
    ap.add_argument("--clients_file", default="data/math_fed/clients.json")
    ap.add_argument("--n_ppl", type=int, default=504, help="train solutions for familiarity, balanced over clients")
    ap.add_argument("--acc_clients", default="gsm8k+cn_k12/Geometry+math+olympiads")
    ap.add_argument("--acc_prompts", type=int, default=50, help="test problems per accuracy client")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--enable_thinking", action="store_true", help="Qwen3 hybrids only; default off")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    stamp = make_stamp()
    run = f"{args.out_dir.rstrip('/')}_{stamp}"
    os.makedirs(os.path.join(run, "evals"), exist_ok=True)
    clients = json.load(open(args.clients_file))["clients"]
    recs = read_jsonl(args.data)
    train, test = defaultdict(list), defaultdict(list)
    for r in recs:
        if r["split"] == "train":
            train[r["client"]].append(r)
        elif r["split"] == "test":
            test[r["client"]].append(r)

    model, tok = load_model(args.model, SteerLoraConfig(), grad_checkpointing=False)
    # the LoRA is injected with both B matrices at zero, i.e. exactly the base model
    fmt = ChatFormatter(tok, template_kwargs={"enable_thinking": args.enable_thinking},
                        max_prompt_tokens=512, max_target_tokens=1024)
    out = {"model": args.model, "familiarity": {}, "headroom": {}}

    # --- familiarity: target-token NLL per client (alpha is irrelevant: the direction is zero)
    rng = random.Random(0)
    per = max(1, args.n_ppl // len(clients))
    nlls = []
    for c in clients:
        ex = [dict(r, alpha=0.0) for r in rng.sample(train[c], min(per, len(train[c])))]
        nll = mean_loss(model, fmt, ex, batch_size=8)
        out["familiarity"][c] = {"nll": nll, "ppl": float(np.exp(nll)), "n": len(ex)}
        nlls.append(nll)
        print(f"familiarity {c:30s} nll {nll:.4f} ppl {np.exp(nll):.3f}", flush=True)
    out["familiarity_mean"] = {"nll": float(np.mean(nlls)), "ppl": float(np.exp(np.mean(nlls)))}
    print(f"FAMILIARITY mean nll {np.mean(nlls):.4f} ppl {np.exp(np.mean(nlls)):.3f}", flush=True)

    # --- headroom: base accuracy at a long cap
    results = {}
    for c in args.acc_clients.split("+"):
        rs = test[c][: args.acc_prompts]
        gens = generate_at_alpha(model, fmt, [r["prompt"] for r in rs], 0.0,
                                 max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        texts = [g[0] for g in gens]
        toks = np.array([cot_tokens(t) for t in texts])
        row = {"n": len(rs),
               "accuracy": float(np.mean([is_correct(t, r["answer"]) for t, r in zip(texts, rs)])),
               "boxed": float(np.mean([extract_boxed(t) is not None for t in texts])),
               "truncated": float(np.mean(toks >= args.max_new_tokens - 1)),
               "loop": float(np.mean([repetition_loop(t) for t in texts])),
               "tokens_median": float(np.median(toks)),
               "ref_tokens_median": float(np.median([r["score"] for r in rs]))}
        out["headroom"][c] = row
        results[c] = {"outputs": [[t] for t in texts], "grid": [[float(x)] for x in toks],
                      "record_ids": [r["url"] for r in rs]}
        print(f"headroom {c:30s} " + " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                              for k, v in row.items()), flush=True)
    acc = [v["accuracy"] for v in out["headroom"].values()]
    out["headroom_mean_accuracy"] = float(np.mean(acc))
    print(f"HEADROOM mean accuracy {np.mean(acc):.3f}", flush=True)

    out["provenance"] = provenance(args=vars(args))
    json.dump(out, open(os.path.join(run, "screen.json"), "w"), indent=1)
    json.dump({"snapshot": None, "round": 0, "split": "test", "alphas": [0.0], "clients": results,
               "model": args.model, "provenance": provenance(args=vars(args))},
              open(os.path.join(run, "evals", f"eval_base_test__{stamp}.json"), "w"), indent=1)
    print(f"RUN={run}", flush=True)


if __name__ == "__main__":
    main()
