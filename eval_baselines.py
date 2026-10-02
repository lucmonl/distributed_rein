"""Evaluate baselines B1 (prompting) and B4 (federated activation steering) on the client
models of a run (each client's private adapter and calibration, learned direction removed).

    python eval_baselines.py --run runs/X --snapshot runs/X/snapshots/round_0100.pt --baseline prompt --shots 3
    python eval_baselines.py --run runs/X --snapshot ... --baseline prompt --shots 3 --model base
    python eval_baselines.py --run runs/X --snapshot ... --baseline caa --caa_layer 8

Output: <run>/evals/eval_round_XXXX_<tag>__<stamp>.json in the standard format (metrics,
near-ties, all generated texts), so scripts/score_quality.py and tie_report.py apply.
B3 (merged direction): scripts/merge_local_directions.py + eval_direction.py --shared.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from fedsteer.baselines import (caa_vector, pick_shots, prompt_with_level, result_from_grid,  # noqa: E402
                                score_grid_custom, zero_adapter, zero_direction)
from fedsteer.data import (ChatFormatter, ClientQuantiles, alpha_reference_from_json,  # noqa: E402
                           build_clients, client_support, read_jsonl)
from fedsteer.evaluate import assemble, brief, control_state  # noqa: E402
from fedsteer.fed import load_snapshot_into  # noqa: E402
from fedsteer.lora import SteerLoraConfig  # noqa: E402
from fedsteer.metrics import SCORERS  # noqa: E402
from fedsteer.model import load_model  # noqa: E402
from fedsteer.runinfo import make_stamp, provenance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--baseline", required=True, choices=["prompt", "caa"])
    ap.add_argument("--model", default="client", choices=["client", "base"],
                    help="client: private adapter of each client (direction removed); base: no adapter")
    ap.add_argument("--shots", type=int, default=0, help="B1: examples from the client's own train data")
    ap.add_argument("--caa_layer", type=int, default=None, help="B4: decoder layer (default: middle)")
    ap.add_argument("--caa_examples", type=int, default=200, help="B4: examples per side per client")
    ap.add_argument("--caa_gains", default="0.05,0.1,0.2,0.3,0.5,0.8")
    ap.add_argument("--fit_prompts", type=int, default=30, help="B4: dev articles used to fit the gain")
    ap.add_argument("--split", default="test")
    ap.add_argument("--scorer", default="density", choices=sorted(SCORERS))
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--max_prompts", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(args.run, "config.yaml")))
    local_q = {c: ClientQuantiles(v) for c, v in json.load(open(os.path.join(args.run, "client_quantiles.json"))).items()}
    ref_path = os.path.join(args.run, "alpha_reference.json")
    gref = alpha_reference_from_json(json.load(open(ref_path))) if os.path.exists(ref_path) else None
    refs = {c: (gref if gref is not None else local_q[c]) for c in local_q}
    supports = {c: client_support(local_q[c], gref) for c in local_q} if gref is not None else {}
    alphas = [float(a) for a in args.alphas.split(",")]
    score = SCORERS[args.scorer]

    by = defaultdict(lambda: defaultdict(list))
    for r in read_jsonl(cfg["data_path"]):
        if r["client"] in cfg["clients"]:
            by[r.get("split", "train")][r["client"]].append(r)
    # exactly the training examples the run used (same cap, shuffle seed and alpha protocol), so the
    # baselines see no more data than the method did
    records = [r for sp in by.values() for rs in sp.values() for r in rs]
    train, _ = build_clients(records, cfg["clients"], tie_break=cfg.get("tie_break", "average"),
                             max_train=cfg.get("max_train_per_client"), seed=cfg["fed"]["seed"],
                             alpha_mode=cfg.get("alpha_mode", "local"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(cfg["model_name"], SteerLoraConfig(**cfg["lora"]), device=device,
                            grad_checkpointing=False, attn_implementation=cfg.get("attn_implementation", "sdpa"))
    fmt = ChatFormatter(tok, system_prompt=cfg.get("system_prompt"), template_kwargs=cfg.get("chat_template_kwargs"), max_prompt_tokens=4096,
                        max_target_tokens=cfg.get("max_target_tokens", 256))
    snap = torch.load(args.snapshot, map_location="cpu", weights_only=False)

    def setup(c):
        load_snapshot_into(model, snap, c)
        zero_direction(model)
        if args.model == "base":
            zero_adapter(model)

    results, extra = {}, {}
    if args.baseline == "prompt":
        tag = f"b1_prompt_k{args.shots}_{args.model}"
        for c in cfg["clients"]:
            setup(c)
            recs = by[args.split][c][: args.max_prompts]
            prompt_fn = lambda r, a, c=c: prompt_with_level(r, a, pick_shots(train[c], a, args.shots, r.get("url")))
            grid, texts = score_grid_custom(model, fmt, recs, alphas, score, prompt_fn=prompt_fn,
                                            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
            results[c] = result_from_grid(grid, texts, recs, alphas, refs[c], supports.get(c))
            results[c].update(control_state(model))
            print(c, brief(results[c]), flush=True)
    else:
        layer = args.caa_layer if args.caa_layer is not None else len(model.model.layers) // 2
        tag = f"b4_caa_L{layer}_{args.model}"
        vecs, norms = {}, {}
        for c in cfg["clients"]:                    # client side: local CAA vectors on own data
            setup(c)
            vecs[c], norms[c] = caa_vector(model, fmt, train[c], layer, args.caa_examples)
            print(f"{c}: CAA vector norm {vecs[c].norm().item():.3f}, mean hidden norm {norms[c]:.2f}", flush=True)
        shared = torch.stack([vecs[c] for c in cfg["clients"]]).mean(0)      # server: uniform average
        unit = shared / shared.norm().clamp_min(1e-12)
        cos = {c: float(torch.nn.functional.cosine_similarity(vecs[c], shared, dim=0)) for c in vecs}
        gains = [float(g) for g in args.caa_gains.split(",")]
        for c in cfg["clients"]:
            setup(c)
            steer = lambda a, g, c=c: (g * (2 * a - 1) * norms[c] * unit).to(device)
            # client side: fit the scalar gain on a few dev articles (lowest percentile error)
            fit_recs = by["dev"][c][: args.fit_prompts]
            fit = {}
            for g in gains:
                grid, texts = score_grid_custom(model, fmt, fit_recs, alphas, score,
                                                steer_fn=lambda a, g=g: steer(a, g), layer=layer,
                                                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
                fit[g] = result_from_grid(grid, texts, fit_recs, alphas, refs[c])["pct_calib_err"]
            g_best = min(fit, key=fit.get)
            recs = by[args.split][c][: args.max_prompts]
            grid, texts = score_grid_custom(model, fmt, recs, alphas, score,
                                            steer_fn=lambda a: steer(a, g_best), layer=layer,
                                            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
            results[c] = result_from_grid(grid, texts, recs, alphas, refs[c], supports.get(c))
            results[c].update({"caa_gain": g_best, "caa_fit_pct_err": fit, "caa_cos_to_shared": cos[c],
                               "caa_layer": layer})
            print(c, brief(results[c]), flush=True)
        extra = {"caa_shared_norm": float(shared.norm()), "caa_client_cos_to_shared": cos}

    res = assemble(results, alphas, args.split, args.snapshot, snap["round"])
    res["baseline"] = {"name": args.baseline, "tag": tag, **{k: v for k, v in vars(args).items()}, **extra}
    res["provenance"] = provenance(args=vars(args))
    os.makedirs(os.path.join(args.run, "evals"), exist_ok=True)
    split = "" if args.split == "test" else f"_{args.split}"
    out = os.path.join(args.run, "evals", f"eval_round_{snap['round']:04d}{split}_{tag}__{make_stamp()}.json")
    json.dump(res, open(out, "w"), indent=1)
    print("SUMMARY", json.dumps(res["summary"]), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
