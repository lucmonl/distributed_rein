"""Evaluate a learned steering direction on each client (plan §4, E1).

    python eval_direction.py --run runs/toy_length_fedavg --scorer words
    python eval_direction.py --run runs/toy_local --shared runs/toy_local/merged_shared.pt   # baseline B3

For every client and test prompt, generate at each alpha on the grid, score the
outputs, and report steerability (order rate, Spearman), normalized range and
calibration MAE against the client's own quantiles F_i^{-1}(alpha).
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

from fedsteer.data import ChatFormatter, ClientQuantiles, read_jsonl
from fedsteer.fed import load_snapshot_into
from fedsteer.lora import SteerLoraConfig
from fedsteer.model import generate_at_alpha, load_model

SCORERS = {
    "words": lambda text: float(len(text.split())),
}


def latest_snapshot(run: str) -> str:
    d = os.path.join(run, "snapshots")
    return os.path.join(d, sorted(os.listdir(d))[-1])


def metrics_for_client(scores: np.ndarray, alphas: list[float], q: ClientQuantiles) -> dict:
    """scores: [n_prompts, n_alphas]"""
    order = np.all(np.diff(scores, axis=1) > 0, axis=1).mean()
    rhos = [spearmanr(alphas, s).correlation for s in scores if np.ptp(s) > 0]
    rng = (scores[:, -1] - scores[:, 0]).mean() / max(q.iqr(), 1e-9)
    targets = np.array([q.quantile(a) for a in alphas])
    mae = np.abs(scores - targets[None, :]).mean() / max(q.iqr(), 1e-9)
    return {
        "order_rate": float(order),
        "spearman": float(np.mean(rhos)) if rhos else 0.0,
        "norm_range": float(rng),
        "calib_mae_iqr": float(mae),
        "mean_score_by_alpha": scores.mean(axis=0).round(3).tolist(),
        "target_by_alpha": targets.round(3).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--snapshot", default=None, help="defaults to the latest snapshot")
    ap.add_argument("--shared", default=None, help="override direction (e.g. merged_shared.pt)")
    ap.add_argument("--scorer", default="words", choices=sorted(SCORERS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--max_prompts", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(os.path.join(args.run, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(args.run, "client_quantiles.json")) as f:
        quantiles = {c: ClientQuantiles(v) for c, v in json.load(f).items()}
    snap_path = args.snapshot or latest_snapshot(args.run)
    snap = torch.load(snap_path, map_location="cpu", weights_only=False)
    shared = None
    if args.shared:
        shared = torch.load(args.shared, map_location="cpu")
        shared = {k: v for k, v in shared.items() if k.endswith("lora_B_d")}
    alphas = [float(a) for a in args.alphas.split(",")]
    score = SCORERS[args.scorer]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(cfg["model_name"], SteerLoraConfig(**cfg["lora"]), device=device,
                            grad_checkpointing=False, attn_implementation=cfg.get("attn_implementation", "sdpa"))
    fmt = ChatFormatter(tok, system_prompt=cfg.get("system_prompt"),
                        max_prompt_tokens=cfg.get("max_prompt_tokens", 1024),
                        max_target_tokens=cfg.get("max_target_tokens", 256))

    by_client = defaultdict(list)
    for r in read_jsonl(cfg["data_path"]):
        if r.get("split") == args.split and r["client"] in cfg["clients"]:
            by_client[r["client"]].append(r)

    results, samples = {}, {}
    for c in cfg["clients"]:
        load_snapshot_into(model, snap, c, shared=shared)
        prompts = [r["prompt"] for r in by_client[c][: args.max_prompts]]
        grid = np.zeros((len(prompts), len(alphas)))
        samples[c] = []
        for j, a in enumerate(alphas):
            outs = generate_at_alpha(model, fmt, prompts, a, max_new_tokens=args.max_new_tokens,
                                     batch_size=args.batch_size)
            grid[:, j] = [score(o[0]) for o in outs]
            samples[c].append({"alpha": a, "output": outs[0][0]})
        results[c] = metrics_for_client(grid, alphas, quantiles[c])
        results[c]["gain"] = float(torch.exp(snap["clients"][c]["gain"]["steer_control.u"]).item())
        print(c, json.dumps(results[c]), flush=True)

    keys = ["order_rate", "spearman", "norm_range", "calib_mae_iqr"]
    summary = {k: {"mean": float(np.mean([results[c][k] for c in results])),
                   "worst": float(min(results[c][k] for c in results) if k != "calib_mae_iqr"
                                  else max(results[c][k] for c in results))} for k in keys}
    print("SUMMARY", json.dumps(summary), flush=True)
    out = args.out or os.path.join(args.run, f"eval_{os.path.basename(snap_path).removesuffix('.pt')}"
                                   f"{'_' + os.path.basename(args.shared).removesuffix('.pt') if args.shared else ''}.json")
    with open(out, "w") as f:
        json.dump({"snapshot": snap_path, "shared": args.shared, "alphas": alphas, "clients": results,
                   "summary": summary, "samples": samples}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
