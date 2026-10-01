"""Evaluate a learned steering direction on each client (plan §4, E1).

    python eval_direction.py --run runs/toy_length_fedavg --scorer words
    python eval_direction.py --run runs/toy_local --shared runs/toy_local/merged_shared.pt   # baseline B3

    # warp option F (ablation): remap alpha post hoc using the dev split, then evaluate on test
    python eval_direction.py --run runs/X --scorer density --posthoc_remap

    # checkpoint sweep on the dev split (one model load, several snapshots)
    python eval_direction.py --run runs/X --split dev --dev_loss \
        --snapshot runs/X/snapshots/round_0020.pt runs/X/snapshots/round_0040.pt

For every client and prompt, generate at each alpha on the grid, score the outputs
and report direction-quality metrics (fedsteer.metrics).  Select checkpoints on
``--split dev``; report on ``--split test``.

Results go to ``<run>/evals/eval_<snapshot>[_<split>][_remap][_<suffix>]__<stamp>.json``
(stamp = time plus SLURM job id), so repeated evaluations never overwrite each other.
"""

import argparse
import os
import sys

# import the fedsteer package that sits next to this script (job code snapshots)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import os
from collections import defaultdict

import numpy as np
import torch
import yaml

from fedsteer.data import ChatFormatter, ClientQuantiles, alpha_reference_from_json, client_support, read_jsonl
from fedsteer.fed import load_snapshot_into
from fedsteer.lora import SteerLoraConfig
from fedsteer.evaluate import assemble, brief, evaluate_loaded_client
from fedsteer.metrics import SCORERS
from fedsteer.model import load_model
from fedsteer.runinfo import make_stamp, provenance


def latest_snapshot(run: str) -> str:
    d = os.path.join(run, "snapshots")
    return os.path.join(d, sorted(os.listdir(d))[-1])


def evaluate_snapshot(model, fmt, snap_path, shared, by_client, clients, quantiles, alphas, score, args,
                      remap_by_client=None, supports=None):
    """``quantiles[c]``: the reference CDF that defines alpha for client c;
    ``supports[c]``: (lo, hi) part of the alpha axis covered by c's own data (global mode)."""
    snap = torch.load(snap_path, map_location="cpu", weights_only=False)
    results = {}
    for c in clients:
        load_snapshot_into(model, snap, c, shared=shared)
        remap_recs = remap_by_client[c][: args.remap_prompts] if remap_by_client is not None else None
        results[c] = evaluate_loaded_client(
            model, fmt, by_client[c][: args.max_prompts], alphas, score, quantiles[c],
            support=supports.get(c) if supports else None, max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size, dev_loss=args.dev_loss, remap_recs=remap_recs,
            remap_grid=getattr(args, "remap_grid", 11))
        print(c, brief(results[c]), flush=True)
    return assemble(results, alphas, args.split, snap_path, snap["round"], shared=args.shared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--snapshot", nargs="*", default=None, help="one or more snapshots (default: latest)")
    ap.add_argument("--shared", default=None, help="override direction (e.g. merged_shared.pt)")
    ap.add_argument("--scorer", default="words", choices=sorted(SCORERS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--max_prompts", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dev_loss", action="store_true", help="also report held-out loss on --split")
    ap.add_argument("--posthoc_remap", action="store_true",
                    help="warp option F: isotonic alpha remap fitted on --remap_split, applied on --split")
    ap.add_argument("--remap_split", default="dev")
    ap.add_argument("--remap_grid", type=int, default=11, help="alphas probed on the calibration split")
    ap.add_argument("--remap_prompts", type=int, default=50)
    ap.add_argument("--suffix", default="", help="appended to output file names, e.g. test200")
    ap.add_argument("--out", default=None, help="output file (single snapshot only)")
    args = ap.parse_args()

    with open(os.path.join(args.run, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(args.run, "client_quantiles.json")) as f:
        local_q = {c: ClientQuantiles(v) for c, v in json.load(f).items()}
    ref_path = os.path.join(args.run, "alpha_reference.json")
    ref = alpha_reference_from_json(json.load(open(ref_path))) if os.path.exists(ref_path) else None
    if ref is None:                                   # local alpha: each client's own CDF
        quantiles, supports = local_q, None
    else:                                             # global alpha: one shared reference
        quantiles = {c: ref for c in local_q}
        supports = {c: client_support(local_q[c], ref) for c in local_q}
        print("alpha mode: global; client supports:",
              {c: [round(x, 3) for x in v] for c, v in supports.items()}, flush=True)
    snaps = args.snapshot or [latest_snapshot(args.run)]
    if args.out and len(snaps) > 1:
        raise ValueError("--out only works with a single snapshot")
    shared = None
    if args.shared:
        shared = torch.load(args.shared, map_location="cpu")
        shared = {k: v for k, v in shared.items() if k.endswith("lora_B_d")}
    alphas = [float(a) for a in args.alphas.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(cfg["model_name"], SteerLoraConfig(**cfg["lora"]), device=device,
                            grad_checkpointing=False, attn_implementation=cfg.get("attn_implementation", "sdpa"))
    fmt = ChatFormatter(tok, system_prompt=cfg.get("system_prompt"),
                        max_prompt_tokens=cfg.get("max_prompt_tokens", 1024),
                        max_target_tokens=cfg.get("max_target_tokens", 256))

    if args.posthoc_remap and args.remap_split == args.split:
        raise ValueError("fit the remap on a different split than the one evaluated")
    by_client, remap_by_client = defaultdict(list), (defaultdict(list) if args.posthoc_remap else None)
    for r in read_jsonl(cfg["data_path"]):
        if r["client"] not in cfg["clients"]:
            continue
        if r.get("split") == args.split:
            by_client[r["client"]].append(r)
        elif args.posthoc_remap and r.get("split") == args.remap_split:
            remap_by_client[r["client"]].append(r)

    stamp = make_stamp()
    os.makedirs(os.path.join(args.run, "evals"), exist_ok=True)
    for snap_path in snaps:
        print(f"== {snap_path} ({args.split})", flush=True)
        res = evaluate_snapshot(model, fmt, snap_path, shared, by_client, cfg["clients"], quantiles,
                                alphas, SCORERS[args.scorer], args, remap_by_client, supports)
        print("SUMMARY", json.dumps(res["summary"]), flush=True)
        tag = os.path.basename(snap_path).removesuffix(".pt")
        tag += f"_{os.path.basename(args.shared).removesuffix('.pt')}" if args.shared else ""
        tag += "" if args.split == "test" else f"_{args.split}"
        tag += "_remap" if args.posthoc_remap else ""
        tag += f"_{args.suffix}" if args.suffix else ""
        out = args.out or os.path.join(args.run, "evals", f"eval_{tag}__{stamp}.json")
        if os.path.exists(out):
            raise FileExistsError(out)
        res["provenance"] = provenance(args=vars(args))
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
