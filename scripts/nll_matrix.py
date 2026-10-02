"""Cross-client NLL matrix: does each client's model specialize to its own publication?

    python scripts/nll_matrix.py --run Method=runs/A::round_0050,round_0100 --run A2=runs/B::round_0070

For each run and snapshot, each client's model (adapter, calibration, direction) scores every
participant's references (teacher forced, at the reference's own global alpha).  Row = model of
client i, column = references of client j.  Specialization = how much better a client's model
fits its own references than other clients' models do:
    own_advantage_j = mean_{i != j} NLL[i, j] - NLL[j, j]
A shared adapter with shared calibration has identical rows, so its own_advantage is 0.
Writes <run>/evals/nll_matrix_<split>_round_XXXX.json and prints a summary.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedsteer.data import ChatFormatter, alpha_reference_from_json, read_jsonl  # noqa: E402
from fedsteer.fed import load_snapshot_into  # noqa: E402
from fedsteer.lora import SteerLoraConfig  # noqa: E402
from fedsteer.model import load_model  # noqa: E402
from fedsteer.monitor import mean_loss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=path::round_XXXX[,round_YYYY]")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_refs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    specs = []
    for spec in args.run:
        name, rest = spec.split("=", 1)
        path, _, snaps = rest.partition("::")
        specs.append((name, path, snaps.split(",")))
    cfg = yaml.safe_load(open(os.path.join(specs[0][1], "config.yaml")))
    clients = cfg["clients"]
    ref = alpha_reference_from_json(json.load(open(os.path.join(specs[0][1], "alpha_reference.json"))))
    records = read_jsonl(cfg["data_path"])
    refs = {c: [dict(r, alpha=ref.cdf(r["score"])) for r in records if r["client"] == c and r.get("split") == args.split]
            [: args.max_refs] for c in clients}

    model, tok = load_model(cfg["model_name"], SteerLoraConfig(**cfg["lora"]), device="cuda",
                            grad_checkpointing=False, attn_implementation=cfg.get("attn_implementation", "sdpa"))
    fmt = ChatFormatter(tok, system_prompt=cfg.get("system_prompt"), template_kwargs=cfg.get("chat_template_kwargs"),
                        max_prompt_tokens=cfg.get("max_prompt_tokens", 1024),
                        max_target_tokens=cfg.get("max_target_tokens", 256))
    summary = []
    for name, path, snaps in specs:
        rcfg = yaml.safe_load(open(os.path.join(path, "config.yaml")))
        assert rcfg["model_name"] == cfg["model_name"] and rcfg["clients"] == clients, f"{name}: different setup"
        for sn in snaps:
            snap = torch.load(os.path.join(path, "snapshots", f"{sn}.pt"), map_location="cpu", weights_only=False)
            fc = snap["fed_config"]
            identical = fc.get("adapter") == "shared" and fc.get("calibration") == "shared" and fc["mode"] == "fedavg"
            mat = np.zeros((len(clients), len(clients)))
            for i, ci in enumerate(clients):
                if identical and i > 0:
                    mat[i] = mat[0]
                    continue
                load_snapshot_into(model, snap, ci)
                for j, cj in enumerate(clients):
                    mat[i, j] = mean_loss(model, fmt, refs[cj], batch_size=args.batch_size)
                print(name, sn, ci, " ".join(f"{x:.3f}" for x in mat[i]), flush=True)
            diag = np.diag(mat)
            off = np.array([np.mean([mat[i, j] for i in range(len(clients)) if i != j]) for j in range(len(clients))])
            out = {"run": path, "snapshot": sn, "split": args.split, "clients": clients, "nll": mat.tolist(),
                   "own_nll": diag.tolist(), "others_on_own_refs": off.tolist(),
                   "own_advantage": (off - diag).tolist(), "identical_rows": identical}
            json.dump(out, open(os.path.join(path, "evals", f"nll_matrix_{args.split}_{sn}.json"), "w"), indent=1)
            summary.append((name, sn, diag, off))

    print(f"\n{'run':12s} {'snapshot':11s} {'own NLL':>8s} {'others on own refs':>18s} {'own advantage':>13s}")
    for name, sn, diag, off in summary:
        print(f"{name:12s} {sn:11s} {diag.mean():8.3f} {off.mean():18.3f} {(off - diag).mean():13.3f}")
    print("\nper-client own advantage (others' NLL on the client's refs - own NLL)")
    print(f"{'run':24s} " + " ".join(f"{c[:9]:>9s}" for c in clients))
    for name, sn, diag, off in summary:
        print(f"{name + ' ' + sn:24s} " + " ".join(f"{x:9.3f}" for x in off - diag))


if __name__ == "__main__":
    main()
