"""Federated training of a shared steering direction.

    python train_fed.py --config configs/toy_length.yaml
    python train_fed.py --config configs/toy_length.yaml --set fed.mode=local out_dir=runs/toy_local

Re-running with the same ``out_dir`` resumes from ``out_dir/state.pt``.
"""

import argparse
import json
import os
from dataclasses import asdict, fields

import torch
import yaml

from fedsteer.data import ChatFormatter, build_clients, read_jsonl
from fedsteer.fed import FedConfig, FedSteerTrainer
from fedsteer.lora import SteerLoraConfig
from fedsteer.model import load_model
from fedsteer.monitor import MonitorConfig, make_monitor


def _parse_value(v: str):
    return yaml.safe_load(v)


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(val)
    return cfg


def _dataclass_from(cls, d: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. fed.rounds=10")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    lora_cfg = _dataclass_from(SteerLoraConfig, cfg.get("lora", {}))
    fed_cfg = _dataclass_from(FedConfig, cfg.get("fed", {}))
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(fed_cfg.seed)

    records = read_jsonl(cfg["data_path"])
    clients = cfg.get("clients")
    if not clients and cfg.get("clients_file"):
        # participants of a held-out rotation, e.g. data/newsroom_fed/clients.json
        with open(cfg["clients_file"]) as f:
            meta = json.load(f)
        rot = cfg.get("rotation")
        clients = meta["clients"] if rot is None else meta["rotations"][rot]["participants"]
    clients = clients or sorted({r["client"] for r in records})
    examples, quantiles = build_clients(records, clients, tie_break=cfg.get("tie_break", "average"),
                                        max_train=cfg.get("max_train_per_client"), seed=fed_cfg.seed)

    resolved = dict(cfg, clients=clients, lora=asdict(lora_cfg), fed=asdict(fed_cfg))
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)
    with open(os.path.join(out_dir, "client_quantiles.json"), "w") as f:
        json.dump({c: q.sorted_scores for c, q in quantiles.items()}, f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(cfg["model_name"], lora_cfg, device=device,
                            grad_checkpointing=cfg.get("grad_checkpointing", True),
                            attn_implementation=cfg.get("attn_implementation", "sdpa"))
    fmt = ChatFormatter(tok, system_prompt=cfg.get("system_prompt"),
                        max_prompt_tokens=cfg.get("max_prompt_tokens", 1024),
                        max_target_tokens=cfg.get("max_target_tokens", 256))

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"clients: {clients}")
    print(f"train examples: { {c: len(v) for c, v in examples.items()} }")
    print(f"trainable params (one client's view): {n_train / 1e6:.2f}M", flush=True)

    eval_fn = None
    if cfg.get("monitor"):
        mon_cfg = _dataclass_from(MonitorConfig, cfg["monitor"])
        eval_fn = make_monitor(records, clients, quantiles, fmt, mon_cfg)
        print(f"monitor: {asdict(mon_cfg)}", flush=True)

    trainer = FedSteerTrainer(model, fmt, examples, fed_cfg, out_dir, eval_fn=eval_fn)
    trainer.fit()


if __name__ == "__main__":
    main()
