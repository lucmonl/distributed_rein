"""Federated training of a shared steering direction.

    python train_fed.py --config configs/toy_length.yaml
    python train_fed.py --config configs/toy_length.yaml --set fed.mode=local out_dir=runs/toy_local
    python train_fed.py --resume runs/toy_local_20260929-221530 --set fed.rounds=60

``out_dir`` in the config is a prefix: every run writes to a new directory
``<out_dir>_<YYYYmmdd-HHMMSS>[_j<SLURM_JOB_ID>]`` (fedsteer/runinfo.py), so runs never
overwrite each other.  ``--resume`` continues an existing run from its state.pt, using
the config stored in that directory (plus any ``--set`` overrides).
"""

import argparse
import os
import sys

# import the fedsteer package that sits next to this script (job code snapshots)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import os
from dataclasses import asdict, fields

import torch
import yaml

from fedsteer.data import ChatFormatter, build_clients, client_support, fit_local_quantiles, read_jsonl
from fedsteer.fed import FedConfig, FedSteerTrainer
from fedsteer.lora import SteerLoraConfig
from fedsteer.model import load_model
from fedsteer.monitor import MonitorConfig, make_monitor
from fedsteer.regularize import RegConfig
from fedsteer.runinfo import make_stamp, record_run_info


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
    ap.add_argument("--config", help="new run from this config")
    ap.add_argument("--resume", help="continue this existing run directory")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. fed.rounds=10")
    args = ap.parse_args()
    if bool(args.config) == bool(args.resume):
        ap.error("give exactly one of --config or --resume")

    if args.resume:
        out_dir = args.resume.rstrip("/")
        if not os.path.exists(os.path.join(out_dir, "config.yaml")):
            ap.error(f"{out_dir} has no config.yaml")
        cfg = load_config(os.path.join(out_dir, "config.yaml"), args.set)
        cfg["out_dir"] = out_dir
        event = "resumed"
    else:
        cfg = load_config(args.config, args.set)
        cfg["out_prefix"] = cfg["out_dir"]
        out_dir = f"{cfg['out_dir'].rstrip('/')}_{make_stamp()}"
        if os.path.exists(out_dir) and os.listdir(out_dir):
            raise FileExistsError(f"{out_dir} already exists; use --resume to continue it")
        cfg["out_dir"] = out_dir
        event = "created"

    lora_cfg = _dataclass_from(SteerLoraConfig, cfg.get("lora", {}))
    fed_cfg = _dataclass_from(FedConfig, cfg.get("fed", {}))
    reg_cfg = _dataclass_from(RegConfig, cfg.get("reg") or {})
    mon_cfg = _dataclass_from(MonitorConfig, cfg["monitor"]) if cfg.get("monitor") else None
    if mon_cfg and mon_cfg.full_every and mon_cfg.full_every % fed_cfg.save_every:
        raise ValueError(f"monitor.full_every ({mon_cfg.full_every}) must be a multiple of fed.save_every "
                         f"({fed_cfg.save_every}) so every evaluated round has a snapshot")
    os.makedirs(out_dir, exist_ok=True)
    record_run_info(out_dir, event, config=args.config or args.resume, overrides=args.set)
    print(f"run directory: {out_dir} ({event})", flush=True)
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
    alpha_mode = cfg.get("alpha_mode", "local")
    examples, quantiles = build_clients(records, clients, tie_break=cfg.get("tie_break", "average"),
                                        max_train=cfg.get("max_train_per_client"), seed=fed_cfg.seed,
                                        alpha_mode=alpha_mode)
    local_q = fit_local_quantiles(records, clients)

    resolved = dict(cfg, clients=clients, lora=asdict(lora_cfg), fed=asdict(fed_cfg), reg=asdict(reg_cfg))
    if mon_cfg:
        resolved["monitor"] = asdict(mon_cfg)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)
    with open(os.path.join(out_dir, "client_quantiles.json"), "w") as f:      # each client's own CDF
        json.dump({c: q.sorted_scores for c, q in local_q.items()}, f)
    with open(os.path.join(out_dir, "alpha_reference.json"), "w") as f:       # the CDF that defines alpha
        ref = next(iter(quantiles.values()))
        json.dump(ref.to_json() if alpha_mode == "global" else {"mode": "local"}, f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(cfg["model_name"], lora_cfg, device=device,
                            grad_checkpointing=cfg.get("grad_checkpointing", True),
                            attn_implementation=cfg.get("attn_implementation", "sdpa"))
    fmt = ChatFormatter(tok, system_prompt=cfg.get("system_prompt"), template_kwargs=cfg.get("chat_template_kwargs"),
                        max_prompt_tokens=cfg.get("max_prompt_tokens", 1024),
                        max_target_tokens=cfg.get("max_target_tokens", 256))

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"clients: {clients}")
    print(f"train examples: { {c: len(v) for c, v in examples.items()} }")
    print(f"trainable params (one client's view): {n_train / 1e6:.2f}M", flush=True)

    eval_fn = None
    if mon_cfg:
        supports = ({c: client_support(local_q[c], quantiles[c]) for c in clients}
                    if alpha_mode == "global" else None)
        eval_fn = make_monitor(records, clients, quantiles, fmt, mon_cfg, supports=supports, out_dir=out_dir)
        print(f"monitor: {asdict(mon_cfg)}", flush=True)
    print(f"reg: {asdict(reg_cfg)}", flush=True)

    trainer = FedSteerTrainer(model, fmt, examples, fed_cfg, out_dir, eval_fn=eval_fn, reg=reg_cfg)
    trainer.fit()
    from fedsteer.model import report_peak_memory
    report_peak_memory('train_fed.py')


if __name__ == "__main__":
    main()
