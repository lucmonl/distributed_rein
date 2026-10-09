"""E2 (claim C2, portability): a held-out client joins with the frozen federated direction.

    python e2_heldout.py --run runs/FED --snapshot runs/FED/snapshots/round_0100.pt --ns 16+64+256

For each held-out client of the run's rotation (never seen in training) and each data size
n (its first n training pairs under a fixed shuffle, so the subsets are nested), alpha on
the run's global scale (free to compute: it is the reference CDF of the pair's density):
  frozen_D   the participants' objective (alpha per example, steering on) with D frozen:
             trains the private adapter P from scratch, plus the calibration if the run's
             calibration is private (shared calibration stays frozen at the run's value).
  local_D    baseline, no federation: the same objective and step budget, but the client
             trains its own direction from zero (P + D + calibration).
  plugin     ablation: P trained by plain SFT with the steering module off (a client that
             fine-tunes without knowing the knob), then D attached; calibration as the
             run has it (shared: frozen; private: fitted on the same n pairs).
  prompt     baseline: the plugin adapter with D removed, target level stated in the prompt,
             3 shots from the same n pairs.
Steps per n: clip(ceil(epochs * n / batch), min_steps, max_steps), the same for every setting.
local_D and prompt do not depend on the federated run (only on its rotation and alpha
reference), so they need running once per rotation.

--alpha_window LO HI keeps only the client's pairs with alpha in [LO, HI] (a client with a
skewed attribute distribution; its support then comes from those pairs).

fed.private_offset runs (shared h_l + private o_i, NR-68): the new client gets the server's frozen
shape and gain and fits its own o_i from 0 (frozen_D: P + o_i; plugin: o_i only).
calibration=aligned runs (NR-68): the new client gets the run's frozen inference table h-bar_l
(exactly as load_snapshot_into sets it) and, with lora.offset, fits its own offset o_i from 0;
the gain stays at s = 1 (u is never trained) and the warps are not used (the table replaces
them).  So frozen_D trains P + o_i; plugin fits only o_i after the SFT adapter; local_D trains
P + D_i + its own per-layer warps (+ o_i), gain fixed.  --respect_fix_gain applies the same
"gain fixed" rule to private/shared runs with fed.fix_gain (opt-in; off = the original
behaviour, which trains the whole gain group u + o).
Writes <run>/evals/eval_round_XXXX_e2_{setting}_n{n}[_wLO-HI]__<stamp>.json (standard format).
"""

import argparse
import json
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import yaml  # noqa: E402

from fedsteer.adapt import fit_calibration, train_adapter_sft, train_steered  # noqa: E402
from fedsteer.baselines import (pick_shots, prompt_with_level, result_from_grid, score_grid_custom,  # noqa: E402
                                zero_direction)
from fedsteer.data import (ChatFormatter, ClientQuantiles, alpha_reference_from_json, build_clients,  # noqa: E402
                           client_support, read_jsonl)
from fedsteer.evaluate import assemble, brief, control_state, evaluate_loaded_client  # noqa: E402
from fedsteer.lora import SteerLoraConfig, get_gain_state, get_private_adapter_state, load_state  # noqa: E402
from fedsteer.metrics import SCORERS  # noqa: E402
from fedsteer.model import load_model  # noqa: E402
from fedsteer.runinfo import make_stamp, provenance  # noqa: E402

SETTINGS = ("frozen_D", "local_D", "plugin", "prompt")


def e2_plan(fc: dict, lora: dict, snap: dict, init_control: dict, respect_fix_gain: bool = False) -> dict:
    """What a new client trains in each setting, and the frozen calibration it starts from.

    Returns {"frozen_control", "frozen_table" (grid, z) or None, "frozen": components,
    "local": components, "plugin": calibration components fitted after the SFT adapter
    (() = none), "calibration": description}.  Components are fedsteer.adapt keys."""
    cal = fc.get("calibration", "private")
    aligned = cal == "aligned"
    priv_off = cal == "shared" and bool(fc.get("private_offset", False))   # NR-60: shared h_l, own o_i
    fix_u = (respect_fix_gain or aligned or priv_off) and bool(fc.get("fix_gain", False))
    offset = bool(lora.get("offset", False))

    def calib(with_warp):                              # the client's own calibration parameters
        if fix_u:
            return (("offset",) if offset else ()) + (("warp",) if with_warp else ())
        return ("gain", "warp") if with_warp else ("gain",)

    local = ("private", "shared") + calib(True)
    if aligned:
        cov = snap.get("coverage")
        if not cov or "z" not in cov:
            raise SystemExit("calibration=aligned snapshot has no coverage table")
        return {"frozen_control": init_control, "frozen_table": (cov["grid"], cov["z"]),
                "frozen": ("private",) + calib(False), "local": local, "plugin": calib(False),
                "calibration": "aligned (frozen table h-bar_l" + (", own offset o_i trained)" if offset else ")")}
    if cal == "shared":
        ctl = {k: v for k, v in snap["server"].items() if k.startswith("steer_control.")}
        if priv_off:
            # the server holds the shared shape and gain but no offset: the new client starts its
            # own o_i at the initial value (0) and fits it, like a participant
            ctl = dict(ctl, **{k: v for k, v in init_control.items() if k == "steer_control.o"})
            return {"frozen_control": ctl, "frozen_table": None, "frozen": ("private", "offset"),
                    "local": local, "plugin": ("offset",),
                    "calibration": "shared (frozen h_l), own offset o_i trained"}
        return {"frozen_control": ctl, "frozen_table": None, "frozen": ("private",), "local": local,
                "plugin": (), "calibration": "shared (frozen)"}
    return {"frozen_control": init_control, "frozen_table": None, "frozen": ("private",) + calib(True),
            "local": local, "plugin": calib(True), "calibration": "private (trained)"}


def split_list(s):
    return [x for x in re.split(r"[,+]", s) if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--clients", default=None, help="comma/+ list; default: the rotation's held-out clients")
    ap.add_argument("--ns", default="16+64+256+1024", help="data sizes, comma/+ separated; 'all' = every pair")
    ap.add_argument("--settings", default="+".join(SETTINGS))
    ap.add_argument("--epochs", type=float, default=4.0, help="participants: 2000 steps x 8 / 4000 pairs = 4")
    ap.add_argument("--min_steps", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=None, help="default: participants' rounds * local_steps")
    ap.add_argument("--cal_steps", type=int, default=100, help="plugin, private-calibration runs only")
    ap.add_argument("--alpha_window", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--max_prompts", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--respect_fix_gain", action="store_true",
                    help="private/shared runs with fed.fix_gain: never train u (opt-in; always on for aligned)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(args.run, "config.yaml")))
    fc = cfg["fed"]
    if fc["mode"] != "fedavg":
        sys.exit("E2 needs a federated run (one shared direction)")
    ref = alpha_reference_from_json(json.load(open(os.path.join(args.run, "alpha_reference.json"))))
    if ref is None:
        sys.exit("E2 needs a global-alpha run (alpha_reference.json with mode=global)")
    if args.clients:
        held = split_list(args.clients)
    else:
        meta = json.load(open(cfg["clients_file"]))
        held = meta["rotations"][cfg.get("rotation", 0)]["held_out"]
    assert not set(held) & set(cfg["clients"]), "held-out clients must not be participants"
    settings = split_list(args.settings)
    assert set(settings) <= set(SETTINGS), settings
    if fc.get("calibration") in ("coverage", "consensus"):
        # plan 2.1: the server table is not a new client's warp, and the private-calibration path
        # here would also train the gain, which coverage fixes at 1
        raise SystemExit("calibration=coverage runs need their own held-out-client protocol (plan 2.1)")
    bs = fc["batch_size"]
    max_steps = args.max_steps or fc["rounds"] * fc["local_steps"]
    lrs = {"private": fc["lr_private"], "shared": fc["lr_shared"], "gain": fc.get("lr_gain", 1e-2),
           "warp": fc.get("lr_warp", 1e-2)}
    warmup = fc.get("warmup_steps", 20)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    score = SCORERS["density"]
    wtag = f"_w{args.alpha_window[0]:g}-{args.alpha_window[1]:g}" if args.alpha_window else ""

    records = read_jsonl(cfg["data_path"])
    # the new clients' training pairs, alpha on the run's scale, in a fixed shuffled order
    # (prefixes = nested subsets); an alpha window keeps a skewed part of the client's data
    train, _ = build_clients(records, held, seed=fc["seed"])
    pools = {}
    for c in held:
        rs = [dict(r, alpha=ref.cdf(r["score"])) for r in train[c]]
        if args.alpha_window:
            rs = [r for r in rs if args.alpha_window[0] <= r["alpha"] <= args.alpha_window[1]]
        random.Random(args.seed).shuffle(rs)
        pools[c] = rs
    test = {c: [r for r in records if r["client"] == c and r.get("split") == "test"][: args.max_prompts] for c in held}
    ns = []
    for x in split_list(args.ns):
        ns.append(x if x == "all" else int(x))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model, tok = load_model(cfg["model_name"], SteerLoraConfig(**cfg["lora"]), device=device,
                            grad_checkpointing=True, attn_implementation=cfg.get("attn_implementation", "sdpa"))
    fmt = ChatFormatter(tok, system_prompt=cfg.get("system_prompt"), template_kwargs=cfg.get("chat_template_kwargs"),
                        max_prompt_tokens=cfg.get("max_prompt_tokens", 1024),
                        max_target_tokens=cfg.get("max_target_tokens", 256))
    fmt_prompt = ChatFormatter(tok, system_prompt=cfg.get("system_prompt"), template_kwargs=cfg.get("chat_template_kwargs"), max_prompt_tokens=4096,
                               max_target_tokens=cfg.get("max_target_tokens", 256))
    init_private = get_private_adapter_state(model)     # A_p random (seeded), B_p = 0
    init_control = get_gain_state(model)                # s = 1, o = 0, h = identity
    zero_dir = {n: torch.zeros_like(p).cpu() for n, p in model.named_parameters() if n.endswith("lora_B_d")}
    snap = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    direction = {k: v for k, v in snap["server"].items() if k.endswith("lora_B_d")}
    plan = e2_plan(fc, cfg["lora"], snap, init_control, args.respect_fix_gain)
    run_control = plan["frozen_control"]
    print(f"E2 plan: calibration {plan['calibration']}; frozen_D trains {plan['frozen']}, local_D "
          f"{plan['local']}, plugin fits {plan['plugin'] or 'nothing'}", flush=True)
    tag_model = re.sub(r"[^A-Za-z0-9.-]+", "_", cfg["model_name"])
    cache_dir = os.path.join("runs", "_e2_adapters")
    os.makedirs(cache_dir, exist_ok=True)
    stamp = make_stamp()
    out_dir = os.path.join(args.run, "evals")

    def reset(direction_state, control, table=None):
        load_state(model, init_private)
        load_state(model, direction_state)
        load_state(model, control)
        model.steer_control.set_table(*table) if table is not None else model.steer_control.set_table(None)

    def eval_loaded(c, support):
        model.eval()
        return evaluate_loaded_client(model, fmt, test[c], alphas, score, ref, support=support,
                                      max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)

    def train_info(n, steps, losses):
        return {"n": n, "steps": steps, "loss_first": losses[0] if losses else None,
                "loss_last": sum(losses[-20:]) / min(20, len(losses)) if losses else None}

    for n in ns:
        res = {s: {} for s in settings}
        for c in held:
            ex = pools[c] if n == "all" else pools[c][:n]
            if n != "all" and len(ex) < n:
                print(f"{c}: only {len(ex)} pairs (< n = {n}), skipped", flush=True)
                continue
            steps = min(max_steps, max(args.min_steps, math.ceil(args.epochs * len(ex) / bs)))
            support = client_support(ClientQuantiles.fit([r["score"] for r in ex]), ref)
            print(f"== {c} n={n} ({len(ex)} pairs, {steps} steps) support {[round(x, 3) for x in support]}",
                  flush=True)

            if "frozen_D" in res:
                reset(direction, run_control, plan["frozen_table"])
                losses = train_steered(model, fmt, ex, steps, plan["frozen"], lrs, bs, warmup, seed=args.seed)
                r = eval_loaded(c, support)
                r.update(control_state(model))
                r["train"] = train_info(len(ex), steps, losses)
                res["frozen_D"][c] = r
                print(c, f"frozen_D n={n}", brief(r), flush=True)

            if "local_D" in res:
                reset(zero_dir, init_control)
                losses = train_steered(model, fmt, ex, steps, plan["local"], lrs, bs, warmup, seed=args.seed)
                r = eval_loaded(c, support)
                r.update(control_state(model))
                r["train"] = train_info(len(ex), steps, losses)
                res["local_D"][c] = r
                print(c, f"local_D n={n}", brief(r), flush=True)

            if "plugin" in res or "prompt" in res:
                # plain SFT adapter, steering off (cached: independent of the federated run)
                path = os.path.join(cache_dir, f"{tag_model}_{c}_n{len(ex)}{wtag}_steps{steps}"
                                               f"_r{cfg['lora']['rank_private']}_lr{lrs['private']}_bs{bs}"
                                               f"_seed{args.seed}_sft.pt")
                if os.path.exists(path):
                    sft = torch.load(path, map_location="cpu")
                    sft_info = {"n": len(ex), "steps": steps, "cached": path}
                else:
                    reset(zero_dir, init_control)
                    losses = train_adapter_sft(model, fmt, ex, steps, lr=lrs["private"], batch_size=bs, seed=args.seed)
                    sft = get_private_adapter_state(model)
                    torch.save(sft, path)
                    sft_info = train_info(len(ex), steps, losses)
                if "plugin" in res:
                    reset(direction, run_control, plan["frozen_table"])
                    load_state(model, sft)
                    if plan["plugin"]:
                        fit_calibration(model, fmt, ex, steps=args.cal_steps, lr=lrs["gain"],
                                        batch_size=min(8, len(ex)), seed=args.seed, components=plan["plugin"])
                    r = eval_loaded(c, support)
                    r.update(control_state(model))
                    r["train"] = sft_info
                    res["plugin"][c] = r
                    print(c, f"plugin n={n}", brief(r), flush=True)
                if "prompt" in res:
                    reset(zero_dir, init_control)
                    load_state(model, sft)
                    zero_direction(model)
                    model.eval()
                    pf = lambda rec, a, ex=ex: prompt_with_level(rec, a, pick_shots(ex, a, 3, rec.get("url")))
                    grid, texts = score_grid_custom(model, fmt_prompt, test[c], alphas, score, prompt_fn=pf,
                                                    max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
                    r = result_from_grid(grid, texts, test[c], alphas, ref, support)
                    r["train"] = sft_info
                    res["prompt"][c] = r
                    print(c, f"prompt n={n}", brief(r), flush=True)

        for s, rr in res.items():
            if not rr:
                continue
            out = assemble(rr, alphas, "test", args.snapshot, snap["round"])
            out["e2"] = {"setting": s, "n": n, "alpha_window": args.alpha_window, "held_out": held,
                         "calibration": plan["calibration"],
                         "components": {"frozen_D": plan["frozen"], "local_D": plan["local"],
                                        "plugin": plan["plugin"]},
                         "args": vars(args)}
            out["provenance"] = provenance(args=vars(args))
            path = os.path.join(out_dir, f"eval_round_{snap['round']:04d}_e2_{s}_n{n}{wtag}__{stamp}.json")
            json.dump(out, open(path, "w"), indent=1)
            print(f"SUMMARY {s} n={n}", json.dumps(out["summary"]), flush=True)
            print(f"wrote {path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
