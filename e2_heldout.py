"""E2 (claim C2, portability): steering held-out clients with the frozen federated direction.

    python e2_heldout.py --run runs/FED --snapshot runs/FED/snapshots/round_0100.pt --ks 16,64

For each held-out client of the run's rotation (never seen in training):
  1. its private adapter is trained by plain SFT with the steering module off (no D, no
     alpha), on its own training data, with the participants' budget (cached in
     runs/_e2_adapters/: it does not depend on the federated run);
  2. settings evaluated on its test articles (standard eval files, one per setting):
       frozen_k0        frozen federated D + the run's calibration as is (shared calibration),
                        or the identity calibration (s = 1, o = 0, h = id) for private runs
       frozen_cal_k{k}  frozen D, calibration fitted on k labelled examples (from the above)
       localdir_k{k}    baseline: the client's own direction trained from zero on the same k
                        examples (adapter frozen)
       prompt_k{k}      baseline: prompting (target level stated) with 3 shots from the same k
Alpha uses the run's global reference (participants' mixture), as it would for a new client.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import yaml  # noqa: E402

from fedsteer.adapt import fit_calibration, sample_k, train_adapter_sft, train_local_direction  # noqa: E402
from fedsteer.baselines import (pick_shots, prompt_with_level, result_from_grid, score_grid_custom,  # noqa: E402
                                zero_direction)
from fedsteer.data import (ChatFormatter, alpha_reference_from_json, build_clients, client_support,  # noqa: E402
                           fit_local_quantiles, read_jsonl)
from fedsteer.evaluate import assemble, brief, control_state, evaluate_loaded_client  # noqa: E402
from fedsteer.lora import SteerLoraConfig, get_gain_state, get_private_adapter_state, load_state  # noqa: E402
from fedsteer.metrics import SCORERS  # noqa: E402
from fedsteer.model import load_model  # noqa: E402
from fedsteer.runinfo import make_stamp, provenance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--clients", default=None, help="comma list; default: the rotation's held-out clients")
    ap.add_argument("--ks", default="16,64")
    ap.add_argument("--adapter_steps", type=int, default=None,
                    help="SFT steps for the new client's adapter (default: participants' rounds*local_steps)")
    ap.add_argument("--cal_steps", type=int, default=100)
    ap.add_argument("--dir_steps", type=int, default=200)
    ap.add_argument("--settings", default="frozen,frozen_cal,localdir,prompt")
    ap.add_argument("--max_prompts", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(args.run, "config.yaml")))
    if cfg["fed"]["mode"] != "fedavg":
        sys.exit("E2 needs a federated run (one shared direction)")
    ref = alpha_reference_from_json(json.load(open(os.path.join(args.run, "alpha_reference.json"))))
    if ref is None:
        sys.exit("E2 needs a global-alpha run (alpha_reference.json with mode=global)")
    if args.clients:
        held = args.clients.split(",")
    else:
        meta = json.load(open(cfg["clients_file"]))
        held = meta["rotations"][cfg.get("rotation", 0)]["held_out"]
    assert not set(held) & set(cfg["clients"]), "held-out clients must not be participants"
    ks = [int(k) for k in args.ks.split(",") if k]
    settings = args.settings.split(",")
    steps = args.adapter_steps or cfg["fed"]["rounds"] * cfg["fed"]["local_steps"]
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    score = SCORERS["density"]

    records = read_jsonl(cfg["data_path"])
    # the new clients' training data: same cap/shuffle as participants; alpha on the run's scale
    train, _ = build_clients(records, held, max_train=cfg.get("max_train_per_client"), seed=cfg["fed"]["seed"])
    train = {c: [dict(r, alpha=ref.cdf(r["score"])) for r in rs] for c, rs in train.items()}
    test = {c: [r for r in records if r["client"] == c and r.get("split") == "test"][: args.max_prompts] for c in held}
    local_q = fit_local_quantiles(records, held)
    support = {c: client_support(local_q[c], ref) for c in held}
    print("held-out clients and their support on the run's alpha scale:",
          {c: [round(x, 3) for x in v] for c, v in support.items()}, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model, tok = load_model(cfg["model_name"], SteerLoraConfig(**cfg["lora"]), device=device,
                            grad_checkpointing=True, attn_implementation=cfg.get("attn_implementation", "sdpa"))
    fmt = ChatFormatter(tok, system_prompt=cfg.get("system_prompt"),
                        max_prompt_tokens=cfg.get("max_prompt_tokens", 1024),
                        max_target_tokens=cfg.get("max_target_tokens", 256))
    fmt_prompt = ChatFormatter(tok, system_prompt=cfg.get("system_prompt"), max_prompt_tokens=4096,
                               max_target_tokens=cfg.get("max_target_tokens", 256))
    init_private = get_private_adapter_state(model)
    init_control = get_gain_state(model)            # s = 1, o = 0, h = identity
    snap = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    direction = {k: v for k, v in snap["server"].items() if k.endswith("lora_B_d")}
    shared_cal = cfg["fed"].get("calibration", "private") == "shared"
    run_control = ({k: v for k, v in snap["server"].items() if k.startswith("steer_control.")}
                   if shared_cal else init_control)

    # 1. new clients' adapters: plain SFT, steering off (cached; independent of the federated run)
    cache_dir = os.path.join("runs", "_e2_adapters")
    os.makedirs(cache_dir, exist_ok=True)
    tag_model = re.sub(r"[^A-Za-z0-9.-]+", "_", cfg["model_name"])
    adapters = {}
    for c in held:
        path = os.path.join(cache_dir, f"{tag_model}_{c}_cap{cfg.get('max_train_per_client')}_steps{steps}"
                                       f"_r{cfg['lora']['rank_private']}_lr{cfg['fed']['lr_private']}_bs{cfg['fed']['batch_size']}"
                                       f"_seed{args.seed}.pt")
        if os.path.exists(path):
            adapters[c] = torch.load(path, map_location="cpu")
            print(f"{c}: cached adapter {path}", flush=True)
            continue
        load_state(model, init_private)
        load_state(model, init_control)
        losses = train_adapter_sft(model, fmt, train[c], steps, lr=cfg["fed"]["lr_private"],
                                   batch_size=cfg["fed"]["batch_size"], seed=args.seed)
        adapters[c] = get_private_adapter_state(model)
        torch.save(adapters[c], path)
        print(f"{c}: trained adapter ({steps} steps, loss {losses[0]:.3f} -> {sum(losses[-50:]) / 50:.3f}) -> {path}",
              flush=True)

    def setup(c, control):
        load_state(model, adapters[c])
        load_state(model, direction)
        load_state(model, control)

    stamp = make_stamp()
    out_dir = os.path.join(args.run, "evals")

    def write(setting, results, extra=None):
        res = assemble(results, alphas, "test", args.snapshot, snap["round"])
        res["e2"] = {"setting": setting, "held_out": held, "adapter_steps": steps, "args": vars(args), **(extra or {})}
        res["provenance"] = provenance(args=vars(args))
        path = os.path.join(out_dir, f"eval_round_{snap['round']:04d}_e2_{setting}__{stamp}.json")
        json.dump(res, open(path, "w"), indent=1)
        print(f"SUMMARY {setting}", json.dumps(res["summary"]), flush=True)
        print(f"wrote {path}", flush=True)

    def eval_loaded(c):
        return evaluate_loaded_client(model, fmt, test[c], alphas, score, ref, support=support[c],
                                      max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)

    model.eval()
    if "frozen" in settings:
        res = {}
        for c in held:
            setup(c, run_control)
            res[c] = eval_loaded(c)
            print(c, "frozen_k0", brief(res[c]), flush=True)
        write("frozen_k0", res, {"calibration": "shared (as trained)" if shared_cal else "identity"})
    for k in ks:
        kex = {c: sample_k(train[c], k, seed=args.seed) for c in held}
        if "frozen_cal" in settings:
            res = {}
            for c in held:
                setup(c, run_control)
                fit_calibration(model, fmt, kex[c], steps=args.cal_steps, lr=cfg["fed"].get("lr_gain", 1e-2),
                                batch_size=min(8, k), seed=args.seed)
                res[c] = eval_loaded(c)
                res[c].update(control_state(model))
                print(c, f"frozen_cal_k{k}", brief(res[c]), flush=True)
            write(f"frozen_cal_k{k}", res, {"k": k, "cal_steps": args.cal_steps})
        if "localdir" in settings:
            res = {}
            for c in held:
                setup(c, init_control)
                train_local_direction(model, fmt, kex[c], steps=args.dir_steps, lr=cfg["fed"]["lr_shared"],
                                      lr_calibration=cfg["fed"].get("lr_gain", 1e-2), batch_size=min(8, k),
                                      seed=args.seed)
                res[c] = eval_loaded(c)
                print(c, f"localdir_k{k}", brief(res[c]), flush=True)
            write(f"localdir_k{k}", res, {"k": k, "dir_steps": args.dir_steps})
        if "prompt" in settings and k >= 3:
            res = {}
            for c in held:
                setup(c, run_control)
                zero_direction(model)
                pf = lambda r, a, c=c: prompt_with_level(r, a, pick_shots(kex[c], a, 3, r.get("url")))
                grid, texts = score_grid_custom(model, fmt_prompt, test[c], alphas, score, prompt_fn=pf,
                                                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
                res[c] = result_from_grid(grid, texts, test[c], alphas, ref, support[c])
                print(c, f"prompt_k{k}", brief(res[c]), flush=True)
            write(f"prompt_k{k}", res, {"k": k, "shots": 3})
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
