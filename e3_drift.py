"""E3 (claim C3, stability): does the direction keep working while clients keep fine-tuning?

    python e3_drift.py --run runs/X --snapshot runs/X/snapshots/round_0100.pt

For each participant, starting from the given checkpoint (its adapter, calibration and
direction; works for federated and local runs):
  stage 0   evaluate as trained
  stage s   continue plain, label-free SFT of the private adapter on drift stage s-1
            (the client's later-year data, 200 pairs per stage) with the steering module
            off (fine-tuning that knows nothing about the knob), then reattach the
            direction and evaluate on the same test articles:
              asis   as is
              refit  after refitting the calibration on k labelled examples from that
                     drift stage (a branch: the drift chain continues from "asis")
Writes <run>/evals/eval_round_XXXX_e3_s{stage}_{asis|refit}__<stamp>.json (standard format).
Compare federated vs local runs to ask whether a shared direction is more robust.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import yaml  # noqa: E402

from fedsteer.adapt import fit_calibration, sample_k, train_adapter_sft  # noqa: E402
from fedsteer.data import (ChatFormatter, ClientQuantiles, alpha_reference_from_json,  # noqa: E402
                           client_support, read_jsonl)
from fedsteer.evaluate import assemble, brief, control_state, evaluate_loaded_client  # noqa: E402
from fedsteer.fed import load_snapshot_into  # noqa: E402
from fedsteer.lora import SteerLoraConfig, get_gain_state, load_state  # noqa: E402
from fedsteer.metrics import SCORERS  # noqa: E402
from fedsteer.model import load_model  # noqa: E402
from fedsteer.runinfo import make_stamp, provenance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--steps_per_stage", type=int, default=100, help="SFT steps per drift stage (batch = run's)")
    ap.add_argument("--refit_k", type=int, default=16, help="0 disables the refit branch")
    ap.add_argument("--cal_steps", type=int, default=100)
    ap.add_argument("--max_prompts", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(args.run, "config.yaml")))
    local_q = {c: ClientQuantiles(v) for c, v in json.load(open(os.path.join(args.run, "client_quantiles.json"))).items()}
    rp = os.path.join(args.run, "alpha_reference.json")
    gref = alpha_reference_from_json(json.load(open(rp))) if os.path.exists(rp) else None
    refs = {c: gref or local_q[c] for c in cfg["clients"]}
    support = {c: client_support(local_q[c], gref) for c in cfg["clients"]} if gref else {}
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    score = SCORERS["density"]

    records = read_jsonl(cfg["data_path"])
    test = {c: [r for r in records if r["client"] == c and r.get("split") == "test"][: args.max_prompts]
            for c in cfg["clients"]}
    drift = {c: {s: [dict(r, alpha=refs[c].cdf(r["score"])) for r in records
                     if r["client"] == c and r.get("split") == "drift" and r.get("drift_stage") == s]
                 for s in range(args.stages)} for c in cfg["clients"]}
    print("drift pairs per stage:", {c: [len(drift[c][s]) for s in range(args.stages)] for c in cfg["clients"]},
          flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model, tok = load_model(cfg["model_name"], SteerLoraConfig(**cfg["lora"]), device=device,
                            grad_checkpointing=True, attn_implementation=cfg.get("attn_implementation", "sdpa"))
    fmt = ChatFormatter(tok, system_prompt=cfg.get("system_prompt"), template_kwargs=cfg.get("chat_template_kwargs"),
                        max_prompt_tokens=cfg.get("max_prompt_tokens", 1024),
                        max_target_tokens=cfg.get("max_target_tokens", 256))
    snap = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    if args.refit_k > 0 and cfg["fed"].get("calibration") == "coverage":
        raise SystemExit("--refit_k fits a gain, which calibration=coverage fixes at 1 (plan 2.1)")

    stamp = make_stamp()
    results = {(s, v): {} for s in range(args.stages + 1) for v in ("asis", "refit")}
    for c in cfg["clients"]:
        load_snapshot_into(model, snap, c)
        model.eval()
        ev = lambda: evaluate_loaded_client(model, fmt, test[c], alphas, score, refs[c], support=support.get(c),
                                            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        results[(0, "asis")][c] = ev()
        print(c, "stage 0", brief(results[(0, "asis")][c]), flush=True)
        for s in range(1, args.stages + 1):
            losses = train_adapter_sft(model, fmt, drift[c][s - 1], args.steps_per_stage, lr=cfg["fed"]["lr_private"],
                                       batch_size=cfg["fed"]["batch_size"], seed=args.seed + s)
            results[(s, "asis")][c] = ev()
            results[(s, "asis")][c]["drift_loss"] = [losses[0], sum(losses[-10:]) / min(10, len(losses))] if losses else None
            print(c, f"stage {s} asis", brief(results[(s, "asis")][c]), flush=True)
            if args.refit_k > 0:
                saved = get_gain_state(model)
                kex = sample_k(drift[c][s - 1], args.refit_k, seed=args.seed + s)
                fit_calibration(model, fmt, kex, steps=args.cal_steps, lr=cfg["fed"].get("lr_gain", 1e-2),
                                batch_size=min(8, args.refit_k), seed=args.seed + s)
                results[(s, "refit")][c] = ev()
                results[(s, "refit")][c].update(control_state(model))
                print(c, f"stage {s} refit", brief(results[(s, "refit")][c]), flush=True)
                load_state(model, saved)          # the drift chain continues without the refit

    for (s, v), res in results.items():
        if not res:
            continue
        out = assemble(res, alphas, "test", args.snapshot, snap["round"])
        out["e3"] = {"stage": s, "variant": v, "steps_per_stage": args.steps_per_stage, "args": vars(args)}
        out["provenance"] = provenance(args=vars(args))
        path = os.path.join(args.run, "evals", f"eval_round_{snap['round']:04d}_e3_s{s}_{v}__{stamp}.json")
        json.dump(out, open(path, "w"), indent=1)
        print(f"SUMMARY stage {s} {v}", json.dumps(out["summary"]), flush=True)
        print(f"wrote {path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
