"""E2 zero-shot: a held-out client uses a run's GLOBAL model as-is, with no training (user decision for
pure FedAvg, NR-78).

    python scripts/e2_zeroshot.py --run runs/<run> --snapshot runs/<run>/snapshots/round_0070.pt

For a shared-adapter run (fed.adapter=shared) the global model is the shared adapter + the direction + the
shared calibration (incl. a shared offset); `load_snapshot_into` loads exactly that for any participant, so
one participant's view is the global model. The held-out clients, test articles (first --max_prompts test
records), alpha grid, scorer, generation settings and support (the client's full training pool on the
run's global alpha scale, as e2_heldout.py at n = all) are e2_heldout.py's.
Writes <run>/evals/eval_round_XXXX_e2_zeroshot_nall__<stamp>.json (standard eval format).
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import yaml  # noqa: E402

from fedsteer.data import ChatFormatter, ClientQuantiles, alpha_reference_from_json, build_clients, client_support, read_jsonl  # noqa: E402
from fedsteer.evaluate import assemble, brief, control_state, evaluate_loaded_client  # noqa: E402
from fedsteer.fed import load_snapshot_into  # noqa: E402
from fedsteer.lora import SteerLoraConfig  # noqa: E402
from fedsteer.metrics import SCORERS  # noqa: E402
from fedsteer.model import load_model  # noqa: E402
from fedsteer.runinfo import make_stamp, provenance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--max_prompts", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(args.run, "config.yaml")))
    fc = cfg["fed"]
    ref = alpha_reference_from_json(json.load(open(os.path.join(args.run, "alpha_reference.json"))))
    held = json.load(open(cfg["clients_file"]))["rotations"][cfg.get("rotation", 0)]["held_out"]
    assert not set(held) & set(cfg["clients"])
    records = read_jsonl(cfg["data_path"])
    train, _ = build_clients(records, held, seed=fc["seed"])
    test = {c: [r for r in records if r["client"] == c and r.get("split") == "test"][: args.max_prompts] for c in held}
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    score = SCORERS["density"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model, tok = load_model(cfg["model_name"], SteerLoraConfig(**cfg["lora"]), device=device,
                            grad_checkpointing=False, attn_implementation=cfg.get("attn_implementation", "sdpa"))
    fmt = ChatFormatter(tok, system_prompt=cfg.get("system_prompt"), template_kwargs=cfg.get("chat_template_kwargs"),
                        max_prompt_tokens=cfg.get("max_prompt_tokens", 1024),
                        max_target_tokens=cfg.get("max_target_tokens", 256))
    snap = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    load_snapshot_into(model, snap, cfg["clients"][0])          # the global model (shared adapter + D + calibration)
    model.eval()
    print(f"global model loaded from {args.snapshot} (adapter {fc.get('adapter')}, calibration {fc.get('calibration')}); "
          f"control {control_state(model)}", flush=True)

    res = {}
    for c in held:
        support = client_support(ClientQuantiles.fit([r["score"] for r in train[c]]), ref)
        r = evaluate_loaded_client(model, fmt, test[c], alphas, score, ref, support=support,
                                   max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        r.update(control_state(model))
        r["train"] = {"n": 0, "steps": 0, "zero_shot": True}
        res[c] = r
        print(c, "zeroshot", brief(r), flush=True)
    out = assemble(res, alphas, "test", args.snapshot, snap["round"])
    out["e2"] = {"setting": "zeroshot", "n": "all", "held_out": held, "calibration": f"global ({fc.get('calibration')})",
                 "adapter": fc.get("adapter"), "args": vars(args)}
    out["provenance"] = provenance(args=vars(args))
    path = os.path.join(args.run, "evals", f"eval_round_{snap['round']:04d}_e2_zeroshot_nall__{make_stamp()}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("SUMMARY zeroshot", json.dumps(out["summary"]), flush=True)
    print(f"wrote {path}\nDONE", flush=True)


if __name__ == "__main__":
    main()
