"""Baseline B3: one-shot merged direction from a local-only run.

    python scripts/merge_local_directions.py --snapshot runs/LOCAL/snapshots/round_0090.pt
    python eval_direction.py --run runs/LOCAL --snapshot runs/LOCAL/snapshots/round_0090.pt \\
        --shared runs/LOCAL/merged_round_0090.pt --scorer density --suffix b3

Averages the clients' independently trained directions (B_d; A_d is shared, so this is
exactly the average of the direction products) once, uniformly, at the given checkpoint.
Each client then uses the merged direction with its own adapter and calibration.
"""

import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", default=None, help="default: <run>/merged_round_XXXX.pt")
    args = ap.parse_args()
    snap = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    if snap["fed_config"]["mode"] != "local":
        raise SystemExit("B3 needs a local-only run (fed.mode=local)")
    dirs = [{k: v for k, v in cs["shared_local"].items() if k.endswith("lora_B_d")}
            for cs in snap["clients"].values()]
    merged = {k: sum(d[k] for d in dirs) / len(dirs) for k in dirs[0]}
    run = os.path.dirname(os.path.dirname(os.path.abspath(args.snapshot)))
    out = args.out or os.path.join(run, f"merged_round_{snap['round']:04d}.pt")
    torch.save(merged, out)
    print(f"merged {len(dirs)} client directions from round {snap['round']} -> {out}")


if __name__ == "__main__":
    main()
