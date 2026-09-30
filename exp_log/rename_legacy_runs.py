"""One-off (2026-09-29): rename run folders created before timestamped run directories
existed, move their eval files into evals/ with the stamp of the job that produced
them, point config.yaml at the new path, and backfill run_info.json.  File contents
are unchanged (paths recorded *inside* old eval JSONs still use the old names).

    python exp_log/rename_legacy_runs.py toy        # the two toy runs
    python exp_log/rename_legacy_runs.py pilot      # nr_pilot_llama1b (after job 11006740 ends)
"""
import json
import os
import sys

import yaml

JOBS = {  # job id -> (start stamp, what it did), from sacct
    "10985846": ("20260929-112613", "toy smoke test: train fedavg + local, eval"),
    "10987927": ("20260929-122920", "Newsroom pilot, 10 rounds, eval (old metrics)"),
    "10996826": ("20260929-161525", "re-eval round 10 with updated metrics (100 test prompts)"),
    "10997796": ("20260929-164218", "resumed pilot to 100 rounds (user-submitted), eval 50 test prompts"),
    "11006740": ("20260929-211737", "checkpoint sweep: dev rounds 10..100, test rounds 10/60/100"),
}
PLANS = {
    "toy": [("runs/toy_fedavg", "10985846", ["10985846"], {"eval_round_0030.json": "10985846"}),
            ("runs/toy_local", "10985846", ["10985846"], {"eval_round_0030.json": "10985846",
                                                          "eval_round_0030_merged_shared.json": "10985846"})],
    "pilot": [("runs/nr_pilot_llama1b", "10987927", ["10987927", "10997796"],
               {"eval_round_0010.json": "10987927", "eval_round_0010_v2.json": "10996826",
                "eval_round_0100.json": "10997796",
                **{f"eval_round_{r:04d}_dev.json": "11006740" for r in range(10, 101, 10)},
                **{f"eval_round_{r:04d}_test200.json": "11006740" for r in (10, 60, 100)}})],
}


def stamp(job):
    return f"{JOBS[job][0]}_j{job}"


def main(which):
    for old, creator, train_jobs, evals in PLANS[which]:
        new = f"{old}_{stamp(creator)}"
        if not os.path.isdir(old):
            print(f"skip {old}: not found (already renamed to {new}?)")
            continue
        os.rename(old, new)
        os.makedirs(os.path.join(new, "evals"), exist_ok=True)
        for fname, job in evals.items():
            src = os.path.join(new, fname)
            if os.path.exists(src):
                dst = os.path.join(new, "evals", f"{fname.removesuffix('.json')}__{stamp(job)}.json")
                os.rename(src, dst)
                print(f"  {fname} -> evals/{os.path.basename(dst)}")
        cfg_path = os.path.join(new, "config.yaml")
        cfg = yaml.safe_load(open(cfg_path))
        cfg.setdefault("out_prefix", cfg.get("out_dir", old))
        cfg["out_dir"] = new
        yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
        events = [{"event": "created" if i == 0 else "resumed", "slurm_job_id": j, "time": JOBS[j][0],
                   "note": JOBS[j][1] + " (backfilled on rename)"} for i, j in enumerate(train_jobs)]
        events.append({"event": "renamed", "from": old, "to": new, "note": "legacy run, see exp_log"})
        with open(os.path.join(new, "run_info.json"), "w") as f:
            json.dump({"events": events}, f, indent=1)
        print(f"{old} -> {new}")


if __name__ == "__main__":
    main(sys.argv[1])
