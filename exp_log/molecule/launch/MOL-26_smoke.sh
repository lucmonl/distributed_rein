#!/bin/bash
# MOL-26 pre-flight smoke (2026-10-09). NOT an experiment: SMOKE=1 forces 4 prompts, 8 CAA examples
# and a 2-point gain grid. One job exercising all three baseline paths end to end before the real
# five go out, per MOL-22b's lesson that only an end-to-end run finds these failures.
#
# What can break here and nowhere else:
#   - B1/B4 load an ALIGNED snapshot (pooled table h-bar_l + private o_i), which no baseline run has
#     done before; with lora_B_d zeroed the output must be the adapter alone, so the score grid must
#     be alpha-INVARIANT. Checked from the smoke's own eval json afterwards.
#   - B3 merges local-off's per-layer directions and evaluates with each client's own adapter,
#     per-layer shapes and offset.
#   - The strict deco scorer and the pruned dataset reach every step (SCORER/DATA are passed, not
#     defaulted: the file's defaults are clogp_residual and data/chembl_fed, neither of which is MOL-25's).
set -u
cd /u/lucmon/rein
F=$(ls -d runs/mol25-fed-aligned-off_*_j11237283)
L=$(ls -d runs/mol25-local-off_*_j11237285)
sbatch --parsable --job-name=mol26-smoke --time=1:00:00 --gres=gpu:A100:1 --exclude=ccc0387 \
  --export=ALL,SMOKE=1,STEPS=b1+b4+b3,SCORER=clogp_residual_deco_strict,DATA=data/chembl_deco_skew_pruned/data.jsonl,FED_RUN=$F,FED_SNAP=$F/snapshots/round_0100.pt,LOCAL_RUN=$L,LOCAL_SNAP=$L/snapshots/round_0100.pt \
  sbatch/eval_baselines_chembl.sbatch
