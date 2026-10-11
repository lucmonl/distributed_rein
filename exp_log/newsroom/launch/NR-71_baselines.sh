#!/bin/bash
# Experiment log NR-71 (2026-10-09): baselines B1 / B3 / B4 for fed-aligned-off (coordinator instruction,
# user request). Archive protocol of entries 19/22 via sbatch/eval_baselines.sbatch; test, 200 articles per
# client, quality scoring after each step.
#   B1 prompting on fed-aligned-off's client models (adapter + calibration, direction removed): k = 0, k = 3.
#      k = 3 base model: reused from the archive (11065061; same data subset, alpha reference, template,
#      max_new_tokens 128, 200 prompts).
#   B4 federated CAA on the same client models, widened gain grid (archive: 6/8 clients at the top gain).
#   B3 one-shot merge of a local run's directions (each client keeps its adapter, per-layer shapes, offset):
#      local-off (dev-selected r60) and local (r100).
# Variables go through the environment and --export=ALL: sbatch --export splits values on commas (entry 19a).
# cc only, --gres=gpu:A100:1, --exclude=ccc0387. Smoke first; the real jobs depend on it.
set -u
cd /u/lucmon/rein
FED=runs/nr62-fed-aligned-off_20261007-231639_j11209264
LOFF=runs/nr60-local-off_20261007-181308_j11206105
LOC=runs/nr60-local_20261007-231639_j11206106
export FED_RUN=$FED FED_SNAP=$FED/snapshots/round_0100.pt
COMMON="--time=12:00:00 --partition=dali,IllinoisComputes-GPU --gres=gpu:A100:1 --exclude=ccc0387 --export=ALL"

S=$(STEPS=b1+b4+b3 SMOKE=1 LOCAL_RUN=$LOFF LOCAL_SNAP=$LOFF/snapshots/round_0060.pt \
    sbatch --parsable --job-name=nr71-smoke $COMMON sbatch/eval_baselines.sbatch)
echo "nr71-smoke $S"
DEP="--dependency=afterok:$S"
J=$(STEPS=b1 B1_SETS="--shots 0 --model client" sbatch --parsable --job-name=nr71-b1-k0 $COMMON $DEP sbatch/eval_baselines.sbatch)
echo "nr71-b1-k0 $J"
J=$(STEPS=b1 B1_SETS="--shots 3 --model client" sbatch --parsable --job-name=nr71-b1-k3 $COMMON $DEP sbatch/eval_baselines.sbatch)
echo "nr71-b1-k3 $J"
J=$(STEPS=b4 CAA_ARGS="--caa_gains 0.05,0.1,0.2,0.3,0.5,0.8,1.2,1.6,2.4,3.2" \
    sbatch --parsable --job-name=nr71-b4-caa $COMMON $DEP sbatch/eval_baselines.sbatch)
echo "nr71-b4-caa $J"
J=$(STEPS=b3 LOCAL_RUN=$LOFF LOCAL_SNAP=$LOFF/snapshots/round_0060.pt \
    sbatch --parsable --job-name=nr71-b3-local-off $COMMON $DEP sbatch/eval_baselines.sbatch)
echo "nr71-b3-local-off $J"
J=$(STEPS=b3 LOCAL_RUN=$LOC LOCAL_SNAP=$LOC/snapshots/round_0100.pt \
    sbatch --parsable --job-name=nr71-b3-local $COMMON $DEP sbatch/eval_baselines.sbatch)
echo "nr71-b3-local $J"
