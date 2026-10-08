#!/bin/bash
# Experiment log NR-58 (2026-10-07): aligned calibration (user design, after NR-57).
#   training : client i's forward pass uses g-bar itself (per layer): pooled with saturating weights
#              c/(tau+c) from its OWN live warp values (trained) and the other clients' values frozen
#              from the last round, then a differentiable weighted isotonic projection
#              (fedsteer/coverage.pool_and_project); gradients reach g_i only where it has weight
#   penalty  : tiny tie lambda_max * c/(tau+c) * (g_i - stopgrad g-bar)^2, lambda_max = 0.01, only to
#              pin the directions of g_i that g-bar (a weighted average) does not determine
#   inference: the same operator on everyone's final values (one table per layer, linear interpolation)
# Same setting as NR-54/56 (Llama-3.2-1B, rotation 0, 4k per client, 100 rounds, gain 1, no offset,
# warp_reg 0, lora.warp_scope=module, K = 21, b = 0.2, tau = 100). Compared with NR-54 A_L / B_L and
# NR-56 consensus. Raced on cc + Anvil.
set -u
cd /u/lucmon/rein
COMMON="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 max_train_per_client=4000 lora.offset=false lora.warp=kumaraswamy_mix lora.warp_scope=module fed.fix_gain=true fed.warp_reg=0 fed.calibration=aligned fed.cov_grid=21 fed.cov_pool=saturating fed.cov_lambda_max=0.01"
N=nr58_aligned_l0p01
CC=$(sbatch --parsable --job-name=$N --time=24:00:00 \
  --partition=dali,IllinoisComputes-GPU,scavenger --exclude=ccc0387,ccc0089,ccc0090,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236 \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="$COMMON out_dir=runs/$N" \
  sbatch/train_eval.sbatch)
AN=$(ssh -o BatchMode=yes -l x-zchen17 anvil.rcac.purdue.edu "cd /home/x-zchen17/lucmon/rein
sbatch --parsable --job-name=${N}_anvil --time=24:00:00 \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES=\"$COMMON out_dir=runs/${N}_anvil\" \
  sbatch/train_eval_anvil.sbatch")
echo "cc $CC anvil $AN"
