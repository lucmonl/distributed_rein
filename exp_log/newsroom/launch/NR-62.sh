#!/bin/bash
# Experiment log NR-62 (2026-10-07): fed-aligned-off = aligned calibration (NR-58) with a PRIVATE offset
# (user request). Coefficient g_{i,l}(alpha) = o_i + g-bar_l(alpha): the shapes h_{i,l} are pooled into
# g-bar_l exactly as in NR-58 (training and inference, saturating weights, K = 21, b = 0.2, tau = 100,
# lambda_max = 0.01); o_i stays per client, trained from the gain warm-up (round 2), |o_i| <= 2.
# Matched to NR-58 (fed-aligned) and NR-60 (fed-shared-off): Llama-3.2-1B, rotation 0, 4k per client,
# 100 rounds, gain fixed at 1, per-layer shapes, warp_reg 0. Raced on cc + Anvil.
set -u
cd /u/lucmon/rein
COMMON="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 max_train_per_client=4000 lora.warp=kumaraswamy_mix lora.warp_scope=module fed.fix_gain=true fed.warp_reg=0"
ALIGNED="fed.calibration=aligned fed.cov_grid=21 fed.cov_pool=saturating fed.cov_lambda_max=0.01 lora.offset=true"
N=nr62-fed-aligned-off
CC=$(sbatch --parsable --job-name=$N --time=24:00:00 \
  --partition=dali,IllinoisComputes-GPU,scavenger --exclude=ccc0387,ccc0089,ccc0090,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236 \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="$COMMON $ALIGNED out_dir=runs/$N" \
  sbatch/train_eval.sbatch)
AN=$(ssh -o BatchMode=yes -l x-zchen17 anvil.rcac.purdue.edu "cd /home/x-zchen17/lucmon/rein
sbatch --parsable --job-name=$N --time=24:00:00 \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES=\"$COMMON $ALIGNED out_dir=runs/$N\" \
  sbatch/train_eval_anvil.sbatch")
echo "cc $CC anvil $AN"
