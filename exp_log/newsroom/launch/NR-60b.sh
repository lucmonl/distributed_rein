#!/bin/bash
# Experiment log NR-60b (2026-10-07, 22:20): fed-aligned-soff = aligned calibration (NR-58) with ONE offset
# shared by all clients (user correction: the shared-offset arm builds on aligned; replaces the cancelled
# fed-shared-soff of NR-60a). g_{i,l}(alpha) = o + h-bar_l(alpha): h-bar_l pooled exactly as in NR-58/62
# (saturating weights, K = 21, b = 0.2, tau = 100, lambda_max = 0.01, in training and inference); o is
# averaged by the server every round (fed.shared_offset=true), trained from round 2, |o| <= 2.
# NR-60 common settings. Contrasts: fed-aligned-off (NR-62, private o_i), fed-aligned (NR-58, no offset).
set -u
cd /u/lucmon/rein
COMMON="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 max_train_per_client=4000 lora.warp=kumaraswamy_mix lora.warp_scope=module fed.fix_gain=true fed.warp_reg=0"
ARM="fed.calibration=aligned fed.cov_grid=21 fed.cov_pool=saturating fed.cov_lambda_max=0.01 lora.offset=true fed.shared_offset=true"
N=nr60-fed-aligned-soff
CC=$(sbatch --parsable --job-name=$N --time=24:00:00 \
  --partition=dali,IllinoisComputes-GPU,scavenger --exclude=ccc0387,ccc0089,ccc0090,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236 \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="$COMMON $ARM out_dir=runs/$N" \
  sbatch/train_eval.sbatch)
AN=$(ssh -o BatchMode=yes -l x-zchen17 anvil.rcac.purdue.edu "cd /home/x-zchen17/lucmon/rein
sbatch --parsable --job-name=$N --time=24:00:00 \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES=\"$COMMON $ARM out_dir=runs/$N\" \
  sbatch/train_eval_anvil.sbatch")
echo "cc $CC anvil $AN"
