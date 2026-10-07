#!/bin/bash
# Experiment log NR-50 (2026-10-06): replacement for NR-46 arm C10 (lambda_max = 10).
# The original, 11174461, landed on scavenger node ccc0234, a Quadro RTX 6000 (Turing: no native
# bf16), and runs ~985 s/round vs ~87 s on A100 -> 100 rounds cannot fit its 24 h limit.
# Same overrides as NR-46 C10; cc copy excludes the Turing nodes ccc0232-ccc0236; raced on Anvil.
set -u
cd /u/lucmon/rein
COMMON="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 max_train_per_client=4000 lora.offset=false lora.warp=kumaraswamy_mix fed.fix_gain=true fed.warp_reg=0 fed.calibration=coverage"
N=nr46_cal_coverage_l10b
CC=$(sbatch --parsable --job-name=$N --time=24:00:00 \
  --partition=dali,IllinoisComputes-GPU,scavenger --exclude=ccc0387,ccc0089,ccc0090,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236 \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="$COMMON out_dir=runs/$N fed.cov_lambda_max=10" \
  sbatch/train_eval.sbatch)
AN=$(ssh -o BatchMode=yes -l x-zchen17 anvil.rcac.purdue.edu "cd /home/x-zchen17/lucmon/rein
sbatch --parsable --job-name=${N}_anvil --time=24:00:00 \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES=\"$COMMON out_dir=runs/${N}_anvil fed.cov_lambda_max=10\" \
  sbatch/train_eval_anvil.sbatch")
echo "cc $CC  anvil $AN"
