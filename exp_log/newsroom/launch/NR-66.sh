#!/bin/bash
# Experiment log NR-66 (2026-10-08): pooling-weight sweep for fed-aligned-off (user request: the offset
# version of NR-64). g_{i,l}(alpha) = o_i + h-bar_l(alpha), o_i private; only the pooling weight of h-bar_l
# varies: c/(tau_pool + c) with tau_pool in {300, 1000, 3000}, or count pooling. tau_pool = 100 is NR-62.
# Everything else = NR-62 (lambda_max = 0.01, K = 21, b = 0.2, tau_local = 100) and NR-60 common settings.
# cc ONLY: Anvil's conda environment is broken (NR-65); no Anvil copies until it is rebuilt (CAL-11).
set -u
cd /u/lucmon/rein
COMMON="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 max_train_per_client=4000 lora.warp=kumaraswamy_mix lora.warp_scope=module fed.fix_gain=true fed.warp_reg=0"
ALIGNED="fed.calibration=aligned fed.cov_grid=21 fed.cov_lambda_max=0.01 lora.offset=true"
ARMS=(
  "nr66-fed-aligned-off-tp300|fed.cov_pool=saturating fed.cov_tau_pool=300"
  "nr66-fed-aligned-off-tp1000|fed.cov_pool=saturating fed.cov_tau_pool=1000"
  "nr66-fed-aligned-off-tp3000|fed.cov_pool=saturating fed.cov_tau_pool=3000"
  "nr66-fed-aligned-off-cnt|fed.cov_pool=count"
)
for a in "${ARMS[@]}"; do
  N=${a%%|*}; O=${a#*|}
  J=$(sbatch --parsable --job-name=$N --time=24:00:00 \
    --partition=dali,IllinoisComputes-GPU,scavenger --exclude=ccc0387,ccc0089,ccc0090,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236 \
    --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="$COMMON $ALIGNED $O out_dir=runs/$N" \
    sbatch/train_eval.sbatch)
  echo "cc $N $J"
done
