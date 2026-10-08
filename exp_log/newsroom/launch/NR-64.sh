#!/bin/bash
# Experiment log NR-64 (2026-10-08): pooling-weight sweep for fed-aligned (user request: tune tau).
# Pooling weight of client i at grid point k: c_ik / (tau_pool + c_ik) (saturating) or c_ik (count).
# NR-58 used tau_pool = 100: with in-support counts in the hundreds to thousands every weight saturates,
# so nypost.com + reuters.com hold 0.33 of h-bar_l at alpha = 0.9 while owning 0.66 of the data there.
# Grid (share at alpha = 0.9): 300 (0.39), 1000 (0.49), 3000 (0.58), count (0.66); 10000 (0.63) ~ count.
# Everything else = NR-58 (fed-aligned, no offset, lambda_max = 0.01, K = 21, b = 0.2, tau_local = 100),
# NR-60 common settings. Raced on cc + Anvil, all four arms ONE group (same host for the sweep).
set -u
cd /u/lucmon/rein
COMMON="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 max_train_per_client=4000 lora.warp=kumaraswamy_mix lora.warp_scope=module fed.fix_gain=true fed.warp_reg=0"
ALIGNED="fed.calibration=aligned fed.cov_grid=21 fed.cov_lambda_max=0.01 lora.offset=false"
ARMS=(
  "nr64-fed-aligned-tp300|fed.cov_pool=saturating fed.cov_tau_pool=300"
  "nr64-fed-aligned-tp1000|fed.cov_pool=saturating fed.cov_tau_pool=1000"
  "nr64-fed-aligned-tp3000|fed.cov_pool=saturating fed.cov_tau_pool=3000"
  "nr64-fed-aligned-cnt|fed.cov_pool=count"
)
for a in "${ARMS[@]}"; do
  N=${a%%|*}; O=${a#*|}
  J=$(sbatch --parsable --job-name=$N --time=24:00:00 \
    --partition=dali,IllinoisComputes-GPU,scavenger --exclude=ccc0387,ccc0089,ccc0090,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236 \
    --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="$COMMON $ALIGNED $O out_dir=runs/$N" \
    sbatch/train_eval.sbatch)
  echo "cc $N $J"
done
for a in "${ARMS[@]}"; do
  N=${a%%|*}; O=${a#*|}
  J=$(ssh -o BatchMode=yes -l x-zchen17 anvil.rcac.purdue.edu "cd /home/x-zchen17/lucmon/rein
sbatch --parsable --job-name=$N --time=24:00:00 \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES=\"$COMMON $ALIGNED $O out_dir=runs/$N\" \
  sbatch/train_eval_anvil.sbatch")
  echo "anvil $N $J"
done
