#!/bin/bash
# Experiment log NR-54 (2026-10-06): the NR-46/49 calibration suite rerun with PER-LAYER warps.
# Until now every adapted matrix of a client used the same h_i (one SteerControl warp); with
# lora.warp_scope=module each of the 112 adapted matrices of Llama-3.2-1B (16 blocks x 7
# projections) has its own kumaraswamy_mix warp h_{i,l} (user request, 2026-10-06).
# Everything else as NR-46/49: rotation 0, 4k per client, 100 rounds, gain fixed at 1, no offset,
# warp_reg = 0, warp warm-up 5 rounds; coverage K = 11, b = 0.2, tau_local = tau_peer = 100.
# Arms: A shared (server averages every layer's warp) | B private | C coverage, lambda_max in
# {0.01, 0.1, 1, 10} (one server table per layer; counts and lambda per grid point as before).
# Raced on cc + Anvil; all six arms are ONE group so the comparison comes from one host.
set -u
cd /u/lucmon/rein
COMMON="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 max_train_per_client=4000 lora.offset=false lora.warp=kumaraswamy_mix lora.warp_scope=module fed.fix_gain=true fed.warp_reg=0"
ARMS=(
  "nr54_layer_shared|fed.calibration=shared"
  "nr54_layer_private|fed.calibration=private"
  "nr54_layer_cov_l0p01|fed.calibration=coverage fed.cov_lambda_max=0.01"
  "nr54_layer_cov_l0p1|fed.calibration=coverage fed.cov_lambda_max=0.1"
  "nr54_layer_cov_l1|fed.calibration=coverage fed.cov_lambda_max=1"
  "nr54_layer_cov_l10|fed.calibration=coverage fed.cov_lambda_max=10"
)
# --- cc: Turing (ccc0232-0236) and V100 (ccc0089/0090) scavenger nodes excluded (NR-50)
for a in "${ARMS[@]}"; do
  N=${a%%|*}; O=${a#*|}
  J=$(sbatch --parsable --job-name=$N --time=24:00:00 \
    --partition=dali,IllinoisComputes-GPU,scavenger --exclude=ccc0387,ccc0089,ccc0090,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236 \
    --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="$COMMON out_dir=runs/$N $O" \
    sbatch/train_eval.sbatch)
  echo "cc $N $J"
done
# --- Anvil (-A cis260796-ai, -p ai only)
for a in "${ARMS[@]}"; do
  N=${a%%|*}; O=${a#*|}
  J=$(ssh -o BatchMode=yes -l x-zchen17 anvil.rcac.purdue.edu "cd /home/x-zchen17/lucmon/rein
sbatch --parsable --job-name=${N}_anvil --time=24:00:00 \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES=\"$COMMON out_dir=runs/${N}_anvil $O\" \
  sbatch/train_eval_anvil.sbatch")
  echo "anvil $N $J"
done
