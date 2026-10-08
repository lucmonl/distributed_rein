#!/bin/bash
# Experiment log NR-60 (2026-10-07): private offset on Newsroom, fed and local, matched to NR-54
# (exp_log/launch/NR-54_per_layer_calibration.sh): Llama-3.2-1B, rotation 0, 4k per client, 100 rounds,
# gain fixed at 1, per-layer shape (lora.warp_scope=module), warp_reg 0; same train -> dev selection
# -> test -> quality -> judge pipeline (sbatch/train_eval*.sbatch).
#   nr60-fed-shared-off   shared shape + PRIVATE offset (new flag fed.private_offset)
#   nr60-fed-private-off  private shape + private offset
#   nr60-local-off        local, private offset
#   nr60-local            local, no offset (the matched local baseline at gain 1, per-layer shapes)
#   nr60-local-off-reset  local, private offset, direction's Adam state reset every round as in fed
#                         (new flag fed.local_reset_opt_state; coordinator addendum, after MOL-23)
# Raced on cc + Anvil; all five arms are ONE group. Same job name on both hosts (CONVENTIONS §4).
set -u
cd /u/lucmon/rein
COMMON="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 max_train_per_client=4000 lora.warp=kumaraswamy_mix lora.warp_scope=module fed.fix_gain=true fed.warp_reg=0"
ARMS=(
  "nr60-fed-shared-off|fed.calibration=shared lora.offset=true fed.private_offset=true"
  "nr60-fed-private-off|fed.calibration=private lora.offset=true"
  "nr60-local-off|fed.mode=local fed.calibration=private lora.offset=true"
  "nr60-local|fed.mode=local fed.calibration=private lora.offset=false"
  "nr60-local-off-reset|fed.mode=local fed.calibration=private lora.offset=true fed.local_reset_opt_state=true"
)
# --- cc: Turing (ccc0232-0236) and V100 (ccc0089/0090) scavenger nodes excluded (NR-50), plus ccc0387
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
sbatch --parsable --job-name=$N --time=24:00:00 \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES=\"$COMMON out_dir=runs/$N $O\" \
  sbatch/train_eval_anvil.sbatch")
  echo "anvil $N $J"
done
