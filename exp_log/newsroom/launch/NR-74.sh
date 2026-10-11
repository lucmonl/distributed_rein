#!/bin/bash
# Experiment log NR-74 (2026-10-09): two arms in the new parameterization (coordinator instruction, user request).
#   nr74-fed-linear-off           g_i(alpha) = o_i + alpha: no learned shape (lora.warp=none), private offset.
#                                 (lora.warp_scope=module is ignored with warp=none: lora.inject falls back to one map.)
#   nr74-fed-shared-soff-sadapter pure FedAvg: shared adapter, per-layer shape averaged, ONE shared offset o.
# Matched to NR-62/NR-60: Llama-3.2-1B, rotation 0, 4k per client, 100 rounds, gain fixed at 1, warp_reg 0.
# cc only (Anvil broken); --gres=gpu:A100:1 to match NR-62's GPU type family; ccc0387 excluded.
set -u
cd /u/lucmon/rein
COMMON="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 max_train_per_client=4000 fed.fix_gain=true fed.warp_reg=0"
ARMS=(
  "nr74-fed-linear-off|lora.warp=none lora.warp_scope=module fed.calibration=shared lora.offset=true fed.private_offset=true"
  "nr74-fed-shared-soff-sadapter|fed.adapter=shared fed.calibration=shared lora.offset=true lora.warp=kumaraswamy_mix lora.warp_scope=module"
)
for a in "${ARMS[@]}"; do
  N=${a%%|*}; O=${a#*|}
  J=$(sbatch --parsable --job-name=$N --time=24:00:00 --partition=dali,IllinoisComputes-GPU \
    --gres=gpu:A100:1 --exclude=ccc0387 \
    --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="$COMMON out_dir=runs/$N $O" \
    sbatch/train_eval.sbatch)
  echo "cc $N $J"
done
