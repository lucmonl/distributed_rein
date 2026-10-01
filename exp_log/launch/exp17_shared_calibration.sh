#!/bin/bash
# Experiment log entry 17 (2026-10-01): shared calibration under the global alpha scale.
# 4k examples per client, no regularizers (same as exp11_cap4k_base, 11041282), 100 rounds.
#   B           = fed.calibration=shared (gain, offset, warp shared by all clients)
#   B_nooff     = B with lora.offset=false
#   local arms  : local mode cannot share calibration (nothing is aggregated), so
#                 "local B" == local with per-client calibration == job 11061046 (exp16,
#                 already queued; not resubmitted). "local B_nooff" = local without offset.
cd /u/lucmon/rein
COMMON="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 max_train_per_client=4000"
submit () {
  sbatch --job-name=$1 --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="$COMMON out_dir=runs/$1 $2" \
    sbatch/train_eval.sbatch
}
submit exp17_fed_calshared_cap4k        "fed.calibration=shared"
submit exp17_fed_calshared_nooff_cap4k  "fed.calibration=shared lora.offset=false"
submit exp17_local_nooff_cap4k          "fed.mode=local lora.offset=false"
