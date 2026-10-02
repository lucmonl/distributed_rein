#!/bin/bash
# Experiment log entry 25 (2026-10-01): ablation / non-personalized baseline A2.
# The method configuration (exp17 F_shared_noO, 11063910: federated, shared calibration,
# no offset, 4k per client, 100 rounds) with fed.adapter=shared: the "private" adapter is
# averaged by the server every round like D, so nothing stays on the client (one global
# alpha-conditioned FedAvg model).  Same pipeline: dev selection, test, quality, judge.
cd /u/lucmon/rein
COMMON="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 max_train_per_client=4000"
submit () {
  sbatch --job-name=$1 --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="$COMMON out_dir=runs/$1 $2" \
    sbatch/train_eval.sbatch
}
submit exp25_fed_adaptershared_calshared_nooff_cap4k "fed.calibration=shared lora.offset=false fed.adapter=shared"
