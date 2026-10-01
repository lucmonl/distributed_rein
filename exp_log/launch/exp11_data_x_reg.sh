#!/bin/bash
# Experiment log entry 11 (2026-09-30): overfitting study, federated arm.
#   {equal cap 2k, equal cap 4k, natural sizes (<=5k)} x {base, regularized}
#   (natural sizes run only regularized) -> 5 jobs.
# Common: configs/newsroom_fedavg.yaml as of this date (global alpha, offset, warp B,
# private adapter, in-training full dev eval every 10 rounds), Llama-3.2-1B, 100 rounds.
# Pipeline: sbatch/train_eval.sbatch (train -> select on dev -> test -> quality -> judge).
cd /u/lucmon/rein
COMMON="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100"
REG="fed.lr_schedule=cosine lora.dropout=0.05 reg.private_wd=0.05 reg.shared_wd=0.01 reg.decorr=0.1 reg.fedprox_mu=0.1 reg.gain_l2=0.01 reg.offset_l2=0.01"

submit () {  # name, overrides
  sbatch --job-name=$1 --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="$COMMON out_dir=runs/$1 $2" \
    sbatch/train_eval.sbatch
}
submit exp11_cap2k_base    "max_train_per_client=2000"
submit exp11_cap2k_reg     "max_train_per_client=2000 $REG"
submit exp11_cap4k_base    "max_train_per_client=4000"
submit exp11_cap4k_reg     "max_train_per_client=4000 $REG"
submit exp11_natural_reg   "max_train_per_client=null $REG"
