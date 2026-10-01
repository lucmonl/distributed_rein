#!/bin/bash
# Experiment log entry 11/14: resubmit the two exp11 cells that failed at round 10
# (jobs 11041283, 11041284: ImportError from a code edit made while they ran; entry 14).
# Same overrides as exp11_data_x_reg.sh. Jobs now run from a private code snapshot.
cd /u/lucmon/rein
COMMON="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100"
REG="fed.lr_schedule=cosine lora.dropout=0.05 reg.private_wd=0.05 reg.shared_wd=0.01 reg.decorr=0.1 reg.fedprox_mu=0.1 reg.gain_l2=0.01 reg.offset_l2=0.01"
submit () {
  sbatch --job-name=$1 --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="$COMMON out_dir=runs/$1 $2" \
    sbatch/train_eval.sbatch
}
submit exp11_cap4k_reg     "max_train_per_client=4000 $REG"
submit exp11_natural_reg   "max_train_per_client=null $REG"
