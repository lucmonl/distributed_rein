#!/bin/bash
# Experiment log entry 9 (2026-09-30): global alpha, federated vs. local-only.
# Exact submission commands. Both use the generic pipeline sbatch/train_eval.sbatch
# (train -> dev sweep over snapshots -> select checkpoint -> test eval); only the
# overrides differ.  Submitted as jobs 11011538 (fedavg) and 11011539 (local).
cd /u/lucmon/rein

# federated (method)
sbatch --job-name=nr_glob_fedavg \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="model_name=meta-llama/Llama-3.2-1B-Instruct out_dir=runs/nr_global_fedavg fed.rounds=100 fed.save_every=5" \
  sbatch/train_eval.sbatch

# local-only baseline (B2); also writes merged_shared.pt for B3
sbatch --job-name=nr_glob_local \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="model_name=meta-llama/Llama-3.2-1B-Instruct out_dir=runs/nr_global_local fed.mode=local fed.rounds=100 fed.save_every=5" \
  sbatch/train_eval.sbatch
