#!/bin/bash
# Experiment log entry 16 (2026-09-30): local-only baseline (B2) at 4k examples per client,
# the fair counterpart of exp11_cap4k_base (11041282). Identical overrides except fed.mode=local.
# Local mode also writes merged_shared.pt (one-shot merged direction, baseline B3).
cd /u/lucmon/rein
sbatch --job-name=exp16_local_cap4k \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 out_dir=runs/exp16_local_cap4k max_train_per_client=4000 fed.mode=local" \
  sbatch/train_eval.sbatch
