#!/bin/bash
# Experiment log entry 31 (2026-10-02): ChEMBL gate G0 -- can Qwen3-4B-Instruct-2507 emit
# valid, core-preserving SMILES after SFT on one client?  One client (CHEMBL240, hERG,
# 4000 train pairs), 20 rounds x 20 steps = 400 steps ~ 1 epoch at batch 8.
# Pass condition: validity >= 0.90 and scaffold retention >= 0.80 on dev.
cd /u/lucmon/rein
RR=/u/lucmon/lucmon/rein_runs
sbatch --parsable --job-name=exp31_chembl_g0 --time=04:00:00 \
  --export=ALL,CONFIG=configs/chembl_fedavg.yaml,GPU_LOG=1,TEST_PROMPTS=100,RUNS_ROOT=$RR,\
OVERRIDES="clients=[CHEMBL240] fed.rounds=20 fed.save_every=10 monitor.full_every=10 monitor.full_prompts=50 out_dir=$RR/exp31_chembl_g0_qwen3_4b" \
  sbatch/train_eval_chembl.sbatch
