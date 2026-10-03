#!/bin/bash
# Experiment log entry 33 (2026-10-02): G0 retry with 5x the training.
# G0 (entry 33, job 11117359) failed: validity 0.82 (gate 0.90), retention 0.58 (gate 0.80),
# and retention collapses with alpha (0.81 -> 0.32).  Dev loss was still falling steeply at
# round 20 (0.651 -> 0.545 over rounds 10-20, no plateau) and 400 steps x batch 8 is only
# ~0.8 epochs on 4k pairs, so undertraining is the leading hypothesis.
# Same client (CHEMBL240), 100 rounds x 20 steps ~ 4 epochs, snapshot every 20 rounds so the
# validity/retention trend over training is visible.
cd /u/lucmon/rein
RR=/u/lucmon/lucmon/rein_runs
sbatch --parsable --job-name=exp33_chembl_g0b --time=06:00:00 \
  --export=ALL,CONFIG=configs/chembl_fedavg.yaml,GPU_LOG=1,TEST_PROMPTS=100,RUNS_ROOT=$RR,\
OVERRIDES="clients=[CHEMBL240] fed.rounds=100 fed.save_every=20 monitor.full_every=20 monitor.full_prompts=50 out_dir=$RR/exp33_chembl_g0b_qwen3_4b" \
  sbatch/train_eval_chembl.sbatch
