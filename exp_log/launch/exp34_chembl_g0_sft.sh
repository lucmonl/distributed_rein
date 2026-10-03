#!/bin/bash
# Experiment log entry 34 (2026-10-02): gate G0 as the PLAN specifies it -- plain SFT, no steering.
# Entry 33's "G0" used rank_shared=16 with an alpha-conditioned loss, so it measured steering, not
# backbone competence, and cannot attribute the retention failure.
# rank_shared=0 is unsupported (scale_d = lora_alpha_shared / rank_shared divides by zero), but
# B_d initializes to ZEROS, so fed.lr_shared=0 leaves the direction exactly zero forever: the model
# is alpha-independent and this is plain SFT of the private adapter.
# Sanity assertions for the log: direction_norm == 0 every round, and the dev eval must show
# no_effect_rate == 1.0 / text_tie_rate == 1.0 (identical output at every alpha).
cd /u/lucmon/rein
RR=/u/lucmon/lucmon/rein_runs
sbatch --parsable --job-name=exp34_chembl_g0_sft --time=06:00:00 \
  --export=ALL,CONFIG=configs/chembl_fedavg.yaml,GPU_LOG=1,TEST_PROMPTS=100,RUNS_ROOT=$RR,\
OVERRIDES="clients=[CHEMBL240] fed.lr_shared=0 fed.rounds=100 fed.save_every=20 monitor.full_every=20 monitor.full_prompts=50 out_dir=$RR/exp34_chembl_g0_sft_qwen3_4b" \
  sbatch/train_eval_chembl.sbatch
