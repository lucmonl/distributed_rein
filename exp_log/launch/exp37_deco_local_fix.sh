#!/bin/bash
# Experiment log entry 38 (2026-10-03): resubmit the deco LOCAL run.
# 11135487 died in 2m17s: configs/chembl_deco.yaml bakes in fed.calibration=shared (the method
# default from entry 20) and fed.py rejects shared calibration in local mode -- nothing is
# aggregated, so local training always has per-client calibration. The launcher overrode
# fed.mode=local but not the calibration. Adding fed.calibration=private.
cd /u/lucmon/rein
RR=/u/lucmon/lucmon/rein_runs
sbatch --parsable --job-name=exp37_deco_local --time=2-00:00:00 \
  --export=ALL,CONFIG=configs/chembl_deco.yaml,GPU_LOG=1,RUNS_ROOT=$RR,DATA=data/chembl_deco/data.jsonl,RESCORE=none,SCORER=clogp_residual_deco,TEST_PROMPTS=150,\
OVERRIDES="fed.rounds=30 fed.mode=local fed.calibration=private monitor.full_prompts=25 monitor.full_max_new_tokens=96 monitor.max_new_tokens=96 out_dir=$RR/exp37_deco_local_qwen3_4b" \
  sbatch/train_eval_chembl.sbatch
