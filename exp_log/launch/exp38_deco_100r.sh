#!/bin/bash
# Experiment log entry 38b (2026-10-03): deco fed/local pair at 100 rounds.
# The 30-round federated run (11135486) was clearly undertrained: dev pctErr improved every
# snapshot (0.259 -> 0.249 -> 0.237), dev loss was still falling (0.368 -> 0.305) and the
# dev-selected checkpoint was the FINAL round. Newsroom used 100 rounds. Round time 178 s, so
# 100 rounds is ~5 h of training plus evals -- inside the 2-day limit.
# The 30-round pair still runs, to keep a matched-budget fed-vs-local comparison.
cd /u/lucmon/rein
RR=/u/lucmon/lucmon/rein_runs
COMMON="CONFIG=configs/chembl_deco.yaml,GPU_LOG=1,RUNS_ROOT=$RR,DATA=data/chembl_deco/data.jsonl,RESCORE=none,SCORER=clogp_residual_deco,TEST_PROMPTS=150"
EVAL="monitor.full_prompts=25 monitor.full_max_new_tokens=96 monitor.max_new_tokens=96 fed.save_every=20 monitor.full_every=20"
F=$(sbatch --parsable --job-name=exp38_deco_fed100 --time=2-00:00:00 \
  --export=ALL,$COMMON,OVERRIDES="fed.rounds=100 fed.calibration=shared $EVAL out_dir=$RR/exp38_deco_fed100_qwen3_4b" \
  sbatch/train_eval_chembl.sbatch)
L=$(sbatch --parsable --job-name=exp38_deco_local100 --time=2-00:00:00 \
  --export=ALL,$COMMON,OVERRIDES="fed.rounds=100 fed.mode=local fed.calibration=private $EVAL out_dir=$RR/exp38_deco_local100_qwen3_4b" \
  sbatch/train_eval_chembl.sbatch)
echo "fed100 $F  local100 $L"
