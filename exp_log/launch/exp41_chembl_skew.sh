#!/bin/bash
# Experiment log entry 41 (2026-10-04): NATURAL coverage skew (replaces entry 40's hard cut).
# Entry 40 censored each client at a hard alpha threshold (keep alpha<=0.6 or >=0.4). That was
# artificial in three ways: molecule-level rather than programme-level censoring, exactly zero
# mass beyond the cut, and all 8 clients given identical window widths -- which is MORE uniform
# than Newsroom, where only 2 of 8 clients are strongly skewed (widths 0.44, 0.50) and the rest
# are broad (0.67-0.96). Jobs 11152030/11152031 were cancelled before they started.
# scripts/skew_chembl.py instead drops whole chemical series (a company's programmes) with a
# smooth propensity, thins each surviving series' disfavoured tail, and gives every client its own
# skew strength beta_i. Result: widths 0.53-0.94 (median 0.76) vs Newsroom 0.44-0.96 (median 0.81),
# one strongly skewed client per side, and every client keeping a real tail (5-444 molecules).
cd /u/lucmon/rein
RR=/u/lucmon/lucmon/rein_runs
COMMON="CONFIG=configs/chembl_deco.yaml,GPU_LOG=1,RUNS_ROOT=$RR,DATA=data/chembl_deco_skew/data.jsonl,RESCORE=none,SCORER=clogp_residual_deco,TEST_PROMPTS=150"
D="data_path=data/chembl_deco_skew/data.jsonl clients_file=data/chembl_deco_skew/clients.json"
EVAL="monitor.full_prompts=25 monitor.full_max_new_tokens=96 monitor.max_new_tokens=96 fed.save_every=20 monitor.full_every=20"
F=$(sbatch --parsable --job-name=exp41_skew_fed --time=2-00:00:00 \
  --export=ALL,$COMMON,OVERRIDES="$D fed.rounds=100 fed.calibration=shared $EVAL out_dir=$RR/exp41_skew_fed_qwen3_4b" \
  sbatch/train_eval_chembl.sbatch)
L=$(sbatch --parsable --job-name=exp41_skew_local --time=2-00:00:00 \
  --export=ALL,$COMMON,OVERRIDES="$D fed.rounds=100 fed.mode=local fed.calibration=private $EVAL out_dir=$RR/exp41_skew_local_qwen3_4b" \
  sbatch/train_eval_chembl.sbatch)
echo "skew_fed $F  skew_local $L"
