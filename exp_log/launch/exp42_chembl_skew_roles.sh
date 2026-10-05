#!/bin/bash
# Experiment log entry 42 (2026-10-04): natural skew, v2 -- stronger high side + middle clients.
# v1 (11152110/11152111, cancelled) under-skewed the high side: [0.37, 0.98]. Cause: the natural
# alpha medians only span 0.27-0.64 (the scale is built from these clients), so a ramp centred at
# 0.5 cannot lift a high-side 5th percentile to ~0.5. The ramp centre is now a per-role parameter.
# Also added MIDDLE-only clients, a regime Newsroom lacks entirely (its clients are all one-sided):
# a client holding only mid-range alpha needs BOTH tails from others, so the direction has to
# extrapolate in both directions for the same client.
# Roles: low/high specialist, strong/mild middle, low/high moderate, broad, untouched.
cd /u/lucmon/rein
RR=/u/lucmon/lucmon/rein_runs
COMMON="CONFIG=configs/chembl_deco.yaml,GPU_LOG=1,RUNS_ROOT=$RR,DATA=data/chembl_deco_skew/data.jsonl,RESCORE=none,SCORER=clogp_residual_deco,TEST_PROMPTS=150"
D="data_path=data/chembl_deco_skew/data.jsonl clients_file=data/chembl_deco_skew/clients.json"
EVAL="monitor.full_prompts=25 monitor.full_max_new_tokens=96 monitor.max_new_tokens=96 fed.save_every=20 monitor.full_every=20"
F=$(sbatch --parsable --job-name=exp42_skew_fed --time=24:00:00 \
  --export=ALL,$COMMON,OVERRIDES="$D fed.rounds=100 fed.calibration=shared $EVAL out_dir=$RR/exp42_skew_fed_qwen3_4b" \
  sbatch/train_eval_chembl.sbatch)
L=$(sbatch --parsable --job-name=exp42_skew_local --time=24:00:00 \
  --export=ALL,$COMMON,OVERRIDES="$D fed.rounds=100 fed.mode=local fed.calibration=private $EVAL out_dir=$RR/exp42_skew_local_qwen3_4b" \
  sbatch/train_eval_chembl.sbatch)
echo "skew_fed $F  skew_local $L"
