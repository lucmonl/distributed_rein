#!/bin/bash
# Experiment log entry 37 (2026-10-03): deco-format method/local pair, resubmitted.
# The original pair (11124739/11124740) died DependencyNeverSatisfied because deco G0
# (11124738) hit its 6 h wall clock -- after finishing training AND the test eval.
# G0 passed (validity/retention/uniqueness/novelty all 1.000, pct err 0.165), so no dependency.
# Changes vs the first attempt:
#   --time 2 days (Lustre was stalling today; the single-client run needed 4.5 h of wall clock
#                  for work that should take ~45 min)
#   monitor.full_max_new_tokens/max_new_tokens 192 -> 96 (observed generation max: 68 tokens)
#   monitor.full_prompts 40 -> 25 (8 clients per full dev eval instead of 1)
cd /u/lucmon/rein
RR=/u/lucmon/lucmon/rein_runs
COMMON="CONFIG=configs/chembl_deco.yaml,GPU_LOG=1,RUNS_ROOT=$RR,DATA=data/chembl_deco/data.jsonl,RESCORE=none,SCORER=clogp_residual_deco,TEST_PROMPTS=150"
EVAL="monitor.full_prompts=25 monitor.full_max_new_tokens=96 monitor.max_new_tokens=96"
F=$(sbatch --parsable --job-name=exp36_deco_fed --time=2-00:00:00 \
  --export=ALL,$COMMON,OVERRIDES="fed.rounds=30 fed.calibration=shared $EVAL out_dir=$RR/exp36_deco_fed_qwen3_4b" \
  sbatch/train_eval_chembl.sbatch)
L=$(sbatch --parsable --job-name=exp36_deco_local --time=2-00:00:00 \
  --export=ALL,$COMMON,OVERRIDES="fed.rounds=30 fed.mode=local $EVAL out_dir=$RR/exp36_deco_local_qwen3_4b" \
  sbatch/train_eval_chembl.sbatch)
echo "fed $F  local $L"
