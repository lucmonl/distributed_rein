#!/bin/bash
# Experiment log entry 40 (2026-10-04): controlled coverage experiment on ChEMBL.
# Entry 39: C1/G3 failed because ChEMBL clients have NO coverage gaps (17.2% of attribute
# variance between clients, supports ~[0.02,0.95]), so federation has nothing to contribute.
# This manufactures the gaps: complementary truncation of ALL 8 participants --
#   4 clients with low alpha medians keep only alpha <= 0.6 (lose the TOP),
#   4 with high medians keep only alpha >= 0.4 (lose the BOTTOM).
# Truncating all clients the SAME way would be vacuous (nobody would hold the missing region);
# complementary truncation keeps the union covering [0,1] and gives 8 clients with real gaps
# instead of the 3-client subset in the plan.
# Dev/test are untouched, so each client is tested in its removed region against real molecules.
# 100 rounds (entry 39: 30 is undertrained). Train sizes drop to 1,040-3,088, so absolute numbers
# are NOT comparable to the untruncated runs -- the comparison is fed vs local within truncation.
cd /u/lucmon/rein
RR=/u/lucmon/lucmon/rein_runs
COMMON="CONFIG=configs/chembl_deco.yaml,GPU_LOG=1,RUNS_ROOT=$RR,DATA=data/chembl_deco_trunc/data.jsonl,RESCORE=none,SCORER=clogp_residual_deco,TEST_PROMPTS=150"
D="data_path=data/chembl_deco_trunc/data.jsonl clients_file=data/chembl_deco_trunc/clients.json"
EVAL="monitor.full_prompts=25 monitor.full_max_new_tokens=96 monitor.max_new_tokens=96 fed.save_every=20 monitor.full_every=20"
F=$(sbatch --parsable --job-name=exp40_trunc_fed --time=2-00:00:00 \
  --export=ALL,$COMMON,OVERRIDES="$D fed.rounds=100 fed.calibration=shared $EVAL out_dir=$RR/exp40_trunc_fed_qwen3_4b" \
  sbatch/train_eval_chembl.sbatch)
L=$(sbatch --parsable --job-name=exp40_trunc_local --time=2-00:00:00 \
  --export=ALL,$COMMON,OVERRIDES="$D fed.rounds=100 fed.mode=local fed.calibration=private $EVAL out_dir=$RR/exp40_trunc_local_qwen3_4b" \
  sbatch/train_eval_chembl.sbatch)
echo "trunc_fed $F  trunc_local $L"
