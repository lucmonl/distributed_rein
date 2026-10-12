#!/bin/bash
# MOL-29 pre-flight smoke (2026-10-11). NOT an experiment. First E2 ever run on ChEMBL.
# 2 of the 4 held-out clients, n=16, frozen_D + local_D, 20 training steps.
#
# The point is the CAL-33 checks, not the numbers:
#   - e2_heldout.py must pick clogp_residual_deco_strict from the run's monitor.scorer (it used to
#     hardcode SCORERS["density"], which would have scored decoration sets by extractive fragment
#     density and completed normally with meaningless percentiles). The log line "E2 scorer: ..."
#     reports which one it chose.
#   - the percentiles must be NON-NULL and sensible, so --max_prompts is 30 rather than the
#     sbatch SMOKE=1 default of 4: at 4 prompts the strict scorer can return all-nan and a null
#     result would prove nothing (MOL-26's smoke did exactly that).
#   - --max_new_tokens 96 matches MOL-25's test settings; the real jobs use --max_prompts 150.
#   - QUALITY/JUDGE default to 0 in sbatch/eval_settings_chembl.sbatch because score_quality.py and
#     judge_quality.py need Newsroom's article/target/url fields.
# Rotation 0 held-out clients: CHEMBL205, CHEMBL236, CHEMBL4005, CHEMBL4409.
set -u
cd /u/lucmon/rein
R=$(ls -d runs/mol28-fed-aligned-off-tp1000_*_j11256740)
sbatch --parsable --job-name=mol29-smoke --time=2:00:00 --gres=gpu:A100:1 --exclude=ccc0387 \
  --export=ALL,STEPS=e2,RUN=$R,SNAP=$R/snapshots/round_0100.pt,E2_ARGS="--clients CHEMBL205+CHEMBL236 --ns 16 --settings frozen_D+local_D --max_prompts 30 --max_new_tokens 96 --min_steps 20 --max_steps 20 --cal_steps 10" \
  sbatch/eval_settings_chembl.sbatch
