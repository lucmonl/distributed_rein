#!/bin/bash
# MOL-24 pre-flight smoke (2026-10-07). NOT an experiment: 2 clients, 3 rounds, 64 train examples.
#
# Purpose: run all three MOL-24 override sets end to end through the REAL production script
# (sbatch/train_eval_chembl.sbatch), so every stage is exercised before 25 SU of Anvil and ~18 h of
# cc GPU go out the door: train -> summarize_sweep (dev select) -> eval_direction (test) ->
# score_molecules -> rescore_eval. MOL-22b's four failed submissions were all caught by neither
# `--help` nor an import test; only an end-to-end run finds them.
#
# What specifically could break here and nowhere else:
#   - fed.private_offset=true validates against lora.offset (fed.py raises if offset is absent);
#   - lora.warp_scope=module with fed.calibration=shared: per-layer shapes aggregated by FedAvg,
#     a combination ChEMBL has never run;
#   - local mode with a private offset and a per-layer private shape (never run on ChEMBL);
#   - clients come from the config, not --export, because a comma inside an --export value is
#     silently truncated (MOL-22b cause 2).
#
# Clients CHEMBL243 (no high side) and CHEMBL228 (no low side) are chosen for OPPOSITE support
# gaps, so the out-of-support path runs in both directions.
set -u
cd /u/lucmon/rein
CFG=configs/chembl_deco_skew_pruned_smoke2.yaml
DATA=data/chembl_deco_skew_pruned/data.jsonl
ENVC="CONFIG=$CFG,DATA=$DATA,SCORER=clogp_residual_deco_strict,RESCORE=clogp_residual_deco,SELECT=pct_calib_err_penalized,TEST_PROMPTS=4"
# CONVENTIONS.md 3: every reference-matched arm passes these literally ("default" there means the
# reference design, NOT the code/config defaults -- the code defaults to warp_scope=model and this
# config ships fed.warp_reg=0.01).
COMMON="lora.warp=kumaraswamy_mix lora.warp_scope=module fed.fix_gain=true fed.warp_reg=0 lora.offset=false fed.calibration=shared"

s () {  # $1 = job/run name, $2 = arm overrides (appended, so they win)
  sbatch --parsable --job-name=$1 --time=1:00:00 \
    --partition=secondary,dali,IllinoisComputes-GPU,scavenger \
    --export=ALL,$ENVC,OVERRIDES="$COMMON out_dir=runs/$1 $2" \
    sbatch/train_eval_chembl.sbatch
}
A=$(s mol24-smoke-fed-shared      "")
B=$(s mol24-smoke-fed-shared-off  "lora.offset=true fed.private_offset=true")
C=$(s mol24-smoke-local-off "fed.mode=local fed.calibration=private lora.offset=true")
echo "smoke  fed-shared $A  fed-shared-off $B  local-off $C"
