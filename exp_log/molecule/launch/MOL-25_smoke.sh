#!/bin/bash
# MOL-25 pre-flight smoke (2026-10-09). NOT an experiment: 2 clients, 6 rounds, 64 train examples.
#
# Covers the two override sets ChEMBL has never run. `fed-shared-off` is skipped: it was already
# smoked end to end as MOL-24's arm 2 (11206294), and its override set is unchanged.
#
#   mol25-smoke-fed-aligned-off   aligned pooled table + private o_i -- new on ChEMBL
#   mol25-smoke-local-off         local, private per-layer shape + private o_i -- still owed
#
# 6 rounds rather than 3, so the aligned table exists at round 5 for the checks CAL-19 asks for:
# 252 rows x 21 points, o_i differing across clients and absent from the server state, and
# tie_loss / proj_adjust logged. Clients CHEMBL243 (no high side) and CHEMBL228 (no low side) come
# from the smoke config, picked for OPPOSITE support gaps so the out-of-support path runs both ways;
# they are in the config and not in --export because a comma inside an --export value is silently
# truncated (MOL-22b cause 2).
set -u
cd /u/lucmon/rein
CFG=configs/chembl_deco_skew_pruned_smoke2.yaml
DATA=data/chembl_deco_skew_pruned/data.jsonl
ENVC="CONFIG=$CFG,DATA=$DATA,SCORER=clogp_residual_deco_strict,RESCORE=clogp_residual_deco,SELECT=pct_calib_err_penalized,TEST_PROMPTS=4"
COMMON="lora.warp=kumaraswamy_mix lora.warp_scope=module fed.fix_gain=true fed.warp_reg=0 fed.rounds=6 fed.save_every=1"

s () { sbatch --parsable --job-name=$1 --time=1:00:00 \
    --partition=secondary,dali,IllinoisComputes-GPU,scavenger \
    --export=ALL,$ENVC,OVERRIDES="$COMMON out_dir=runs/$1 $2" \
    sbatch/train_eval_chembl.sbatch; }
A=$(s mol25-smoke-fed-aligned-off "fed.calibration=aligned fed.cov_grid=21 fed.cov_pool=saturating fed.cov_lambda_max=0.01 lora.offset=true")
B=$(s mol25-smoke-local-off       "fed.mode=local fed.calibration=private lora.offset=true")
echo "smoke  fed-aligned-off $A  local-off $B"
