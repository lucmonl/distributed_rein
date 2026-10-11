#!/bin/bash
# Experiment log MOL-25 (2026-10-09). Spec from the coordinator session (CAL-19), on the user's
# decision to make fed-aligned-off the main algorithm across datasets. Supersedes MOL-24, never run.
#
# E1 on pruned ChEMBL with the aligned calibration as the method under test:
#
#   mol25-fed-aligned-off  g_{i,l}(a) = o_i + h-bar_l(a)   pooled table in training AND inference
#   mol25-fed-shared-off   g_{i,l}(a) = o_i + h_l(a)       FedAvg of the shape's parameters
#   mol25-local-off        g_{i,l}(a) = o_i + h_{i,l}(a)   each client alone
#
# All three carry a private offset o_i, so the contrasts isolate HOW the shape is obtained:
#   aligned-off vs local-off    C1, matched: is a shared calibration better than none?
#   aligned-off vs shared-off   does the pooled table beat FedAvg-of-parameters, o_i held equal?
#
# Matched to MOL-18/MOL-22: same config file, data/chembl_deco_skew_pruned, 100 rounds, rotation 0,
# seed 0, strict scorer at every stage, dev selection on pct_calib_err_penalized, 150 test prompts.
# MOL-22's PRIV_OFF/LOCAL_OFF used ONE shape per client, so they are context, not references.
#
# tau_pool / b / tau_local stay at the code defaults (100 / 0.2 / 100), as in Newsroom's NR-62; NR-69
# found tau did not matter once o_i was present.
#
# fed.private_offset is NOT passed to the aligned arm: under calibration=aligned, lora.offset=true
# already gives a private o_i (fedsteer/fed.py:246), and private_offset is gated on
# calibration=="shared" (fed.py:177), so adding it would be a silent no-op.
#
# CONVENTIONS.md 3: warp_scope and warp_reg are passed LITERALLY -- the code defaults to
# warp_scope=model (one shape per client) and this config ships fed.warp_reg=0.01.
#
# CC ONLY (CAL-11/CAL-16): the Anvil conda env lost stdlib files to a scratch purge and its code is
# stale on three files, so no Anvil copies and no race group -- one host, one scorer version, nothing
# to race. 0 SU. Anvil returns only after newsroom-iter1 reruns setup_anvil.sh --code-only with an
# md5 check. ccc0284 is never excluded (memory reference-slurm); the Turing nodes and 16 GB V100s
# stay excluded in the sbatch itself (NR-50/MOL-18).
set -u
cd /u/lucmon/rein
CFG=configs/chembl_deco_skew_pruned.yaml
DATA=data/chembl_deco_skew_pruned/data.jsonl
ENVC="CONFIG=$CFG,GPU_LOG=1,DATA=$DATA,SCORER=clogp_residual_deco_strict,RESCORE=clogp_residual_deco,SELECT=pct_calib_err_penalized,TEST_PROMPTS=150"
COMMON="lora.warp=kumaraswamy_mix lora.warp_scope=module fed.fix_gain=true fed.warp_reg=0"

cc () { sbatch --parsable --job-name=$1 --time=24:00:00 \
    --export=ALL,$ENVC,OVERRIDES="$COMMON out_dir=runs/$1 $2" \
    sbatch/train_eval_chembl.sbatch; }
A=$(cc mol25-fed-aligned-off "fed.calibration=aligned fed.cov_grid=21 fed.cov_pool=saturating fed.cov_lambda_max=0.01 lora.offset=true")
S=$(cc mol25-fed-shared-off  "fed.calibration=shared lora.offset=true fed.private_offset=true")
L=$(cc mol25-local-off       "fed.mode=local fed.calibration=private lora.offset=true")
echo "cc  fed-aligned-off $A  fed-shared-off $S  local-off $L"
squeue -u lucmon -o "%.10i %.26j %.9T %.6M %R" | grep -E "mol25|JOBID"
