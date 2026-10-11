#!/bin/bash
# Experiment log MOL-28 (2026-10-09). Spec CAL-29 (user request): tau_pool sweep for fed-aligned-off
# on ChEMBL, mirroring Newsroom's NR-66/NR-69. tau = 100 is MOL-25's aligned arm (11237283), not rerun.
#
#   mol28-fed-aligned-off-tp300   saturating, tau_pool = 300
#   mol28-fed-aligned-off-tp1000  saturating, tau_pool = 1000
#   mol28-fed-aligned-off-tp3000  saturating, tau_pool = 3000
#   mol28-fed-aligned-off-cnt     count pooling (no saturation)
#
# Identical to 11237283 in every other respect: pruned data, strict scorer at every stage, dev
# selection on pct_calib_err_penalized, 100 rounds, rotation 0, seed 0, grid 21, lambda_max 0.01,
# per-layer shapes, gain fixed at 1, warp_reg 0, private offset via calibration=aligned.
#
# GPU PINNING -- and why it is not cosmetic. MOL-25's three arms each landed on a DIFFERENT GPU type:
#   11237283 fed-aligned-off  ccc0284  A100 80GB PCIe
#   11237284 fed-shared-off   ccc0389  A100-SXM4-80GB
#   11237285 local-off        ccc0465  H200
# aligned and shared differ by only 0.002 penalized, which is inside the <= 0.005 band the
# coordinator reports GPU mixing made unreadable on Newsroom. This sweep's whole point is differences
# of that size, so all four arms are constrained to ccc0284's feature (AE7713_100g_512G_A100) -- the
# ONLY node carrying it, and the same node tau = 100 ran on, so every paired contrast vs tau = 100 is
# hardware-matched. ccc0284 has 3 A100s, so the fourth arm serialises behind the others; that is the
# deliberate trade (~12 h wall clock, matched) over running concurrently on mixed hardware.
# ccc0284 is never excluded (memory reference-slurm). Anvil stays gated off (CAL-11/CAL-16).
set -u
cd /u/lucmon/rein
CFG=configs/chembl_deco_skew_pruned.yaml
DATA=data/chembl_deco_skew_pruned/data.jsonl
ENVC="CONFIG=$CFG,GPU_LOG=1,DATA=$DATA,SCORER=clogp_residual_deco_strict,RESCORE=clogp_residual_deco,SELECT=pct_calib_err_penalized,TEST_PROMPTS=150"
COMMON="lora.warp=kumaraswamy_mix lora.warp_scope=module fed.fix_gain=true fed.warp_reg=0 fed.calibration=aligned fed.cov_grid=21 fed.cov_lambda_max=0.01 lora.offset=true"

# ccc0284 belongs to the `dali` partition ONLY, while the sbatch requests
# dali,IllinoisComputes-GPU,scavenger -- so a feature constraint is unsatisfiable in two of the
# three and SLURM rejects the whole job with BadConstraints (seen on the first attempt,
# 11256733-36, cancelled). Restricting to -p dali and naming the node directly is unambiguous.
go () { sbatch --parsable --job-name=$1 --time=24:00:00 --gres=gpu:A100:1 \
    --partition=dali --nodelist=ccc0284 \
    --export=ALL,$ENVC,OVERRIDES="$COMMON out_dir=runs/$1 $2" \
    sbatch/train_eval_chembl.sbatch; }
A=$(go mol28-fed-aligned-off-tp300  "fed.cov_pool=saturating fed.cov_tau_pool=300")
B=$(go mol28-fed-aligned-off-tp1000 "fed.cov_pool=saturating fed.cov_tau_pool=1000")
C=$(go mol28-fed-aligned-off-tp3000 "fed.cov_pool=saturating fed.cov_tau_pool=3000")
D=$(go mol28-fed-aligned-off-cnt    "fed.cov_pool=count")
echo "cc  tp300 $A  tp1000 $B  tp3000 $C  cnt $D"
