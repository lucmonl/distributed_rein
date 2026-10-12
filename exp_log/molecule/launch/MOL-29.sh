#!/bin/bash
# Experiment log MOL-29 (2026-10-11). First E2 (held-out-client portability, claim C2) ever run on
# ChEMBL. Spec CAL-33, std protocol CAL-31 via newsroom-iter1's NR-78 route.
#
# Arms (rotation 0 held-out clients: CHEMBL205, CHEMBL236, CHEMBL4005, CHEMBL4409):
#   frozen_D  the claim: a newcomer gets the frozen federated direction, trains only P (+ its own
#             o_i, since tau=1000 is calibration=aligned: frozen h-bar_l, u never trained)
#   local_D   the no-federation baseline, INSIDE the same job -- e2_heldout.py exits for mode=local,
#             so it cannot be a separate run on mol25-local-off. Verified key by key that
#             mol28-tp1000's config matches mol25-local-off on every key local_D reads (all lora.*,
#             lrs, batch, rounds, local_steps, warmup, fix_gain, warp_reg, seed, data, model); the
#             only differences are calibration / cov_* / mode, which local_D does not read.
#   zeroshot  sadapter, the global model as-is with NO per-client training (CAL-33 ruling 2).
#             e2_heldout.py would train a fresh private adapter and destroy the point of a
#             shared-adapter arm. Its in/out-of-support split is NOT comparable with n < all --
#             zero-shot support is the client's full pool -- so compare only overall penalized
#             error, Spearman and reach.
#
# --max_prompts 150 --max_new_tokens 96 match MOL-25's test settings (defaults are 200 / 128).
# QUALITY/JUDGE default to 0 in sbatch/eval_settings_chembl.sbatch (Newsroom-only post-steps).
#
# Smoke 11325513 COMPLETED: logged "E2 scorer: clogp_residual_deco_strict" (it used to hardcode
# density), non-null sensible percentiles, and already discriminating -- frozen_D on CHEMBL205 ran
# 0.008 -> 0.977 across alpha against local_D's flat 0.016 -> 0.024.
set -u
cd /u/lucmon/rein
R=$(ls -d runs/mol28-fed-aligned-off-tp1000_*_j11256740)
S=$(ls -d runs/mol27-fed-shared-soff-sadapter_*_j11254744)
COMMON="--settings frozen_D+local_D --max_prompts 150 --max_new_tokens 96"
go () { sbatch --parsable --job-name=$1 --time=24:00:00 --gres=gpu:A100:1 --exclude=ccc0387 \
    --export=ALL,STEPS=e2,RUN=$R,SNAP=$R/snapshots/round_0100.pt,E2_ARGS="$2" \
    sbatch/eval_settings_chembl.sbatch; }
A=$(go mol29-e2-tp1000-nsmall "$COMMON --ns 16+64+256+1024")
B=$(go mol29-e2-tp1000-nall   "$COMMON --ns all")
# zero-shot needs no training, so it runs as a short command job on the sadapter run
C=$(sbatch --parsable --job-name=mol29-e2-sadapter-zeroshot --time=4:00:00 --gres=gpu:A100:1 \
    --exclude=ccc0387 --export=ALL,CMD="python \$CODE/scripts/e2_zeroshot.py --run $S --snapshot $S/snapshots/round_0100.pt --max_prompts 150 --max_new_tokens 96" \
    sbatch/run_cmd_chembl.sbatch)
echo "cc  nsmall $A  nall $B  sadapter-zeroshot $C"
