#!/bin/bash
# Experiment log MOL-26 (2026-10-09). Spec from the coordinator (CAL-25), on the user's request for
# baselines on ChEMBL. Mirrors Newsroom's NR-71/NR-73 (CAL-23).
#
# Method under test: mol25-fed-aligned-off (11237283, round 100), g_{i,l}(a) = o_i + h-bar_l(a).
# Baselines, all on test with the strict deco scorer and 150 prompts per client (MOL-25's size):
#   mol26-b1-k0        B1 prompting, k=0, client model, direction removed
#   mol26-b1-k3        B1 prompting, k=3, client model, direction removed
#   mol26-b1-k3-base   B1 prompting, k=3, BASE model (adapter removed too)
#   mol26-b4-caa       B4 federated CAA on the same models, widened gain grid 0.05 .. 3.2
#   mol26-b3-local-off B3: merge mol25-local-off's per-layer directions, evaluate with each
#                      client's own adapter, per-layer shapes and offset
#
# THE FOUR TRAPS (three named in CAL-25, the fourth found here), all handled by passing values
# rather than relying on this file's defaults:
#   SCORER   defaults to clogp_residual        -> passed as clogp_residual_deco_strict
#   DATA     defaults to data/chembl_fed       -> passed as data/chembl_deco_skew_pruned
#   PROMPTS  the file hard-coded 200           -> new opt-in var, passed as 150
#   LOCAL_SNAP=best selects on pct_calib_err   -> round_0100.pt passed explicitly
#   CAA_GAINS: eval_baselines.py defaults to the NARROW grid (0.05..0.8) with no 1.2/1.6/2.4/3.2,
#              so the widened grid must be passed or the "none at the grid top" check is vacuous
# CAA_GAINS and B1_SETS use "+" not "," because --export is itself comma-separated and truncates a
# value at its first comma (MOL-22b lost a run to exactly that).
#
# Pre-flight: smoke 11242038 COMPLETED (all three paths, strict scorer and pruned data confirmed
# reaching every step). B1's molecule template verified to carry the scaffold, and the records carry
# `scaffold` (not `problem`/`article`), so prompt_with_level's auto-dispatch picks the molecule
# template. B1's score grid is alpha-invariant where cells are scorable, as it must be with
# lora_B_d zeroed.
#
# cc only, A100, ccc0387 excluded. Anvil stays gated off (CAL-11/CAL-16: broken env, stale code).
set -u
cd /u/lucmon/rein
F=$(ls -d runs/mol25-fed-aligned-off_*_j11237283)
L=$(ls -d runs/mol25-local-off_*_j11237285)
COMMON="SCORER=clogp_residual_deco_strict,DATA=data/chembl_deco_skew_pruned/data.jsonl,PROMPTS=150"
FEDV="FED_RUN=$F,FED_SNAP=$F/snapshots/round_0100.pt"
LOCV="LOCAL_RUN=$L,LOCAL_SNAP=$L/snapshots/round_0100.pt"
GAINS="CAA_GAINS=0.05+0.1+0.2+0.3+0.5+0.8+1.2+1.6+2.4+3.2"

go () {  # $1 = job name, $2 = extra exports, $3 = time
  sbatch --parsable --job-name=$1 --time=$3 --gres=gpu:A100:1 --exclude=ccc0387 \
    --export=ALL,$COMMON,$FEDV,$LOCV,$2 sbatch/eval_baselines_chembl.sbatch
}
B1A=$(go mol26-b1-k0        "STEPS=b1,B1_SETS=k0c" 6:00:00)
B1B=$(go mol26-b1-k3        "STEPS=b1,B1_SETS=k3c" 6:00:00)
B1C=$(go mol26-b1-k3-base   "STEPS=b1,B1_SETS=k3b" 6:00:00)
B4=$(go  mol26-b4-caa       "STEPS=b4,$GAINS"      10:00:00)
B3=$(go  mol26-b3-local-off "STEPS=b3"             6:00:00)
echo "cc  b1-k0 $B1A  b1-k3 $B1B  b1-k3-base $B1C  b4-caa $B4  b3-local-off $B3"
