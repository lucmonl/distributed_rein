#!/bin/bash
# Experiment log MOL-26a (2026-10-09). Reruns MOL-26's B1 x3 and B4 after the CAL-26 fixes.
# MOL-26's B1 numbers are VOID: the template predated the decoration format and asked the client
# models a question they were never trained on. B3 stands as reported in MOL-26.
#
#   mol26a-b1-k0        B1 k=0, client model, direction removed
#   mol26a-b1-k3        B1 k=3, client model
#   mol26a-b1-k3-base   B1 k=3, base model
#   mol26a-b4-caa       B4 federated CAA, widened grid, gain chosen on the PENALIZED metric
#
# Fixes being exercised (grant from CAL-26; 83/83 tests pass):
#   fedsteer/baselines.py  B1's molecule prompt is built from the record's own training prompt, so it
#                          shows core_attached with [n*] points and closes with the record's own
#                          "Answer with the decorations only" contract -- LAST, after the level
#                          instruction, which is where training puts it.
#   eval_baselines.py      --caa_select {pct_calib_err,pct_calib_err_penalized}; default unchanged so
#                          NR-71 stays valid, MOL-26a uses penalized (never None, so a gain that
#                          destroys validity ranks worst instead of crashing selection).
#   scripts/compare_runs.py  a null pct_calib_err prints n/a instead of killing the table.
#
# Smoke 11248372 COMPLETED: B1 now emits [n*] notation (40/40, 40/40, 37/40) and unscorable fell
# from 1.000 to 0.00 / 0.21 / 0.67; B4 printed its per-gain dev curve and chose gains 0.1-0.3.
set -u
cd /u/lucmon/rein
F=$(ls -d runs/mol25-fed-aligned-off_*_j11237283)
L=$(ls -d runs/mol25-local-off_*_j11237285)
COMMON="SCORER=clogp_residual_deco_strict,DATA=data/chembl_deco_skew_pruned/data.jsonl,PROMPTS=150"
FEDV="FED_RUN=$F,FED_SNAP=$F/snapshots/round_0100.pt"
LOCV="LOCAL_RUN=$L,LOCAL_SNAP=$L/snapshots/round_0100.pt"
GAINS="CAA_GAINS=0.05+0.1+0.2+0.3+0.5+0.8+1.2+1.6+2.4+3.2,CAA_SELECT=pct_calib_err_penalized"

go () { sbatch --parsable --job-name=$1 --time=$3 --gres=gpu:A100:1 --exclude=ccc0387 \
    --export=ALL,$COMMON,$FEDV,$LOCV,$2 sbatch/eval_baselines_chembl.sbatch; }
A=$(go mol26a-b1-k0      "STEPS=b1,B1_SETS=k0c" 6:00:00)
B=$(go mol26a-b1-k3      "STEPS=b1,B1_SETS=k3c" 6:00:00)
C=$(go mol26a-b1-k3-base "STEPS=b1,B1_SETS=k3b" 6:00:00)
D=$(go mol26a-b4-caa     "STEPS=b4,$GAINS"      10:00:00)
echo "cc  b1-k0 $A  b1-k3 $B  b1-k3-base $C  b4-caa $D"
