#!/bin/bash
# MOL-26a pre-flight smoke (2026-10-09). NOT an experiment. Verifies the CAL-26 fixes end to end
# before the four real jobs: the rebuilt B1 decoration prompt and B4's --caa_select.
# The check that matters: B1's generations must now be in [n*] attachment-point notation, not plain
# SMILES. MOL-26's B1 was 1.000 unscorable because the template asked for "one SMILES string" from a
# bare scaffold, a question the client models were never trained on.
set -u
cd /u/lucmon/rein
F=$(ls -d runs/mol25-fed-aligned-off_*_j11237283)
L=$(ls -d runs/mol25-local-off_*_j11237285)
sbatch --parsable --job-name=mol26a-smoke --time=1:00:00 --gres=gpu:A100:1 --exclude=ccc0387 \
  --export=ALL,SMOKE=1,STEPS=b1+b4,SCORER=clogp_residual_deco_strict,DATA=data/chembl_deco_skew_pruned/data.jsonl,FED_RUN=$F,FED_SNAP=$F/snapshots/round_0100.pt,LOCAL_RUN=$L,LOCAL_SNAP=$L/snapshots/round_0100.pt,CAA_SELECT=pct_calib_err_penalized \
  sbatch/eval_baselines_chembl.sbatch
