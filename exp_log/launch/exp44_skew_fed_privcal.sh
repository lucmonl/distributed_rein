#!/bin/bash
# Experiment log MOL-15 (2026-10-05): the matched fed-vs-local comparison on skewed ChEMBL.
# MOL-14 found my own comparison confounded: the federated run used fed.calibration=shared
# (every client pinned to gain 1.053) while local necessarily uses private calibration (gains
# 1.251-1.670). So it compared {shared direction + shared calibration} against {local direction
# + private calibration} -- two factors at once, and the calibration factor alone explains the
# result (Spearman(local gain, FED-LOC out-of-support error) = +0.76).
# This run holds calibration fixed at private and varies ONLY the direction, which is what C1 is
# about. fed.calibration=private is the one legal matched setting: fed.py rejects local+shared
# (nothing is aggregated in local mode), so a full 2x2 is not available.
# Duplicated across Anvil and cc per the failover rule; cancel the loser once one is RUNNING.
cd /u/lucmon/rein
D="data_path=data/chembl_deco_skew/data.jsonl clients_file=data/chembl_deco_skew/clients.json"
EVAL="monitor.full_prompts=25 monitor.full_max_new_tokens=96 monitor.max_new_tokens=96 fed.save_every=20 monitor.full_every=20"
OV="$D fed.rounds=100 fed.calibration=private $EVAL"
# --- Anvil copy
ssh -o BatchMode=yes -l x-zchen17 anvil.rcac.purdue.edu "cd /home/x-zchen17/lucmon/rein && sbatch --parsable --job-name=exp44_fed_privcal_anvil --time=20:00:00 \
  --export=ALL,CONFIG=configs/chembl_deco.yaml,GPU_LOG=1,DATA=data/chembl_deco_skew/data.jsonl,RESCORE=none,SCORER=clogp_residual_deco,TEST_PROMPTS=150,OVERRIDES=\"$OV out_dir=runs/exp44_skew_fed_privcal_anvil\" \
  sbatch/train_eval_chembl_anvil.sbatch"
# --- cc copy
RR=/u/lucmon/lucmon/rein_runs
sbatch --parsable --job-name=exp44_fed_privcal_cc --time=24:00:00 \
  --export=ALL,CONFIG=configs/chembl_deco.yaml,GPU_LOG=1,RUNS_ROOT=$RR,DATA=data/chembl_deco_skew/data.jsonl,RESCORE=none,SCORER=clogp_residual_deco,TEST_PROMPTS=150,\
OVERRIDES="$OV out_dir=$RR/exp44_skew_fed_privcal_cc" \
  sbatch/train_eval_chembl.sbatch
