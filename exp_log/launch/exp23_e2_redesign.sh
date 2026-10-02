#!/bin/bash
# Experiment log entry 23 (2026-10-01): E2 redesigned. The new client trains its private adapter
# with the participants' objective (alpha per pair, steering on) and D frozen; baseline local_D
# trains its own D on the same n pairs; plugin (old design: adapter SFT with steering off) and
# prompt are kept as ablation / baseline.  Data sizes n = 16, 64, 256, 1024, all.
#   Method run (fed, shared calibration, no offset; 11063910): one job per n, all settings
#     (local_D and prompt do not depend on the federated run, so they run only here).
#   Private-calibration fed run (11041282): frozen_D + plugin, all n in one job.
# Smoke first (11087947, alpha window 0..0.4 on the smoke run); real jobs afterok.
cd /u/lucmon/rein
FS=runs/exp17_fed_calshared_nooff_cap4k_20261001-001022_j11063910
FP=runs/exp11_cap4k_base_20260930-114746_j11041282
S=11087947
sub () { sbatch --parsable --dependency=afterok:$S "$@" sbatch/eval_settings.sbatch; }
for N in 16 64 256 1024; do
  J=$(sub --job-name=exp23_e2_fs_n$N --time=08:00:00 --export=ALL,STEPS=e2,RUN=$FS,SNAP=$FS/snapshots/round_0100.pt,E2_ARGS="--ns $N")
  echo "fed shared n=$N: $J"
done
J=$(sub --job-name=exp23_e2_fs_nall --time=08:00:00 --export=ALL,STEPS=e2,RUN=$FS,SNAP=$FS/snapshots/round_0100.pt,E2_ARGS="--ns all --settings frozen_D+local_D+plugin")
echo "fed shared n=all: $J"
J=$(sub --job-name=exp23_e2_fp --time=12:00:00 --export=ALL,STEPS=e2,RUN=$FP,SNAP=$FP/snapshots/round_0100.pt,E2_ARGS="--ns 16+64+256+1024+all --settings frozen_D+plugin")
echo "fed private all n: $J"
