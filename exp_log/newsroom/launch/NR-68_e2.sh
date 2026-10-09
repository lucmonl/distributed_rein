#!/bin/bash
# Experiment log NR-68 (2026-10-08): E2, held-out clients joining (coordinator instruction, user request).
# Rotation 0 held-out clients (telegraph.co.uk, bbc.com, mashable.com, latimes.com); n in {16, 64, 256, 1024,
# all}, nested subsets, steps clip(ceil(4 n / 8), 200, 2000), 200 test articles per client (archive 23/30).
# Settings per arm (e2_heldout.py e2_plan; u never trained, s = 1):
#   fed-aligned-off   frozen_D (P + own o_i, run's table h-bar_l) + local_D (= the local-off reference:
#                     P + D_i from zero + own per-layer warps + o_i) + plugin (SFT adapter, then fit o_i only)
#   fed-shared-off    frozen_D (P + own o_i, server's per-layer h_l) + plugin (fit o_i only)
#   fed-aligned-tp3000 frozen_D (P, run's table)
#   fed-shared-soff   frozen_D (P; the server's shared o frozen at -0.026)
# prompt: reused from the archive rotation-0 results (11088050-53; same model, tokens, shots, seed, reference).
# cc only (Anvil broken). One GPU type per job: --gres=gpu:A100:1 (dali A100-40GB or ccc0388-0390
# A100-80GB); ccc0387 excluded as always, ccc0284 never. Quality + judge (frozen_D/local_D) run in the job.
set -u
cd /u/lucmon/rein
AO=runs/nr62-fed-aligned-off_20261007-231639_j11209264
SO=runs/nr60-fed-shared-off_20261007-165434_j11206103
TP=runs/nr64-fed-aligned-tp3000_20261008-095918_j11210170
SS=runs/nr60-fed-shared-soff_20261008-055340_j11209750
sub () {  # name run e2-args
  J=$(sbatch --parsable --job-name=$1 --time=12:00:00 --partition=dali,IllinoisComputes-GPU \
    --gres=gpu:A100:1 --exclude=ccc0387 \
    --export=ALL,STEPS=e2,RUN=$2,SNAP=$2/snapshots/round_0100.pt,E2_ARGS="$3" sbatch/eval_settings.sbatch)
  echo "$1 $J"
}
for N in 16 64 256 1024 all; do
  sub nr68-e2-fed-aligned-off-n$N $AO "--ns $N --settings frozen_D+local_D+plugin"
done
sub nr68-e2-fed-shared-off-nsmall $SO "--ns 16+64+256+1024 --settings frozen_D+plugin"
sub nr68-e2-fed-shared-off-nall $SO "--ns all --settings frozen_D+plugin"
sub nr68-e2-fed-aligned-tp3000-nsmall $TP "--ns 16+64+256+1024 --settings frozen_D"
sub nr68-e2-fed-aligned-tp3000-nall $TP "--ns all --settings frozen_D"
sub nr68-e2-fed-shared-soff-nsmall $SS "--ns 16+64+256+1024 --settings frozen_D"
sub nr68-e2-fed-shared-soff-nall $SS "--ns all --settings frozen_D"
