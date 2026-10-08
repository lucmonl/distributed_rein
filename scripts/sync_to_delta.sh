#!/bin/bash
# One-way sync of the code + datasets from cc-login to Delta (delta-login2), then rewrite the
# Delta copy's sbatch headers.  cc-login stays the only place where code is edited: the Delta
# copy is overwritten (and its stale files deleted) on every run.
#
# Needs the SSH master connection:  ssh -fN delta-login2   (Duo, once)
#
# Layout on Delta:
#   /u/lucmon/rein                        code (this repo, no .git)
#   /u/lucmon/rein/runs        -> /work/nvme/bhby/lucmon/rein        (bulk outputs)
#   /u/lucmon/rein/data/<name> -> /work/nvme/bhby/datasets/<name>
#   HF cache: /work/nvme/bhby/lucmon/hf_home (HF_HOME)
#
# Usage: scripts/sync_to_delta.sh [--dry-run] [--data] [--no-code] [--host delta|dtai]   (--data also syncs the datasets)
#
# --no-code: sync everything EXCEPT code (*.py, configs/, fedsteer/, scripts/, tests/), i.e. docs,
# exp_log/ and the sbatch files (headers rewritten as usual).  For a shared-code hold (CONVENTIONS.md
# section 5): pending jobs snapshot the code tree when they start, so code must not move then.
#
# --host dtai: DeltaAI (dtai-1 = gh-login01, aarch64 GH200, partition ghx4).  /work is SHARED with
# Delta (runs/, datasets, HF cache are the same directories) but home is NOT: the code goes to
# dtai's own /u/lucmon/rein.  Its conda is /work/nvme/bhby/lucmon/anaconda3_deltaai with env
# rein-gh (clone of dtai's steer + scipy, scikit-learn, rdkit, math-verify).  Needs ssh -fN dtai-1.
set -euo pipefail
HOST=delta-login2; ACCOUNT=bhby-delta-gpu; PARTITION=gpuA100x4,gpuA100x8,gpuH200x8; ENV=rein
SRC=/u/lucmon/rein
DST=/u/lucmon/rein
RUNS_TARGET=/work/nvme/bhby/lucmon/rein
DATA_ROOT=/work/nvme/bhby/datasets
DATASETS="newsroom chembl newsroom_fed chembl_fed newsroom_stats math_fed math_fed_ans"

DRY=""; DATA=0; NOCODE=0; NEXT=""
for a in "$@"; do
  if [ "$NEXT" = host ]; then
    case $a in delta) ;; dtai) HOST=dtai-1; ACCOUNT=bhby-dtai-gh; PARTITION=ghx4; ENV=rein-gh;;
      *) echo "unknown host $a"; exit 1;; esac
    NEXT=""; continue
  fi
  case $a in --dry-run) DRY="--dry-run";; --data) DATA=1;; --no-code) NOCODE=1;; --host) NEXT=host;; esac
done

ssh -O check $HOST 2>/dev/null || { echo "no SSH master: run 'ssh -fN $HOST' first"; exit 1; }

ssh $HOST "mkdir -p $DST $DATA_ROOT $RUNS_TARGET"

# --- code (data/ and runs/ are symlinks on both sides: never synced here)
# Excluded paths are also protected from --delete, so --no-code leaves the remote code untouched.
CODE_EXCL=()
[ $NOCODE = 1 ] && CODE_EXCL=(--exclude='*.py' --exclude='/configs' --exclude='/fedsteer'
                              --exclude='/scripts' --exclude='/tests')
rsync -az $DRY --delete --itemize-changes \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='/runs' --exclude='/data' --exclude='/sbatch/logs' "${CODE_EXCL[@]}" \
  $SRC/ $HOST:$DST/

# loose files directly under data/ (e.g. toy_length.jsonl) are small: they live in the code tree
ssh $HOST "mkdir -p $DST/data"
[ $NOCODE = 1 ] || rsync -a $DRY $SRC/data/*.jsonl $HOST:$DST/data/

# --- datasets: real directories go to /work, symlinked from data/
if [ $DATA = 1 ]; then
  for d in $DATASETS; do
    rsync -azL $DRY --info=progress2 --delete $SRC/data/$d/ $HOST:$DATA_ROOT/$d/
  done
fi

[ -n "$DRY" ] && { echo "dry run: skipping remote setup"; exit 0; }

# --- remote setup + sbatch rewrite (remote copy only)
ssh $HOST bash -s <<EOF
set -e
cd $DST
mkdir -p sbatch/logs sbatch/logs/math-cot   # SLURM does not create --output dirs (CONVENTIONS.md section 2)
# runs/ -> /work
if [ -e runs ] && [ ! -L runs ]; then echo "runs exists and is not a symlink; leaving it"; else ln -sfn $RUNS_TARGET runs; fi
# data/: link each dataset dir into the repo
mkdir -p data
for d in $DATASETS; do ln -sfn $DATA_ROOT/\$d data/\$d; done
# sbatch headers: account/partition/mem of the host, env, HF cache
sed -i \
  -e 's/^#SBATCH --account=lucmon-ic/#SBATCH --account=$ACCOUNT/' \
  -e 's/^#SBATCH --partition=.*/#SBATCH --partition=$PARTITION/' \
  -e '/^#SBATCH --exclude=/d' \
  -e 's/^#SBATCH --gres=gpu:.*/#SBATCH --gpus-per-node=1/' \
  -e 's/^#SBATCH --ntasks-per-node=8/#SBATCH --ntasks-per-node=16\n#SBATCH --mem=64G/' \
  -e 's/^source ~\/.bashrc$/set +e  # Delta: \/etc\/bashrc returns non-zero on compute nodes, fatal under set -e\nsource ~\/.bashrc/' \
  -e 's/^source activate steer$/source activate $ENV\nset -e/' \
  -e 's#^export HF_HUB_OFFLINE=1#export HF_HOME=/work/nvme/bhby/lucmon/hf_home\nexport HF_HUB_OFFLINE=1#' \
  sbatch/*.sbatch
# pilot scripts have no --account line (cc default account): make it explicit
for f in \$(grep -L '^#SBATCH --account' sbatch/*.sbatch); do
  sed -i '/^#SBATCH --partition/i #SBATCH --account=$ACCOUNT' \$f
done
echo "$HOST setup done"; ls -l runs; ls -l data
EOF
