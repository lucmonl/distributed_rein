#!/bin/bash
# One-time setup of Purdue Anvil as a third execution host for the ChEMBL (molecule) workstream.
# Anvil was offered because the cc queue is the bottleneck: 17 of ~20 `ai` nodes sit partly free
# while cc jobs wait hours on Priority.
#
#   Allocation : -A cis260796-ai, 500 SU, TRESBillingWeights=GRES/gpu=1.0 -> 1 SU per GPU-hour,
#                so 500 GPU-hours. A 100-round fed/local pair costs ~12 SU (~2.4%).
#   Account    : x-zchen17 (NOT the user's own) -- everything stays under lucmon/ subdirectories.
#   Partitions : ai | gpu (4 GPUs/node, --gpus-per-node=1), max 48 h; gpu-debug for 30 min tests.
#   Quotas     : home 25 GB (0.0% used), scratch 100 TB (empty), projects 5 TB. The relayed
#                warning that /home was 85% full does not match `myquota`; still, the env, HF
#                cache and run outputs all go to scratch, not home.
#
# Usage: scripts/setup_anvil.sh [--code-only]
set -euo pipefail
H=anvil; USER_A=x-zchen17
# Layout mirrors cc-login1: code in HOME, all bulk on scratch and reached through symlinks.
#   cc:    /u/lucmon/rein (home)            data/<n> -> /projects/.../lucmon/data/<n>, runs, HF cache, conda env in project space
#   anvil: /home/x-zchen17/lucmon/rein      data/<n> -> $SCRATCH/lucmon/data/<n>, runs -> $SCRATCH/lucmon/runs, env + HF cache on scratch
# Code in home also keeps it clear of scratch's retention policy; bulk there is regenerable.
SCR=/anvil/scratch/$USER_A/lucmon
ROOT=/home/$USER_A/lucmon
SSH="ssh -o BatchMode=yes -l $USER_A anvil.rcac.purdue.edu"

$SSH "mkdir -p $ROOT/rein/sbatch/logs $SCR/hf_home $SCR/runs $SCR/envs $SCR/data"

# --- code (data/ and runs/ are symlinks locally; synced separately)
rsync -az --delete --itemize-changes -e "ssh -o BatchMode=yes -l $USER_A" \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='/runs' --exclude='/data' --exclude='/sbatch/logs' \
  /u/lucmon/rein/ anvil.rcac.purdue.edu:$ROOT/rein/ | tail -5

# --- datasets the molecule workstream needs (small: jsonl + parquet)
for d in chembl_deco chembl_deco_skew chembl_deco_skew_pruned; do
  rsync -az --itemize-changes -e "ssh -o BatchMode=yes -l $USER_A" \
    /u/lucmon/rein/data/$d/ anvil.rcac.purdue.edu:$SCR/data/$d/ | tail -3
done
$SSH "cd $ROOT/rein && mkdir -p data && for d in chembl_deco chembl_deco_skew chembl_deco_skew_pruned; do ln -sfn $SCR/data/\$d data/\$d; done; ln -sfn $SCR/runs runs; ln -sfn $SCR/hf_home hf_home"
[ "${1:-}" = "--code-only" ] && { echo "code+data synced; skipping env/model"; exit 0; }

# --- conda env + model (slow: ~2.5 GB of wheels, 7.6 GB of weights)
$SSH "bash -lc '
set -e
module load conda
export CONDA_PKGS_DIRS=$SCR/.conda_pkgs
conda create -y -p $SCR/envs/rein python=3.11 >/dev/null
source activate $SCR/envs/rein
# Versions are PINNED to match cc's `steer` env (torch 2.5.1, transformers 4.56.0, numpy 1.26.3).
# An unpinned install gave torch 2.14.1 / transformers 5.18.0 -- a major-version jump that risks
# breaking the hand-rolled LoRA wrapping and, worse, makes Anvil numbers non-comparable with the
# cc runs they will sit beside in the same table.
pip install -q "torch==2.5.1" "transformers==4.56.0" "numpy==1.26.3" accelerate safetensors \
    scipy pyyaml "pyarrow==19.0.0" pandas rdkit
python -c \"import torch,transformers,rdkit;print(torch.__version__,transformers.__version__)\"
export HF_HOME=$SCR/hf_home
python -c \"
from huggingface_hub import snapshot_download
p=snapshot_download(\\\"Qwen/Qwen3-4B-Instruct-2507\\\", allow_patterns=[\\\"*.json\\\",\\\"*.safetensors\\\",\\\"*.txt\\\",\\\"*.jinja\\\"])
print(\\\"model at\\\", p)\"
'"
echo "ANVIL SETUP DONE: code=$ROOT/rein bulk=$SCR"
