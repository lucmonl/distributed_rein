#!/bin/bash
# NR-57 (2026-10-07): test evaluation of the NR-56 consensus checkpoints with each client's OWN warps
# (eval_direction.py --local_warp) instead of the server table g-bar: measures the train/inference
# mismatch of consensus. Same settings as the pipeline's test eval (200 prompts, 128 tokens, density),
# plus quality scoring. cc only (short eval job; no Anvil eval script exists).
# Submitted: cons0.1 11202869, cons1 11202870 (--dependency=afterok:11191901)
CMD='set +e; source ~/.bashrc; source activate steer; set -e; cd /u/lucmon/rein; export HF_HUB_OFFLINE=1; S=$(date +%Y%m%d-%H%M%S)_j${SLURM_JOB_ID}; C=runs/_code/$S; mkdir -p $C; cp -r fedsteer eval_direction.py scripts $C/; nvidia-smi -L; python $C/eval_direction.py --run $RUN --scorer density --split test --max_prompts 200 --max_new_tokens 128 --snapshot $RUN/snapshots/round_0100.pt --local_warp; E=$(ls -t $RUN/evals/eval_round_0100_localwarp__*.json | head -1); python $C/scripts/score_quality.py --run $RUN --eval $E; echo DONE $E'
EX=ccc0387,ccc0089,ccc0090,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236
# sbatch --parsable --job-name=nr57_localwarp_<arm> [--dependency=afterok:<train job>] --time=6:00:00 --account=lucmon-ic \
#   --partition=dali,IllinoisComputes-GPU,scavenger --exclude=$EX --gres=gpu:1 --ntasks-per-node=8 \
#   --output=/u/lucmon/rein/sbatch/logs/nr57_localwarp.o%j --export=ALL,RUN=<run dir> --wrap="$CMD"
