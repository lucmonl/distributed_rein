#!/bin/bash
# Log entry MATH-4 (2026-10-04): user rule -- math-CoT runs on Delta and DeltaAI (dtai) only. The cc
# copies from MATH-2/MATH-3 were withdrawn while still pending, and Delta copies replace them in the
# race with dtai (exp_log/launch/exp40_math_qwen25_7b.sh). Same overrides. 7B -> 80 GB+ GPUs only
# (gpuA100x8, gpuH200x8; gpuA100x4 is 40 GB, about the 7B's ~40 GB peak measured on dtai).
# Sync first: scripts/sync_to_delta.sh   Run on cc-login1 (SSH master to delta-login2).
set -e
MODEL="model_name=Qwen/Qwen2.5-7B-Instruct"
ONE="$MODEL clients=[math] fed.rounds=100 fed.save_every=10 monitor.full_every=20 monitor.full_prompts=50 monitor.full_batch_size=64"
E1="$MODEL fed.rounds=100 fed.save_every=10 monitor.full_every=20 monitor.full_prompts=20 monitor.full_batch_size=64"
P="--partition=gpuA100x8,gpuH200x8"
timeout 300 ssh -o BatchMode=yes delta-login2 "cd /u/lucmon/rein && \
G0=\$(sbatch --parsable --job-name=math4_g0_sft_delta $P \
  --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,OVERRIDES=\"$ONE fed.lr_shared=0 out_dir=runs/exp40_g0_sft_qwen25_7b\" \
  sbatch/train_eval_math.sbatch) && \
G2=\$(sbatch --parsable --job-name=math4_g2_single_delta $P \
  --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,OVERRIDES=\"$ONE out_dir=runs/exp40_g2_single_qwen25_7b\" \
  sbatch/train_eval_math.sbatch) && \
G1=\$(sbatch --parsable --job-name=math4_g1_b1_delta $P --dependency=afterok:\$G2 \
  --export=ALL,RUN_JOB=\$G2 sbatch/eval_b1_math.sbatch) && \
F=\$(sbatch --parsable --job-name=math4_e1_fed_delta $P --time=2-00:00:00 \
  --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,OVERRIDES=\"$E1 out_dir=runs/exp40_e1_fed_qwen25_7b\" \
  sbatch/train_eval_math.sbatch) && \
L=\$(sbatch --parsable --job-name=math4_e1_local_delta $P --time=2-00:00:00 \
  --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,OVERRIDES=\"$E1 fed.mode=local fed.calibration=private out_dir=runs/exp40_e1_local_qwen25_7b\" \
  sbatch/train_eval_math.sbatch) && \
echo \$G0 \$G2 \$G1 \$F \$L" </dev/null
