#!/bin/bash
# Experiment log entry 40 (2026-10-04): math CoT with the screened backbone Qwen2.5-7B-Instruct, on
# DeltaAI (dtai-1, GH200, ghx4). Gates + C1. Sync first: scripts/sync_to_delta.sh --host dtai --data
# Focus (user, 10-04): the length of the CoT; accuracy is reported, not gated. Backup if accuracy
# still degrades: data/math_fed_ans (gold answer stated in the prompt).
#   smoke  2 rounds x 5 steps, 4-prompt evals: training has never run on dtai (torch 2.13, aarch64)
#   G0     one client (math), plain SFT: fed.lr_shared=0 keeps B_d == 0
#   G2     one client (math), steering on
#   G1     B1 prompting on G2's checkpoint (k=0/3 client at 1,280; k=3 base at 4,096)
#   E1     rotation 0 (8 participants): federated (shared calibration, no offset) vs local-only
# 100 rounds x 20 steps x batch 8 (4 epochs over 4k pairs; single-client G2 on Qwen3-4B was still
# improving at round 60). Dev evals every 20 rounds; E1 dev evals use 20 prompts per client.
# 10-04: the afterok:$S dependencies below were removed after submission (scontrol), since the smoke
# test was estimated to start a day later; the jobs now queue in parallel with it.
set -e
MODEL="model_name=Qwen/Qwen2.5-7B-Instruct"
ONE="$MODEL clients=[math] fed.rounds=100 fed.save_every=10 monitor.full_every=20 monitor.full_prompts=50 monitor.full_batch_size=64"
E1="$MODEL fed.rounds=100 fed.save_every=10 monitor.full_every=20 monitor.full_prompts=20 monitor.full_batch_size=64"
timeout 300 ssh -o BatchMode=yes dtai-1 "cd /u/lucmon/rein && \
S=\$(sbatch --parsable --job-name=exp40_smoke_qwen25_7b --time=02:00:00 \
  --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,TEST_PROMPTS=4,EVAL_BATCH=64,MAX_NEW=128,OVERRIDES=\"$MODEL clients=[math,olympiads] fed.rounds=2 fed.local_steps=5 fed.save_every=1 monitor.full_every=1 monitor.full_prompts=4 monitor.full_max_new_tokens=128 out_dir=runs/exp40_smoke_qwen25_7b\" \
  sbatch/train_eval_math.sbatch) && \
G0=\$(sbatch --parsable --job-name=exp40_g0_sft --dependency=afterok:\$S \
  --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,OVERRIDES=\"$ONE fed.lr_shared=0 out_dir=runs/exp40_g0_sft_qwen25_7b\" \
  sbatch/train_eval_math.sbatch) && \
G2=\$(sbatch --parsable --job-name=exp40_g2_single --dependency=afterok:\$S \
  --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,OVERRIDES=\"$ONE out_dir=runs/exp40_g2_single_qwen25_7b\" \
  sbatch/train_eval_math.sbatch) && \
G1=\$(sbatch --parsable --job-name=exp40_g1_b1 --dependency=afterok:\$G2 \
  --export=ALL,RUN_JOB=\$G2 sbatch/eval_b1_math.sbatch) && \
F=\$(sbatch --parsable --job-name=exp40_e1_fed --time=2-00:00:00 --dependency=afterok:\$S \
  --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,OVERRIDES=\"$E1 out_dir=runs/exp40_e1_fed_qwen25_7b\" \
  sbatch/train_eval_math.sbatch) && \
L=\$(sbatch --parsable --job-name=exp40_e1_local --time=2-00:00:00 --dependency=afterok:\$S \
  --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,OVERRIDES=\"$E1 fed.mode=local fed.calibration=private out_dir=runs/exp40_e1_local_qwen25_7b\" \
  sbatch/train_eval_math.sbatch) && \
echo smoke=\$S G0=\$G0 G2=\$G2 G1=\$G1 E1_fed=\$F E1_local=\$L" </dev/null
