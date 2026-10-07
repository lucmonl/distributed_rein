#!/bin/bash
# Log entry MATH-5 (2026-10-05): the Qwen2.5-7B math suite, resubmitted after every run of entry 40 /
# MATH-4 went NaN (SDPA backward on long batches, dtai torch 2.13). Now attn_implementation=eager
# (configs/math_fedavg.yaml) plus a NaN guard in fedsteer/fed.py (skips a step with a non-finite loss or
# gradient and counts it). Raced on dtai (ghx4) and Delta (gpuA100x8,gpuH200x8) -- math-CoT is Delta/dtai
# only -- losers cancelled by scripts/race_watch.py once a copy RUNS (spec MATH-5_race.json).
# Sync first: scripts/sync_to_delta.sh && scripts/sync_to_delta.sh --host dtai.  Run on cc-login1.
# 10-06 (MATH-10): gpuA100x8 is A100-SXM4-40GB, not 80 GB as assumed; the 7B OOMs there, so Delta
# copies use --partition=gpuH200x8 only.
set -e
MODEL="model_name=Qwen/Qwen2.5-7B-Instruct"
ONE="$MODEL clients=[math] fed.rounds=100 fed.save_every=10 monitor.full_every=20 monitor.full_prompts=50 monitor.full_batch_size=64"
E1="$MODEL fed.rounds=100 fed.save_every=10 monitor.full_every=20 monitor.full_prompts=20 monitor.full_batch_size=64"
submit () {  # host tag extra_partition_flag
  timeout 300 ssh -o BatchMode=yes $1 "cd /u/lucmon/rein && \
G0=\$(sbatch --parsable --job-name=math5_g0_sft_$2 $3 \
  --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,OVERRIDES=\"$ONE fed.lr_shared=0 out_dir=runs/math5_g0_sft_qwen25_7b\" \
  sbatch/train_eval_math.sbatch) && \
G2=\$(sbatch --parsable --job-name=math5_g2_single_$2 $3 \
  --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,OVERRIDES=\"$ONE out_dir=runs/math5_g2_single_qwen25_7b\" \
  sbatch/train_eval_math.sbatch) && \
G1=\$(sbatch --parsable --job-name=math5_g1_b1_$2 $3 --dependency=afterok:\$G2 \
  --export=ALL,RUN_JOB=\$G2 sbatch/eval_b1_math.sbatch) && \
F=\$(sbatch --parsable --job-name=math5_e1_fed_$2 $3 --time=2-00:00:00 \
  --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,OVERRIDES=\"$E1 out_dir=runs/math5_e1_fed_qwen25_7b\" \
  sbatch/train_eval_math.sbatch) && \
L=\$(sbatch --parsable --job-name=math5_e1_local_$2 $3 --time=2-00:00:00 \
  --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,OVERRIDES=\"$E1 fed.mode=local fed.calibration=private out_dir=runs/math5_e1_local_qwen25_7b\" \
  sbatch/train_eval_math.sbatch) && \
echo \$G0 \$G2 \$G1 \$F \$L" </dev/null 2>&1 | grep -v OpenSSL
}
read T0 T2 T1 TF TL < <(submit dtai-1 dtai "")
read D0 D2 D1 DF DL < <(submit delta-login2 delta "--partition=gpuH200x8")
echo "dtai:  $T0 $T2 $T1 $TF $TL"; echo "delta: $D0 $D2 $D1 $DF $DL"
cat > exp_log/launch/MATH-5_race.json <<JSON
{"hosts": {"delta": "delta-login2", "dtai": "dtai-1"},
 "races": [
  {"name": "G0", "copies": {"delta": "$D0", "dtai": "$T0"}},
  {"name": "G2", "copies": {"delta": "$D2", "dtai": "$T2"}, "dependents": {"delta": ["$D1"], "dtai": ["$T1"]}},
  {"name": "E1fed", "group": "E1", "copies": {"delta": "$DF", "dtai": "$TF"}},
  {"name": "E1local", "group": "E1", "copies": {"delta": "$DL", "dtai": "$TL"}}
 ]}
JSON
