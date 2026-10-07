#!/bin/bash
# Log entry MATH-7 (2026-10-05): the Qwen2.5-7B math suite in the ANSWER-IN-PROMPT setting (user,
# 10-05): data/math_fed_ans -- same records, splits, targets and length labels as data/math_fed, but
# each prompt ends "The final answer is $X$. Please reason step by step to reach it, ...", so the
# study is about the length of the CoT only. Everything else identical to MATH-5
# (exp_log/launch/MATH-5_qwen25_7b_eager.sh: eager attention, NaN guard, 100 rounds, same evals).
# MATH-5's E1-fed (dtai 3311356, original prompts) keeps running as the reference.
# Raced on dtai (ghx4) and Delta (gpuA100x8,gpuH200x8); losers cancelled by scripts/race_watch.py
# (spec MATH-7_race.json) once a copy has RUN for 20 min. Plus a base-model reference with the answer
# in the prompt (math client, 4,096 cap) on dtai only. Run on cc-login1 after syncing both hosts.
# 10-06 (MATH-10): gpuA100x8 is A100-SXM4-40GB, not 80 GB as assumed; the 7B OOMs there, so Delta
# copies use --partition=gpuH200x8 only.
set -e
MODEL="model_name=Qwen/Qwen2.5-7B-Instruct data_path=data/math_fed_ans/data.jsonl clients_file=data/math_fed_ans/clients.json max_prompt_tokens=576"  # 7 of 49,800 answer-in-prompt prompts exceed 512 tokens (max 549)
ONE="$MODEL clients=[math] fed.rounds=100 fed.save_every=10 monitor.full_every=20 monitor.full_prompts=50 monitor.full_batch_size=64"
E1="$MODEL fed.rounds=100 fed.save_every=10 monitor.full_every=20 monitor.full_prompts=20 monitor.full_batch_size=64"
COMMON_EXPORT="CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,DATA=data/math_fed_ans/data.jsonl"
submit () {  # host tag extra_partition_flag
  timeout 300 ssh -o BatchMode=yes $1 "cd /u/lucmon/rein && \
G0=\$(sbatch --parsable --job-name=math7_ans_g0_sft_$2 $3 \
  --export=ALL,$COMMON_EXPORT,OVERRIDES=\"$ONE fed.lr_shared=0 out_dir=runs/math7_ans_g0_sft_qwen25_7b\" \
  sbatch/train_eval_math.sbatch) && \
G2=\$(sbatch --parsable --job-name=math7_ans_g2_single_$2 $3 \
  --export=ALL,$COMMON_EXPORT,OVERRIDES=\"$ONE out_dir=runs/math7_ans_g2_single_qwen25_7b\" \
  sbatch/train_eval_math.sbatch) && \
G1=\$(sbatch --parsable --job-name=math7_ans_g1_b1_$2 $3 --dependency=afterok:\$G2 \
  --export=ALL,RUN_JOB=\$G2,DATA=data/math_fed_ans/data.jsonl sbatch/eval_b1_math.sbatch) && \
F=\$(sbatch --parsable --job-name=math7_ans_e1_fed_$2 $3 --time=2-00:00:00 \
  --export=ALL,$COMMON_EXPORT,OVERRIDES=\"$E1 out_dir=runs/math7_ans_e1_fed_qwen25_7b\" \
  sbatch/train_eval_math.sbatch) && \
L=\$(sbatch --parsable --job-name=math7_ans_e1_local_$2 $3 --time=2-00:00:00 \
  --export=ALL,$COMMON_EXPORT,OVERRIDES=\"$E1 fed.mode=local fed.calibration=private out_dir=runs/math7_ans_e1_local_qwen25_7b\" \
  sbatch/train_eval_math.sbatch) && \
echo \$G0 \$G2 \$G1 \$F \$L" </dev/null 2>&1 | grep -v OpenSSL
}
read T0 T2 T1 TF TL < <(submit dtai-1 dtai "")
read D0 D2 D1 DF DL < <(submit delta-login2 delta "--partition=gpuH200x8")
B=$(timeout 300 ssh -o BatchMode=yes dtai-1 "cd /u/lucmon/rein && sbatch --parsable --job-name=math7_ans_base_dtai \
  --export=ALL,OUT=runs/math7_ans_base_qwen25_7b,MODEL=Qwen/Qwen2.5-7B-Instruct,MAX_NEW=4096,BATCH=16,CLIENTS=math,EXTRA='--data data/math_fed_ans/data.jsonl --clients_file data/math_fed_ans/clients.json' \
  sbatch/eval_base_math.sbatch" </dev/null 2>&1 | grep -v OpenSSL)
echo "dtai:  $T0 $T2 $T1 $TF $TL   base $B"; echo "delta: $D0 $D2 $D1 $DF $DL"
cat > exp_log/launch/MATH-7_race.json <<JSON
{"hosts": {"delta": "delta-login2", "dtai": "dtai-1"},
 "races": [
  {"name": "G0", "copies": {"delta": "$D0", "dtai": "$T0"}},
  {"name": "G2", "copies": {"delta": "$D2", "dtai": "$T2"}, "dependents": {"delta": ["$D1"], "dtai": ["$T1"]}},
  {"name": "E1fed", "group": "E1", "copies": {"delta": "$DF", "dtai": "$TF"}},
  {"name": "E1local", "group": "E1", "copies": {"delta": "$DL", "dtai": "$TL"}}
 ]}
JSON
