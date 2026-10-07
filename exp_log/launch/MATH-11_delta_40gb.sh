#!/bin/bash
# Log entry MATH-11 (2026-10-06): Delta copies of the answer-in-prompt suite (MATH-7), resubmitted after
# the 40 GB OOM (MATH-9/10) with memory-safe settings that now live in configs/math_fedavg.yaml:
# fed.batch_size=2 x fed.grad_accum=4 (same 8 examples per optimizer step), monitor.batch_size=4
# (dev-loss batch), B1 batch 8 for long-prompt rows. Verified under a 39.5 GiB cap on dtai (probe
# 3322211: train peak 21.9 GiB, test eval 19.9 GiB). So all Delta GPU partitions are usable again
# (the sbatch default gpuA100x4,gpuA100x8,gpuH200x8). The dtai copies 3316647-52 (pending) read the same
# config/code at start. Race: scripts/race_watch.py, spec MATH-11_race.json. Sync Delta first.
set -e
MODEL="model_name=Qwen/Qwen2.5-7B-Instruct data_path=data/math_fed_ans/data.jsonl clients_file=data/math_fed_ans/clients.json max_prompt_tokens=576"
ONE="$MODEL clients=[math] fed.rounds=100 fed.save_every=10 monitor.full_every=20 monitor.full_prompts=50 monitor.full_batch_size=64"
E1="$MODEL fed.rounds=100 fed.save_every=10 monitor.full_every=20 monitor.full_prompts=20 monitor.full_batch_size=64"
EXP="CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,DATA=data/math_fed_ans/data.jsonl"
read D0 D2 D1 DF DL < <(timeout 300 ssh -o BatchMode=yes delta-login2 "cd /u/lucmon/rein && \
G0=\$(sbatch --parsable --job-name=math11_ans_g0_sft_delta --export=ALL,$EXP,OVERRIDES=\"$ONE fed.lr_shared=0 out_dir=runs/math7_ans_g0_sft_qwen25_7b\" sbatch/train_eval_math.sbatch) && \
G2=\$(sbatch --parsable --job-name=math11_ans_g2_single_delta --export=ALL,$EXP,OVERRIDES=\"$ONE out_dir=runs/math7_ans_g2_single_qwen25_7b\" sbatch/train_eval_math.sbatch) && \
G1=\$(sbatch --parsable --job-name=math11_ans_g1_b1_delta --dependency=afterok:\$G2 --export=ALL,RUN_JOB=\$G2,DATA=data/math_fed_ans/data.jsonl sbatch/eval_b1_math.sbatch) && \
F=\$(sbatch --parsable --job-name=math11_ans_e1_fed_delta --time=2-00:00:00 --export=ALL,$EXP,OVERRIDES=\"$E1 out_dir=runs/math7_ans_e1_fed_qwen25_7b\" sbatch/train_eval_math.sbatch) && \
L=\$(sbatch --parsable --job-name=math11_ans_e1_local_delta --time=2-00:00:00 --export=ALL,$EXP,OVERRIDES=\"$E1 fed.mode=local fed.calibration=private out_dir=runs/math7_ans_e1_local_qwen25_7b\" sbatch/train_eval_math.sbatch) && \
echo \$G0 \$G2 \$G1 \$F \$L" </dev/null)
echo "delta: $D0 $D2 $D1 $DF $DL"
cat > exp_log/launch/MATH-11_race.json <<JSON
{"hosts": {"delta": "delta-login2", "dtai": "dtai-1"},
 "races": [
  {"name": "G0", "copies": {"delta": "$D0", "dtai": "3316647"}},
  {"name": "G2", "copies": {"delta": "$D2", "dtai": "3316648"}, "dependents": {"delta": ["$D1"], "dtai": ["3316649"]}},
  {"name": "E1fed", "group": "E1", "copies": {"delta": "$DF", "dtai": "3316650"}},
  {"name": "E1local", "group": "E1", "copies": {"delta": "$DL", "dtai": "3316651"}}
 ]}
JSON
