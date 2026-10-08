#!/bin/bash
# Log entry MATH-14 (2026-10-07): per-layer calibration functions for math (user request, as on Newsroom
# NR-54): lora.warp_scope=module gives each of Qwen2.5-7B's 196 adapted matrices its own warp h_l
# (shared calibration: each layer's warp averaged; one gain). Single-client G2, answer-in-prompt, matched
# to MATH-11's G2 (Delta 22703347: test pct err 0.253, gain pinned at 4.0) -- only the overrides below differ.
#   P   lora.warp_scope=module
#   PG  lora.warp_scope=module lora.gain_max=16   (per-layer shapes cannot move the endpoints h(0)=0,
#       h(1)=1 and share the clamped gain, MATH-13; PG tells whether P was held back by the clamp)
# Overrides only -- NOT config defaults: the queued E1 jobs read configs/math_fedavg.yaml at start.
# Raced on Delta (all GPU partitions; 40 GB-safe since MATH-11) and dtai; spec MATH-14_race.json.
set -e
BASE="model_name=Qwen/Qwen2.5-7B-Instruct data_path=data/math_fed_ans/data.jsonl clients_file=data/math_fed_ans/clients.json max_prompt_tokens=576 clients=[math] fed.rounds=100 fed.save_every=10 monitor.full_every=20 monitor.full_prompts=50 monitor.full_batch_size=64"
EXP="CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,DATA=data/math_fed_ans/data.jsonl"
submit () {  # host tag
  timeout 300 ssh -o BatchMode=yes $1 "cd /u/lucmon/rein && \
P=\$(sbatch --parsable --job-name=math14_g2_perlayer_$2 --export=ALL,$EXP,OVERRIDES=\"$BASE lora.warp_scope=module out_dir=runs/math14_ans_g2_perlayer_qwen25_7b\" sbatch/train_eval_math.sbatch) && \
G=\$(sbatch --parsable --job-name=math14_g2_perlayer_gmax16_$2 --export=ALL,$EXP,OVERRIDES=\"$BASE lora.warp_scope=module lora.gain_max=16 out_dir=runs/math14_ans_g2_perlayer_gmax16_qwen25_7b\" sbatch/train_eval_math.sbatch) && \
echo \$P \$G" </dev/null 2>&1 | grep -v OpenSSL
}
read DP DG < <(submit delta-login2 delta)
read TP TG < <(submit dtai-1 dtai)
echo "delta: P=$DP PG=$DG   dtai: P=$TP PG=$TG"
cat > exp_log/launch/MATH-14_race.json <<JSON
{"hosts": {"delta": "delta-login2", "dtai": "dtai-1"},
 "races": [
  {"name": "G2_perlayer", "copies": {"delta": "$DP", "dtai": "$TP"}},
  {"name": "G2_perlayer_gmax16", "copies": {"delta": "$DG", "dtai": "$TG"}}
 ]}
JSON
