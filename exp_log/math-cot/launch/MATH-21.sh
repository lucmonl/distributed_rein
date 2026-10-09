#!/bin/bash
# DRAFT (2026-10-08), NOT SUBMITTED: ON HOLD by the user (relayed by the coordinator, 2026-10-08). Do not run until released.
# MATH-21: does a fixed gain s = 1 (CAL-12) cost single-client math anything?
# Matched to MATH-14 P (dtai 3328129, `local-gain`: g_l(a) = s*h_l(a), 196 per-layer shapes, s learned and
# clamped at 4; test pct err 0.213). Same BASE/EXP as exp_log/launch/MATH-14_per_layer_warps.sh; only the
# overrides after BASE differ.
#   (i)  local-wr0p01  g_l(a) = h_l(a), s = 1, warp_reg 0.01 as in P   -> isolates s
#   (ii) local         g_l(a) = h_l(a), s = 1, warp_reg 0               -> the reference design
# Single client: "local" in CONVENTIONS.md terms. Run with fed.mode=fedavg on one client, exactly like P, so the
# per-round Adam reset is matched.
# dtai ONLY (no Delta race): P ran on dtai (torch 2.13), and a Delta copy would not be host-matched (MATH-18).
set -e
BASE="model_name=Qwen/Qwen2.5-7B-Instruct data_path=data/math_fed_ans/data.jsonl clients_file=data/math_fed_ans/clients.json max_prompt_tokens=576 clients=[math] fed.rounds=100 fed.save_every=10 monitor.full_every=20 monitor.full_prompts=50 monitor.full_batch_size=64 lora.warp_scope=module fed.fix_gain=true"
EXP="CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,DATA=data/math_fed_ans/data.jsonl"
timeout 300 ssh -o BatchMode=yes dtai-1 "cd /u/lucmon/rein && mkdir -p sbatch/logs/math-cot && \
A=\$(sbatch --parsable --job-name=math21-local-wr0p01 --export=ALL,$EXP,OVERRIDES=\"$BASE fed.warp_reg=0.01 out_dir=runs/math21-local-wr0p01\" sbatch/train_eval_math.sbatch) && \
B=\$(sbatch --parsable --job-name=math21-local --export=ALL,$EXP,OVERRIDES=\"$BASE fed.warp_reg=0 out_dir=runs/math21-local\" sbatch/train_eval_math.sbatch) && \
echo dtai: local-wr0p01=\$A local=\$B" </dev/null 2>&1 | grep -v OpenSSL
