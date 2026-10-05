#!/bin/bash
# Experiment log entry 39 (2026-10-03): backbone screen for the math-CoT task, run on DELTA.
# G0 (entry 38) showed that fine-tuning Qwen3-4B-Instruct-2507 on the GPT-4o NuminaMath solutions
# costs accuracy (0.98 -> 0.79 on problems the base finished), a teacher-style mismatch. Screen
# candidate backbones for familiarity (perplexity of the targets) and headroom (base accuracy at a
# 4,096 cap) before choosing one for G0. No training.
# HOST=delta-login2 (default; sync: scripts/sync_to_delta.sh) or HOST=dtai-1 (DeltaAI GH200, ghx4;
# sync: scripts/sync_to_delta.sh --host dtai). The first submission on Delta (22647630-38) never
# started (queue) and was cancelled; the screen ran on dtai-1 (10-04).
set -e
HOST=${HOST:-delta-login2}
submit () {  # name model [extra]
  timeout 120 ssh -o BatchMode=yes $HOST "cd /u/lucmon/rein && sbatch --parsable --job-name=exp39_screen_$1 \
    --export=ALL,MODEL=$2,OUT=runs/exp39_screen_$1,EXTRA='$3' sbatch/backbone_screen.sbatch" </dev/null
}
submit qwen3_4b_2507   Qwen/Qwen3-4B-Instruct-2507        # reference: the G0 backbone
submit qwen3_8b_nothink Qwen/Qwen3-8B                      # hybrid, thinking off (template default here)
submit qwen25_7b       Qwen/Qwen2.5-7B-Instruct
submit qwen25_3b       Qwen/Qwen2.5-3B-Instruct
submit llama31_8b      meta-llama/Llama-3.1-8B-Instruct
submit llama32_1b      meta-llama/Llama-3.2-1B-Instruct
