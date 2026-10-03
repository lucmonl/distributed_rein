#!/bin/bash
# Experiment log entry 37 (2026-10-03): math CoT (NuminaMath-CoT) gates, run on DELTA.
# Code is edited on cc-login1 only; sync first:  scripts/sync_to_delta.sh --data
# Run from cc-login1 (needs the SSH master: ssh -fN delta-login2).  Outputs land in
# Delta's runs/ -> /work/nvme/bhby/lucmon/rein.
#
#   E0a  base Qwen3-4B-Instruct-2507 on all 12 clients' test problems (no training)
#   G0   plain SFT on one client (math), no steering: fed.lr_shared=0 keeps B_d == 0 (entry 34)
#   G2   the same client with steering on (single-client steerability)
# Single-client runs: 60 rounds x 20 steps x batch 8 = 9,600 examples = 2.4 epochs over 4k pairs
# (ChEMBL G0 was undertrained at 0.8 epochs, entry 33); dev eval (50 problems x 5 alphas) every 10.
set -e
if [ -z "$G2_JOB" ] && [ -z "$E0A_LONG" ] && [ -z "$EXTEND" ]; then
ONE="clients=[math] fed.rounds=60 fed.save_every=10 monitor.full_every=10 monitor.full_prompts=50"
ssh delta-login2 "cd /u/lucmon/rein && \
  sbatch --parsable --job-name=exp37_math_e0a_base \
    --export=ALL,OUT=runs/exp37_math_e0a_base_qwen3_4b sbatch/eval_base_math.sbatch && \
  sbatch --parsable --job-name=exp37_math_g0_sft \
    --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,OVERRIDES=\"$ONE fed.lr_shared=0 out_dir=runs/exp37_math_g0_sft_qwen3_4b\" \
    sbatch/train_eval_math.sbatch && \
  sbatch --parsable --job-name=exp37_math_g2_single \
    --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,OVERRIDES=\"$ONE out_dir=runs/exp37_math_g2_single_qwen3_4b\" \
    sbatch/train_eval_math.sbatch"
exit 0
fi

# G1 (B1 prompting) on G2's client model, chained after G2 (afterok). The first submission
# (22640626-28, 10-03) failed in setup on Delta and was resubmitted as 22640760-62.
if [ -n "$G2_JOB" ]; then
ssh delta-login2 "cd /u/lucmon/rein && sbatch --parsable --job-name=exp37_math_g1_b1 \
  --dependency=afterok:$G2_JOB --export=ALL,RUN_JOB=$G2_JOB sbatch/eval_b1_math.sbatch"
fi

# --- E0a-long (10-03): E0a at the 1,280 cap truncated 42% of base outputs, so its accuracy measured
# the cap (accuracy on untruncated outputs: math 0.98, olympiads 0.81). Rerun at 4,096 tokens,
# 3 jobs x 4 clients (round-robin over the median order), batch 16 for the longer KV cache.
# Run as:  E0A_LONG=1 bash exp_log/launch/exp37_math_gates.sh
if [ "$E0A_LONG" = "1" ]; then
  for SET in "orca_math/Logic and Puzzles+synthetic_math/Algebra+synthetic_math/Geometry+synthetic_amc" \
             "cn_k12/Logic and Puzzles+orca_math/Algebra+cn_k12/Geometry+olympiads" \
             "gsm8k+cn_k12/Inequalities+math+aops_forum"; do
    ssh delta-login2 "cd /u/lucmon/rein && sbatch --parsable --job-name=exp37_math_e0a_long \
      --export=ALL,OUT=runs/exp37_math_e0a_long4096_qwen3_4b,MAX_NEW=4096,BATCH=16,CLIENTS='$SET' sbatch/eval_base_math.sbatch"
  done
fi

# --- G0/G2 extended to 120 rounds (10-03): at round 50 G2 was still improving ~0.04 pct err per 10
# rounds (0.299 -> 0.273 -> 0.235) with the direction norm still growing. Resume both runs in place
# (dev eval every 20 rounds) after the 60-round jobs finish; G1 then runs on the extended G2.
# Run as:  EXTEND=1 bash exp_log/launch/exp37_math_gates.sh
if [ "$EXTEND" = "1" ]; then
  ssh delta-login2 'cd /u/lucmon/rein && \
    G0R=$(ls -d runs/exp37_math_g0_sft_qwen3_4b_*_j22640761) && G2R=$(ls -d runs/exp37_math_g2_single_qwen3_4b_*_j22640762) && \
    A=$(sbatch --parsable --job-name=exp37_math_g0_sft_r120 --dependency=afterany:22640761 \
      --export=ALL,RESUME=$G0R,OVERRIDES="fed.rounds=120 monitor.full_every=20" sbatch/train_eval_math.sbatch) && \
    B=$(sbatch --parsable --job-name=exp37_math_g2_single_r120 --dependency=afterany:22640762 \
      --export=ALL,RESUME=$G2R,OVERRIDES="fed.rounds=120 monitor.full_every=20" sbatch/train_eval_math.sbatch) && \
    C=$(sbatch --parsable --job-name=exp37_math_g1_b1 --dependency=afterok:$B \
      --export=ALL,RUN_JOB=22640762 sbatch/eval_b1_math.sbatch) && echo "G0_r120=$A G2_r120=$B G1=$C"'
fi
