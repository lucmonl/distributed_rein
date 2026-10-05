#!/bin/bash
# Log entry MATH-2 (2026-10-04): race the Qwen2.5-7B math suite (entry "40. Math CoT with
# Qwen2.5-7B-Instruct [dtai]") on cc as well, because dtai's ghx4 queue estimates a start on
# 2026-10-05 17:37. Same overrides as exp_log/launch/exp40_math_qwen25_7b.sh. The losing copies are
# cancelled by scripts/race_watch.py only once a copy is RUNNING (memory: feedback-cluster-failover);
# the E1 fed/local pair is kept on one host. cc: dali,IllinoisComputes-GPU (A100 80 GB / H200),
# env steer + math-verify from /u/lucmon/lucmon/pylib/mathverify. Run on cc-login1.
set -e
cd /u/lucmon/rein
if [ "$SCAV" != "1" ]; then
MODEL="model_name=Qwen/Qwen2.5-7B-Instruct"
ONE="$MODEL clients=[math] fed.rounds=100 fed.save_every=10 monitor.full_every=20 monitor.full_prompts=50 monitor.full_batch_size=64"
E1="$MODEL fed.rounds=100 fed.save_every=10 monitor.full_every=20 monitor.full_prompts=20 monitor.full_batch_size=64"
G0=$(sbatch --parsable --job-name=math2_g0_sft_cc \
  --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,OVERRIDES="$ONE fed.lr_shared=0 out_dir=runs/exp40_g0_sft_qwen25_7b" \
  sbatch/train_eval_math.sbatch)
G2=$(sbatch --parsable --job-name=math2_g2_single_cc \
  --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,OVERRIDES="$ONE out_dir=runs/exp40_g2_single_qwen25_7b" \
  sbatch/train_eval_math.sbatch)
G1=$(sbatch --parsable --job-name=math2_g1_b1_cc --dependency=afterok:$G2 \
  --export=ALL,RUN_JOB=$G2 sbatch/eval_b1_math.sbatch)
F=$(sbatch --parsable --job-name=math2_e1_fed_cc --time=3-00:00:00 \
  --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,OVERRIDES="$E1 out_dir=runs/exp40_e1_fed_qwen25_7b" \
  sbatch/train_eval_math.sbatch)
L=$(sbatch --parsable --job-name=math2_e1_local_cc --time=3-00:00:00 \
  --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,OVERRIDES="$E1 fed.mode=local fed.calibration=private out_dir=runs/exp40_e1_local_qwen25_7b" \
  sbatch/train_eval_math.sbatch)
echo "cc: G0=$G0 G2=$G2 G1=$G1 E1_fed=$F E1_local=$L"
cat > exp_log/launch/MATH-2_race.json <<JSON
{"hosts": {"cc": null, "dtai": "dtai-1"},
 "races": [
  {"name": "G0", "copies": {"cc": "$G0", "dtai": "3308444"}},
  {"name": "G2", "copies": {"cc": "$G2", "dtai": "3308445"}, "dependents": {"cc": ["$G1"], "dtai": ["3308446"]}},
  {"name": "E1fed", "group": "E1", "copies": {"cc": "$F", "dtai": "3308447"}},
  {"name": "E1local", "group": "E1", "copies": {"cc": "$L", "dtai": "3308448"}}
 ]}
JSON
fi

# --- (10-04, later) third copies of the single-client gates on cc's preemptible `scavenger` partition
# (24 h max). Only nodes with >= 48 GB and a GPU supported by torch 2.5.1+cu118: H100, H200, L40S.
# Excluded: ccc0089/90 (V100 16 GB), ccc0232-0236 (Quadro RTX 6000 24 GB), ccc0496-0499 (RTX6000B,
# Blackwell: needs CUDA >= 12.8). race_watch.py treats this host as preemptible: a scavenger copy
# wins only by COMPLETING, so a preemption cannot cost the other copies.
# Run as:  SCAV=1 bash exp_log/launch/MATH-2_qwen25_7b_cc_race.sh   (skips the block above)
if [ "$SCAV" = "1" ]; then
  MODEL="model_name=Qwen/Qwen2.5-7B-Instruct"
  ONE="$MODEL clients=[math] fed.rounds=100 fed.save_every=10 monitor.full_every=20 monitor.full_prompts=50 monitor.full_batch_size=64"
  BAD=ccc0089,ccc0090,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236,ccc0496,ccc0497,ccc0498,ccc0499
  G0=$(sbatch --parsable --job-name=math2_g0_sft_scav --partition=scavenger --exclude=$BAD --time=12:00:00 \
    --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,OVERRIDES="$ONE fed.lr_shared=0 out_dir=runs/exp40_g0_sft_qwen25_7b" \
    sbatch/train_eval_math.sbatch)
  G2=$(sbatch --parsable --job-name=math2_g2_single_scav --partition=scavenger --exclude=$BAD --time=12:00:00 \
    --export=ALL,CONFIG=configs/math_fedavg.yaml,GPU_LOG=1,EVAL_BATCH=64,OVERRIDES="$ONE out_dir=runs/exp40_g2_single_qwen25_7b" \
    sbatch/train_eval_math.sbatch)
  G1=$(sbatch --parsable --job-name=math2_g1_b1_scav --dependency=afterok:$G2 --export=ALL,RUN_JOB=$G2 sbatch/eval_b1_math.sbatch)
  echo "scavenger: G0=$G0 G2=$G2 G1(after scav G2, regular partitions)=$G1"
fi
