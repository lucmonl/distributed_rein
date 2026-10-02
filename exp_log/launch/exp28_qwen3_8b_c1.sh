#!/bin/bash
# Experiment log entry 28 (2026-10-02): C1 on a second model family at 8B: Qwen/Qwen3-8B
# (thinking off via chat_template_kwargs.enable_thinking=false), A100 80GB.
# Same data and hyperparameters as the 1B method / local pair (exp17: 11063910 / 11063911):
# 4k per client, 100 rounds x 20 steps, batch 8, rank 16, LRs unchanged.
#   fed   = method: federated, shared calibration, no offset
#   local = local-only, no offset (per-client calibration)
# Smoke first (2 rounds x 10 steps, 8-article evals, GPU memory log); real runs afterok.
cd /u/lucmon/rein
MODEL="model_name=Qwen/Qwen3-8B chat_template_kwargs.enable_thinking=false"
COMMON="$MODEL fed.rounds=100 max_train_per_client=4000 lora.offset=false"
S=$(sbatch --parsable --job-name=exp28_smoke_qwen3_8b --time=04:00:00 \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,GPU_LOG=1,TEST_PROMPTS=8,OVERRIDES="$MODEL max_train_per_client=4000 lora.offset=false fed.calibration=shared fed.rounds=2 fed.local_steps=10 fed.save_every=1 monitor.full_every=1 monitor.full_prompts=8 out_dir=runs/smoke_qwen3_8b" \
  sbatch/train_eval.sbatch)
F=$(sbatch --parsable --job-name=exp28_fed_qwen3_8b --time=3-00:00:00 --dependency=afterok:$S \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,GPU_LOG=1,OVERRIDES="$COMMON fed.calibration=shared out_dir=runs/exp28_fed_calshared_nooff_cap4k_qwen3_8b" \
  sbatch/train_eval.sbatch)
L=$(sbatch --parsable --job-name=exp28_local_qwen3_8b --time=3-00:00:00 --dependency=afterok:$S \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,GPU_LOG=1,OVERRIDES="$COMMON fed.mode=local out_dir=runs/exp28_local_nooff_cap4k_qwen3_8b" \
  sbatch/train_eval.sbatch)
echo "smoke $S  fed $F  local $L"

# --- Resubmission (2026-10-02 ~00:30): /u/lucmon has a 100 GB quota (68 GB used) and an 8B run writes
# ~25-30 GB of snapshots, so the two real runs were cancelled before they started and resubmitted with
# their run directories in project space (/u/lucmon/lucmon -> /projects/illinois/eng/cs/arindamb/lucmon),
# linked into runs/ by train_eval.sbatch.  The smoke run (small) stays in runs/.
# RR=/u/lucmon/lucmon/rein_runs; S=11097847
# F=$(sbatch --parsable --job-name=exp28_fed_qwen3_8b --time=3-00:00:00 --dependency=afterok:$S \
#   --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,GPU_LOG=1,RUNS_ROOT=$RR,OVERRIDES="$COMMON fed.calibration=shared out_dir=$RR/exp28_fed_calshared_nooff_cap4k_qwen3_8b" \
#   sbatch/train_eval.sbatch)
# L=$(sbatch --parsable --job-name=exp28_local_qwen3_8b --time=3-00:00:00 --dependency=afterok:$S \
#   --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,GPU_LOG=1,RUNS_ROOT=$RR,OVERRIDES="$COMMON fed.mode=local out_dir=$RR/exp28_local_nooff_cap4k_qwen3_8b" \
#   sbatch/train_eval.sbatch)
