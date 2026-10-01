#!/bin/bash
# Experiment log entry 10: GPU smoke test of the new pipeline (in-training full dev eval,
# selection, test eval, quality metrics, LLM judge, all regularizers on) on tiny settings.
cd /u/lucmon/rein
sbatch --job-name=smoke_pipeline --time=02:00:00 \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,TEST_PROMPTS=4,OVERRIDES="model_name=meta-llama/Llama-3.2-1B-Instruct out_dir=runs/smoke_pipeline max_train_per_client=50 fed.rounds=2 fed.local_steps=2 fed.save_every=1 monitor.full_every=1 monitor.full_prompts=4 fed.lr_schedule=cosine lora.dropout=0.05 reg.private_wd=0.05 reg.shared_wd=0.01 reg.decorr=0.1 reg.fedprox_mu=0.1 reg.gain_l2=0.01 reg.offset_l2=0.01" \
  sbatch/train_eval.sbatch
