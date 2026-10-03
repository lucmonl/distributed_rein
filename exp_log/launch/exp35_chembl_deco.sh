#!/bin/bash
# Experiment log entry 35 (2026-10-03): decoration-format G0 (gate G0 re-run) + the method/local
# pair, following the prior-art review.  Entry 34 showed the whole-molecule format loses the
# requested core on 42% of generations, rising with alpha; Arus-Pous et al. 2020 and SAFE both fix
# this by having the model emit ONLY the decorations, with the core fixed in the prompt.
# Data: data/chembl_deco (core carries [n*] attachment points; target = decorations; molzip
# reassembly).  Core preservation is now STRUCTURAL, and alpha labels are unchanged from
# data/chembl_fed so the two formats are directly comparable.
cd /u/lucmon/rein
RR=/u/lucmon/lucmon/rein_runs
COMMON="CONFIG=configs/chembl_deco.yaml,GPU_LOG=1,RUNS_ROOT=$RR,DATA=data/chembl_deco/data.jsonl,RESCORE=none,SCORER=clogp_residual_deco"
# G0 in the new format: one client, 100 rounds, same budget as exp33 so the formats are comparable
G=$(sbatch --parsable --job-name=exp35_deco_g0 --time=06:00:00 \
  --export=ALL,$COMMON,TEST_PROMPTS=100,\
OVERRIDES="clients=[CHEMBL240] fed.rounds=100 fed.save_every=20 monitor.full_every=20 monitor.full_prompts=40 out_dir=$RR/exp35_deco_g0_qwen3_4b" \
  sbatch/train_eval_chembl.sbatch)
# the method / local pair (G2, G3), 8 participants of rotation 0, gated on G0 finishing cleanly
F=$(sbatch --parsable --job-name=exp35_deco_fed --time=1-00:00:00 --dependency=afterok:$G \
  --export=ALL,$COMMON,TEST_PROMPTS=150,\
OVERRIDES="fed.rounds=30 fed.calibration=shared out_dir=$RR/exp35_deco_fed_qwen3_4b" \
  sbatch/train_eval_chembl.sbatch)
L=$(sbatch --parsable --job-name=exp35_deco_local --time=1-00:00:00 --dependency=afterok:$G \
  --export=ALL,$COMMON,TEST_PROMPTS=150,\
OVERRIDES="fed.rounds=30 fed.mode=local out_dir=$RR/exp35_deco_local_qwen3_4b" \
  sbatch/train_eval_chembl.sbatch)
echo "g0 $G  fed $F  local $L"
