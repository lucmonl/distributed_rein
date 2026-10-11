#!/bin/bash
# MOL-27 pre-flight smoke (2026-10-09). NOT an experiment: 2 clients, 6 rounds, 64 train examples.
# Spec CAL-27 (user request): Newsroom's NR-74 ablations on ChEMBL.
#
#   mol27-smoke-fed-linear-off           g_i(a) = o_i + a       learned shape ABLATED (warp=none)
#   mol27-smoke-fed-shared-soff-sadapter g(a)  = o + h_l(a)     pure FedAvg: adapter, shape and
#                                                               offset all shared
# What the smoke must establish, per CAL-27, and what only a real run can show:
#   arm 1: the server holds ONLY the gain u, and each client its own o  (private_offset with
#          calibration=shared). With warp=none there are no shape tensors to share at all.
#   arm 2: the server holds the adapter B_p, the offset o, the 252 shape tensors and u; the clients
#          hold nothing private. fed.py:196 adds the private adapter params to the shared set when
#          adapter=shared, and fed.py:185 notes calibration=shared ALREADY shares the offset -- so
#          no fed.shared_offset flag is wanted here (that one is only for calibration=aligned).
set -u
cd /u/lucmon/rein
CFG=configs/chembl_deco_skew_pruned_smoke2.yaml
DATA=data/chembl_deco_skew_pruned/data.jsonl
ENVC="CONFIG=$CFG,DATA=$DATA,SCORER=clogp_residual_deco_strict,RESCORE=clogp_residual_deco,SELECT=pct_calib_err_penalized,TEST_PROMPTS=4"
COMMON="fed.fix_gain=true fed.warp_reg=0 fed.rounds=6 fed.save_every=1 monitor.full_every=6"

s () { sbatch --parsable --job-name=$1 --time=1:00:00 --gres=gpu:A100:1 --exclude=ccc0387 \
    --export=ALL,$ENVC,OVERRIDES="$COMMON out_dir=runs/$1 $2" \
    sbatch/train_eval_chembl.sbatch; }
A=$(s mol27-smoke-fed-linear-off \
      "lora.warp=none fed.calibration=shared lora.offset=true fed.private_offset=true")
B=$(s mol27-smoke-fed-shared-soff-sadapter \
      "fed.adapter=shared fed.calibration=shared lora.offset=true lora.warp=kumaraswamy_mix lora.warp_scope=module")
echo "smoke  fed-linear-off $A  fed-shared-soff-sadapter $B"
