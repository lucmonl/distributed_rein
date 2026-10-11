#!/bin/bash
# Experiment log MOL-27 (2026-10-09). Spec CAL-27 (user request): Newsroom's NR-74 ablations on ChEMBL.
#
#   mol27-fed-linear-off            g_i(a) = o_i + a    the learned shape ABLATED (warp=none), o_i private
#   mol27-fed-shared-soff-sadapter  g(a)   = o + h_l(a) pure FedAvg: adapter, per-layer shape and
#                                                        offset all shared, nothing private
# Matched to MOL-25: configs/chembl_deco_skew_pruned.yaml, 100 rounds, rotation 0, seed 0, strict
# scorer at every stage, dev selection on pct_calib_err_penalized, 150 test prompts.
# COMMON: fed.fix_gain=true fed.warp_reg=0
#
# Verified off-GPU and by 2-client smoke (11253935 / 11253936, both COMPLETED, DONE, no tracebacks):
#   arm 1  warp=none forces scope `model` at source -- warp_indices(targets, cfg.warp_scope if
#          cfg.warp != "none" else "model") -- so warp_scope is genuinely inert and is not passed.
#          Server holds the 252 direction tensors and u; NO shape tensors anywhere; each client its
#          OWN o (0.04544 vs 0.06338 at round 6) and the server's o list is empty, i.e. private.
#   arm 2  Server holds direction (252) + adapter B_p/A_p (504) + u + o + 756 shape tensors
#          (= 252 shapes x 3 kumaraswamy params). Nothing is private.
#          fed.py:196 folds the private adapter params into the shared set when adapter=shared;
#          fed.py:185 notes calibration=shared ALREADY shares the offset, so fed.shared_offset is
#          deliberately NOT passed -- it requires calibration=aligned and would raise here.
#   ⚠️ The arm-2 round-6 snapshot shows per-client o = 0.0, which looked like a frozen parameter.
#      It is bookkeeping: the SERVER holds the trained shared o (0.04875) and the per-round log
#      shows it training (0 -> 0.0074 -> 0.017 -> 0.034 -> 0.041), so the client copy simply is not
#      the authoritative one for a shared offset. Checked before committing 2 x ~6 h.
#
# cc only, A100, ccc0387 excluded; ccc0284 never excluded (memory reference-slurm). Anvil stays
# gated off (CAL-11/CAL-16).
set -u
cd /u/lucmon/rein
CFG=configs/chembl_deco_skew_pruned.yaml
DATA=data/chembl_deco_skew_pruned/data.jsonl
ENVC="CONFIG=$CFG,GPU_LOG=1,DATA=$DATA,SCORER=clogp_residual_deco_strict,RESCORE=clogp_residual_deco,SELECT=pct_calib_err_penalized,TEST_PROMPTS=150"
COMMON="fed.fix_gain=true fed.warp_reg=0"

go () { sbatch --parsable --job-name=$1 --time=24:00:00 --gres=gpu:A100:1 --exclude=ccc0387 \
    --export=ALL,$ENVC,OVERRIDES="$COMMON out_dir=runs/$1 $2" \
    sbatch/train_eval_chembl.sbatch; }
A=$(go mol27-fed-linear-off \
      "lora.warp=none fed.calibration=shared lora.offset=true fed.private_offset=true")
B=$(go mol27-fed-shared-soff-sadapter \
      "fed.adapter=shared fed.calibration=shared lora.offset=true lora.warp=kumaraswamy_mix lora.warp_scope=module")
echo "cc  fed-linear-off $A  fed-shared-soff-sadapter $B"
