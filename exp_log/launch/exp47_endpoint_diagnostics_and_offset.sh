#!/bin/bash
# Experiment log MOL-21 (2026-10-07). Two parts, both following from MOL-20's decomposition, which
# found the alpha=0 cell is the worst for every arm (~0.30 vs ~0.18 interior) and that no arm moves
# it, because with gain 1 / no offset / h(0)=0 the coefficient at alpha=0 is exactly zero and the
# output is the private adapter alone.
#
# PART 1 -- the two diagnostics the plan asks for, on existing checkpoints, no training:
#   (a) eval_direction.py --posthoc_remap: an oracle isotonic alpha remap fitted on dev and applied
#       on test. Bounds how much error ANY recalibration can remove. Prediction: ~0 at the
#       endpoints, because the remap only reparameterizes alpha within [0,1] while h(0)=0 and
#       h(1)=1 are fixed. If that holds, warp-side variants are exhausted and the endpoint problem
#       is reach, not calibration.
#   (b) scripts/coeff_sweep.py (new): the raw-coefficient sweep. Needed because every nonlinear
#       warp hard-clamps its input (`x = alpha.float().clamp(0.0, 1.0)`), so coefficients outside
#       [0,1] are unreachable through alpha and `--alphas 1.5` silently reproduces alpha 1. The
#       script replaces the warp with the identity (which does not clamp) and sets gain 1 /
#       offset 0, so the requested value IS the coefficient. Separately labelled: no calibration
#       error is reported, since a coefficient of 1.25 has no correct percentile. It answers
#       whether the frozen direction can reach percentile 0 / 1 at all, and at what validity cost.
# Both run on `secondary` among others: a 4 h cap is ample for evaluation and keeps them out of the
# queue the 100-round runs are in.
#
# PART 2 -- the alpha=0 fix, as a training experiment: `lora.offset=true`.
# With an offset the coefficient is o_i + s*h(alpha), so at alpha=0 it is o_i rather than 0 and the
# model can apply the direction NEGATIVELY, below the private adapter's natural level. This is the
# only mechanism in the current code that can move that endpoint (`offset_max` is already 2.0, and
# per fed.py:303 the offset trains even with fix_gain=true). Everything else is identical to
# MOL-18, so PRIV_OFF - PRIV and LOCAL_OFF - LOCAL isolate the offset.
#
# The offset is PRIVATE, for two reasons, one forced and one measured:
#   * Forced: there is no flag to share only the offset. It sits in the "gain" parameter group
#     (lora.py:342), and `shared_params` takes `groups["gain"] + groups["warp"]` only when
#     `calibration == "shared"` -- so sharing the offset means sharing the gain and warp too, which
#     is the configuration MOL-14 identified as the confound and the harmful regime when support
#     widths differ (here 0.40-0.94).
#   * Measured: the alpha=0 endpoint sits in a strongly client-specific place. Achieved percentile
#     at alpha=0 ranges from 0.169 (CHEMBL204) to 0.536 (CHEMBL228) across clients, a spread of
#     0.367 with per-arm std ~0.10. One shared scalar would be correcting eight adapters that start
#     in very different places.
# ⚠️ No COV arm: `_check_coverage_config` rejects an offset outright ("lora.offset must be false"),
# because plan 2.1 defines coverage as gain 1 + no offset + private warp. Extending borrowing to
# the offset is a method change, not a flag, and is left as a follow-up -- a real one, since the
# clients needing the largest low-end correction (CHEMBL228: 12 training molecules below alpha
# 0.25) are exactly those with the least data to fit a private offset on.
set -u
cd /u/lucmon/rein
CFG=configs/chembl_deco_skew_pruned.yaml
DATA=data/chembl_deco_skew_pruned/data.jsonl
SC=clogp_residual_deco_strict

# ---------------------------------------------------------------- Part 1: diagnostics
COV=runs/exp46_cov_borrow_l1_20261006-154820_j11180420
PRIV=runs/exp46_cov_private_20261006-163645_j11180421
LOCAL=runs/exp46_cov_local_20261006-170041_j11180422
diag () {   # $1 = short name, $2 = run dir, $3 = dev-selected snapshot round (zero-padded)
  SNAP=$2/snapshots/round_$3.pt
  CMD="python \$CODE/eval_direction.py --run $2 --snapshot $SNAP --scorer $SC --split test \
--posthoc_remap --remap_split dev --remap_grid 11 --remap_prompts 50 \
--max_prompts 150 --max_new_tokens 96 --suffix remapdiag
python \$CODE/scripts/coeff_sweep.py --run $2 --snapshot $SNAP --scorer $SC --split test \
--coeffs -0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5 --max_prompts 60 --max_new_tokens 96"
  sbatch --parsable --job-name=exp47_diag_$1 --export=ALL,CMD="$CMD" sbatch/run_cmd_chembl.sbatch
}
D1=$(diag cov   $COV   0100)
D2=$(diag priv  $PRIV  0080)
D3=$(diag local $LOCAL 0100)
echo "diagnostics: cov $D1  priv $D2  local $D3"

# ---------------------------------------------------------------- Part 2: offset arms
ENVC="CONFIG=$CFG,GPU_LOG=1,DATA=$DATA,SCORER=$SC,RESCORE=clogp_residual_deco,SELECT=pct_calib_err_penalized,TEST_PROMPTS=150"
COMMON="lora.offset=true lora.warp=kumaraswamy_mix fed.fix_gain=true fed.warp_reg=0"
cc () { sbatch --parsable --job-name=$1 --time=24:00:00 \
    --export=ALL,$ENVC,OVERRIDES="$COMMON out_dir=runs/$1 $2" sbatch/train_eval_chembl.sbatch; }
P_CC=$(cc exp47_off_private "fed.calibration=private")
L_CC=$(cc exp47_off_local   "fed.mode=local fed.calibration=private")
echo "cc    PRIV_OFF $P_CC  LOCAL_OFF $L_CC"

A="ssh -o BatchMode=yes -l x-zchen17 anvil.rcac.purdue.edu"
$A "cd /home/x-zchen17/lucmon/rein
E=\"$ENVC\"; C=\"$COMMON\"
s () { sbatch --parsable --job-name=\$1_anvil --time=20:00:00 \
  --export=ALL,\$E,OVERRIDES=\"\$C out_dir=runs/\$1_anvil \$2\" sbatch/train_eval_chembl_anvil.sbatch; }
P=\$(s exp47_off_private 'fed.calibration=private')
L=\$(s exp47_off_local   'fed.mode=local fed.calibration=private')
echo \"anvil PRIV_OFF \$P  LOCAL_OFF \$L\"
squeue -A cis260796-ai -o \"%.10i %.30j %.9T %R\""
