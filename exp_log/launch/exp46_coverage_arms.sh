#!/bin/bash
# Experiment log MOL-18 (2026-10-06): the experiment asked for by
# chembl-experiment-plan.md "2026-10-06 - What coverage calibration can diagnose or repair":
# "compare matched gain-1, no-offset joint-training arms with/without coverage borrowing and
# local-only, controlling shared-optimizer-state resets."
#
# This is PROSPECTIVE. That entry is explicit that coverage calibration is a new hypothesis and
# must be tested against local, not assumed to explain away Exp45's negative result. Nothing in
# the scoring definition is changed to favour any arm.
#
# All four arms are identical except for the calibration / optimizer line, and all follow
# federated-steering-plan.md 2.1 the way NR-46 did on Newsroom:
#   gain fixed at 1 (fed.fix_gain=true, so u=0 and s=exp(0)=1), no offset,
#   kumaraswamy_mix warp, identity penalty off (fed.warp_reg=0), full participation.
#
#   COV      fedavg + coverage borrowing, lambda_max = 1   fed.calibration=coverage
#   PRIV     fedavg + private warps, no borrowing          fed.calibration=private
#   LOCAL    local-only, private warps                     fed.mode=local
#   NORESET  fedavg + private warps, Adam state kept       fed.reset_shared_opt_state=false
#
# Why NORESET: `reset_shared_opt_state=True` drops the direction's Adam state every round, but
# only in fedavg mode (fed.py:270 `mode == "fedavg" and ...`), so every fed-vs-local comparison so
# far has confounded sharing with an optimizer-state reset. local mode cannot be made to reset
# without a code change, so the control runs in the other direction: a fedavg arm that keeps it.
# PRIV vs NORESET measures that factor alone; COV vs PRIV measures the borrowing alone.
#
# Data: data/chembl_deco_skew_pruned (Exp45's cleaned single-parent-ligand dataset), 100 rounds,
# rotation 0, seed 0, strict scorer throughout, dev selection on pct_calib_err_penalized -- all
# as in Exp45, so these arms are comparable with each other. They are NOT comparable with Exp45's
# own numbers, which used a LEARNED gain (0.606-0.956 FED, 1.191-1.544 LOCAL); here gain is 1.
#
# Measured pre-flight (login node, free): borrowing is in an active regime on this dataset and
# targets the designed gaps -- lambda reaches .77-.89 on CHEMBL243's missing high side, .71-.93 on
# CHEMBL228's missing low side, and is high at BOTH ends for the middle-only CHEMBL4078
# (.92/.83 low, .59/.76 high), while the broad donors CHEMBL240/CHEMBL204 stay at <=.34.
# No grid point has zero total evidence. lambda_max=1 is the value NR-48 found decisive on
# Newsroom; it is untuned for ChEMBL, and NR-49's sweep should inform any follow-up here.
#
# Cluster rule (memory `reference-slurm`): molecule workstream -> cc + Anvil. The four arms are
# ONE race group, so they all land on the same host: cc has rdkit 2025.09.6 and Anvil 2026.03.6,
# and a split would confound the arms with the scorer's version.
set -u
cd /u/lucmon/rein
CFG=configs/chembl_deco_skew_pruned.yaml
DATA=data/chembl_deco_skew_pruned/data.jsonl
ENVC="CONFIG=$CFG,GPU_LOG=1,DATA=$DATA,SCORER=clogp_residual_deco_strict,RESCORE=clogp_residual_deco,SELECT=pct_calib_err_penalized,TEST_PROMPTS=150"
COMMON="lora.offset=false lora.warp=kumaraswamy_mix fed.fix_gain=true fed.warp_reg=0"

# --- cc copies
cc () {  # $1 = run name, $2 = arm overrides
  sbatch --parsable --job-name=$1 --time=24:00:00 \
    --export=ALL,$ENVC,OVERRIDES="$COMMON out_dir=runs/$1 $2" \
    sbatch/train_eval_chembl.sbatch
}
C_COV=$(cc   exp46_cov_borrow_l1       "fed.calibration=coverage fed.cov_lambda_max=1")
C_PRIV=$(cc  exp46_cov_private         "fed.calibration=private")
C_LOC=$(cc   exp46_cov_local           "fed.mode=local fed.calibration=private")
C_NOR=$(cc   exp46_cov_private_noreset "fed.calibration=private fed.reset_shared_opt_state=false")
echo "cc    COV $C_COV  PRIV $C_PRIV  LOCAL $C_LOC  NORESET $C_NOR"

# --- Anvil copies (-A cis260796-ai can only use -p ai; no multi-partition requests)
A="ssh -o BatchMode=yes -l x-zchen17 anvil.rcac.purdue.edu"
$A "cd /home/x-zchen17/lucmon/rein
E=\"$ENVC\"; C=\"$COMMON\"
s () { sbatch --parsable --job-name=\$1_anvil --time=20:00:00 \
  --export=ALL,\$E,OVERRIDES=\"\$C out_dir=runs/\$1_anvil \$2\" \
  sbatch/train_eval_chembl_anvil.sbatch; }
COV=\$(s   exp46_cov_borrow_l1       'fed.calibration=coverage fed.cov_lambda_max=1')
PRIV=\$(s  exp46_cov_private         'fed.calibration=private')
LOC=\$(s   exp46_cov_local           'fed.mode=local fed.calibration=private')
NOR=\$(s   exp46_cov_private_noreset 'fed.calibration=private fed.reset_shared_opt_state=false')
echo \"anvil COV \$COV  PRIV \$PRIV  LOCAL \$LOC  NORESET \$NOR\"
squeue -A cis260796-ai -o \"%.10i %.30j %.9T %R\""
