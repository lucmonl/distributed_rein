#!/bin/bash
# Experiment log MOL-24 (2026-10-07). Spec from the coordinator session (CAL-4).
#
# Question: on ChEMBL, does a private offset help under a SHARED shape, at gain 1, with per-layer
# shapes? ChEMBL has never had a shared-shape run at gain 1 (CAL-1's "missing" list), and it has
# never had a per-layer local arm with an offset to compare against.
#
# Design decision by the user (2026-10-07), overriding CAL-3/CAL-4: the per-round reset of the
# direction's Adam state is PART OF federated training, not a confound to control away -- FedAvg
# overwrites the client's shared parameters with the server average every round, so carrying stale
# Adam moments across that overwrite is the anomaly, not resetting them. fed (with its reset) vs
# local (as local training is naturally implemented) is therefore already the fair comparison, and
# the `local-off-reset` control from CAL-3 is dropped. MOL-23's finding stands as a statement about
# WHY FedAvg helps on this task (it buys reach), not as a confound.
#
#   mol24-fed-shared        reference calibration: gain 1, per-layer shape, shared shape, no offset
#   mol24-fed-shared-off    + private offset under the shared shape (needs NR-60's fed.private_offset)
#   mol24-local-off        local, private per-layer shape, private offset, no reset
#
# Matched to MOL-18/MOL-22 in everything else: same config file, 100 rounds, rotation 0, seed 0,
# strict scorer at every stage, dev selection on pct_calib_err_penalized, fedavg keeps its
# per-round reset. MOL-22's runs used ONE shape per client, so they are context, not references.
#
# CONVENTIONS.md 3: the reference overrides are passed LITERALLY, because "default" there means the
# reference design and NOT the code/config defaults -- the code defaults to lora.warp_scope=model
# (one shape per client) and this config ships fed.warp_reg=0.01.
#
# ONE RACE GROUP, all three arms together (MOL-18's rule): cc has rdkit 2025.09.6 and Anvil
# 2026.03.6, so splitting the arms across hosts would confound the arms with the scorer version.
# The group means whichever host starts ANY member first gets ALL three.
#
# Anvil budget: ~6 SU per 100-round arm (archive: ~12 SU per fed/local pair), so ~18 SU of the
# 25 SU the coordinator allocated, leaving room for one retry. Balance before launch: 473.1/500.
#
# Pre-flight done before this script ran: 74/74 shared tests pass on cc; all three override sets
# resolved against FedConfig/SteerLoraConfig/MonitorConfig; all three run end to end through this
# same production sbatch on 2 clients x 3 rounds (MOL-24_smoke.sh); Anvil code md5-identical on 27
# shared files.
set -u
cd /u/lucmon/rein
CFG=configs/chembl_deco_skew_pruned.yaml
DATA=data/chembl_deco_skew_pruned/data.jsonl
ENVC="CONFIG=$CFG,GPU_LOG=1,DATA=$DATA,SCORER=clogp_residual_deco_strict,RESCORE=clogp_residual_deco,SELECT=pct_calib_err_penalized,TEST_PROMPTS=150"
COMMON="lora.warp=kumaraswamy_mix lora.warp_scope=module fed.fix_gain=true fed.warp_reg=0 lora.offset=false fed.calibration=shared"
OFF="lora.offset=true fed.private_offset=true"
LOC="fed.mode=local fed.calibration=private lora.offset=true"

# --- cc copies (same job name on both clusters; the job id tells the copies apart, CONVENTIONS 4)
cc () { sbatch --parsable --job-name=$1 --time=24:00:00 \
    --export=ALL,$ENVC,OVERRIDES="$COMMON out_dir=runs/$1 $2" \
    sbatch/train_eval_chembl.sbatch; }
C_FS=$(cc  mol24-fed-shared       "")
C_FO=$(cc  mol24-fed-shared-off   "$OFF")
C_LO=$(cc  mol24-local-off  "$LOC")
echo "cc     fed-shared $C_FS  fed-shared-off $C_FO  local-off $C_LO"

# --- Anvil copies: GATED OFF by default (CAL-11, 2026-10-08)
# /anvil/scratch/x-zchen17/lucmon/envs/rein lost stdlib files to what looks like a scratch purge,
# and every Anvil job since 03:09 dies in seconds with
#   cannot import name '_parser' from partially initialized module 're'
# Submitting there would burn queue slots on jobs that cannot run. The race is safe to watch even
# so -- scripts/race_watch.py requires RUNNING past --min_running_s or COMPLETED to declare a
# winner (lines 103-111; the FAILED-containing STARTED set at line 31 is dead code), so a failing
# Anvil copy cannot cancel the cc copies -- but there is no reason to create them.
# Re-enable with ANVIL=1 ONLY after the env is rebuilt and a short job completes there.
# The rebuild is the user's call; do not touch the Anvil environment.
ANVIL=${ANVIL:-0}
if [ "$ANVIL" != "1" ]; then
  echo "anvil  SKIPPED (env broken, CAL-11). cc-only race, 0 SU. Re-run with ANVIL=1 once fixed."
  exit 0
fi

# --- Anvil copies (-A cis260796-ai can only use -p ai; no multi-partition requests)
A="ssh -o BatchMode=yes -l x-zchen17 anvil.rcac.purdue.edu"
$A "cd /home/x-zchen17/lucmon/rein
E=\"$ENVC\"; C=\"$COMMON\"
s () { sbatch --parsable --job-name=\$1 --time=20:00:00 \
  --export=ALL,\$E,OVERRIDES=\"\$C out_dir=runs/\$1 \$2\" \
  sbatch/train_eval_chembl_anvil.sbatch; }
FS=\$(s mol24-fed-shared      '')
FO=\$(s mol24-fed-shared-off  '$OFF')
LO=\$(s mol24-local-off '$LOC')
echo \"anvil  fed-shared \$FS  fed-shared-off \$FO  local-off \$LO\"
squeue -A cis260796-ai -o \"%.10i %.24j %.9T %R\""
