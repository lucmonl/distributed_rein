#!/bin/bash
# Experiment log MOL-17 (2026-10-05): matched federated-vs-local pair on the PRUNED skew dataset.
#
# Why: MOL-16 found Exp43's outputs could move cLogP with *unattached* Cl/[Br-] components, i.e.
# without changing the attached ligand at all. `data/chembl_deco_skew_pruned/` removes those
# records (28,732 of 30,114 retained; 1,382 dropped) and the strict scorer enforces the
# single-parent-ligand contract. MOL-16's reversal under the strict score was post-hoc on old
# generations; this is the matched training run that can actually test it.
#
# Design, per MOL-14's lesson: BOTH arms use fed.calibration=private, so only the direction
# varies. (fed.py rejects local+shared, so private is the one legal matched setting.)
# Monitoring, checkpoint selection and the test evaluation all use the strict contract:
#   monitor.scorer = clogp_residual_deco_strict   (in the config)
#   SELECT         = pct_calib_err_penalized      (counts unscorable cells as error 1
#                                                  instead of dropping them from the mean)
#   SCORER         = clogp_residual_deco_strict   (test evaluation)
#   RESCORE        = clogp_residual_deco          (legacy score on the SAME generations, so the
#                                                  strict-vs-legacy gap is measured, not assumed)
#
# Cluster rule (memory `reference-slurm`): molecule workstream runs on cc + Anvil only. Each of
# the two jobs is duplicated across both, and the loser is cancelled only once the winner is
# RUNNING *and past startup* (model loaded, round 1 logged) -- never to resubmit (`feedback-cluster-failover`).
# The Anvil duplicate of MOL-15 (21092869) was cancelled first: its cc copy 11165980 had been
# running 5 h 43 m, and the pending duplicate competed with these jobs for the same `ai` allocation.
#
# Caveat for the eventual table: cc has rdkit 2025.09.6, Anvil 2026.03.6 (torch 2.5.1 on both,
# cu118 vs cu124). Both reproduce every stored label in the pruned dataset to <1e-6, so the race
# winner's identity does not change the attribute definition -- but note which host produced a
# number before putting cc and Anvil runs in one row.
set -u
cd /u/lucmon/rein
CFG=configs/chembl_deco_skew_pruned.yaml
DATA=data/chembl_deco_skew_pruned/data.jsonl
ENVCOMMON="CONFIG=$CFG,GPU_LOG=1,DATA=$DATA,SCORER=clogp_residual_deco_strict,RESCORE=clogp_residual_deco,SELECT=pct_calib_err_penalized,TEST_PROMPTS=150"

# --- cc copies. runs/ is already a symlink into project space (entry 39 correction), so no
# RUNS_ROOT is needed. --time=24:00:00 because `scavenger` is in the partition list (MaxTime 1 d).
F_CC=$(sbatch --parsable --job-name=exp45_pruned_fed_cc --time=24:00:00 \
  --export=ALL,$ENVCOMMON,OVERRIDES="out_dir=runs/exp45_pruned_fed_cc" \
  sbatch/train_eval_chembl.sbatch)
L_CC=$(sbatch --parsable --job-name=exp45_pruned_local_cc --time=24:00:00 \
  --export=ALL,$ENVCOMMON,OVERRIDES="fed.mode=local out_dir=runs/exp45_pruned_local_cc" \
  sbatch/train_eval_chembl.sbatch)
echo "cc    fed $F_CC  local $L_CC"

# --- Anvil copies (-A cis260796-ai can only use -p ai; no multi-partition requests there)
A="ssh -o BatchMode=yes -l x-zchen17 anvil.rcac.purdue.edu"
$A "cd /home/x-zchen17/lucmon/rein
E=\"$ENVCOMMON\"
F=\$(sbatch --parsable --job-name=exp45_pruned_fed_anvil --time=20:00:00 \
  --export=ALL,\$E,OVERRIDES=\"out_dir=runs/exp45_pruned_fed_anvil\" \
  sbatch/train_eval_chembl_anvil.sbatch)
L=\$(sbatch --parsable --job-name=exp45_pruned_local_anvil --time=20:00:00 \
  --export=ALL,\$E,OVERRIDES=\"fed.mode=local out_dir=runs/exp45_pruned_local_anvil\" \
  sbatch/train_eval_chembl_anvil.sbatch)
echo \"anvil fed \$F  local \$L\"
squeue -A cis260796-ai -o \"%.10i %.28j %.9T %R\""
