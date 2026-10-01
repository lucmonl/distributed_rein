#!/bin/bash
# Experiment log entry 19 (2026-10-01): baselines B1 (prompting), B4 (federated CAA), B3 (one-shot merged).
#   B1/B4 use the client models of the federated 4k run (exp11_cap4k_base, dev-selected round 100),
#   with the learned direction removed. B3 merges the local-only runs' directions at their
#   dev-selected checkpoints: local 2k (round 90, entry 9) and local 4k (exp16, when it finishes).
# A smoke job runs first; the real jobs only start if it succeeds (afterok).
cd /u/lucmon/rein
FED=runs/exp11_cap4k_base_20260930-114746_j11041282
L2=runs/nr_global_local_20260930-001106_j11011539
L4=runs/exp16_local_cap4k_20260930-232203_j11061046
SMOKE_RUN=runs/smoke_pipeline_20260930-111641_j11039406
sub () { sbatch --parsable "$@" sbatch/eval_baselines.sbatch; }

S=$(sub --job-name=exp19_smoke --time=01:30:00 --export=ALL,STEPS=b1+b4+b3,SMOKE=1,FED_RUN=$SMOKE_RUN,FED_SNAP=$SMOKE_RUN/snapshots/round_0002.pt,LOCAL_RUN=$L2,LOCAL_SNAP=$L2/snapshots/round_0090.pt)
echo "smoke $S"
B1=$(sub --job-name=exp19_b1_prompt --dependency=afterok:$S --export=ALL,STEPS=b1,FED_RUN=$FED,FED_SNAP=$FED/snapshots/round_0100.pt)
B4=$(sub --job-name=exp19_b4_caa    --dependency=afterok:$S --export=ALL,STEPS=b4,FED_RUN=$FED,FED_SNAP=$FED/snapshots/round_0100.pt)
B32=$(sub --job-name=exp19_b3_local2k --dependency=afterok:$S --export=ALL,STEPS=b3,LOCAL_RUN=$L2,LOCAL_SNAP=$L2/snapshots/round_0090.pt)
B34=$(sub --job-name=exp19_b3_local4k --dependency=afterok:$S:11061046 --export=ALL,STEPS=b3,LOCAL_RUN=$L4,LOCAL_SNAP=best)
echo "b1 $B1  b4 $B4  b3(2k) $B32  b3(4k) $B34"
