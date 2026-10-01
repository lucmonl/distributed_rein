#!/bin/bash
# Experiment log entry 21 (2026-10-01): evaluation settings E2 (held-out clients) and E3 (drift).
# Smoke first; real jobs only if it succeeds (afterok).
#   E2 on the method run (federated, shared calibration, no offset; 11063910, round 100):
#      frozen_k0 (zero-shot), frozen_cal_k16/k64, baselines localdir_k16/k64 and prompt_k16/k64.
#   E2 on the private-calibration federated run (11041282, round 100): k=0 = identity calibration.
#      Waits for the first E2 job to reuse the cached held-out adapters (runs/_e2_adapters/).
#   E3 on the method run and on its local counterpart (11063911, round 100): 3 drift stages
#      x 100 SFT steps (steering off), evaluated as is and after a k=16 calibration refit.
cd /u/lucmon/rein
FS=runs/exp17_fed_calshared_nooff_cap4k_20261001-001022_j11063910
LN=runs/exp17_local_nooff_cap4k_20261001-013643_j11063911
FP=runs/exp11_cap4k_base_20260930-114746_j11041282
SR=runs/smoke_pipeline_20260930-111641_j11039406
sub () { sbatch --parsable "$@" sbatch/eval_settings.sbatch; }
S=$(sub --job-name=exp21_smoke --time=01:30:00 --export=ALL,STEPS=e2+e3,SMOKE=1,RUN=$SR,SNAP=$SR/snapshots/round_0002.pt)
E2=$(sub --job-name=exp21_e2_fed_shared --dependency=afterok:$S --export=ALL,STEPS=e2,RUN=$FS,SNAP=$FS/snapshots/round_0100.pt)
E3F=$(sub --job-name=exp21_e3_fed_shared --dependency=afterok:$S --export=ALL,STEPS=e3,RUN=$FS,SNAP=$FS/snapshots/round_0100.pt)
E3L=$(sub --job-name=exp21_e3_local --dependency=afterok:$S --export=ALL,STEPS=e3,RUN=$LN,SNAP=$LN/snapshots/round_0100.pt)
E2P=$(sub --job-name=exp21_e2_fed_private --dependency=afterok:$S:$E2 --export=ALL,STEPS=e2,RUN=$FP,SNAP=$FP/snapshots/round_0100.pt)
echo "smoke $S  e2(fed shared) $E2  e3(fed shared) $E3F  e3(local) $E3L  e2(fed private) $E2P"

# --- Resubmission (2026-10-01 ~13:10): the smoke job 11082124 passed E2 and E3 but failed in quality
# scoring (score_quality.py only looked up participants' records; E2 evaluates held-out clients).
# Fixed and verified on the smoke's E2 output; the blocked jobs were cancelled and resubmitted
# without the smoke dependency (E2 on the private run still waits for the first E2 job).
# E2=$(sub --job-name=exp21_e2_fed_shared --export=ALL,STEPS=e2,RUN=$FS,SNAP=$FS/snapshots/round_0100.pt)
# E3F=$(sub --job-name=exp21_e3_fed_shared --export=ALL,STEPS=e3,RUN=$FS,SNAP=$FS/snapshots/round_0100.pt)
# E3L=$(sub --job-name=exp21_e3_local --export=ALL,STEPS=e3,RUN=$LN,SNAP=$LN/snapshots/round_0100.pt)
# E2P=$(sub --job-name=exp21_e2_fed_private --dependency=afterok:$E2 --export=ALL,STEPS=e2,RUN=$FP,SNAP=$FP/snapshots/round_0100.pt)
