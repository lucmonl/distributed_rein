#!/bin/bash
# Experiment log NR-56 (2026-10-06): consensus calibration (user design, after NR-53).
#   training : each client keeps its own warps g_{i,l} (per layer, lora.warp_scope=module) and is
#              tied to the server table g-bar ONLY where it has data:
#              lambda_ik = lambda_max * c_ik / (tau + c_ik), no penalty outside its support
#   server   : g-bar pooled per layer from the clients' warp values with SATURATING weights
#              c/(tau_pool + c) (one dense client cannot dominate), then monotone projection
#   inference: every client uses g-bar (linear interpolation of the table), never its own warp;
#              the in-training monitor (dev selection) and the test evaluation both use g-bar
# Finer grid K = 21 (user); b = 0.2, tau_local = tau_pool = 100. Same setting as NR-54 otherwise
# (Llama-3.2-1B, rotation 0, 4k per client, 100 rounds, gain 1, no offset, warp_reg 0).
# Compared with NR-54's per-layer A (shared) and B (private). Raced on cc + Anvil, in the same
# race group as NR-54 so all per-layer arms run on one host.
set -u
cd /u/lucmon/rein
COMMON="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 max_train_per_client=4000 lora.offset=false lora.warp=kumaraswamy_mix lora.warp_scope=module fed.fix_gain=true fed.warp_reg=0 fed.calibration=consensus fed.cov_grid=21 fed.cov_pool=saturating"
ARMS=(
  "nr56_consensus_l0p1|fed.cov_lambda_max=0.1"
  "nr56_consensus_l1|fed.cov_lambda_max=1"
)
for a in "${ARMS[@]}"; do
  N=${a%%|*}; O=${a#*|}
  J=$(sbatch --parsable --job-name=$N --time=24:00:00 \
    --partition=dali,IllinoisComputes-GPU,scavenger --exclude=ccc0387,ccc0089,ccc0090,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236 \
    --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="$COMMON out_dir=runs/$N $O" \
    sbatch/train_eval.sbatch)
  echo "cc $N $J"
done
for a in "${ARMS[@]}"; do
  N=${a%%|*}; O=${a#*|}
  J=$(ssh -o BatchMode=yes -l x-zchen17 anvil.rcac.purdue.edu "cd /home/x-zchen17/lucmon/rein
sbatch --parsable --job-name=${N}_anvil --time=24:00:00 \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES=\"$COMMON out_dir=runs/${N}_anvil $O\" \
  sbatch/train_eval_anvil.sbatch")
  echo "anvil $N $J"
done
