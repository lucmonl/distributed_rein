#!/bin/bash
# Experiment log NR-60, added arm (user, 2026-10-07 22:00): fed-shared-soff = shared shape AND shared offset.
# g_{i,l}(alpha) = o + h_l(alpha): one offset o for all clients, FedAvg'd every round with the shape
# (calibration=shared, lora.offset=true, fed.private_offset=false = the code's default shared path).
# Contrast: fed-shared-soff vs fed-shared-off (only how o is shared differs) and vs fed-shared (NR-54 A_L).
# NR-60 common settings; raced on cc + Anvil as its own race (the NR60 group was already decided).
set -u
cd /u/lucmon/rein
COMMON="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 max_train_per_client=4000 lora.warp=kumaraswamy_mix lora.warp_scope=module fed.fix_gain=true fed.warp_reg=0"
ARM="fed.calibration=shared lora.offset=true fed.private_offset=false"
N=nr60-fed-shared-soff
CC=$(sbatch --parsable --job-name=$N --time=24:00:00 \
  --partition=dali,IllinoisComputes-GPU,scavenger --exclude=ccc0387,ccc0089,ccc0090,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236 \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="$COMMON out_dir=runs/$N $ARM" \
  sbatch/train_eval.sbatch)
AN=$(ssh -o BatchMode=yes -l x-zchen17 anvil.rcac.purdue.edu "cd /home/x-zchen17/lucmon/rein
sbatch --parsable --job-name=$N --time=24:00:00 \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES=\"$COMMON out_dir=runs/$N $ARM\" \
  sbatch/train_eval_anvil.sbatch")
echo "cc $CC anvil $AN"
