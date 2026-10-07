#!/bin/bash
# Experiment log NR-46 (2026-10-05): coverage-aware local nonlinear calibration (plan section 2.1)
# on Newsroom extractiveness, with the matched gain-1 controls the plan asks for.
#
# Setting = the calibration-design runs of entries 38-41 (Llama-3.2-1B, rotation 0, 4k per
# client, 100 rounds, standard pipeline), changed as section 2.1 prescribes for ALL arms:
#   gain fixed at 1 (fed.fix_gain=true), no offset, kumaraswamy_mix warp, identity penalty off
#   (fed.warp_reg=0), warp warm-up 5 rounds (config default), uniform FedAvg on D.
# Arms (only the calibration differs):
#   A    shared nonlinear warp, gain 1              fed.calibration=shared
#   B    private nonlinear warps, gain 1, no borrow  fed.calibration=private
#   C1   coverage-aware borrowing, lambda_max = 1    fed.calibration=coverage
#   C10  coverage-aware borrowing, lambda_max = 10   fed.calibration=coverage
# Coverage settings: K = 11, b = 0.2, tau_local = tau_peer = 100 (smoothed counts; with 4k
# examples per client the in-support counts are 200-2300, the gaps 5-50). lambda_max is
# untuned, hence two values; choose between them on dev (the in-training monitor), never test.
# The old learned-gain runs (11063910 shared, 11145309 private) are historical references,
# not matched ablations.
#
# Cluster: news workstream -> cc (memory `reference-slurm`). scavenger widens the pool; its
# 16 GB V100 nodes are excluded; --time must fit scavenger's 24 h cap.
set -u
cd /u/lucmon/rein
COMMON="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 max_train_per_client=4000 lora.offset=false lora.warp=kumaraswamy_mix fed.fix_gain=true fed.warp_reg=0"
submit () {
  sbatch --parsable --job-name=$1 --time=24:00:00 \
    --partition=dali,IllinoisComputes-GPU,scavenger --exclude=ccc0387,ccc0089,ccc0090,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236 \
    --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="$COMMON out_dir=runs/$1 $2" \
    sbatch/train_eval.sbatch
}
A=$(submit nr46_cal_shared_g1     "fed.calibration=shared")
B=$(submit nr46_cal_private_g1    "fed.calibration=private")
C1=$(submit nr46_cal_coverage_l1  "fed.calibration=coverage fed.cov_lambda_max=1")
C10=$(submit nr46_cal_coverage_l10 "fed.calibration=coverage fed.cov_lambda_max=10")
echo "A $A  B $B  C1 $C1  C10 $C10"

# --- Anvil copies (added 2026-10-06 00:5x): the news workstream races cc against Anvil
# (memory `reference-slurm`, `feedback-cluster-failover`). Needed a one-time Anvil setup for
# Newsroom: data/newsroom_fed, HF weights (Llama-3.2-1B-Instruct, roberta-large, AlignScore,
# judge Qwen2.5-7B-Instruct) copied from cc's cache, and sbatch/train_eval_anvil.sbatch.
# -A cis260796-ai can only use -p ai. The race watcher cancels the losing host's copies once
# one copy has RUNNING >= 20 min; the four arms form ONE group so they all run on one host.
A="ssh -o BatchMode=yes -l x-zchen17 anvil.rcac.purdue.edu"
$A "cd /home/x-zchen17/lucmon/rein
C=\"$COMMON\"
s () { sbatch --parsable --job-name=\$1_anvil --time=24:00:00 \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES=\"\$C out_dir=runs/\$1_anvil \$2\" \
  sbatch/train_eval_anvil.sbatch; }
A=\$(s nr46_cal_shared_g1 'fed.calibration=shared')
B=\$(s nr46_cal_private_g1 'fed.calibration=private')
C1=\$(s nr46_cal_coverage_l1 'fed.calibration=coverage fed.cov_lambda_max=1')
C10=\$(s nr46_cal_coverage_l10 'fed.calibration=coverage fed.cov_lambda_max=10')
echo \"anvil A \$A  B \$B  C1 \$C1  C10 \$C10\"
squeue -A cis260796-ai -o \"%.10i %.28j %.9T %R\""
