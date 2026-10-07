#!/bin/bash
# Experiment log NR-49 (2026-10-06): smaller borrowing strengths for coverage calibration.
# NR-48 found lambda_max = 1 decisive for nypost.com (the borrowing loss is only ~1e-4 per step),
# so the sweep extends downward: lambda_max in {0.01, 0.1}. With the running 1 and 10 (NR-46)
# that is a full decade grid. Everything else is identical to NR-46 arm C (same overrides; the
# code is byte-identical to C1's snapshot runs/_code/20261006-082124_j11174460).
# Raced on cc + Anvil (memory `feedback-cluster-failover`); watcher spec NR-49_race.json, the two
# arms are one group so they land on one host.
set -u
cd /u/lucmon/rein
COMMON="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 max_train_per_client=4000 lora.offset=false lora.warp=kumaraswamy_mix fed.fix_gain=true fed.warp_reg=0 fed.calibration=coverage"

# --- cc: dali, IllinoisComputes-GPU, scavenger (16 GB V100 nodes excluded; 24 h cap)
submit () {
  sbatch --parsable --job-name=$1 --time=24:00:00 \
    --partition=dali,IllinoisComputes-GPU,scavenger --exclude=ccc0387,ccc0089,ccc0090,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236 \
    --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="$COMMON out_dir=runs/$1 $2" \
    sbatch/train_eval.sbatch
}
L01=$(submit nr46_cal_coverage_l0p1  "fed.cov_lambda_max=0.1")
L001=$(submit nr46_cal_coverage_l0p01 "fed.cov_lambda_max=0.01")
echo "cc     l0.1 $L01  l0.01 $L001"

# --- Anvil (-A cis260796-ai, -p ai only)
A="ssh -o BatchMode=yes -l x-zchen17 anvil.rcac.purdue.edu"
$A "cd /home/x-zchen17/lucmon/rein
C=\"$COMMON\"
s () { sbatch --parsable --job-name=\$1_anvil --time=24:00:00 \
  --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES=\"\$C out_dir=runs/\$1_anvil \$2\" \
  sbatch/train_eval_anvil.sbatch; }
L01=\$(s nr46_cal_coverage_l0p1 'fed.cov_lambda_max=0.1')
L001=\$(s nr46_cal_coverage_l0p01 'fed.cov_lambda_max=0.01')
echo \"anvil  l0.1 \$L01  l0.01 \$L001\""
