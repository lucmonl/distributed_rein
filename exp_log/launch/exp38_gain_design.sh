#!/bin/bash
# Experiment log entry 38 (2026-10-03): design of the calibration g(alpha) = s * h(alpha) (no offset).
# Reference = the method run 11063910 (federated, shared calibration, learnable gain s + warp h
# kumaraswamy_mix, no offset, 4k per client, 100 rounds, Llama-3.2-1B).  Each arm changes only g:
#   const   g(alpha) = alpha             lora.warp=none, fed.fix_gain=true   (Q1: learnable non-linear vs constant)
#   linear  g(alpha) = s * alpha         lora.warp=none, s learnable, shared (Q1: splits scale from non-linearity)
#   private g_i = s_i * h_i(alpha)       fed.calibration=private             (Q2: shared vs per-client)
# Federated per-client gains stay well inside the clamp [1/4, 4] (0.97-2.2 in 11041282), so gain_max stays 4.
# Run directories in project space (home quota), linked into runs/.
cd /u/lucmon/rein
RR=/u/lucmon/lucmon/rein_runs
COMMON="model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 max_train_per_client=4000 lora.offset=false"
submit () {
  sbatch --parsable --job-name=$1 --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,RUNS_ROOT=$RR,OVERRIDES="$COMMON out_dir=$RR/$1 $2" \
    sbatch/train_eval.sbatch
}
A=$(submit exp38_fed_gain_const_cap4k   "fed.calibration=shared lora.warp=none fed.fix_gain=true")
B=$(submit exp38_fed_gain_linear_cap4k  "fed.calibration=shared lora.warp=none")
C=$(submit exp38_fed_calpriv_nooff_cap4k "fed.calibration=private")
echo "const $A  linear $B  private $C"
