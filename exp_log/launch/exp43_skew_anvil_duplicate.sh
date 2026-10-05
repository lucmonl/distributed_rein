#!/bin/bash
# Experiment log entry MOL-13 (2026-10-04): DUPLICATE the exp42 natural-skew pair onto Anvil, so
# the same experiment races on two clusters and whichever gets scheduled first wins.
# cc copies: 11156942 (fed) / 11156950 (local). Anvil copies submitted here.
# The Anvil smoke test (21067708) is cancelled first: it was never started, and it competes with
# these real jobs for the same `ai` allocation. The real run's own first rounds validate the stack,
# and a stack failure shows up within minutes at ~0 SU.
# Cancel the loser only once the winner is RUNNING *and past startup* (model loaded, round 1
# logged) -- not merely RUNNING, because the Anvil stack is newly built.
cd /u/lucmon/rein
A="ssh -o BatchMode=yes -l x-zchen17 anvil.rcac.purdue.edu"
$A 'scancel 21067708 2>/dev/null; cd /home/x-zchen17/lucmon/rein
D="data_path=data/chembl_deco_skew/data.jsonl clients_file=data/chembl_deco_skew/clients.json"
EVAL="monitor.full_prompts=25 monitor.full_max_new_tokens=96 monitor.max_new_tokens=96 fed.save_every=20 monitor.full_every=20"
C="CONFIG=configs/chembl_deco.yaml,GPU_LOG=1,DATA=data/chembl_deco_skew/data.jsonl,RESCORE=none,SCORER=clogp_residual_deco,TEST_PROMPTS=150"
F=$(sbatch --parsable --job-name=exp43_skew_fed_anvil --time=20:00:00 \
  --export=ALL,$C,OVERRIDES="$D fed.rounds=100 fed.calibration=shared $EVAL out_dir=runs/exp43_skew_fed_anvil" \
  sbatch/train_eval_chembl_anvil.sbatch)
L=$(sbatch --parsable --job-name=exp43_skew_local_anvil --time=20:00:00 \
  --export=ALL,$C,OVERRIDES="$D fed.rounds=100 fed.mode=local fed.calibration=private $EVAL out_dir=runs/exp43_skew_local_anvil" \
  sbatch/train_eval_chembl_anvil.sbatch)
echo "anvil fed $F  anvil local $L"
squeue -A cis260796-ai -h -o "%.10i %.24j %.9T %R"'
