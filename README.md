# fedsteer: federated learning of steering directions

Implements the plan in `federated-steering-plan.md`.

Per client *i*, every targeted linear layer is

    y = W0 x + B_p,i A_p,i x + s_i · α · B_d A_d x

- `A_p, B_p`: private adapter (never leaves the client)
- `s_i = exp(u_i)`: private gain, clamped to [1/4, 4]
- `A_d`: shared, frozen, generated from a common seed. Because A_d never changes, averaging `B_d` averages the direction products exactly.
- `B_d`: the shared direction, the only thing aggregated
- `α`: a percentile of the attribute score, under one of two protocols (`alpha_mode`):
  - **`global` (the method):** the percentile under the equal-weight mixture of the participants' score distributions. The server builds it by averaging per-client normalized histograms, so no examples are shared. α then means the same behaviour on every client, and each client's data covers only part of the α axis (its *support*).
  - **`local` (ablation):** each client's own percentile. The same α can mean different behaviour on different clients.
- Optional private **offset** (`lora.offset`): the coefficient becomes `o_i + s_i · h_i(α)`, so a client's α = 0 can start partway along the direction.
- **Adapter mode** (`fed.adapter`): `private` (the method: P_i stays on the client), `shared` (P is aggregated too; formerly `share_private`), or `none` (no task adapter; clients have only the gain, offset and warp scalars).

**Private α warp (optional; `lora.warp`).** The coefficient `s_i · α` becomes `s_i · h_i(α)`, where h_i is a private increasing map with h(0) = 0 and h(1) = 1 ([fedsteer/warp.py](fedsteer/warp.py)). It lets each client place where its behaviour changes fastest; for Newsroom, that's where rewriting switches to copying the lead sentence.

| `warp` | h(α) | Private params |
|---|---|---|
| `none` | α (the original linear model) | 0 |
| `kumaraswamy` (A) | 1 − (1 − α^p)^q | 2 |
| `kumaraswamy_mix` (B, the method) | (1 − w)·α + w·[1 − (1 − α^p)^q] | 3 |
| `step` (C) | (1 − w)·α + w·S((α − c)/τ), c = switch point | 3 |

- Every warp starts as the identity, so it doesn't change the model until it is trained.
- Training settings in `fed:`: `lr_warp`, `warp_warmup_rounds` (the warp stays the identity until D carries signal) and `warp_reg` (penalty toward the identity).
- Warp parameters are stored with the client's gain and are never aggregated.
- Option F is a post-hoc isotonic remap with no training change, used as an ablation: `eval_direction.py --posthoc_remap`. It fits the remap on dev and evaluates on test.

## Outputs and experiment log

- **Experiment log:** [`exp_log/EXPERIMENT_LOG.md`](exp_log/EXPERIMENT_LOG.md) records every change and job: job id, run folder, results.
- **Run folders:** a config's `out_dir` is a prefix. Each run writes to `runs/<prefix>_<YYYYmmdd-HHMMSS>[_j<jobid>]/`, and `run_info.json` records the job, host, command and git commit. Continue a run with `python train_fed.py --resume <run_dir>`.
- **Evaluations:** written to `<run>/evals/eval_…__<stamp>.json`. Nothing is overwritten.
- **Standard job:** `sbatch --export=ALL,CONFIG=configs/newsroom_fedavg.yaml,OVERRIDES="fed.rounds=30" sbatch/train_eval.sbatch` trains, sweeps the snapshots on dev, selects a checkpoint and evaluates it on test.

## Layout

| Path | What |
|---|---|
| `fedsteer/lora.py` | `SteerLinear`, `inject_steer_lora`, state partitioning (shared / private / gain) |
| `fedsteer/data.py` | JSONL client data, client quantiles (α labels and calibration targets), chat formatting with the loss masked to the answer |
| `fedsteer/fed.py` | `FedSteerTrainer`: modes `fedavg` (proposed) and `local` (baseline B2, which also writes the B3 merged direction); ablations `fix_gain` (A1) and `share_private` (A2); checkpoint and resume |
| `fedsteer/model.py` | Model loading and `generate_at_alpha` |
| `train_fed.py` | Training entry point (YAML config plus `--set key=value` overrides) |
| `eval_direction.py` | Direction quality per client: order rate, Spearman, normalized range, calibration MAE |
| `fedsteer/extractive.py` | Fragment coverage/density/compression (Grusky et al. 2018, regex tokenizer), publication-from-URL |
| `scripts/newsroom_stats.py` | Per-publication statistics (gate G0) and scorer validation against Newsroom's precomputed values |
| `scripts/build_newsroom.py` | Builds `data/newsroom_fed/{data.jsonl,clients.json}`: client selection, temporal splits, held-out rotations |
| `fedsteer/warp.py` | Private monotone α warps (A/B/C) |
| `fedsteer/calibrate.py` | Post-hoc isotonic α remap (option F) |
| `fedsteer/metrics.py` | Scorers and direction-quality metrics, shared by the monitor and `eval_direction.py` |
| `fedsteer/monitor.py` | Held-out monitor during training: dev loss every round, a small steering check every few rounds |
| `scripts/summarize_sweep.py` | Table of a checkpoint sweep and dev-based checkpoint selection |
| `scripts/make_toy_data.py` | Toy clients with different length ranges (smoke tests) |
| `tests/test_fedsteer.py` | CPU tests: `python tests/test_fedsteer.py` |

## Data format

One JSONL file, one record per example:

```json
{"client": "nytimes", "split": "train", "prompt": "...", "target": "...", "score": 3.7}
```

`score` is the attribute value of `target`. α is computed from the scores of each client's own `train` split.

## Newsroom (flagship task)

```bash
python scripts/newsroom_stats.py --src data/newsroom/release --out data/newsroom_stats
python scripts/build_newsroom.py --stats data/newsroom_stats --out data/newsroom_fed
python train_fed.py --config configs/newsroom_fedavg.yaml       # -> runs/newsroom_fedavg_rot0_<stamp>/
python eval_direction.py --run runs/newsroom_fedavg_rot0_<stamp> --scorer density
```

- **Clients:** 12 publications, evenly spaced by median density among publications with at least 5k usable pairs. They run from telegraph.co.uk (median density 1.3) to nypost.com (32).
- **Rotations:** `clients.json` defines 3 held-out rotations of 4 clients each, stratified by density. Configs choose one with `rotation: k`.
- **Splits per client:**
  - `train`: up to 5k pairs; training uses 2k by default via `max_train_per_client`.
  - `dev`: 100 pairs.
  - `test`: 200 articles from the official test split.
  - `drift`: 600 pairs in 3 stages of 200. Drift pairs come from the latest year(s); the other splits come from strictly earlier years. aol.com is the only exception (almost no pre-2016 data) and is flagged `temporal_split: false`.
- **Articles:** truncated to 400 words. `score` is density recomputed with `fedsteer.extractive` on the truncated article, the same scorer used at evaluation. Pairs whose summary depends on the removed part are dropped.
- **Scorer vs. Newsroom's values:** Spearman 0.99 on density. Absolute values differ where spaCy treats unusual whitespace (`\xa0`, tabs) as tokens, which cuts copied fragments apart in the reference values. Our tokenizer ignores whitespace.

## Choosing checkpoints (held-out monitoring and sweeps)

Training loss only measures fit to data the model trains on. Over several passes through the data it mostly measures memorization. Checkpoints are chosen on the **dev** split and reported on **test**.

- **During training:** the `monitor:` config section logs dev loss every round and a steering check every `steer_every` rounds (20 dev articles × α ∈ {0.1, 0.5, 0.9}). It prints them on each round's log line and stores them in `train_log.jsonl` under `eval`.
- **After training:**
  ```bash
  python eval_direction.py --run runs/X --scorer density --split dev --dev_loss --max_prompts 100 \
      --snapshot runs/X/snapshots/round_00{2,4,6}0.pt
  python scripts/summarize_sweep.py --run runs/X --select pct_calib_err
  ```
  `sbatch/sweep_pilot.sbatch` runs the whole procedure, including the test evaluation of the selected checkpoint.
- **Global α:** evaluation splits the α grid into *in-support* and *out-of-support* values for each client. It reports `pct_err_in_support`, `pct_err_out_support` and `reach_rate`: the share of outputs at out-of-support α that leave the client's own range in the requested direction. This is where federated vs. local-only training is decided.
- **Reading the metrics:** the percentile calibration error of a model that always outputs the client median is 0.30 on the α grid {0, .25, .5, .75, 1} and 0.27 on {0.1, 0.5, 0.9}. A useful knob must beat that.

## Commands

```bash
python tests/test_fedsteer.py
python scripts/make_toy_data.py --out data/toy_length.jsonl --clients 4
python train_fed.py --config configs/toy_length.yaml            # -> runs/toy_length_fedavg_<stamp>/
python eval_direction.py --run runs/toy_length_fedavg_<stamp> --scorer words
sbatch sbatch/toy_smoke.sbatch
```

## Not yet implemented
- The Amazon task
- Utility (AlignScore) and specificity metrics
- E2 (held-out clients with only the gain fitted) and E3 (drift)
- B1 (prompting), B4 (federated CAA), B5 (pooled reference), A3 (PFL structure with a control token)
