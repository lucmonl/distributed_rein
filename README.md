# fedsteer: federated learning of steering directions

Implements the plan in `federated-steering-plan.md`.

Per client *i*, every targeted linear layer is

    y = W0 x + B_p,i A_p,i x + s_i · α · B_d A_d x

- `A_p, B_p`: private adapter (never leaves the client)
- `s_i = exp(u_i)`: private gain, clamped to [1/4, 4]
- `A_d`: shared, frozen, generated from a common seed. Because A_d never changes, averaging `B_d` averages the direction products exactly.
- `B_d`: the shared direction, the only thing aggregated
- `α`: each training example's client-local percentile of the attribute score

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
python train_fed.py --config configs/newsroom_fedavg.yaml
python eval_direction.py --run runs/newsroom_fedavg_rot0 --scorer density
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

## Commands

```bash
python tests/test_fedsteer.py
python scripts/make_toy_data.py --out data/toy_length.jsonl --clients 4
python train_fed.py --config configs/toy_length.yaml
python eval_direction.py --run runs/toy_length_fedavg --scorer words
sbatch sbatch/toy_smoke.sbatch
```

## Not yet implemented
- The Amazon task
- Utility (AlignScore) and specificity metrics
- E2 (held-out clients with only the gain fitted) and E3 (drift)
- B1 (prompting), B4 (federated CAA), B5 (pooled reference), A3 (PFL structure with a control token)
