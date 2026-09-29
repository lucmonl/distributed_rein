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
| `scripts/make_toy_data.py` | Toy clients with different length ranges (smoke tests) |
| `tests/test_fedsteer.py` | CPU tests: `python tests/test_fedsteer.py` |

## Data format

One JSONL file, one record per example:

```json
{"client": "nytimes", "split": "train", "prompt": "...", "target": "...", "score": 3.7}
```

`score` is the attribute value of `target`. α is computed from the scores of each client's own `train` split.

## Commands

```bash
python tests/test_fedsteer.py
python scripts/make_toy_data.py --out data/toy_length.jsonl --clients 4
python train_fed.py --config configs/toy_length.yaml
python eval_direction.py --run runs/toy_length_fedavg --scorer words
sbatch sbatch/toy_smoke.sbatch
```

## Not yet implemented
- The Newsroom data pipeline and density scorer (the flagship task), and the Amazon task
- Utility (AlignScore) and specificity metrics
- E2 (held-out clients with only the gain fitted) and E3 (drift)
- B1 (prompting), B4 (federated CAA), B5 (pooled reference), A3 (PFL structure with a control token)
