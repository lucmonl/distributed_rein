# Conventions shared by all datasets (from 2026-10-07)

Every session reads this before launching anything. The coordinator session ("Calibration function design") owns this file; ask it before changing anything here.

## 1. Sessions, datasets, clusters

| Session | Dataset folder | Experiment IDs | Clusters | Owns |
|---|---|---|---|---|
| newsroom-iter1 | `newsroom` | `NR-60`, `NR-61`, … | cc + Anvil | `configs/newsroom*`, `sbatch/train_eval.sbatch`, `sbatch/train_eval_anvil.sbatch`, `sbatch/eval_*.sbatch` except `*_chembl`/`*_math`, `sbatch/judge_evals.sbatch` |
| molecule-iter1 | `molecule` | `MOL-23`, `MOL-24`, … | cc + Anvil | `configs/chembl*`, `sbatch/*chembl*`, `chembl-experiment-plan.md` |
| math-cot-iter1 | `math-cot` | `MATH-16`, `MATH-17`, … | Delta + dtai | `configs/math*`, `sbatch/*math*`, `math-cot-experiment-plan.md`, `scripts/sync_to_delta.sh` |
| coordinator | `coordinator` | `CAL-2`, `CAL-3`, … | none | this file, `federated-steering-plan.md` §2 (the method) |

- Each session numbers only its own prefix, so IDs never collide. The next free number is the highest in your own log plus one.
- **cc and Anvil are shared by Newsroom and molecule.** The Anvil allocation (`cis260796-ai`) is one budget for both: check the balance before submitting and say in your log how many SU a launch will use.

## 2. Where things go

| What | Path |
|---|---|
| Experiment log (one per dataset) | `exp_log/<dataset>/LOG.md` |
| Launch scripts | `exp_log/<dataset>/launch/<ID>.sh` (+ `<ID>_race.json`) |
| Reports | `exp_log/<dataset>/reports/<id>_<what>.txt` |
| SLURM stdout/stderr | `sbatch/logs/<dataset>/<jobname>.o<jobid>` |
| GPU-memory logs | `sbatch/logs/<dataset>/gpumem.o<jobid>` |
| Race-watcher logs | `sbatch/logs/<dataset>/race-<ID>.log` |
| Cross-dataset decisions | `exp_log/coordinator/LOG.md` |

`<dataset>` is exactly `newsroom`, `molecule` or `math-cot`. In each sbatch file use
`#SBATCH --output=<repo>/sbatch/logs/<dataset>/%x.o%j` (`%x` = job name). SLURM does not create missing directories, so create `sbatch/logs/<dataset>/` on every host before submitting.

`exp_log/EXPERIMENT_LOG.md`, `exp_log/launch/` and `exp_log/reports/` are the **archive up to 2026-10-07**: read them, never append to them. Old job logs stay in `sbatch/logs/` (flat), because the archive refers to them by those paths.

## 3. Terminology (use these words in logs, tables and job names)

The model, per adapted matrix l of client i:

  output = base + **adapter**_i + **coefficient**_{i,l}(α) · **direction**

| Term | Meaning | Code / config |
|---|---|---|
| α, the **knob** | requested level on the global percentile scale, in [0, 1] | `alpha_mode=global` |
| **adapter** | client's private LoRA P_i | `fed.adapter=private` |
| **direction** | shared steering LoRA D (federated) or D_i (local) | `B^D` |
| **calibration** | the map α → coefficient: coefficient = **offset** + **gain** · **shape**(α) | |
| **shape** | monotone curve with shape(0) = 0, shape(1) = 1 (was "warp", h) | `lora.warp=kumaraswamy_mix` |
| **per-layer shape** | one shape per adapted matrix (default) vs. **one shape** per client | `lora.warp_scope=module` / `model` |
| **gain** | scale s; **fixed** at 1 (default) or **learned** | `fed.fix_gain=true` / `false` |
| **offset** | additive level o; **none** (default), **private** (one scalar per client) or **shared** | `lora.offset` (+ how it is shared, see below) |
| **fed** / **local** | FedAvg on the direction / each client alone | `fed.mode=fedavg` / `local` |

**Notation (use exactly these symbols):** client i, adapted matrix (layer) l, knob α.

| Symbol | Meaning |
|---|---|
| g_{i,l}(α) = o_i + s · h_{i,l}(α) | calibration of client i at layer l (its value is the coefficient on the direction) |
| h_{i,l}(α) | shape of client i at layer l; monotone, h(0) = 0, h(1) = 1 |
| o_i | offset: one scalar per client, clamped to ±`offset_max` (2.0). A per-layer o_{i,l} is not implemented |
| s | gain (fixed at 1 in the reference; s_i if per client) |
| h̄_l, ḡ_l | server-pooled shape / pooled calibration at layer l |
| P_i, D | adapter of client i, direction |

Drop an index only when the object truly does not depend on it (e.g. h_l only for a shape that is literally shared by all clients, as in `fed-shared`).

How the shape is learned (fed only; local is always private):

| Name | Meaning | Code |
|---|---|---|
| **shared** | one shape per layer, FedAvg of its parameters | `fed.calibration=shared` |
| **private** | one shape per client, never shared | `fed.calibration=private` |
| **borrow** | private shapes pulled toward a coverage-weighted table out of support (plan §2.1) | `fed.calibration=coverage` |
| **consensus** | private shapes in training, tied in support; pooled table at inference | `fed.calibration=consensus` |
| **aligned** | pooled table in both training and inference | `fed.calibration=aligned` |

**Reference calibration (the default every new arm is compared against):** gain fixed at 1, per-layer shape, shared shape, no offset. On Newsroom that is NR-54 `A_L`.

⚠️ "Default" here means the reference design, **not** the code/config defaults. In the code, `lora.warp_scope` defaults to `model` (one shape per client) and no file in `configs/` sets it; the identity penalty is on by default (`FedConfig.warp_reg = 1e-2`), and every config either leaves it there or sets the same value. Every reference-matched arm must pass these overrides literally:
`lora.warp=kumaraswamy_mix lora.warp_scope=module fed.fix_gain=true fed.warp_reg=0 lora.offset=false fed.calibration=shared`
(then change only what the arm label says). Check `config.yaml` in the run dir after the first round.

## 4. Names

**Arm label:** `<mode>-<shape>[-<deviation>…]`, lowercase, hyphen-separated.
- `<mode>`: `fed` or `local`. For `local` omit `<shape>` (always private).
- `<shape>`: `shared`, `private`, `borrow`, `consensus`, `aligned`.
- `<deviation>` only for what differs from the reference: `off` (private offset), `soff` (shared offset), `gain` (learned gain), `1shape` (one shape per client), hyperparameters as `l0p1` (λ = 0.1), `cnt` (count pooling), `reset` (local run that resets its direction's Adam state every round, as fed does; `fed.local_reset_opt_state=true`), `noreset` (fed run that keeps it) — both only for reading old runs: the per-round reset is part of FedAvg and is not controlled for (user decision, CAL-6), `s1` (seed 1; seed 0 is implicit).
- Examples: `fed-shared` (the reference), `fed-shared-off`, `fed-private-off`, `local-off`, `local`, `fed-aligned-cnt`, `fed-borrow-l0p1-1shape`.

**SLURM job name:** `<id>-<arm>` with the ID lowercased and its hyphen dropped: `nr60-fed-shared-off`, `mol23-local-off`, `math16-fed-shared`. Same name on every cluster of a race (the job ID tells the copies apart).

**Run directory:** `out_dir=runs/<jobname>` → `runs/nr60-fed-shared-off_<stamp>_j<jobid>`.

**In tables** use the arm label, not "A_L", "PRIV_OFF", etc. The first table of a new entry maps any archive names it cites (e.g. `fed-shared` = NR-54 A_L).

## 5. Shared code

`fedsteer/`, `train_fed.py`, `eval_direction.py`, `e2_heldout.py`, `e3_drift.py`, `scripts/compare_runs.py`, `scripts/race_watch.py`, `tests/` are used by all three datasets.
- Changes must be **opt-in**: a new flag whose default reproduces the current behaviour exactly.
- `python tests/test_fedsteer.py` must pass in full before any job is submitted with the change.
- Record the change in your own LOG.md under a "Shared code change" heading, and message the coordinator with the file list.
- Only the session the coordinator names may edit a shared file at a time. Current grants: **newsroom-iter1: `e2_heldout.py`, `fedsteer/adapt.py`, `tests/test_fedsteer.py` (NR-68, E2 for aligned / private-offset / fixed-gain runs; opened 2026-10-08)**. Closed: NR-60b (`fed.shared_offset` for aligned, 76/76); NR-60 (`fed.private_offset`, `fed.local_reset_opt_state`, 74/74); NR-62 (aligned accepts `lora.offset`, 75/75; made at the user's direct request and recorded afterwards). Ask the coordinator before editing shared code.
- Before syncing code to another cluster, make sure the full test suite passes on cc; jobs run from a code snapshot, so running jobs are never affected.
- Never `git commit` (user rule); list changed files instead.
