# Experiment log: math-cot

Updated after every code change and every job; newest entries at the bottom. Entry IDs: `MATH-<n>`, starting at MATH-16.
Conventions (paths, terminology, job names): `../../CONVENTIONS.md`. History before 2026-10-07: `../EXPERIMENT_LOG.md` (archive, read-only).


## MATH-16. Switch to the per-dataset conventions (CAL-2): math sbatch log paths, docs-only sync mode; no new jobs (2026-10-07, 15:00) [cc+delta+dtai]

Coordinator instruction (CAL-2). No experiment, no shared-code edit, no code sync (hold while newsroom-iter1 holds the shared-code grant for NR-60).

**sbatch headers (cc; the Delta/dtai copies via the sync rewrite).**
- `sbatch/train_eval_math.sbatch`, `eval_b1_math.sbatch`, `eval_base_math.sbatch`, `backbone_screen.sbatch`: `#SBATCH --output=/u/lucmon/rein/sbatch/logs/math-cot/%x.o%j` (was flat `sbatch/logs/<fixed name>.o%j`). With `%x` the log is named after the job name given at submission (`math16-fed-shared.o<jobid>`).
- `train_eval_math.sbatch`: the `GPU_LOG=1` file goes to `sbatch/logs/math-cot/gpumem.o<jobid>`.
- Old job logs, and the running watchers' logs (`sbatch/logs/race_MATH-14.log`), stay where they are. New race logs: `sbatch/logs/math-cot/race-<ID>.log`.

**`scripts/sync_to_delta.sh --no-code`** (new flag, mine per CONVENTIONS.md).
- Syncs everything except code: excludes `*.py`, `configs/`, `fedsteer/`, `scripts/`, `tests/` (and skips `data/*.jsonl`). Excluded paths are also protected from `--delete`, so the remote code is left exactly as it was.
- The remote setup step now also creates `sbatch/logs/math-cot/` (SLURM does not create `--output` directories).
- Run on Delta and dtai at ~14:56. Verified: `--no-code --dry-run` then lists only the 17 `sbatch/*.sbatch` files (header rewrite, by design); `sbatch/logs/math-cot/` exists on both; the remote `train_eval_math.sbatch` has the new `--output` and gpumem paths with the host's account/partition.
- **Held code delta** (full `--dry-run`, not synced): `e2_heldout.py`, `e3_drift.py`, `fedsteer/{coverage,fed,lora}.py`, `scripts/coeff_sweep.py`, `scripts/sync_to_delta.sh`, `tests/test_fedsteer.py`. These are other sessions' cc edits since the last code sync (plus this flag). They go out with the first code sync after the coordinator releases the hold, once `python tests/test_fedsteer.py` passes on cc.
- Pending Delta/dtai jobs snapshot the remote code tree at start (`runs/_code/<stamp>`). That tree is unchanged, so MATH-14 and G1 still run the code they were submitted against. The batch script itself was copied by SLURM at submission, so the header change does not affect them (their logs stay flat).

**Job states (15:00), all kept:**

| job | arm label | host / id | state |
|---|---|---|---|
| answer-in-prompt E1 fed (MATH-11/15) | `fed-shared-gain-1shape` | Delta 22703349 | RUNNING 5 h 40 (gpua087) |
| answer-in-prompt E1 local (MATH-11/15) | `local-gain-1shape` | Delta 22703350 | RUNNING 5 h 40 (gpua090) |
| G1, B1 baseline (MATH-11) | — | Delta 22705844 | PENDING (Priority) |
| MATH-14 P, single-client G2, per-layer shape | (single client) learned gain, per-layer | Delta 22718559 / dtai 3328129 | PENDING both |
| MATH-14 PG, as P with `gain_max=16` | as P | Delta 22718560 / dtai 3328130 | PENDING both |

- MATH-14 race watcher pid 977434 (log `sbatch/logs/race_MATH-14.log`) still running.
- Analyses of these go in new entries (MATH-17+) with the CONVENTIONS.md arm labels.

## MATH-17. Code sync to Delta/dtai after the CAL-2 hold was released; suite passes on all three hosts (2026-10-07, 17:00) [cc+delta+dtai]

Coordinator released the hold (newsroom-iter1's NR-60 change: `fed.private_offset`, `fed.local_reset_opt_state`, both opt-in, default false and bit-identical). No new math experiment.

- `python tests/test_fedsteer.py` on cc (`steer`): **74/74**.
- `scripts/sync_to_delta.sh --host delta` and `--host dtai` at ~16:00. Files moved: `e2_heldout.py`, `e3_drift.py`, `fedsteer/{coverage,fed,lora}.py`, `scripts/{coeff_sweep,sync_to_delta}.*`, `tests/test_fedsteer.py`, plus docs and a molecule config. A dry run afterwards lists only sbatch files and docs that other sessions wrote after the sync.
- Imports (`fedsteer.fed`, `fedsteer.lora`, `train_fed`) OK on both; `SteerLoraConfig().warp_scope` still defaults to `model`.
- Suite on the remote login nodes:
  - **dtai** (`rein-gh`, torch 2.13): **74/74** (41 min on the login node).
  - **Delta** (`rein`): 73/74. The one failure, `test_b3_merge_equals_average_of_local_directions`, is an environment error in the subprocess it starts (`scripts/merge_local_directions.py`): "mkl-service + Intel MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp". Run alone, it passes, both with and without `MKL_THREADING_LAYER=GNU`. The test dates from commit a2c0de8, and the script is not in the math pipeline. Not caused by the new code.
- **Effect on queued jobs:** MATH-14 P / PG (Delta 22718559/60, dtai 3328129/30) and G1 (Delta 22705844) are still PENDING. They will copy this new tree when they start. With the new flags off they are bit-identical to the old code, so they stay comparable to MATH-11 G2. The running E1 pair (Delta 22703349/50, RUNNING 7 h 46 at 17:03) runs its own snapshot and is unaffected.
- Write-up notes for the E1 analysis (coordinator, CAL-6): the per-round Adam reset in fed is part of FedAvg. Fed (reset) vs. local (state carried) is the comparison, and the reset can be named as a mechanism but not as a confound. Calibration notation: g_{i,l}(α) = o_i + s·h_{i,l}(α), pooled h̄_l / ḡ_l (CONVENTIONS.md §3).
