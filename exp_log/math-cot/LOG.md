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

## MATH-18. MATH-14 results: per-layer shapes improve single-client G2 at the top of the range; the α = 0 level does not move; the gain clamp was not the limit (2026-10-08, 10:10) [dtai]

**Arms** (single client `math`, answer in prompt, Qwen2.5-7B, 100 rounds, all with `fed.warp_reg=0.01`, no offset). Single client, so the client index is dropped.

| arm label | archive name | calibration | job | test round (dev-selected) |
|---|---|---|---|---|
| `local-gain-1shape` | MATH-11 G2 | g(α) = s·h(α), one shape, s ≤ 4 | Delta 22703347 | 60 |
| `local-gain` | MATH-14 P | g_l(α) = s·h_l(α), 196 shapes, s ≤ 4 | **dtai 3328129** (01:01–05:51) | 60 |
| `local-gain-gm16` | MATH-14 PG | as `local-gain`, s ≤ 16 | **dtai 3328130** (01:02–05:51) | 100 |

Race: the watcher cancelled Delta 22718559 / 60 at 01:21 / 01:27, once the dtai copies were running. Both runs: 0 skipped steps. They ran the tree synced at MATH-17, from before `fed.shared_offset`; defaults are identical, so this does not matter.
Runs: `runs/math14_ans_g2_perlayer_qwen25_7b_20261008-010107_j3328129`, `runs/math14_ans_g2_perlayer_gmax16_qwen25_7b_20261008-010208_j3328130`; logs `sbatch/logs/fedsteer_math.o33281{29,30}` on dtai (submitted before MATH-16, so the flat path).

**Test (100 problems, α ∈ {0, .25, .5, .75, 1}):**

| metric | `local-gain-1shape` | `local-gain` | `local-gain-gm16` |
|---|---:|---:|---:|
| pct err (penalized = plain) | 0.253 | **0.213** | 0.225 |
| in support / out of support | 0.263 / 0.237 | **0.237 / 0.178** | 0.248 / 0.191 |
| Spearman | 0.797 | **0.870** | 0.844 |
| generated range (pct) / reach | 0.525 / 0.190 | **0.645 / 0.345** | 0.619 / 0.310 |
| mean percentile at α = 0, .25, .5, .75, 1 | .224 .486 .608 .663 .749 | .221 .443 .549 .638 **.866** | .289 .485 .574 .671 **.908** |
| mean tokens at α = 0 … 1 | 272 … 601 | 267 … 744 | 306 … 754 |
| s (final) | 4.0 (clamped) | 4.0 (clamped from round 50) | 5.1 (plateau 4.8–5.4 from round 50) |
| shape: h(0.5) (pooled mean for per-layer) | 0.735 | 0.578 | 0.450 |
| per-layer h_l(0.5) min / max | — | 0.035 / 0.991 | 0.013 / 0.998 |
| accuracy mean (worst α) | 0.982 (0.98) | 0.950 (0.90) | 0.936 (0.89) |
| truncated / loop mean | 0.010 / 0.016 | 0.020 / 0.024 | 0.004 / 0.000 |

Base model with the answer in the prompt: accuracy 0.96 (MATH-15).

**Reading:**
- **Per-layer shapes help** (`local-gain` vs `local-gain-1shape`: pct err −0.040, Spearman +0.07, reach 0.19 → 0.35). Almost all of the gain is at the top: α = 1 reaches the 0.87th percentile instead of 0.75, and out-of-support error falls 0.237 → 0.178.
- The per-layer shapes diverge to the extremes. Some layers are nearly step functions at α ≈ 0 (h_l(0.5) ≈ 0.99), others nearly off until α = 1 (h_l(0.5) ≈ 0.03). So "which layers switch on when" acts as extra range, not as a gentle per-layer bend.
- **The bottom does not move.** At α = 0 every arm lands near the 0.22nd percentile (target 0), about 270 tokens vs. the 77-token target. With h_l(0) = 0 for every layer and no offset, g_l(0) = 0, so the α = 0 output is fixed by the adapter P alone. This is the level problem diagnosed in MATH-13, and only an offset o (or a different α = 0 behaviour of the adapter) can address it.
- **The clamp was not the binding limit.** Without it (s ≤ 16), s rises to ~5.1 and stops. It is no better on test (0.225 vs 0.213) and slightly better on dev at its own selected round (0.207 vs 0.218). Within single-seed noise this is a tie. PG's α = 0 level is also higher (0.29 vs 0.22), which hurts. Since g_l(0) = 0 in both, that difference comes from the co-trained adapter P, not from s directly.
- Cost: accuracy drops 0.98 → 0.95 / 0.94 (worst α 0.90 / 0.89), concentrated out of support (0.915). Still close to the base model's 0.96.
- **Confound:** `local-gain-1shape` ran on Delta (A100, torch 2.5.1) and the two MATH-14 arms on dtai (GH200, torch 2.13). P vs. PG is clean; P vs. G2 is not strictly host-matched. A −0.040 gap with the mechanism visible in the per-α table (top end only) makes a host artefact unlikely, but it is not excluded.

**Implication for E1:** carry the per-layer shape into the next math federated design (as in the reference `fed-shared`). Keep the gain limit at 4 or fix s; raising it buys nothing. The α = 0 level needs the offset arm (`-off`, MATH-13 / CAL-9).

## MATH-19. Code sync after the NR-60b release (`fed.shared_offset`); E1 status (2026-10-08, 10:15) [cc+delta+dtai]

- The sync released on 10-07 evening was interrupted (session ended). Done now: `python tests/test_fedsteer.py` on cc **76/76**; `scripts/sync_to_delta.sh` to Delta and dtai. Code moved: `fedsteer/fed.py`, `tests/test_fedsteer.py`, plus docs and paper files. A dry run afterwards is empty apart from sbatch headers. Imports OK on both hosts, with `FedConfig().shared_offset == False`.
- MATH-14 had already run on the MATH-17 tree (defaults identical; see MATH-18). G1 (Delta 22705844, still PENDING) will copy this tree when it starts.
- E1 (Delta 22703349 / 50, RUNNING 25 h): training finished; both selected round 100 on dev; test eval running (10:20). Dev mean pct err at rounds 20 / 40 / 60 / 80 / 100: `fed-shared-gain-1shape` .335 / .302 / .280 / .259 / .256; `local-gain-1shape` .324 / .271 / .265 / .258 / .253. The local lead at round 40 has closed. Full analysis after the test eval (MATH-20).

## MATH-20. E1 answer-in-prompt, test: `fed-shared-gain-1shape` ties `local-gain-1shape` (0.255 vs 0.257); both cover only ~0.37 of the percentile range, and the endpoints are the error (2026-10-08, 14:00) [delta]

**Arms** (8 clients, answer in prompt, Qwen2.5-7B, private adapter P_i, no offset, gain clamp 4, `warp_reg` 0.01, 100 rounds). Archive names: MATH-11 / MATH-15 "answer-in-prompt E1-fed / E1-local".

| arm label | calibration | job | elapsed | test checkpoint |
|---|---|---|---|---|
| `fed-shared-gain-1shape` | g_i(α) = s·h(α); s, h FedAvg'd (shared by all clients and layers) | Delta 22703349 | 28 h 20 m | round 100 (dev-selected) |
| `local-gain-1shape` | g_i(α) = s_i·h_i(α), own direction D_i | Delta 22703350 | 28 h 23 m | round 100 (dev-selected) |

Runs `runs/math7_ans_e1_{fed,local}_qwen25_7b_20261007-091715_j2270334{9,50}`; logs `sbatch/logs/fedsteer_math.o2270334{9,50}` (flat, submitted before MATH-16). Full per-client table: `exp_log/math-cot/reports/math20_e1_ans_test.txt`.

**Test, mean over 8 clients (100 problems each):**

| metric | fed | local |
|---|---:|---:|
| pct err (= penalized; unscorable 0) | **0.255** | 0.257 |
| worst client | 0.320 (orca_math) | 0.323 (olympiads) |
| fed better / worse than local, by client | 5 / 3 | |
| in support / out of support | 0.160 / 0.314 | 0.164 / 0.319 |
| Spearman | **0.789** | 0.763 |
| generated range (pct) / reach | 0.370 / 0.169 | 0.374 / **0.201** |
| s | 3.17 (shared) | s_i = 4.0 (the clamp) for 7 of 8; orca_math 2.10 |
| accuracy (worst cell) | 0.942 (0.86) | 0.945 (0.88) |
| truncated / loop | 0.006 / 0.005 | 0.005 / 0.005 |

**Per α** (mean over clients of the mean output percentile; target = α):

| α | 0 | .25 | .5 | .75 | 1 |
|---|---:|---:|---:|---:|---:|
| fed: mean pct | .311 | .423 | .530 | .613 | .682 |
| local: mean pct | .255 | .424 | .555 | .599 | .628 |
| fed: mean \|pct − α\| | **.311** | .189 | .135 | .173 | **.318** |
| local: mean \|pct − α\| | **.255** | .175 | .133 | .200 | **.372** |

**Reading:**
- **A tie on the mean and on the worst client** (single seed). Per client the gaps are ±0.01–0.05 in both directions:
  - fed better: Logic and Puzzles −0.042, Inequalities −0.021, Geometry −0.018, math −0.010, olympiads −0.008;
  - local better: aops_forum +0.048, gsm8k +0.021, orca_math +0.014.
- **The two arms fail differently at the ends.**
  - The shared shape stays near the identity (h(0.5) = 0.52), so fed keeps rising up to α = 1: top level .682, Spearman .789.
  - The local shapes are strongly concave (e.g. Logic and Puzzles: h_i(0.25) = 0.73, flat after α ≈ 0.5), so local saturates early: top level .628, and .37 → .39 for Logic between α = .75 and 1.
  - Local reaches a lower α = 0 level (.255 vs .311), mainly aops_forum (.17 vs .33). With g_i(0) = 0 in both, the direction is off at α = 0, so that level comes from the adapter P_i alone; it differs because each P_i was co-trained with a different direction.
- **The endpoints are the error, as in MATH-13 / MATH-18.** The mean-level miss is ~0.31 at α = 0 and 0.32–0.37 at α = 1, against 0.13–0.20 in the interior.
  - Both arms cover only ~0.37 of the [0, 1] percentile range between α = 0 and α = 1, with s at 3.2–4.
  - orca_math barely steers at all (range 0.10 / 0.14; .38 → .48).
  - The long-response clients (olympiads, aops_forum, support starting at 0.54–0.56; math at 0.18) sit far above α = 0 at α = 0. Olympiads is at .60 in both arms. This is the per-client level problem of MATH-13: g_i(0) = 0 cannot lower it, while an offset o_i (or a negative coefficient) could.
- **The Adam reset:** fed resets the direction's Adam state every round, as part of FedAvg (CAL-6). Here it gives no net federated edge.
- **Quality is unchanged between the arms:** accuracy 0.94, truncation and loops ≤ 0.6 %.

**Implications for the s = 1 design (CAL-12):**
1. Even with s ≈ 3–4, one direction spans only ~37 % of the range. At s = 1, D must grow ≥ 3× on its own, so the single-client s = 1 check is needed before any federated s = 1 arm.
2. The α = 0 level differs by client (.03–.62), so the offset o_i is the expected lever here, as on Newsroom.
3. Per-layer shapes added range at the top in MATH-18. A one-shape design (both E1 arms) leaves that unused.
