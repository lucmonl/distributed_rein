# Experiment log: federated learning of steering directions

Updated after every code change and every job. Newest entries at the bottom.
The plan is in `../federated-steering-plan.md`.

## Where to look

| What | Where |
|---|---|
| Job stdout/stderr | `sbatch/logs/<name>.o<jobid>` |
| Run directory (one per training run) | `runs/<prefix>_<YYYYmmdd-HHMMSS>_j<jobid>/` |
| Inside a run | `config.yaml` (resolved config), `run_info.json` (who/when/git commit, resumes), `train_log.jsonl` (per round: train loss, gains, dev monitor), `snapshots/round_XXXX.pt`, `state.pt` (resume only; safe to delete) |
| Evaluations | `<run>/evals/eval_<snapshot>[_dev][_remap][_<suffix>]__<YYYYmmdd-HHMMSS>_j<jobid>.json` |
| How a job was launched | `exp_log/launch/expNN_*.sh` (exact `sbatch` commands; `sbatch/` itself is git-ignored). Each job log's first line also prints its `CONFIG` and `OVERRIDES` |
| Sweep table | `python scripts/summarize_sweep.py --run <run> [--match j<jobid>]` |

Since 2026-09-29 (entry 8), runs and evals never overwrite each other. Continue a run with `python train_fed.py --resume <run_dir>`.

## Job index

| Job | Date | Purpose | Run directory | Status / headline |
|---|---|---|---|---|
| 10985846 | 09-29 | Toy smoke test (entry 1) | `runs/toy_fedavg_20260929-112613_j10985846`, `runs/toy_local_20260929-112613_j10985846` | Done; pipeline works |
| 10987927 | 09-29 | Newsroom pilot, 10 rounds, 1B (entry 3) | `runs/nr_pilot_llama1b_20260929-122920_j10987927` | Done; steering on 7 of 8 clients |
| 10996826 | 09-29 | Re-eval of round 10 with fixed metrics (entry 5) | same run | Done |
| 10997791 | 09-29 | (cancelled after 6 s) | — | Cancelled |
| 10997796 | 09-29 | Resume pilot to 100 rounds (user-submitted) | same run | Done; train loss 0.03 = memorization |
| 11006740 | 09-29 | Checkpoint sweep, dev then test (entry 6) | same run | Done: best round 60 (test Spearman 0.84, percentile error 0.196) |
| 11011538 | 09-30 | Global α, **federated** (entry 9) | `runs/nr_global_fedavg_20260930-001106_j11011538` | Done (10 h): test percentile error 0.165, out-of-support 0.171 |
| 11011539 | 09-30 | Global α, **local-only** baseline B2 (entry 9) | `runs/nr_global_local_20260930-001106_j11011539` | Done (10 h): test percentile error 0.180, out-of-support 0.181 |
| 11039406 | 09-30 | Pipeline smoke test (entry 10) | `runs/smoke_pipeline_20260930-111641_j11039406` | Passed |
| 11041280 | 09-30 | exp11 cap 2k, base (entry 11) | `runs/exp11_cap2k_base_20260930-114747_j11041280` | Done (6.3 h), selected round 60 |
| 11041281 | 09-30 | exp11 cap 2k, regularized | `runs/exp11_cap2k_reg_20260930-114747_j11041281` | Done (6.8 h), selected round 60 |
| 11041282 | 09-30 | exp11 cap 4k, base | `runs/exp11_cap4k_base_20260930-114746_j11041282` | Done (6.2 h), selected round 100 |
| 11041283 | 09-30 | exp11 cap 4k, regularized | `runs/exp11_cap4k_reg_20260930-180318_j11041283` | **Failed** at round 10 (ImportError, entry 14); resubmitted as 11060306 |
| 11041284 | 09-30 | exp11 natural sizes (≤5k), regularized | `runs/exp11_natural_reg_20260930-180542_j11041284` | **Failed** at round 10 (ImportError, entry 14); resubmitted as 11060307 |
| 11060306 | 09-30 | exp11 cap 4k, regularized (resubmit) | `runs/exp11_cap4k_reg_20260930-231707_j11060306` | **Cancelled by user** at round 14 (10-01 00:09) |
| 11060307 | 09-30 | exp11 natural sizes, regularized (resubmit) | `runs/exp11_natural_reg_20260930-232106_j11060307` | **Cancelled by user** at round 13 (10-01 00:09) |
| 11061046 | 09-30 | Local-only at 4k (entry 16), fair counterpart of 11041282 | `runs/exp16_local_cap4k_20260930-232203_j11061046` | Done (10-01 06:28, 7.1 h); also the local arm of exp17-B |
| 11063906 | 10-01 | exp17 B: federated, shared calibration, 4k (entry 17) | `runs/exp17_fed_calshared_cap4k_20261001-001022_j11063906` | Done (10-01 06:43, 6.5 h) |
| 11063910 | 10-01 | exp17 B, no offset: federated, shared calibration, 4k | `runs/exp17_fed_calshared_nooff_cap4k_20261001-001022_j11063910` | Done (10-01 06:43, 6.5 h) |
| 11063911 | 10-01 | exp17 local, no offset, 4k | `runs/exp17_local_nooff_cap4k_20261001-013643_j11063911` | Done (10-01 10:18, 8.7 h) |
| 11065060 | 10-01 | exp19 baselines smoke test (entry 19) | — | Failed to start on ccc0387 (user env retrieval), held; released on dali 09:45, then **cancelled** (it was testing only B1, see 19a) |
| 11077014 | 10-01 | exp19 baselines smoke test, resubmitted (all steps) | (writes into the smoke run and local-2k run) | Done (results: entry 22) |
| 11065061 | 10-01 | exp19 B1 prompting (k=0, k=3, k=3 base) on fed 4k client models | `runs/exp11_cap4k_base_…j11041282/evals/*b1_prompt*` | Done (results: entry 22) |
| 11065062 | 10-01 | exp19 B4 federated CAA on fed 4k client models | `runs/exp11_cap4k_base_…j11041282/evals/*b4_caa*` | Done (results: entry 22) |
| 11065063 | 10-01 | exp19 B3 merged direction, local 2k (round 90) | `runs/nr_global_local_…j11011539/evals/*merged*b3*` | Done (results: entry 22) |
| 11065064 | 10-01 | exp19 B3 merged direction, local 4k (dev-selected) | `runs/exp16_local_cap4k_…j11061046/evals/*merged*b3*` | Done (results: entry 22) |
| 11082124 | 10-01 | exp21 E2/E3 smoke test (entry 21) | writes into the smoke run | E2 + E3 passed; **failed in quality scoring** (fixed, see 21a) |
| 11082125 | 10-01 | exp21 E2 on method run (fed, shared cal, no offset) | `runs/exp17_fed_calshared_nooff_cap4k_…j11063910/evals/*_e2_*` | Cancelled (blocked by failed smoke); resubmitted, see 21a |
| 11082128 | 10-01 | exp21 E3 on method run | `runs/exp17_fed_calshared_nooff_cap4k_…j11063910/evals/*_e3_*` | Cancelled (blocked by failed smoke); resubmitted, see 21a |
| 11082129 | 10-01 | exp21 E3 on local counterpart (no offset) | `runs/exp17_local_nooff_cap4k_…j11063911/evals/*_e3_*` | Cancelled (blocked by failed smoke); resubmitted, see 21a |
| 11082130 | 10-01 | exp21 E2 on private-calibration fed run | `runs/exp11_cap4k_base_…j11041282/evals/*_e2_*` | Cancelled (blocked by failed smoke); resubmitted, see 21a |
| 11083572 | 10-01 | exp21 E2 on method run (resubmit), **old design** (superseded by entry 23) | `runs/exp17_fed_calshared_nooff_cap4k_…j11063910/evals/*_e2_*` | Done (superseded design; numbers in entry 26) |
| 11083573 | 10-01 | exp21 E3 on method run (resubmit) | `runs/exp17_fed_calshared_nooff_cap4k_…j11063910/evals/*_e3_*` | Done (10-01 18:21, 4 h) |
| 11083574 | 10-01 | exp21 E3 on local counterpart (resubmit) | `runs/exp17_local_nooff_cap4k_…j11063911/evals/*_e3_*` | Done (10-01 19:41; entry 26) |
| 11083575 | 10-01 | exp21 E2 on private-calibration fed run (resubmit) | `runs/exp11_cap4k_base_…j11041282/evals/*_e2_*` | **Cancelled** before it started (old design, entry 23) |
| 11087947 | 10-01 | exp23 E2 redesign smoke (α window 0–0.4, smoke run) | writes into the smoke run | **Passed** (all four settings, window filter, quality scoring) |
| 11088050–53 | 10-01 | exp23 E2 on method run, n = 16 / 64 / 256 / 1024 (all four settings) | `runs/exp17_fed_calshared_nooff_cap4k_…j11063910/evals/*_e2_*_n{n}__*` | Done (entries 26, 30) |
| 11088054 | 10-01 | exp23 E2 on method run, n = all (frozen_D, local_D, plugin) | same, `*_nall__*` | Done (10-02 02:04; entry 30) |
| 11088055 | 10-01 | exp23 E2 on private-calibration run, all n (frozen_D, plugin) | `runs/exp11_cap4k_base_…j11041282/evals/*_e2_*` | Done (10-02 04:02; entry 30) |
| 11094904 | 10-01 | exp25 A2: shared adapter (non-personalized FedAvg), shared calibration, no offset, 4k | `runs/exp25_fed_adaptershared_calshared_nooff_cap4k_<stamp>_j11094904` | Done (10-02 04:14; selected round 70; entry 30) |
| 11097085 | 10-01 | exp27 judge smoke (refactored `judge_quality.py` on the smoke run's held-out E2 files) | `runs/smoke_pipeline_…j11039406/evals/judge_*e2_*` | **Passed** (held-out lookup, two files with one model load, references judged once) |
| 11097083 | 10-01 | exp27 LLM judge on E2 `frozen_D` / `local_D` (all n) of the method run | `runs/exp17_fed_calshared_nooff_cap4k_…j11063910/evals/judge_*e2_*` | Done (10-02 02:51; entry 30) |
| 11097847 | 10-01 | exp28 Qwen3-8B smoke (2 rounds × 10 steps, 8-article evals, GPU memory log) | `runs/smoke_qwen3_8b_20261001-232600_j11097847` | **Passed** (full pipeline incl. judge; 2.6 s/training step; peak 72 GB) |
| 11097848 / 11097849 | 10-01 | exp28 Qwen3-8B fed / local (first submission) | — | **Cancelled** before starting (home quota; resubmitted to project space) |
| 11097906 / 11097907 | 10-02 | exp28 Qwen3-8B fed / local (second submission) | — | **Cancelled** before starting (resubmitted with the expandable-segments allocator) |
| 11098308 | 10-02 | exp28 **Qwen3-8B federated** (shared calibration, no offset), 4k, 100 rounds | `runs/exp28_fed_calshared_nooff_cap4k_qwen3_8b_<stamp>_j11098308` → `/u/lucmon/lucmon/rein_runs/` | Running (round 75 at 10-02 15:20; ≈ 6.5 min/round) |
| 11098309 | 10-02 | exp28 **Qwen3-8B local** (no offset), 4k, 100 rounds | `runs/exp28_local_nooff_cap4k_qwen3_8b_<stamp>_j11098309` → `/u/lucmon/lucmon/rein_runs/` | Running (from 10-02 02:23; round 65 at 15:20) |
| 11116479 | 10-02 | exp31 cross-client NLL matrix (method / A2 / local, rounds 50–100) | `<run>/evals/nll_matrix_*` | **Cancelled** before starting (user: skip it) |

---

## 1. Federated training framework (2026-09-29)

**Change:** new package `fedsteer/`.
- Steering LoRA layers: W0 + B_p A_p + s·α·B_d A_d. A_d is a frozen, seed-shared matrix, so FedAvg on B_d is exactly an average of the direction products.
- Client-local percentile α.
- FedAvg trainer with modes `fedavg` and `local` (baseline B2; also writes the merged direction for B3), ablations `fix_gain` and `share_private`, checkpoint/resume.
- `train_fed.py`, `eval_direction.py`, toy data `scripts/make_toy_data.py`, CPU tests.

**Tests:** 19/19.

**Job 10985846** (`sbatch/toy_smoke.sbatch`): Llama-3.2-1B, 4 toy clients (attribute = length, different ranges per client), 30 rounds.

| Direction | Spearman | Calibration error (IQR units) | Order rate |
|---|---|---|---|
| fedavg | 0.99 | 0.16 | 0.76 |
| local (B2) | 0.99 | 0.12 | 0.73 |
| merged (B3) | 0.98 | 0.26 | 0.63 |

Takeaway: the pipeline works. The toy task is too easy to show a benefit from collaboration.

## 2. Newsroom integration (2026-09-29)

**Change:**
- `fedsteer/extractive.py`: fragment coverage/density reimplemented with a regex tokenizer; Spearman 0.99 against Newsroom's precomputed density.
- `scripts/newsroom_stats.py`: per-publication statistics (gate G0 passes; median density ranges from 1.1 to 22.6).
- `scripts/build_newsroom.py` → `data/newsroom_fed/{data.jsonl, clients.json}`:
  - 12 publications and 3 held-out rotations;
  - year-level temporal drift split (11 of 12 clients; aol.com falls back);
  - articles truncated to 400 words and rescored.
- Evaluator: scorers take the source record; `density` scorer added.

## 3. Newsroom pilot, 10 rounds (2026-09-29)

**Job 10987927** (`sbatch/newsroom_pilot.sbatch`): Llama-3.2-1B, rotation 0 (8 participants), 10 rounds × 20 steps × batch 8.
- Eval: `evals/eval_round_0010__20260929-122920_j10987927.json` (old metrics).
- Density rises with α for 7 of 8 clients (forbes.com 1.8 → 64.8). nypost.com and reuters.com barely move.

## 4. Metric fixes, part 1 (2026-09-29)

**Change:** percentile calibration error |F(score) − α| (the constant-output reference is 0.30 on the 5-point grid), pairwise concordance, per-prompt score grid saved.

**Why:** calibration error in IQR units was dominated by the α = 0/1 extremes (min/max of heavy-tailed density), and strict order rate was too harsh.

## 5. Metric fixes, part 2: Spearman bias (2026-09-29)

**Change:** articles whose outputs don't change with α now count as Spearman 0; they used to be skipped, which inflated the mean. Added no-effect rate, adjacent increase/tie/decrease rates and endpoint increase rate.

**Job 10996826** (`sbatch/eval_pilot.sbatch`): re-eval of round 10, 100 test prompts.
- File: `evals/eval_round_0010_v2__20260929-161525_j10996826.json`.
- Mean Spearman 0.74, concordance 0.81, α=1 above α=0 on 89% of articles, adjacent ties 37%, percentile error 0.27.
- The low order rate is mostly **ties** from greedy decoding, not reversals.
- nypost.com fails: 77% ties, 33% of articles show no effect at all.

**Job 10997796** (user-submitted): resumed the same run to 100 rounds.
- Eval: `evals/eval_round_0100__20260929-164218_j10997796.json` (50 test prompts).
- Train loss 0.03, dropping at each pass over the data: memorization. The direction's norm grows like a random walk, and clients' updates are uncorrelated (cosine ≈ 0).

## 6. Held-out monitoring and checkpoint sweep (2026-09-29)

**Change:**
- `fedsteer/metrics.py` (shared scorers and metrics).
- `fedsteer/monitor.py`: dev loss every round, steering check every 5 rounds (`monitor:` config section).
- `eval_direction.py`: multiple snapshots, `--dev_loss`, `--suffix`.
- `scripts/summarize_sweep.py`.

**Tests:** 22/22.

**Job 11006740** (`sbatch/sweep_pilot.sbatch`): dev sweep of rounds 10–100, then test eval (200 prompts) of the best (60), 10 and 100. Files: `evals/eval_round_XXXX_dev__20260929-211737_j11006740.json`, `evals/eval_round_XXXX_test200__…`.

| Round | Train loss | Dev loss | Spearman (worst) | Percentile error (worst) |
|---|---|---|---|---|
| 10 | 1.12 | **1.09** | 0.76 (0.35) | 0.27 (0.39) |
| 30 | 0.49 | 1.27 | **0.85 (0.69)** | 0.21 (0.27) |
| 60 | 0.11 | 1.68 | 0.84 (0.67) | **0.19** (0.27) |
| 100 | 0.03 | 1.90 | 0.81 (0.59) | 0.20 (0.28) |

Takeaway: dev loss rises after round 10 (memorization), while steering plateaus at rounds 30–60. **Use about 30 rounds.**

Test results (200 articles per client; job finished 2026-09-30 00:04):

| Round | Spearman (worst) | Concordance | α=1 above α=0 | Adjacent ties | Percentile error (worst) | Percentile range |
|---|---|---|---|---|---|---|
| 10 | 0.74 (0.24) | 0.81 | 90% | 37% | 0.271 (0.41) | 0.45 |
| **60** (selected) | **0.84 (0.64)** | **0.87** | **97%** | **23%** | **0.196 (0.29)** | **0.63** |
| 100 | 0.83 (0.54) | 0.86 | 96% | 24% | 0.200 (0.30) | 0.59 |

Test confirms dev: rounds beyond 60 add nothing. Worst client at round 60 is nypost.com (60% ties; mean output percentile 0.34 → 0.73 across α, i.e. compressed). Best is wsj.com (Spearman 0.94, 0.23 → 0.97). These are *local-α* results from before entry 7.

## 7. Private α warp, offset, adapter modes and global α (2026-09-29)

**Change:**
- `fedsteer/warp.py`: private monotone α warps, `lora.warp: none | kumaraswamy | kumaraswamy_mix | step`, trained with `fed.lr_warp`, `warp_warmup_rounds` and `warp_reg`.
- `fedsteer/calibrate.py` plus `eval_direction.py --posthoc_remap`: post-hoc isotonic remap (option F).
- `lora.offset`: private offset, coefficient o + s·h(α).
- `fed.adapter: private | shared | none` (`share_private` kept as an alias).
- `alpha_mode: global | local`: global = equal-weight mixture of the participants' CDFs (a shared protocol); runs save `alpha_reference.json`.
- Evaluation in global mode: per-client support, in/out-of-support percentile error, `reach_rate`.
- Newsroom config is now `alpha_mode: global`, `offset: true`, `adapter: private`, `warp: kumaraswamy_mix`.

**Why:**
- Under local α, the same α meant different behaviour on different clients: nypost.com copies at local α 0.13–1.
- The motivation is now "each client's data is skewed, and the skews differ".
- Rotation-0 supports on the global scale: nypost.com [0.51, 0.95], theguardian.com [0.04, 0.55], people.com [0.03, 0.70]; the others span almost the whole axis.

**Known confound:** at the top of the density tail, density partly measures summary length (Spearman 0.64 with length). Length will be tracked as an off-target metric.

**Tests:** 36/36. **No jobs yet.**

## 8. Timestamped runs and this log (2026-09-29)

**Change:**
- `fedsteer/runinfo.py`: a config's `out_dir` is now a prefix, and each run writes to `<prefix>_<YYYYmmdd-HHMMSS>[_j<jobid>]`. `run_info.json` records the job id, host, command and git commit (with uncommitted files). `train_fed.py --resume <run_dir>` continues a run.
- Evals are written to `<run>/evals/…__<stamp>.json` with provenance and are never overwritten.
- `scripts/summarize_sweep.py --match j<jobid>`.
- `sbatch/train_eval.sbatch`: the standard pipeline (train → dev sweep → select → test).
- Legacy run folders renamed with `exp_log/rename_legacy_runs.py`: `runs/toy_fedavg`, `runs/toy_local` and `runs/nr_pilot_llama1b` now carry the start stamp of the job that created them. Their eval files moved into `evals/`, stamped with the job that produced each one. `config.yaml` points to the new path, and `run_info.json` is backfilled. Old eval JSONs still contain the old paths internally; `summarize_sweep.py` falls back to the current folder.
- `sbatch/toy_smoke.sbatch` and `sbatch/newsroom_pilot.sbatch` now use stamped run folders; `sbatch/sweep_pilot.sbatch` and `sbatch/eval_pilot.sbatch` point to the renamed pilot run.

**Tests:** 37/37.

## 9. Global α: federated vs. local-only (2026-09-30)

**Question:** with a shared α scale, can clients steer toward parts of the scale their own data barely covers (nypost.com below 0.51, theguardian.com above 0.55), and does federated training beat local-only training there?

**Launch:** [`launch/exp09_global_fed_vs_local.sh`](launch/exp09_global_fed_vs_local.sh), the exact submit commands. Both jobs use the generic `sbatch/train_eval.sbatch`; only `OVERRIDES` differs.

**Setup (both jobs):** `sbatch/train_eval.sbatch` with `configs/newsroom_fedavg.yaml` as of this date:
- `alpha_mode: global`, `offset: true`, `warp: kumaraswamy_mix`, `adapter: private`, dev monitor on.
- Overrides: `model_name=meta-llama/Llama-3.2-1B-Instruct` (Qwen3-4B not cached), `fed.rounds=100`, `fed.save_every=5`.
- The job sweeps all 20 snapshots on dev (100 articles, with dev loss), selects the one with the lowest mean percentile error, and evaluates it on test (200 articles).
- Code: uncommitted working tree as of entry 8; `run_info.json` lists the dirty files.

| Job | Arm | Extra override | Log |
|---|---|---|---|
| 11011538 | federated (fedavg) | — | `sbatch/logs/fedsteer.o11011538` |
| 11011539 | local-only (B2; also writes `merged_shared.pt` for B3) | `fed.mode=local` | `sbatch/logs/fedsteer.o11011539` |

**What to compare:** per client, `pct_err_out_support` and `reach_rate` (the decisive metrics), plus `pct_err_in_support`, Spearman and concordance, with a focus on nypost.com and theguardian.com. Also check summary length as an off-target (density–length confound, entry 7).

**Results (dev):**

Dev selection: fedavg → round 60, local → round 90. The mean-over-clients gap is small but consistent from round 30 on: percentile error about 0.17 vs 0.19, Spearman about 0.90 vs 0.87, reach rate about 0.36 vs 0.30. **The means hide a client-specific effect.** Per client, averaged over rounds 60–100 (dev, 100 articles):

| Client | Support | Out-of-support error, fed / local | Reach rate, fed / local | Mean output pct at α = 0 … 1, fed | … local |
|---|---|---|---|---|---|
| theguardian.com | [0.04, 0.55] | **0.150 / 0.273** | **0.63 / 0.36** | 0.19 → **0.88** | 0.10 → **0.55** |
| people.com | [0.03, 0.70] | **0.168 / 0.213** | **0.41 / 0.29** | 0.19 → 0.80 | 0.13 → 0.67 |
| nypost.com | [0.51, 0.95] | 0.304 / **0.265** | 0.26 / **0.35** | **0.44** → 0.89 | **0.34** → 0.90 |
| forbes / wsj / aol / cbc / reuters | ~full | ≈ equal | ≈ equal (fed slightly higher) | ≈ equal | ≈ equal |

- **Federation lets under-covered clients copy more:** theguardian.com's local model caps at its own ceiling (about the 55th percentile); the federated one reaches the 88th.
- **It does not help nypost.com rewrite more:** local is better at the low end (0.34 vs 0.44 at α = 0). The transfer is asymmetric.
- On fully covered clients, fed ≈ local (a slight in-support edge for fed): no harm.

Caveats: one seed; only 3 of 8 participants have out-of-support regions; no confidence intervals yet; summary length not yet measured (the density–length confound could inflate "reaching" high α).

**Test results** (200 articles, selected checkpoints: fedavg round 60, local round 90; files `evals/eval_round_0060__…j11011538.json`, `evals/eval_round_0090__…j11011539.json`):

| | Percentile error | In-support | Out-of-support | Reach rate | Spearman |
|---|---|---|---|---|---|
| fedavg | **0.165** | **0.146** | **0.171** | **0.39** | **0.91** |
| local | 0.180 | 0.165 | 0.181 | 0.34 | 0.89 |

| Client (test) | Out-of-support error, fed / local | Mean output percentile at α = 0 … 1, fed | … local |
|---|---|---|---|
| theguardian.com | **0.138 / 0.233** | 0.18, 0.31, 0.49, 0.75, **0.90** | 0.11, 0.37, 0.48, 0.57, **0.63** |
| people.com | **0.157 / 0.197** | … 0.72, **0.85** | … 0.62, **0.68** |
| nypost.com | 0.285 / **0.247** | **0.40**, 0.56, … | **0.32**, 0.53, … |

Test confirms dev on both counts: the Guardian gain and the nypost reversal. These eval files predate saving generated texts, so they have **no quality metrics**; exp11's cap2k_base cell reproduces the fedavg setting with the new pipeline.

## 10. Quality metrics, in-training evaluation, regularizers (2026-09-30)

**Why:**
1. Steering metrics say *how extractive* an output is, not whether it's a good summary, and out-of-support generation especially needs a quality check.
2. Dev loss rose steadily (memorization).
3. The post-training sweep duplicated the in-training monitor and cost about 4 h per job.

**Changes:**
- **Quality metrics** (`fedsteer/quality.py`, `scripts/score_quality.py`), all reimplemented, no new packages:
  - AlignScore (official AlignScore-large checkpoint, the paper's chunk/sentence protocol). Sanity check: copied lead 0.99, faithful paraphrase 1.00, contradiction / wrong number / unsupported claim 0.00.
  - BERTScore F1 against the reference (roberta-large, layer 17).
  - Length and repeated-trigram rate.
  - **Reference baseline at matched extractiveness** (Ladhak et al., ACL 2022): real summaries of the same split, binned by global α; `gap_*` = generated − reference at the same α.
- **LLM judge** (`scripts/judge_quality.py`): Qwen2.5-7B-Instruct, FineSurE-style sentence-level faithfulness plus relevance and coherence (1–5), on a subsample (25 articles per client × 5 α) and on reference summaries per α bin. Blind to α and method.
- **Every eval now saves all generated texts** (`outputs`, `record_ids`), so quality can be scored after the fact. Shared per-client evaluation code lives in `fedsteer/evaluate.py`.
- **Full dev evaluation inside training** (`monitor.full_every`): at snapshot rounds, the same evaluation as `eval_direction.py`, written to `evals/eval_round_XXXX_dev__<stamp>.json`. **The separate sweep is removed** from `sbatch/train_eval.sbatch`. The pipeline is now train → select on dev → test → quality → judge (`JUDGE=0` to skip). Snapshots every 10 rounds.
- **Per-component regularizers** (`fedsteer/regularize.py`, config section `reg:`, all 0 by default):
  - `private_wd` (low-rank penalty on Pᵢ) and `lora.dropout`;
  - `decorr`: cos²(Pᵢ, D) per layer, keeping the attribute in D;
  - `fedprox_mu`: FedProx on B_d;
  - `shared_wd` on B_d;
  - `gain_l2`, `offset_l2`: priors on the private scalars.
  - Per-round term values are logged under `reg` in `train_log.jsonl`.
- **Evaluation models** are in the HF cache `/u/lucmon/lucmon/hf_home/hub`: Qwen2.5-7B-Instruct (15 GB, already there), AlignScore (4.6 GB, downloaded), roberta-large (1.4 GB, tokenizer files added).

**Tests:** 42/42.

**Job 11039406:** GPU smoke test of the whole pipeline, all regularizers on, 2 rounds ([launch](launch/exp10_pipeline_smoke.sh)).
- Run: `runs/smoke_pipeline_20260930-111641_j11039406`. **Passed**: every stage ran in 30 min; judge parse failures 0/310.
- Observation for later: real summaries in the most abstractive α bin score AlignScore 0.48, against 0.84–0.87 in extractive bins. Newsroom teasers often go beyond the truncated article, so quality must be read against the same-α reference baseline, not against a fixed bar.

## 11. Overfitting study: data size × regularization, federated arm (2026-09-30)

**Question:** does more data per client and/or per-component regularization remove the rising dev loss (memorization), and does it change steering or summary quality?

**Launch:** [`launch/exp11_data_x_reg.sh`](launch/exp11_data_x_reg.sh). Common settings: `configs/newsroom_fedavg.yaml` as of entry 10 (global α, offset, warp B, private adapter, full dev eval every 10 rounds), Llama-3.2-1B, 100 rounds, uniform aggregation. Pipeline: train → select on dev (percentile error) → test (200 articles) → quality → judge.

"Regularized" = `fed.lr_schedule=cosine lora.dropout=0.05 reg.private_wd=0.05 reg.shared_wd=0.01 reg.decorr=0.1 reg.fedprox_mu=0.1 reg.gain_l2=0.01 reg.offset_l2=0.01`. These weights are **untuned first guesses**.

| Job | Cell | Data per client | Regularized |
|---|---|---|---|
| 11041280 | cap2k_base | 2,000 (same as entry 9's fedavg run) | no |
| 11041281 | cap2k_reg | 2,000 | yes |
| 11041282 | cap4k_base | 4,000 | no |
| 11041283 | cap4k_reg | 4,000 | yes |
| 11041284 | natural_reg | all in pool (reuters.com 4,023, bbc.com 4,228, theguardian.com 4,705, others 5,000) | yes |

**What to compare:**
- dev-loss curves (does the rise go away?);
- steering on dev/test (percentile error, out-of-support error, reach rate);
- quality (AlignScore / BERTScore gaps to the same-α reference; judge faithfulness, relevance, coherence);
- per client, especially nypost.com and theguardian.com.

The natural-size cell tells whether unequal client sizes create problems (small clients overfitting faster).

**Expected timing:** about 5–6 h per job; 3 GPUs on `dali`, so 2 jobs queue behind the first 3.

**Next:** the local-only arm (B2) at the winning setting, then the nypost.com adapter test (`adapter: shared/none`).

**Results, partial (2026-09-30; cap2k_base, cap4k_base and cap2k_reg done; cap4k_reg and natural_reg failed and were resubmitted, entry 14):**

cap2k_reg test (round 60): percentile error 0.192, in-support 0.152, out-of-support **0.215**, reach 0.31, Spearman 0.885. That's worse than cap2k_base (0.162 / 0.144 / 0.167 / 0.40 / 0.917), confirming the dev finding. Quality is about unchanged (AlignScore in 0.784 / out 0.754; gaps to the same-α reference −0.02 / +0.07).


*Overfitting (dev loss, mean over clients):*

| Run | Min dev loss (round) | Final dev loss | Final train loss |
|---|---|---|---|
| cap2k_base | 1.079 (13) | 1.896 | 0.028 |
| cap2k_reg | 1.079 (12) | **2.256** | **0.006** |
| cap4k_base | **1.043 (25)** | **1.442** | 0.262 |

- **4k data roughly halves the dev-loss rise** and moves the minimum from round 13 to round 25 (one pass = 12.5 vs 25 rounds).
- The **regularization bundle makes memorization worse**: cosine decay lets training loss reach 0.006.
- `decorr` stayed at about 0: Pᵢ and D are already nearly orthogonal, so that term is inactive and the "Pᵢ absorbs the attribute along D" hypothesis is not supported in that form. FedProx and the priors are tiny.

*Steering, dev* (round 100; best round in parentheses):

| Run | Percentile error | Out-of-support error | Reach | Spearman | Concordance excl. near-ties | Near-tie |
|---|---|---|---|---|---|---|
| cap2k_base | 0.173 (0.163 @60) | 0.186 | 0.35 | 0.90 | 0.89 | 0.23 |
| cap2k_reg | 0.198 (0.192 @60) | 0.225 | 0.29 | 0.87 | 0.86 | 0.30 |
| cap4k_base | **0.156** (@100, still improving) | **0.166** | **0.37** | **0.93** | **0.90** | **0.22** |

The regularized run is worse on every steering metric at every round. The likely cause is the gain/offset priors pinning gains near 1: the unregularized runs use gains 1.4–2.2. That's untested; an ablation is needed.

*Test* (200 articles):

| Run (selected round) | Percentile error | In-support | Out-of-support | Reach | Spearman | Concordance excl. near-ties | Near-tie | Endpoint near-tie |
|---|---|---|---|---|---|---|---|---|
| entry-9 fedavg (60), reproducibility check | 0.165 | 0.146 | 0.171 | 0.39 | 0.910 | – | – | – |
| cap2k_base (60) | 0.162 | 0.144 | 0.167 | 0.40 | 0.917 | 0.899 | 0.222 | 0.012 |
| cap4k_base (100) | **0.155** | **0.134** | 0.165 | 0.40 | **0.926** | 0.904 | 0.215 | 0.009 |

- cap2k_base reproduces the entry-9 run.
- 4k improves in-support steering; out-of-support is about the same.

*Quality, test* (generated vs. real summaries at the same α; α = 0, 0.25, 0.5, 0.75, 1):

| | AlignScore | Length (tokens) | Judge faithfulness | Judge relevance | Judge coherence |
|---|---|---|---|---|---|
| Real summaries | 0.48, 0.70, 0.84, 0.87, 0.84 | 21.9, 23.2, 29.1, 38.0, 46.8 | 0.75, 0.86, 0.99, 0.88, 0.75 | 3.37 … 3.90 | 3.87 … 4.30 |
| cap2k_base | 0.53, 0.66, 0.82, 0.91, 0.91 | 26.0, 27.8, 32.3, 42.6, **54.7** | 0.76, 0.82, 0.85, 0.83, 0.81 | 3.49 … 3.84 | 4.04 … 4.30 |
| cap4k_base | 0.53, 0.65, 0.82, 0.90, 0.90 | 25.9, 27.1, 32.2, 40.9, **50.5** | 0.77, 0.84, 0.84, 0.80, 0.79 | 3.53 … 3.80 | 4.04 … 4.32 |

- Generated summaries track real ones at the same α on consistency, faithfulness, relevance and coherence.
- Low AlignScore at low α is inherent to abstractive summaries: real ones score 0.48 too.
- Generated summaries are **longer** than real ones, especially at α = 1 (+8 tokens), which reflects the density–length coupling.
- Judge parse failures: 0.

*Out-of-support quality, theguardian.com* (4k, test; its support is [0.04, 0.55]): at α = 0.75 and 1, its outputs reach the 80th and 91st percentiles with AlignScore 0.96 / 0.96 [real 0.87 / 0.84], judge faithfulness 0.86 / 0.83 [0.88 / 0.75] and relevance 3.80 / 3.92 [3.70 / 3.90]. **Quality is on par with real summaries at those α**, but length at α = 1 is 63 tokens vs 47 (long leads copied).


## 12. Data finding: nypost.com summaries are truncated article leads in the raw data (2026-09-30)

**Found while inspecting outputs** that end in "…" (e.g. "…pleading with his office to make an arrest. “Her com...").

- These are **not** generation cutoffs: only 0–7% of outputs reach the 128-token limit.
- **They come from the raw Newsroom data, not our preprocessing.** In the raw `dev.jsonl.gz`, 92% of nypost.com summaries end in "…", with median length 198 characters (74% between 190 and 203). The build script only collapses whitespace in summaries.
- In our training data: 93% end in "…", 67% are an exact prefix of the article, and median length is exactly 200 characters. Newsroom took summaries from HTML metadata; nypost's metadata is its first ~200 characters plus "…".
- The model learns this: 73% of generated nypost outputs end in "…".
- Smaller shares of article prefixes elsewhere: latimes.com 16%, reuters.com 14%, cbc.ca 13%, mashable.com 11%; the others ≤ 7%.

**Decision (user): keep the data as is; don't filter raw summaries.** nypost.com is treated as a real client with a hard *format convention* (fixed-length truncated lead), an extreme form of heterogeneity.

**Consequences:**
- Interpretation: nypost's skew is a fixed format convention, not an editorial preference for copying. The nypost results in entries 9 and 11 (failed low-end transfer, federated < local at low α) should be described as "a client with a hard format convention does not pick up rewriting from other clients". The theguardian.com and people.com findings are unaffected.
- **Counting:** near-identical outputs that differ only in where the cut-off falls ("…Her..." vs "…Her com...") are neither text ties nor score ties; density counts the longer copy as an increase and inflates nypost's measured steerability.

**Proposed:** a near-tie metric (normalize trailing cut-offs before comparing), reported next to the exact tie rate. The `adapter: shared/none` test becomes a test of whether the private adapter locks in nypost's format.

## 13. Near-tie metric (2026-09-30)

**Change:**
- `fedsteer/metrics.py`: `normalize_output` / `near_identical` / `text_tie_metrics`. Outputs count as near-identical if they match after normalization (quotes, dashes and whitespace unified; a trailing ellipsis and the possibly cut-off last word removed) up to a small token edit (difflib ratio ≥ 0.95).
- New per-client metrics: `text_tie_rate`, `near_tie_rate`, `near_no_effect_rate`, `endpoint_near_tie_rate`, plus `adjacent_increase_rate_nt` and `concordance_nt` (the score-based rates with near-identical pairs counted as ties).
- Added automatically to every new evaluation (`fedsteer/evaluate.py`); the exp11 jobs' final test evals will include them. The exp11 in-training dev evals were computed with older code, so use `scripts/tie_report.py` (read-only, works on any eval file with saved texts).

**Tests:** 43/43.

**First numbers** (exp11 cap2k_base, dev, round 100; `scripts/tie_report.py --run runs/exp11_cap2k_base_20260930-114747_j11041280 --round 100`):
- Mean over clients: exact tie 0.16 → near tie 0.23; per-step increase 0.74 → 0.68 excluding near-ties; concordance 0.91 → 0.89.
- Most clients change little; endpoints are near-identical in ≤ 1% of articles.
- **nypost.com:** near tie 0.50 (exact 0.30); per-step increase 0.61 → **0.43**; concordance 0.84 → 0.78.
- **reuters.com:** near tie 0.41; per-step increase 0.60 → 0.50.

So their measured steerability was inflated by cut-off-position changes (entry 12).

## 14. Two jobs failed from a mid-run code edit; jobs now run from a code snapshot (2026-09-30)

**Failure:** jobs 11041283 (cap4k_reg) and 11041284 (natural_reg) died at round 10 (22 min in) with:

```
ImportError: cannot import name 'text_tie_metrics' from 'fedsteer.metrics'
```

**Cause:** the training process imported `fedsteer.metrics` at start, but the monitor imported `fedsteer.evaluate` *lazily* at the first full evaluation (round 10). Entry 13's edit, made while the jobs were in rounds 1–9, added `text_tie_metrics` to both files. The jobs then loaded the new `evaluate.py` against their old, in-memory `metrics.py`. Jobs past round 10 (11041281) were unaffected.

**Fix:**
1. `sbatch/train_eval.sbatch` copies the code (`fedsteer/`, `train_fed.py`, `eval_direction.py`, `scripts/`) to `runs/_code/<stamp>/` at job start and runs every stage from there. The path is written to `<run>/code_snapshot.txt`, so the run records exactly which code it used.
2. `train_fed.py` and `eval_direction.py` put their own folder first on `sys.path`, so a snapshot always imports its own package. Verified with `python -v` that every entry point imports the snapshot's `fedsteer`.
3. The monitor binds its evaluation imports when the job starts, not lazily.

**Tests:** 43/43.

**Resubmitted** ([launch](launch/exp11_resubmit_failed.sh)) with identical settings: 11060306 (cap4k_reg), 11060307 (natural_reg). Queued: `dali`'s 3 GPUs are taken by another user's jobs.

## 15. Analysis: federated vs. local-only, per client with paired bootstrap (2026-09-30)

**Comparison** (no new jobs): test, 200 articles per client, the same articles and global α scale in both runs. Local = entry 9's local run (11011539, 2k, round 90). Fed 2k = exp11_cap2k_base (11041280, round 60), the only equal-data contrast. Fed 4k = exp11_cap4k_base (11041282, round 100); it has more data, so its row is not a fair comparison. Each cell is the difference in per-article percentile error (fed − local; negative = federated better), with a 95% paired bootstrap CI over articles.

| Client | Support | Overall error, fed2k − local2k | Out-of-support error, fed2k − local2k | Overall error, fed4k − local2k (unequal data) |
|---|---|---|---|---|
| theguardian.com | [0.04, 0.55] | **−0.073** [−0.084, −0.063] | **−0.099** [−0.116, −0.083] | −0.083 |
| people.com | [0.03, 0.70] | **−0.031** [−0.040, −0.022] | **−0.046** [−0.059, −0.033] | −0.023 |
| forbes.com | [0.03, 0.99] | **−0.026** [−0.035, −0.017] | +0.021 [+0.008, +0.033] | −0.014 |
| reuters.com | [0.10, 0.97] | **−0.023** [−0.033, −0.013] | +0.001 (n.s.) | −0.025 |
| wsj.com | [0.04, 0.89] | **−0.020** [−0.027, −0.012] | −0.016 [−0.028, −0.004] | −0.022 |
| aol.com | [0.06, 0.82] | **−0.009** [−0.017, −0.001] | −0.019 [−0.033, −0.007] | −0.011 |
| cbc.ca | [0.11, 0.95] | +0.016 [+0.008, +0.023] (local better) | +0.023 [+0.012, +0.035] | −0.021 |
| nypost.com | [0.51, 0.95] | +0.017 [+0.005, +0.030] (local better) | +0.028 [+0.012, +0.043] | −0.001 (n.s.) |

- **At equal data, federated is significantly better on 6 of 8 clients and significantly worse on 2** (cbc.ca and nypost.com, the two most copy-heavy). Mean percentile error 0.180 → 0.162. Worst client: **0.247 (fed) vs 0.235 (local)**, so the worst case is not improved.
- Federated 4k beats local 2k on 7 of 8 clients (nypost.com a tie), but that row is confounded by data size; **local 4k is needed**.
- CIs cover article sampling only. With one training seed each, seed variance is not covered.

## 16. The federated 4k run: data effect, and a local-only 4k baseline (2026-09-30)

**Comparisons now possible** (test; paired bootstrap over the same 200 articles; negative = first better; * = 95% CI excludes 0):

| Client | Fed 4k − fed 2k: overall | in-support | out-of-support | Fed 4k − local 2k (confounded): overall | out-of-support |
|---|---|---|---|---|---|
| forbes.com | +0.012* | +0.034* | −0.021* | −0.014* | −0.000 |
| theguardian.com | −0.010* | −0.015* | −0.006 | −0.083* | −0.106* |
| people.com | +0.008 | −0.010* | +0.020* | −0.023* | −0.026* |
| wsj.com | −0.002 | −0.004 | +0.001 | −0.022* | −0.015* |
| aol.com | −0.002 | −0.024* | +0.030* | −0.011* | +0.010 |
| cbc.ca | **−0.037*** | −0.052* | −0.014* | −0.021* | +0.009 |
| reuters.com | −0.003 | −0.003 | −0.003 | −0.025* | −0.001 |
| nypost.com | **−0.018*** | −0.004 | −0.022* | −0.001 | +0.006 |
| significantly better / worse | 3 / 1 | | | **7 / 0** | |

- **Doubling the data within federated training helps the two clients that lost to local at 2k:** cbc.ca (−0.037) and nypost.com (−0.018), the most copy-heavy. For other clients the effect is mixed and small (forbes.com slightly worse overall, aol.com and people.com worse out-of-support).
- Federated 4k vs. local 2k: better on 7 of 8, worse on none. **Confounded by data size**, hence the run below.

**Job 11061046** ([launch](launch/exp16_local_cap4k.sh)): local-only (B2) at 4k with exactly exp11_cap4k_base's overrides except `fed.mode=local`. It also writes the merged direction (B3). Queued behind 11060306/11060307 (`dali` busy).

## 17. Shared calibration under the global α scale (2026-10-01)

**Why:** under the global α scale, α means the same behaviour on every client, so there's no principled reason for a per-client mapping onto D. Private gains/offsets/warps only compensate for a private adapter that has absorbed part of the attribute, and for skewed clients they're fitted only inside their own support, so they extrapolate without data (nypost.com needs coefficient −0.58 at α = 0, below anything other clients train; see the 4k curves in the entry-16 discussion). The offset's original purpose (start a client partway along D under local α) disappears under global α.

**Change:**
- `fed.calibration: private | shared` (`fedsteer/fed.py`). **shared** = one gain, offset and warp for all clients, part of the server state, averaged every round like D (optimizer state reset like D). Clients never store their own; snapshots restore the shared one for any client.
- `calibration=shared` is refused in local mode (nothing is aggregated there).
- `lora.offset: false` removes the offset. The config default stays `calibration: private`, so earlier and queued jobs are unchanged.
- Not implemented (by decision): calibration `none` (g(α) = α).

**Tests:** 46/46.

**Jobs** ([launch](launch/exp17_shared_calibration.sh)): 4k per client, no regularizers, 100 rounds, otherwise identical to exp11_cap4k_base (11041282 = federated, private calibration, with offset).

| Arm | Federated | Local |
|---|---|---|
| private calibration, offset (reference) | 11041282 (done) | **11061046** (exp16, running). Local with "shared" calibration is identical to this, so it isn't resubmitted |
| B: shared calibration, offset | 11063906 | (= 11061046) |
| B, no offset | 11063910 | 11063911 (local, per-client gain + warp, no offset) |

**What to compare:** per client, in-support / out-of-support error, reach rate, near-ties and quality, with the copy-heavy clients (nypost.com, reuters.com, cbc.ca) and theguardian.com as key cases. Also federated vs local within each arm.

**Status:** queued behind 11060306/11060307/11061046 (`dali` has 3 GPUs).

## 18. Plan updated (2026-10-01)

`federated-steering-plan.md` was rewritten to match the current design and results. No code or jobs changed.
- Claims reframed around **coverage** ("each client's data is skewed, and the skews differ").
- Model section: global α, calibration g(α) = o + s·h(α) with private/shared modes, adapter modes, regularizers (not part of the method yet).
- Data facts (budgets, drift 3×200, nypost.com format convention, density–length coupling).
- Current metric suite and selection procedure.
- A ✅/🔄/⏳ status on every claim, metric, baseline, figure and gate; next steps; a changelog pointing to this log.
- The gap review behind the update: G1 (prompting, B1) is untested and the top priority; B3 isn't evaluated; E2, E3, B4, B5, A1–A3, seeds and the Qwen3-4B backbone are pending.

## 19. Baselines B1 (prompting), B3 (one-shot merge), B4 (federated CAA) (2026-10-01)

**Note on exp11:** the resubmitted cap4k_reg (11060306) and natural_reg (11060307) were **cancelled by the user** at 00:09 (rounds 13–14). The regularization bundle had already been shown to hurt (entry 11), and this freed GPUs for exp17. Not resubmitted.

**Change:**
- **`fedsteer/baselines.py` + `eval_baselines.py`:** B1 and B4 run on a client's model **with the learned direction removed** (B_d = 0). The client keeps its private adapter (house style) and calibration; only the control mechanism changes.
  - **B1 prompting:** the target level is stated in the prompt ("Target extractiveness N on a 0–100 scale, 0 = rewrite everything in your own words, 100 = copy whole sentences"). Optionally k examples (150-word article excerpts + summaries) from the client's **own** training data, nearest to the target α. Variants: k=0 client model; k=3 client model; k=3 base model (no adapter).
  - **B4 federated CAA:**
    - each client computes the mean-difference activation vector (output of the middle decoder layer, teacher-forced, averaged over summary tokens) between its own top- and bottom-quartile summaries (200 each);
    - the server averages the vectors uniformly;
    - each client fits one scalar gain s from {0.05, 0.1, 0.2, 0.3, 0.5, 0.8} on 30 dev articles (lowest percentile error);
    - steering adds s·(2α − 1)·(mean hidden norm)·v̂ to the residual stream at every position.
  - The client data used = exactly the run's training subset (same cap and seed), so the baselines see no more data than the method.
- **`scripts/merge_local_directions.py` (B3):** uniform average of a local-only run's per-client directions at one checkpoint; evaluated with `eval_direction.py --shared` (each client keeps its own adapter and calibration).
- All write the standard eval format (metrics, near-ties, all texts); quality scoring runs after each.
- **`sbatch/eval_baselines.sbatch`** (steps selectable; runs from a code snapshot).
- **Cluster:** both job scripts now use `--account=lucmon-ic --partition=dali,IllinoisComputes-GPU --gres=gpu:1` (whichever frees first). The pending exp17 job 11063911 was moved to both partitions.

**Tests:** 49/49.

**Jobs** ([launch](launch/exp19_baselines.sh)): smoke 11065060 (replaced by 11077014, see 19a) → (afterok) B1 11065061, B4 11065062, B3-2k 11065063; B3-4k 11065064 also waits for local 4k (11061046).
- B1/B4 use exp11_cap4k_base's dev-selected round-100 client models.
- B3 compares with the federated run of the same data size (2k: exp11_cap2k_base / entry 9; 4k: exp11_cap4k_base).

**Fairness notes for the write-up:**
- B1 uses at most k=3 labelled examples per prompt; B4 uses 400 training summaries plus 30 dev articles for its gain; the method uses all training labels.
- B1's prompt template is not tuned (one template). The plan calls for tuning on validation, so a weak B1 result needs a second template before concluding gate G1.

**Gates addressed:** G1 (B1), G3 (B3).

### 19a. Cluster incident (2026-10-01)

- Smoke job 11065060 was dispatched to **ccc0387** (IllinoisComputes-GPU) at 02:17. SLURM failed to retrieve the user environment there; the job was requeued and **held** (`user env retrieval failed requeued held`). No script output, so it failed before our code ran. That blocked the dependent B1/B3/B4 jobs for about 7 h.
- The same partition runs our job 11063911 fine on ccc0388, so this is a node-specific problem with ccc0387.
- **Fix:** moved the smoke job to `dali` (idle), `scontrol release`d it (started 09:45 on ccc0284), and set `ExcNodeList=ccc0387` on the dependent jobs 11065061–11065064.
- If the problem recurs on other nodes, add `#SBATCH --exclude=…` to the job scripts or ask the cluster admins about ccc0387.
- **Second bug (mine):** the smoke job received `STEPS=b1` only. `sbatch --export` splits its argument on commas, so `STEPS=b1,b4,b3` lost b4 and b3, and the smoke test would have exercised B1 only. Fix: `sbatch/eval_baselines.sbatch` takes `+`-separated steps (`STEPS=b1+b4+b3`, converted internally); launch file corrected. Smoke 11065060 was cancelled and resubmitted as **11077014** on `dali` (confirmed `STEPS=b1,b4,b3`). Dependent jobs 11065061–11065064 now depend on `afterok:11077014` (and still exclude ccc0387). The real jobs were not affected: each passes a single step.

## 20. Results: shared vs. private calibration, federated vs. local, all at 4k (2026-10-01)

Five runs, 4k examples per client, no regularizers, 100 rounds; **every run selected round 100 on dev** (still improving). Test, 200 articles per client.

Command: `python scripts/compare_runs.py --run F_priv=…j11041282 --run F_shared=…j11063906 --run F_shared_noO=…j11063910 --run L_priv=…j11061046 --run L_noO=…j11063911 --pair …` (new script). Full output is reproducible from the saved evals.

| Run | Calibration | Offset | Pct error (worst) | In-support | Out-of-support (worst) | Reach | Spearman (worst) | Concordance excl. near-ties | Near-tie |
|---|---|---|---|---|---|---|---|---|---|
| F_priv (11041282) | private | yes | 0.155 (0.229) | **0.134** | 0.165 (0.253) | **0.397** | 0.926 (0.832) | 0.904 | 0.215 |
| F_shared (11063906) | shared | yes | 0.158 (0.229) | 0.141 | 0.167 (0.204) | 0.377 | 0.920 (0.855) | 0.899 | 0.217 |
| **F_shared_noO** (11063910) | **shared** | **no** | **0.151 (0.199)** | 0.135 | **0.161 (0.188)** | 0.387 | **0.933 (0.910)** | **0.907** | **0.204** |
| L_priv (11061046) | per client (local) | yes | 0.161 (0.206) | 0.142 | 0.166 (0.228) | 0.353 | 0.905 (0.842) | 0.891 | 0.249 |
| L_noO (11063911) | per client (local) | no | 0.165 (0.205) | 0.148 | 0.165 (0.249) | 0.363 | 0.906 (0.839) | 0.891 | 0.246 |

**Per client** (paired bootstrap; significant = 95% CI excludes 0):
- **F_shared_noO vs L_noO (corrected 10-01; the first version reported only the overall-error column):**

  | Metric | Federated significantly better | Local significantly better |
  |---|---|---|
  | Overall error | 4 (theguardian −0.066, nypost −0.028, wsj, cbc) | 0 |
  | In-support error | 6 | 1 (nypost +0.022) |
  | **Out-of-support error** | 2 (theguardian −0.101, nypost −0.040) | **4** (forbes +0.029, wsj +0.025, aol +0.020, reuters +0.028) |
  | Spearman | 3 (theguardian +0.104, nypost +0.071, aol +0.017) | 0 |

  **Mixed per publication.** Federation helps a lot where a client's data has *large gaps* (theguardian, nypost: 3–4 of 5 test α outside support) and improves in-support calibration for most clients. It is slightly **worse at the extreme α (0, 1)** for clients whose data covers about [0.05, 0.95]. For those clients, "out-of-support" is only the 5% tails, where a local model fits its own extremes better. Proposed: report *gap* regions (essentially no client data) separately from *tail* regions.
- F_priv vs L_priv: better on 3/8, **worse on 2/8** (reuters.com +0.027, nypost.com +0.022), as at 2k (entry 15).
- F_shared vs F_priv: shared calibration **fixes nypost.com** (−0.042; out-of-support −0.053, the predicted extrapolation effect) but **costs** theguardian.com, people.com, wsj.com and reuters.com (+0.011 to +0.024).
- F_shared_noO vs F_shared: dropping the offset helps people.com (−0.014), reuters.com (−0.030) and nypost.com (−0.010), and hurts nobody.
- Out-of-support error for clients with near-full support (forbes.com [0.03, 0.99] etc.) is about the extreme α only; local is slightly better there (+0.016 to +0.029 for federated).

**Quality, test** (gap = generated − real summaries at the same α):

| Run | AlignScore in / out | Gap in / out | BERTScore in / out | Length gap in / out | Judge faithful in / out | Judge relevance gap in / out | Judge coherence in / out |
|---|---|---|---|---|---|---|---|
| F_priv | 0.780 / 0.734 | −0.024 / +0.051 | 0.900 / 0.891 | +2.5 / +4.0 | 0.82 / 0.78 | −0.01 / +0.02 | 4.17 / 4.18 |
| F_shared | 0.776 / 0.735 | −0.029 / +0.052 | 0.900 / 0.890 | +2.9 / +2.2 | 0.83 / 0.80 | −0.01 / +0.00 | 4.21 / 4.19 |
| F_shared_noO | 0.779 / 0.729 | −0.026 / +0.046 | 0.900 / 0.889 | +2.0 / +2.3 | 0.82 / 0.79 | +0.02 / +0.02 | 4.21 / 4.19 |
| L_priv | 0.794 / 0.698 | −0.010 / +0.014 | 0.900 / 0.887 | +3.5 / +0.5 | 0.81 / 0.78 | −0.04 / **−0.12** | 4.17 / 4.11 |
| L_noO | 0.790 / 0.694 | −0.015 / +0.011 | 0.902 / 0.887 | +2.7 / −0.5 | 0.84 / 0.79 | −0.02 / −0.05 | 4.19 / 4.14 |

- Quality is on par everywhere. Faithfulness, BERTScore and coherence are about equal across runs.
- Out-of-support, the federated runs are **more relevant** than local (judge relevance gap ≈ 0 vs −0.05 to −0.12) and more consistent relative to the same-α reference.
- Federated runs are slightly longer out-of-support (+2 to +4 tokens), the density–length coupling as they reach higher α.
- The judge uses 25 articles per client: no CIs, treat as indicative.

**Takeaways:** shared calibration without offset is the best federated design on mean, worst client, out-of-support and ties. Against local it is a **coverage trade-off**: large gains for clients missing large parts of the scale, better in-support calibration for most, slightly worse at the extreme α for clients with near-full coverage. That fits the global-α design (only house style is private). Caveats: one seed; all runs chose their last checkpoint, so longer training could change the ranking; baselines B1/B3/B4 still pending.

## 21. Evaluation settings E2 (held-out clients) and E3 (drift) (2026-10-01)

**Change:**
- **`fedsteer/adapt.py`** — three single-component training routines (everything else frozen and restored):
  - `train_adapter_sft`: plain, label-free SFT of the private adapter. The steering module is switched off during training via `direction_off`.
  - `fit_calibration`: gain/offset/warp on k labelled examples.
  - `train_local_direction`: a client's own direction from zero (+ calibration) on k examples, adapter frozen.
- **`e2_heldout.py`** (claim C2): for each of the rotation's 4 held-out clients (telegraph.co.uk, bbc.com, mashable.com, latimes.com):
  1. Train the adapter by plain SFT with steering off, at the participants' budget (rounds × local_steps). It is **cached** in `runs/_e2_adapters/`, since it doesn't depend on the federated run.
  2. Attach the frozen federated direction and evaluate on test (200 articles), one standard eval file per setting:
     - `frozen_k0`: the run's calibration as is (shared calibration, i.e. **zero-shot**); identity calibration for private-calibration runs;
     - `frozen_cal_k16/k64`: calibration fitted on k labelled examples;
     - baselines `localdir_k16/k64` (own direction from the same k) and `prompt_k16/k64` (3 shots from the same k).

   α uses the run's global reference (participants' mixture).
- **`e3_drift.py`** (claim C3): for each participant, starting from the checkpoint:
  - stage 0, then 3 drift stages (200 later-year pairs each), each with 100 steps of plain SFT with steering off;
  - after each stage: evaluate as is, and after a k=16 calibration refit (a branch; the drift chain continues un-refit);
  - test articles: 100 per client.
- **`sbatch/eval_settings.sbatch`:** steps e2/e3, code snapshot, quality scoring of all produced files.
- All job scripts now exclude ccc0387.

**Tests:** 52/52 (each helper changes only its own component; direction and requires_grad flags restored).

**Jobs** ([launch](launch/exp21_e2_e3.sh)): smoke 11082124 → (afterok)
- E2 on the method run: 11082125;
- E3 on the method run: 11082128;
- E3 on its local counterpart: 11082129;
- E2 on the private-calibration federated run: 11082130 (after 11082125, reusing cached adapters).

**Design choices to note in the write-up:**
- Drift SFT runs with steering off: someone fine-tuning without knowing about the knob.
- E2 adapters use the participants' training budget.
- The E3 refit uses k=16 labelled pairs from the same drift stage.

### 21a. Smoke result and fix (2026-10-01)

- Smoke 11082124: **E2 (all settings) and E3 (stages 0–1, as is and refit) ran correctly.** It then failed in quality scoring with a `KeyError` on a telegraph.co.uk URL: `scripts/score_quality.py` looked up records of the run's participants only, but E2 evaluates **held-out** clients.
- **Fix:** record lookup over all clients. The same-α reference baseline still uses participants' summaries only. Verified by scoring the smoke's E2 output on CPU.
- The four dependent jobs (blocked by `DependencyNeverSatisfied`) were cancelled and resubmitted without the smoke dependency: E2 on the method run **11083572**, E3 on the method run **11083573**, E3 on the local counterpart **11083574**, E2 on the private-calibration run **11083575** (after 11083572).

## 22. Results: baselines B1 (prompting), B3 (one-shot merge), B4 (federated CAA) (2026-10-01)

All on test (200 articles per client, 4k data). B1/B4 use the client models of F_priv (exp11_cap4k_base, round 100) with the direction removed; B3 merges the local-4k run's (exp16) directions at round 100. "Method" = federated, shared calibration, no offset (exp17, 11063910). Produced with `scripts/compare_runs.py` (now accepts `name=run_dir::file_pattern`).

| | Pct error (worst) | In-support | Out-of-support | Reach | Spearman | Near-tie | Endpoint near-tie | AlignScore in / out (gap) | Length gap out |
|---|---|---|---|---|---|---|---|---|---|
| **Method** | **0.151 (0.199)** | 0.135 | **0.161** | 0.39 | **0.933** | 0.204 | 0.002 | 0.779 / 0.729 (+0.05) | +2.3 |
| F_priv (models for B1/B4) | 0.155 (0.229) | 0.134 | 0.165 | 0.40 | 0.926 | 0.215 | 0.009 | 0.780 / 0.734 (+0.05) | +4.0 |
| L_priv (local 4k) | 0.161 (0.206) | 0.142 | 0.166 | 0.35 | 0.905 | 0.249 | 0.004 | 0.794 / 0.698 (+0.01) | +0.5 |
| **B3 merge** (local 4k directions) | 0.184 (0.229) | 0.149 | 0.220 | 0.26 | 0.878 | 0.223 | 0.009 | 0.788 / 0.708 (+0.03) | −1.7 |
| **B4 CAA** (layer 8) | 0.263 (0.317) | 0.212 | 0.312 | 0.22 | 0.752 | 0.048 | 0.000 | 0.594 / 0.476 (**−0.21**) | **+11.5** |
| **B1 prompt k=0** | 0.378 (0.412) | 0.240 | 0.500 | 0.03 | **−0.002** | 0.738 | **0.704** | 0.648 / 0.650 | −6.9 |
| **B1 prompt k=3** | 0.378 (0.401) | 0.247 | 0.495 | 0.05 | 0.040 | 0.196 | 0.182 | 0.608 / 0.605 | −7.2 |
| **B1 prompt k=3, base model** | 0.341 (0.355) | 0.243 | 0.467 | 0.12 | 0.077 | 0.003 | 0.000 | 0.609 / 0.631 | **+72** (≈110 tokens) |

**Per client (paired bootstrap):**
- Method vs B1 (k=3): better on **8/8**.
- Method vs B4: better on **7/8** (reuters.com a tie).
- Method vs B3: better on **7/8**, worse on 1/8 (reuters.com +0.017).
- B3 vs local 4k: B3 **worse on 6/8**, better on 0/8.
- B4 vs B1: B4 better on 8/8.
- At 2k, the ordering is the same (worst client in brackets):
  - federated 2k: 0.162 (0.247), round 60;
  - local 2k: 0.180 (0.235);
  - B3 merge of the local 2k directions at round 90: 0.195 (0.236).
  - Federated vs B3: better on 5/8, worse on 0/8.
  - B3 vs local: better on 2/8 (theguardian, reuters), worse on 4/8.
- **Pattern in both sizes:** merging helps only theguardian.com out-of-support (−0.07 at 4k, −0.11 at 2k vs local). It is the narrowest-support client (α 0.04–0.55), and it borrows the high-α behaviour other clients learned. The merge hurts the broad-support clients' extremes. Federated training gets the theguardian gain without that loss. This is the same coverage argument as entry 20.
- `scripts/compare_runs.py` now reports the text-based tie columns as n/a for older eval files without saved outputs (the local 2k test eval); before, it crashed.

**Readings:**
- **B1 prompting does not steer** extractiveness with this 1B model. With k=0, α has no effect (Spearman ≈ 0; 70% of articles give near-identical text at α = 0 and 1: the fine-tuned adapter ignores the level instruction). Few-shot examples change the text but not in the right direction. The base model writes ~110-token summaries regardless.
  - **Gate G1 passes, with caveats:** one untuned template; a 1B model may follow numeric style instructions poorly. Re-check on the final backbone (Qwen3-4B) with a second template. The stronger "conditioning through the input" baseline is **A3** (shared/private LoRA trained with the level as a text control token), still pending.
- **B3 one-shot merging is worse than both** the method and local training (directions averaged once don't match each client's adapter and calibration). **Gate G3 passes:** iterative federated training is needed.
  - Caveat: B3 keeps each client's local calibration, fitted to its own direction; a merged-direction variant with a refitted gain would be fairer.
- **B4 activation steering** steers partially (Spearman 0.75) but much worse than the method, and **degrades quality**: AlignScore −0.21 against the same-α reference out-of-support, summaries +11 tokens. Clients' CAA vectors agree moderately with the average (cosine 0.64–0.84).
  - **B4 is under-tuned:** 6 of 8 clients picked the largest gain in the grid (0.8), with dev error still falling. It needs a wider gain grid (and maybe a layer choice) before final numbers.

## 23. E2 redesigned: the new client trains with D in the loop (2026-10-01)

**Problem with entry 21's E2 (pointed out by the user):** the held-out client's adapter was trained by plain SFT with the direction **off**, and D was attached only at evaluation. Participants' adapters were trained together with D (α per pair), so this handicaps the frozen direction. It shows in 11083572's first results:
- frozen D with the run's shared calibration: percentile error 0.335; 0.310 after a k = 16 calibration refit;
- outputs at α = 0 already sit at percentile 0.4–0.78 (overshoot).

Also, α for a new client is free to compute: it is the global CDF of the pair's density. So "k labelled examples" was the wrong axis. The scarce resource is the client's **data**, n pairs.

**New E2** (`e2_heldout.py`, rewritten). For each held-out client and n ∈ {16, 64, 256, 1024, all}:
- **Subsets:** the client's first n training pairs under a fixed shuffle, so the subsets are nested.
- **Settings:**
  - `frozen_D` (main): the participants' objective (α per pair, steering on) with D frozen. Trains P from scratch, plus the calibration if the run's calibration is private; shared calibration stays at the run's value.
  - `local_D` (baseline, no federation): same objective and step budget; the client trains its own D from zero (P + D + calibration).
  - `plugin` (ablation = the old design): P by SFT with steering off, then D attached. Calibration as the run has it: shared → frozen; private → fitted on the same n pairs.
  - `prompt` (baseline): the plugin adapter without D; level stated in the prompt; 3 shots from the n pairs.
- **Steps:** clip(⌈4 · n / batch⌉, 200, 2000), the same for every setting. 4 epochs = the participants' budget: 2000 steps × 8 / 4000 pairs.
  - n = 16, 64, 256 → 200 steps; n = 1024 → 512; n = all → 2000.
  - Warmup 20 steps (as in training).
- `local_D` and `prompt` depend only on the rotation and the α reference, so they run only with the method run.
- **New option** `--alpha_window LO HI` keeps only the client's pairs with α in the window, to simulate a skewed client; its support comes from those pairs.
- **Outputs:** `eval_round_0100_e2_{setting}_n{n}[_wLO-HI]__<stamp>.json`.

**Code:**
- `fedsteer/adapt.py`: `_train` gets a linear warmup; new `train_steered(components=...)` (α-conditioned training of chosen parameter groups).
- `sbatch/eval_settings.sbatch`: smoke arguments updated; lists use "+".

**Concern: held-out clients in rotation 0 are broad.** Their data range on the run's α scale:
- telegraph.co.uk 0.02–0.99, latimes.com 0.11–0.98, bbc.com 0.03–0.79, mashable.com 0.08–0.79.
- The narrow, skewed clients (theguardian 0.04–0.55, nypost 0.51–0.95) are participants. They are held out only in rotation 2.
- So rotation-0 E2 tests **data efficiency**, but hardly **coverage** (few out-of-support α; telegraph has almost none).
- Options:
  - (a) `--alpha_window` on rotation-0 held-out clients: controlled skew, no new training;
  - (b) a rotation-2 federated run: natural skew, about 6.5 h of training, then E2.

**Jobs** ([launch](launch/exp23_e2_redesign.sh)):
- smoke 11087947;
- method run n = 16 / 64 / 256 / 1024: 11088050–53;
- method run n = all: 11088054;
- private-calibration run: 11088055;
- 11083575 (old design) cancelled; 11083572 (old design) left running.

## 24. Plan updated (2026-10-01, evening)

`federated-steering-plan.md` brought up to date with entries 19–23 (no code or jobs changed; previous version kept outside the repo):
- **Claims:** C1 reported as a coverage trade-off (4k); C2 reworded as "a joining client needs less data with the frozen direction" (the redesigned E2); C3 in progress.
- **Method:** shared calibration without offset recorded as the proposed default (best in exp17); the config default is still private + offset.
- **Data:** caveat that rotation 0's held-out clients are broad; skewed clients are held out only in rotation 2.
- **Evaluation:** E2 redesign; E3 details; `compare_runs.py`; proposed gap/tail split of out-of-support error.
- **Baselines:** B1, B3, B4 results; fairness follow-ups (B4 gain grid, B3 with refit calibration, second B1 template, A3).
- **Gates:** G1 passed provisionally, G3 passed, G2 partly (coverage trade-off), G4 running.
- New next steps and changelog rows.

## 25. A2: shared adapter, a non-personalized baseline (2026-10-01)

**Why:** with `fed.adapter=shared` the "private" adapter is averaged by the server every round, like D. With shared calibration, nothing stays on the client: one global α-conditioned model trained by plain FedAvg. This is both
- **the non-personalized FL baseline**, and
- **the ablation of the private adapter.**

**Expectations:**
- Steering may improve: no client-specific adapter can absorb a client's skew (nypost.com's truncated leads, §2 of the plan).
- House style should suffer.

**What to compare** (vs. the method, 11063910):
- steering in- and out-of-support, per client;
- **per-client NLL on the client's own reference summaries** (dev loss is logged by the monitor; test NLL via `eval_direction.py --dev_loss` or an addition to `compare_runs.py`);
- BERTScore vs. the reference; quality and judge as usual.

A publication classifier on the outputs (style fidelity) is optional.

**E2 note:** with a shared adapter a new client can use the global model as is (no training), an extra point for the held-out curve.

**Code:** none (mode implemented and unit-tested since entry 7, never run).

**Job** ([launch](launch/exp25_a2_shared_adapter.sh)): 11094904. Overrides of exp17 F_shared_noO + `fed.adapter=shared`; standard pipeline (dev selection → test → quality → judge).

## 26. Results: E2 (redesigned, n = 16 / 64 / 256) and E3 (drift) (2026-10-01, night)

### E2: held-out clients join with the frozen direction (method run 11063910; rotation 0)

Test, 200 articles per held-out client, mean over the 4 clients. Quality = AlignScore; gap = generated − real summaries at the same α.

| Setting | n | Pct error | In | Out | Spearman | Near-tie | Endpoint near-tie | AlignScore gap in / out | Length gap out |
|---|---|---|---|---|---|---|---|---|---|
| **frozen_D** | 16 | **0.179** | 0.203 | 0.153 | 0.893 | 0.23 | 0.000 | −0.02 / +0.05 | +13.8 |
| **frozen_D** | 64 | **0.170** | 0.182 | 0.152 | 0.922 | 0.27 | 0.000 | +0.02 / +0.08 | +5.7 |
| **frozen_D** | 256 | **0.146** | 0.148 | 0.143 | 0.931 | 0.20 | 0.000 | +0.01 / +0.06 | +2.4 |
| local_D | 16 | 0.295 | 0.289 | 0.299 | 0.542 | 0.51 | 0.130 | −0.02 / +0.06 | +5.8 |
| local_D | 64 | 0.260 | 0.256 | 0.266 | 0.735 | 0.48 | 0.026 | +0.02 / +0.12 | +1.0 |
| local_D | 256 | 0.225 | 0.233 | 0.213 | 0.785 | 0.40 | 0.001 | +0.01 / +0.06 | +0.1 |
| plugin | 16 / 64 / 256 | 0.369 / 0.376 / 0.374 | 0.42–0.47 | 0.28–0.31 | 0.77–0.82 | 0.53–0.56 | 0.06–0.08 | +0.13–0.16 / +0.16–0.20 | +33 to +37 |
| prompt | 16 / 64 / 256 | 0.411 / 0.388 / 0.411 | 0.33–0.37 | 0.45–0.47 | 0.04–0.14 | 0.25–0.28 | 0.16–0.28 | −0.32 to −0.16 / −0.20 to −0.01 | +2 to +6 |

**Per client:** `frozen_D` vs `local_D` (paired bootstrap):
- frozen_D is **significantly better for all 4 clients at every n**, overall and out-of-support. Overall differences: −0.08 to −0.16 at n = 16; −0.06 to −0.10 at n = 256.
- frozen_D with 16 pairs (0.179) already beats local_D with 256 pairs (0.225).
- frozen_D with 256 pairs (0.146) is as good as the participants themselves (0.151 in entry 20).
- **Partial n = 1024 / all** (jobs still running, telegraph.co.uk first):
  - n = 1024: frozen 0.157 vs. local 0.221 in-support;
  - n = all: frozen 0.143 vs. local 0.158 in-support, and about equal out-of-support (0.085 / 0.080).
  - So the gap closes once the client has its full data, as expected.

**Training:** both settings get 200 steps at n ≤ 256 and reach training loss ≈ 0 (memorized), so `local_D`'s deficit is **sample efficiency, not under-training**. A reviewer may still ask for a tuned step count for `local_D` at small n.

**Other observations:**
- **`plugin` fails** (0.37): an adapter trained without D does not work with D. Outputs at α = 0 are already far up the scale; the summaries are much longer and more extractive (higher AlignScore, +35 tokens).
- **`prompt` is worse than a constant output** (0.30).
- Old-design numbers (entry 21; 11083572), for the record: frozen_k0 0.335, frozen_cal_k16/64 0.310, localdir_k16/64 0.367, prompt_k16/64 0.40.

**Caveat (entry 23):** rotation 0's held-out clients have broad support, so out-of-support here means mostly the extremes. **Coverage is not yet tested on new clients.**

### E3: drift (method run vs. its local counterpart 11063911)

Each stage = 100 steps of plain SFT of the adapter with **steering off** on 200 later-year pairs. Evaluated on 100 test articles per client.

| Stage | Fed pct error | Fed Spearman | Local pct error | Local Spearman | Fed / local near-tie | Fed length in |
|---|---|---|---|---|---|---|
| 0 | 0.154 | 0.933 | 0.169 | 0.900 | 0.22 / 0.25 | 33.0 |
| 1 as is / refit | 0.307 / 0.298 | 0.752 / 0.672 | 0.320 / 0.286 | 0.597 / 0.610 | 0.51 / 0.53 | 41.4 |
| 2 as is / refit | 0.338 / 0.322 | 0.695 / 0.586 | 0.354 / 0.318 | 0.481 / 0.521 | 0.56 / 0.60 | 44.9 |
| 3 as is / refit | 0.349 / 0.334 | 0.695 / 0.622 | 0.359 / 0.331 | 0.474 / 0.499 | 0.57 / 0.60 | 44.1 |

- **Steering collapses after one drift stage for both runs:** percentile error doubles to about the constant-output level. A 16-pair calibration refit recovers little.
- The federated direction keeps more of the ordering. Stage-3 Spearman is 0.70 vs. 0.47, and higher for 6/8 clients. But calibration is gone in both.
- **Mechanism:**
  - With shared calibration and no offset, g(0) = 0. So the private adapter *alone* is the α = 0 anchor; during co-training it learns the low end of the scale.
  - SFT with D off retrains the adapter to produce the client's *average* summaries. The anchor moves to the client's mean level, and D then pushes further up.
  - Hence outputs pile up at the copy end: forbes.com's α = 0 output goes from percentile 0.17 to 0.52; nypost.com's is ≈ 0.9 at every α. Summaries get longer (+11 tokens) and more extractive (AlignScore up).
  - It is **not** caused by the drift data: drift-period α is similar to or lower than training α (forbes.com mean 0.40 → 0.25).
  - A refit of gain and warp cannot move the anchor, since there is no offset.
- **The dose is also extreme:** 100 steps × 8 = 4 epochs over 200 pairs, drift loss 2.2 → 0.03 (memorized).
- **Same root cause as E2's `plugin`.** Training the adapter without α conditioning breaks the knob. Since α is free to compute for every pair, the realistic continuation is to keep training *with* steering on (α per pair, D frozen), as in E2's `frozen_D`. The steering-off drift is a stress test and a limitation to report.

### What to run next (proposed, not submitted)

1. **E3 with steering-on drift** (D and calibration frozen, α per pair, adapter trains): the realistic protocol. Keep the current steering-off run as the stress test, and add a gentler dose (1 epoch per stage) to get a dose-response curve. Needs a `--drift_mode steer|off` flag in `e3_drift.py`. Run on both the federated and the local run.
2. **E2 coverage:** `--alpha_window` (0–0.4 and 0.6–1) on the held-out clients, `frozen_D` and `local_D` only, n = 64 and 256.
3. **E2 fairness:** a step sweep (e.g. 50 / 100 / 400) for `local_D` (and `frozen_D`) at n = 64 / 256.
4. **Rotation 2 federated run** (method config) and E2 on it: natural skew, plus a second rotation for E1.
5. Baseline fairness: B4 wider gain grid; B3 with refit calibration; second B1 template.
6. Longer training (150–200 rounds) for the method and local.

Still running: E2 n = 1024 / all (method run), E2 on the private-calibration run, A2 (shared adapter).

## 27. LLM judge for E2; E3 and A3 made optional (2026-10-01, night)

**Did E2 get semantic evaluation?**
- **Automatic metrics: yes.** All 19 E2 eval files have `quality_` files: AlignScore, BERTScore, length, repetition, and gaps to the same-α reference.
- **LLM judge: no.** `sbatch/eval_settings.sbatch` never called it.

E2 quality, mean over held-out clients (in / out of support):

| Setting | n | AlignScore | Gap to same-α reference | BERTScore | Length gap |
|---|---|---|---|---|---|
| frozen_D | 16 | 0.770 / 0.742 | −0.017 / +0.048 | 0.885 / 0.882 | +11.3 / +13.8 |
| frozen_D | 64 | 0.827 / 0.741 | +0.022 / +0.083 | 0.893 / 0.886 | +6.7 / +5.7 |
| frozen_D | 256 | 0.812 / 0.722 | +0.007 / +0.063 | 0.891 / 0.884 | +2.9 / +2.4 |
| local_D | 16 | 0.772 / 0.754 | −0.015 / +0.060 | 0.887 / 0.886 | +13.0 / +5.8 |
| local_D | 64 | 0.825 / 0.777 | +0.021 / +0.119 | 0.894 / 0.889 | +7.0 / +1.0 |
| local_D | 256 | 0.809 / 0.714 | +0.005 / +0.056 | 0.896 / 0.887 | +5.6 / +0.1 |
| plugin | 16–256 | 0.93–0.94 / 0.86 | +0.13–0.16 / +0.16–0.20 | 0.889–0.891 / 0.883–0.889 | +34 to +40 / +33 to +37 |
| prompt | 16–256 | 0.47–0.64 / 0.49–0.65 | −0.32 to −0.16 / −0.20 to −0.01 | 0.862–0.880 / 0.858–0.880 | +8 to +9 / +2 to +6 |

- frozen_D and local_D have **equal quality**; the steering difference (entry 26) comes at no quality cost.
- Short-data frozen_D (n = 16) writes longer summaries (+11–14 tokens).
- plugin's high AlignScore comes from long, copied summaries. Prompting is the least faithful.

**Changes:**
- **`scripts/judge_quality.py`:**
  - looks up records of all clients (the same bug `score_quality.py` had in entry 21a: held-out articles raised a KeyError);
  - `--eval` takes several files, and the judge model is loaded once;
  - reference summaries are judged once per split.
- **`sbatch/eval_settings.sbatch`:** an LLM-judge step after quality scoring (`JUDGE=1` by default). `JUDGE_MATCH` sets which files are judged, by default E2 `frozen_D` / `local_D`.
- **New `sbatch/judge_evals.sbatch`:** judges existing eval files of one run (`RUN`, `MATCH` regex) that have no `judge_` file yet.

**Jobs:**
- **11097085** (smoke): the refactored judge on the smoke run's two held-out E2 files, 2 articles per client.
- **11097083**: the judge on E2 `frozen_D` / `local_D` for all n of the method run. It runs after the n = 1024 / all jobs finish (`afterany`).

**Scope decision (user, 10-01):** E3 (claim C3, drift) and A3 (PFL-structured conditional SFT) are **optional**. The paper's core is C1 and C2.
- E3's current result (entry 26) is reported as a limitation unless the optional steering-on drift run is done.
- The "isn't this just PFL?" question is answered by A2 (non-personalized FedAvg) and the positioning.

Plan updated: claims, E3, A3, figures, gates, next steps, claim sentence.

## 28. C1 on a second model family at 8B: Qwen3-8B (2026-10-01/02)

**Model choice:** Qwen/Qwen3-8B.
- A different family from Llama-3.2-1B, and the plan's Qwen3 line.
- A dense standard transformer (36 layers, hidden 4096) supported by the installed transformers 4.56.
- Run with thinking off.

Alternatives considered:
- **Qwen3.5-9B** (cached): `qwen3_5` is not supported by transformers 4.56 (hybrid linear attention, a vision-language wrapper).
- **Llama-3.1-8B:** same family as the 1B.
- **Mistral-7B-Instruct-v0.3:** older; not fully cached.
- **Qwen2.5-7B-Instruct:** it is the LLM judge.
- **Judge note:** the judge (Qwen2.5-7B) and the backbone are both Qwen models. Within-backbone comparisons (federated vs. local) are unaffected; absolute judge scores across backbones should be read with care.

**Hardware:** all GPUs on `dali` / `IllinoisComputes-GPU` are A100 **80 GB**.
- Smoke: about 48 GB used during training (bf16 weights 16 GB, LoRA rank 16 on all projections, gradient checkpointing, batch 8).
- Trainable parameters per client view: 66M.

**Code:**
- `ChatFormatter(template_kwargs=...)` passes extra chat-template arguments. Config `chat_template_kwargs: {}`; Qwen3 runs set `chat_template_kwargs.enable_thinking=false`. All entry points pass it.
- Checked on real pairs: with thinking off, the prompt ends in the empty `<think>\n\n</think>\n\n` block and only the summary plus `<|im_end|>` is supervised. Without it, the empty think block would be part of the target.
- `sbatch/train_eval.sbatch`:
  - `GPU_LOG=1` records GPU memory every 30 s (`sbatch/logs/gpumem.o<job>`);
  - a run directory outside `runs/` (`RUNS_ROOT`) is linked into `runs/`.

**Disk:** `/u/lucmon` has a 100 GB quota (68 GB used; also 455k of 490k files). An 8B run should write about 25–30 GB of snapshots, so the 8B runs live in `/u/lucmon/lucmon/rein_runs/` (project space, 18 TB free) and are linked into `runs/`.

**Jobs** ([launch](launch/exp28_qwen3_8b_c1.sh)): the same data and hyperparameters as the 1B pair exp17 (method 11063910 / local 11063911): 4k per client, 100 rounds × 20 steps, batch 8, rank 16, unchanged learning rates, standard pipeline (dev selection, test, quality, judge).
- Smoke 11097847.
- Federated 11098308 and local 11098309, `afterok` on the smoke, time limit 3 days.
- 11097906/7 were resubmitted as 11098308/11098309 before starting: the smoke's training peaked at 72 of 80 GB, so `train_eval.sbatch` now sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (less fragmentation).
- The first submissions 11097848/9 were cancelled before starting (quota).

**Next:** once the federated run exists, consider B1 prompting on the 8B client models: it checks whether prompting fails only because the 1B model follows instructions poorly (gate G1 caveat).

## 29. Dataset scouting for a second task beyond summarization (2026-10-01, night)

No jobs. Research only; written up in `../dataset-candidates.md`.
(Numbered 29 because entry 28 is reserved for the Qwen3-8B C1 runs launched at 23:29 from another session: jobs 11097847 smoke, 11097906 fed, 11097907 local; see `exp_log/launch/exp28_qwen3_8b_c1.sh`.)

**New screening criterion from the user:** the attribute must **not** be reachable by prompting, because B1 is a baseline and gate G1 depends on it failing. Added to the existing criteria (continuous/fine-grained, cheap deterministic scorer, attribute-level client heterogeneity, not summarization). Implication: semantic attributes (sentiment, star rating, formality, technicality) are out, because instruction models hit them from words alone. Attributes that survive are **corpus statistics** the model cannot measure on its own output — the current flagship's fragment density, token/step counts, edit distance, RDKit descriptors.

A sixth criterion emerged while screening: **the attribute must be free given the input.** This disqualifies code and SQL complexity, where the spec dictates the answer's complexity, so low α means wrong output.

**This drops §3.2 of the plan (Amazon review rating):** star rating is easily prompted, so B1 would not fail on it.

**Top candidates:**
1. **Reasoning-trace length on math CoT** — `open-r1/OpenR1-Math-220k` (2–4 R1 traces per problem, so α varies *within* a problem; clients = NuminaMath 1.5 `source`). Prompt-level token budgets are documented to fail (L1/LCPO, Token-Budget-Aware Reasoning, BudgetThinker), utility = verifiable accuracy. Risk: eval compute (~18M generated tokens per full eval; needs vLLM and probably a 4B base).
2. **Property-conditioned molecule generation** — ChEMBL/TDC, attribute = RDKit cLogP/TPSA/QED, clients = target family (kinase/GPCR/CNS property skews are mechanistic and documented), real federation precedent (MELLODDY). Cheapest eval of all (~40-token outputs). Risk: SMILES validity from a 1B base; fall back to SELFIES.
3. **Paraphrase lexical divergence** (QCPG dimensions) — best "attribute is free given the input", cheap, but a dated framing.

**Measured locally (no download):** `MegaScience` is already in the HF cache and reproduces the Newsroom disjoint-support structure on answer verbosity — `natural_reasoning` covers global length percentiles [0.43, 0.98], `textbook_reasoning/medicine` and `/biology` cover [0.02, 0.71]. 8 usable groups; the skew is provenance-level, and subjects *within* `textbook_reasoning` are nearly homogeneous (medians 0.18–0.32). Good as a fast transfer sanity run, not as a paper task.

**Proposed next step (not started):** two one-day feasibility probes — per-source counts and within-problem length spread in OpenR1-Math-220k's 94k split; SMILES validity after a 5k-molecule, one-hour SFT of Llama-3.2-1B.

## 30. Results: full E2 curve, E2 on the private-calibration run, A2 (shared adapter), 8B progress (2026-10-02)

Numbered 30: entry 29 is the dataset scouting written by another session. Reports saved in `exp_log/reports/`.

**Tools:**
- New `scripts/e2_report.py` (E2 table, per-client errors, per-client paired bootstrap of frozen_D − local_D at each n, quality and judge columns).
- **Fix in `scripts/compare_runs.py`:** the default file pattern matched E2/E3/baseline files too (the newest eval in a run directory is now often an E3 file). It now matches only plain test evals (`eval_round_XXXX__<stamp>.json`).
- `compare_runs.py` also compares only clients present in every run.
- Numbers in earlier entries were produced with explicit patterns and are unaffected.

### E2, complete curve (method run 11063910; rotation 0; test, 200 articles per held-out client)

| n | frozen_D error (in / out) | local_D error (in / out) | frozen_D better on | plugin | prompt |
|---|---|---|---|---|---|
| 16 | **0.179** (0.203 / 0.153) | 0.295 (0.289 / 0.299) | 4/4 | 0.369 | 0.411 |
| 64 | **0.170** (0.182 / 0.152) | 0.260 (0.256 / 0.266) | 4/4 | 0.376 | 0.388 |
| 256 | **0.146** (0.148 / 0.143) | 0.225 (0.233 / 0.213) | 4/4 | 0.374 | 0.411 |
| 1024 | **0.139** (0.141 / 0.136) | 0.192 (0.209 / 0.165) | 4/4 (latimes.com worse out-of-support, +0.018) | 0.362 | 0.387 |
| all (≈ 4–5k) | **0.135** (0.135 / 0.134) | 0.151 (0.159 / 0.138) | 2/4, others tie (telegraph.co.uk, latimes.com) | 0.343 | — |

- **The data-efficiency curve is clean.** Frozen D with 16 pairs (0.179) beats local D with 1024 pairs (0.192). The gap shrinks with n and closes at full data.
- At full data, frozen D (0.135) is *better* than the participants' own models (0.151, entry 20): the new client gets a direction trained on 8 clients plus an adapter trained against it.
- **Quality is on par.** AlignScore gaps to the same-α reference: frozen_D +0.01 / +0.06–0.08, local_D +0.00–0.02 / +0.04–0.12; BERTScore equal.
- **Judge** (25 articles per client, indicative):
  - faithfulness in-support 0.75–0.83 (frozen) vs. 0.71–0.84 (local);
  - out-of-support relevance gap −0.03 to +0.03 (frozen) vs. −0.04 to −0.15 (local);
  - frozen_D is at least as good.
- Small n has a length cost: frozen_D at n = 16 writes +14 tokens out-of-support (+2 to +7 from n = 256).

### E2 on the private-calibration run (11041282; frozen_D trains private gain/offset/warp)

| n | 16 | 64 | 256 | 1024 | all |
|---|---|---|---|---|---|
| frozen_D (private calibration trained) | 0.183 | 0.177 | 0.151 | 0.141 | 0.132 |
| frozen_D (shared calibration frozen, above) | 0.179 | 0.170 | 0.146 | 0.139 | 0.135 |
| plugin (calibration fitted on the n pairs, **with offset**) | 0.244 | 0.244 | 0.234 | 0.256 | 0.276 |

- **Shared calibration costs nothing for new clients:** same curve, with zero calibration parameters to fit.
- plugin is much better with a fitted offset (0.23–0.28) than with the frozen shared calibration (0.34–0.37). This fits the anchor-shift mechanism (entry 26): an offset can move the α = 0 point back. Still far behind frozen_D.

### A2: shared adapter, i.e. non-personalized FedAvg (11094904, selected round 70)

| Run | Pct error (worst) | In-support | Out-of-support (worst) | Reach | Spearman | AlignScore gap in / out | Length gap in / out | Judge faithful in / out |
|---|---|---|---|---|---|---|---|---|
| Method (private adapter) | **0.151** (0.199) | **0.135** | 0.161 (0.188) | 0.39 | 0.933 | −0.03 / +0.05 | +2.0 / +2.3 | 0.82 / 0.79 |
| **A2 (shared adapter)** | 0.159 (**0.176**) | 0.177 | **0.136 (0.168)** | **0.50** | **0.951** | +0.09 / +0.10 | +7.2 / +10.1 | **0.89 / 0.86** |
| Local, no offset | 0.165 (0.205) | 0.148 | 0.165 (0.249) | 0.36 | 0.906 | −0.02 / +0.01 | +2.7 / −0.5 | 0.84 / 0.79 |

**Per client, A2 − method:**
- overall worse on 5/8 (forbes.com, theguardian.com, wsj.com, aol.com, cbc.ca; +0.01 to +0.03);
- better on the copy-heavy reuters.com (−0.023) and nypost.com (−0.031);
- **out-of-support better on 6/8** (−0.02 to −0.05), worse on none.

**Per-client dev NLL** (α-conditioned, on each client's own dev references; house-style fit):

| Round | Method | A2 | Local |
|---|---|---|---|
| 10 | 1.090 | 1.176 | 1.082 |
| 50 | **1.081** | 1.089 | 1.097 |
| 70 | 1.229 | **1.078** | 1.306 |
| 100 (method's selected) | 1.450 | 1.083 | 1.507 |

- At the selected checkpoints A2 has lower NLL on **all 8** clients (1.08 vs. 1.45). The private adapters memorize after round ~50 while the shared one does not.
- At each run's best round the NLL is **equal** (1.081 vs. 1.078). On Newsroom, private adapters buy **no** house-style fit.

**Reading:**
- The private adapter is a trade-off. It gives better in-support calibration (0.135 vs. 0.177).
- But it gives worse extrapolation, because it absorbs part of each client's level (the identifiability problem). A shared adapter cannot, so the attribute must go through D: better reach, Spearman and worst client.
- A2's summaries are longer and more extractive-looking (AlignScore +0.09, +7–10 tokens vs. the same-α reference). The judge rates them more faithful.
- **Implication for the framing:** "private house style" is not supported as a *benefit* on this dataset. It is a design choice that trades in-support precision against coverage.
- Options: (a) keep the private adapter and report A2 honestly as the trade-off; (b) a smaller or regularized private adapter (e.g. rank 4) as the middle ground; (c) make personalization optional in the method.
- The core claims C1 (federated > local) and C2 (frozen D for new clients) do not depend on this choice. With a shared adapter a new client could even start from the global adapter (E2 variant, not run).
- **Selection caveat:** the method is selected on steering error only, at round 100 where its dev NLL is worst. Quality-aware selection (plan §4) would pick earlier rounds.

### Qwen3-8B (exp28), in progress

Dev error, mean over clients (in-training full eval):

| Round | 10 | 30 | 50 | 60 | 70 |
|---|---|---|---|---|---|
| fed 8B | 0.261 | 0.190 | 0.180 | 0.164 | 0.165 |
| local 8B | 0.269 | 0.200 | 0.194 | 0.185 | — |
| fed 1B | 0.233 | 0.192 | 0.175 | 0.160 | 0.164 |
| local 1B | 0.238 | 0.190 | 0.180 | 0.170 | 0.166 |

- The federated run is ahead of local at every checkpoint from round 30, the same pattern as 1B. The absolute error is not lower than 1B's so far.
- About 6.5 min per round, peak 40 GB after the first rounds (72 GB only at the start), 16–19 GB of disk per run so far.
- Expected to finish training about 10-02 19:00 (local) and 20:00 (federated), then test, quality and judge (~2 h).

## 31. House style of the private adapter, controlled for extractiveness (2026-10-02)

**Question (user):** does the private adapter carry the client's writing style, independent of extractiveness? Dev NLL alone cannot show it: equal at best rounds (entry 30), dominated by content tokens, and mixed with steering quality.

**New `scripts/style_eval.py`** (CPU only, run on the login node; uses saved outputs, no generation):

1. **Publication attribution.**
   - A classifier (char 2–5 + word 1–2 tf-idf, logistic regression) is trained on *real* training summaries of the 8 participants, **density-matched**: within each of 5 global-α bins every publication present contributes the same number of summaries (≤ 400), so extractiveness cannot identify the publication. Cached in `runs/_style/`.
   - Metric: share of a client's generated summaries (in-support α) attributed to its own publication, and the mean probability of its own publication.
   - Topic comes from the article and is identical across models (same test articles), so differences between models are the style the model adds.
   - **Probe:** on real summaries, density-matched accuracy is 0.41–0.63 per α bin vs. chance 0.13–0.20. There is plenty of style beyond extractiveness.
2. **Style-feature gap.** 11 interpretable features (sentences, words per sentence, word length, quotes, ellipsis, parentheses, numerals, capitalized words, first/second person, ? and !). Score: |standardized mean difference| between the generated summaries and the client's real training summaries within ±0.125 α, averaged over features. Lower = closer to the house style.

**Results, test** (selected checkpoints; in-support α; mean over 8 clients; chance 0.125):

| Run | Attribution accuracy | P(own) | Feature gap |
|---|---|---|---|
| Real test summaries (ceiling) | 0.587 | 0.388 | — |
| **Method** (federated, private adapter) | **0.552** | **0.378** | **0.108** |
| Local, no offset | 0.553 | 0.375 | 0.104 |
| F_priv (federated, private calibration) | 0.568 | 0.387 | 0.108 |
| L_priv (local) | 0.568 | 0.380 | 0.105 |
| **A2 (shared adapter)** | **0.409** | 0.295 | **0.190** |
| B3 merged direction (local adapters) | 0.454 | 0.314 | 0.156 |
| B4 CAA (client models, D removed) | 0.532 | 0.358 | 0.166 |
| B1 prompting k=3 (client models, D removed) | 0.556 | 0.371 | 0.166 |

- **Private adapters carry house style.**
  - The method's outputs are attributed to their own publication 55% of the time (real summaries: 59%); A2's only 41%.
  - The method's style-feature gap is 0.11 vs. A2's 0.19.
  - **Per client**, A2 is lower for all 8; largest drops: nypost.com 0.67 → 0.25, theguardian.com 0.55 → 0.29, forbes.com 0.47 → 0.35.
- **Federated training keeps the style.** Method ≈ local (0.552 vs. 0.553), so sharing D does not wash out the private style.
- **The style is orthogonal to extractiveness.** Attribution by target α is flat: method 0.52–0.56 over α = 0 → 1, A2 0.36–0.42. Steering the attribute does not move the style.
- B1/B4 keep the client adapter and keep most of the attribution (0.53–0.56). This confirms the style lives in the adapter. Their feature gap is larger (0.17), since prompts and activation additions change the surface form.
- B3's merged direction lowers attribution (0.45 vs. 0.57 for the same local adapters with their own direction), so a mismatched direction disturbs the style too.
- **Check that the style comes from the adapter, not from the control mechanism:**
  - B1 and B4 run on the client models (base + private adapter, D removed), so their attribution is inherited from the adapter.
  - B1 with the same prompts and the same 3 own-publication examples on the **base model** (no adapter): attribution **0.274** vs. 0.556 on the client model; feature gap 0.666 vs. 0.166 (the base model writes ~110-token summaries).
  - The remaining 0.27 is mostly topic. It is high for cbc.ca (0.71) and wsj.com (0.56), near zero for nypost.com (0.02) and people.com (0.06).
  - In-context examples do not transfer the house style; the adapter does.

**Across training** (dev outputs, rounds 10 / 30 / 50 / 70 / 100):
- method attribution 0.57 / 0.56 / 0.58 / 0.59 / 0.59;
- A2 0.40 / 0.40 / 0.41 / 0.44 / 0.45;
- local 0.59 / 0.57 / 0.58 / 0.60 / 0.57.

The advantage is there from round 10, so it is **not** a product of the late-round memorization seen in dev NLL. The feature gap shrinks over training for all runs (method 0.15 → 0.12).

**Conclusion for A2 / the framing:** the private adapter is justified as **house style**: it is orthogonal to the steered attribute, measured with an extractiveness-controlled attribution metric and interpretable features. The trade-off with A2 (better in-support calibration and style vs. A2's better extrapolation) can be reported as such.

**Also added:** `scripts/nll_matrix.py` (cross-client NLL, specialization = others' NLL on a client's references − its own). Not run (job 11116479 cancelled at the user's request); kept for later.

Reports: `exp_log/reports/style_test.txt`, `exp_log/reports/style_dev_rounds.txt`; per-file results in `<run>/evals/style_*.json`.

## 30. ChEMBL molecule generation: data built, statistics measured (2026-10-02)

No training jobs. Data preparation and analysis only. Plan: `../chembl-experiment-plan.md`.
User picked the molecule candidate from entry 29 as the first task to try beyond summarization.

**Data** (`data/chembl` -> `/projects/illinois/eng/cs/arindamb/lucmon/data/chembl`):
- activities from `martinakaduc/ChEMBL_activities` (20.3M rows) filtered to assay_type=B, confidence>=8, relation "=", nM, type in {Ki,Kd,IC50,EC50} -> **1,337,996 activities over 5,995 targets**;
- structures from ChEMBL 37 `chembl_37_chemreps.txt.gz` (EBI FTP), 99.8% of molecule IDs resolved;
- target names/organism/protein class for the top 120 targets via the ChEMBL REST API;
- `rdkit` and `selfies` installed into the `steer` env (rdkit 2026.03.6).
- New scripts: `scripts/chembl_targets.py`, `chembl_target_meta.py`, `chembl_stats.py`, `chembl_report.py`, `chembl_select_clients.py`.
- Target sizes: 17 targets with >=5k distinct molecules, 40 with >=4k, 116 with >=2k, 247 with >=1k.

**Attribute choice — the decisive measurement.** For a scaffold-conditioned task (x = target + Murcko scaffold, y = molecule), every *raw* descriptor is largely determined by the input, because the scaffold is a subgraph of the molecule. Median within-scaffold p5–p95 spread on the global alpha scale (8,261 scaffolds with >=5 molecules), vs. spread of client medians:

| attribute | free given x | client-median spread | corr(scaffold, molecule) |
|---|---|---|---|
| **clogp residual** | **0.31** | **0.81** | — |
| mw residual | 0.30 | 0.51 | — |
| rotb residual | 0.24 | 0.50 | — |
| raw clogp | 0.18 | 0.73 | +0.76 |
| raw mw | 0.18 | 0.79 | +0.88 |
| raw tpsa | 0.15 | 0.79 | +0.85 |
| raw arom_rings | — | 0.55 | **+1.00** (unusable) |
| hbd residual | 0.07 | 0.41 | — |

**Chosen: cLogP residual = cLogP(molecule) − cLogP(Murcko scaffold)** ("how lipophilic are the decorations"). Best on both axes at once. Raw descriptors leave only ~15–18% of the attribute free given the input, which would make D unidentifiable. TPSA is additionally discrete (mass at 9.2 / 18.5 / 26.0), which would wreck a percentile scale.

**Clients** (`data/chembl/clients.json`): 12 targets, one per protein family, spread over the alpha scale, 8 participants + 4 held out (stratified, Newsroom-style). 64,943 rows, 63,581 distinct molecules — pairwise molecule overlap mean **0.005**, max 0.121. Medians span alpha 0.12 (carbonic anhydrase 2) to 0.68 (serotonin transporter); the skew is mechanistic (polar sulfonamides vs. lipophilic hERG/SERT binders), not incidental.

⚠️ **Honest weakness: supports are broad.** Unlike Newsroom (nypost 0.51–0.95 vs. theguardian 0.04–0.55), every ChEMBL target spans most of the range: mean uncovered share of the scale per client **0.19**, and alpha 0.3–0.7 is covered by 100% of clients. A skew-maximizing selection (`--mode skewed`, `clients_skewed.json`) only reaches 0.23. **So C1 here has to rest on the controlled truncation experiment, not on natural skew.** The regime is "different centres, overlapping tails" rather than disjoint supports — a different and arguably more realistic heterogeneity, worth reporting as such.

**Other measured facts:**
- Constant-output percentile error (always emit the client median, 5-point grid): **0.33** (Newsroom: 0.30).
- Generated molecules are a median of **36 Llama-3.2 tokens** (p95 67) — a full 12-client x 200-test x 5-alpha eval is ~480k tokens, so the 11-point grid, T=0.7 sampling and 3 seeds all become affordable here.
- 25,126 distinct scaffolds; median 1 molecule per scaffold; 2,586 scaffolds have >=5 molecules, covering 52% of rows -> the test set should be drawn from those, so every test input has a same-alpha reference.
- 3.7% of rows have residual exactly 0 (molecule == its scaffold); drop them.

**Gates set before any long run:** G0 validity >=90% and scaffold retention >=80% after plain SFT (else SELFIES, then Qwen3-4B); G1 prompting error >=0.30; G2 method <0.25; G3 method beats local on a majority of clients.

**Next (not started):** `scripts/build_chembl_fed.py` (scaffold-disjoint splits, alpha reference), a `fedsteer` data adapter + RDKit scorer, then E0/G0 and B1/G1 — together under half a day of GPU time.

## 32. Baseline definition confirmed; learned activation steering proposed (2026-10-02)

**Decision (user):** keep the existing baseline definition. B1 (prompting) and B4 (CAA) run on each client's model (frozen base + that client's private adapter, D removed), so only the control mechanism differs from the method. They do no training of their own: B1 only prompts; B4 uses mean-difference activations plus one scalar gain fitted on dev. B1 on the plain base model (`b1_prompt_k3_base`) is a reference row. No code or results changed; the plan states this definition explicitly.

**Proposed, not started:** B4′, federated *learned* activation steering. The method with D replaced by learned activation vectors scaled by the shared g(α), trained with the same objective, data and FedAvg, private adapter kept. This is the equal-training test of weight vs. activation space; CAA stays as the training-free version. Variant without the private adapter if wanted.

**Decision (user):** B4′ is not pursued; nothing changes. The existing B4 (CAA) remains the activation-steering baseline.

## 31. ChEMBL: splits built, pipeline wired, gate G0 queued (2026-10-02)

Backbone for this task: **Qwen/Qwen3-4B-Instruct-2507** (user's choice). It was not in the HF
cache (only Qwen3-8B and Qwen3.5-4B were), so it was downloaded, 7.6 GB. The 2507 instruct
variant is non-thinking, which matters: a thinking model would wrap SMILES in `<think>` blocks.

**Splits** (`scripts/build_chembl_fed.py` -> `data/chembl_fed/`, 40,268 records):
- **Scaffold-disjoint.** Dev/test scaffolds never appear in train, or the answer leaks.
- Dev/test scaffolds are drawn from those with **3-30 molecules** in the client: at least 3 so
  every held-out prompt has real molecules at several alphas for a same-alpha reference, at most
  30 so the large congeneric series stay in the training pool. With a `>=5` floor, 300 held-out
  scaffolds were not available (CHEMBL204 had only 195), hence 3-30 with dev 50 / test 200.
- Dropped 2,387 rows with residual exactly 0 (the molecule *is* its own scaffold).
- Train per client: 1,969 (CHEMBL2039) to 4,000 (cap) -- same spread as Newsroom's 4,023-5,000.
- Rotation 0 holds out CHEMBL205, CHEMBL236, CHEMBL4005, CHEMBL4409. Note this holds out the
  most extreme client (CHEMBL205, alpha median 0.12), exactly as Newsroom's rotation 0 holds out
  telegraph.co.uk; rotations 1-2 keep it as a participant.

**New code:**
- `fedsteer/molecules.py` -- RDKit attribute and quality helpers. `clogp_residual` scores the
  generated molecule against **its own** Murcko scaffold, the same definition as the training
  labels, and returns `nan` when the output does not parse. Also `keeps_scaffold` (substructure
  match against the requested core) and `descriptors`.
- `fedsteer/metrics.py` -- registered the `clogp_residual` scorer, and made `metrics_for_client`
  drop prompts with any unscorable output, reporting `unscorable_row_rate` and `n_prompts`.
  Row-wise metrics (ordering, concordance, Spearman, range) need a complete alpha sweep, so a
  partial row cannot be used. Density is never nan, so Newsroom numbers are unchanged.
- `fedsteer/baselines.py` -- B1 molecule template (`prompt_with_level(..., task="auto")`, picked
  automatically for records carrying a scaffold): states a 0-100 decoration-lipophilicity target
  and optional k nearest-alpha ligands from the client's own training data.
- `scripts/score_molecules.py` -- validity / scaffold retention / uniqueness / novelty / QED /
  MW / TPSA per client and per alpha. No LLM judge anywhere on this task.
- `scripts/summarize_sweep.py` -- `spearman_worst` and `pct_err_worst` now tolerate a missing
  summary key, so a run where nothing could be scored still produces a visible row instead of
  a KeyError (precisely the failure G0 is meant to catch).
- `configs/chembl_fedavg.yaml` -- Qwen3-4B, `enable_thinking=false`, max_prompt 256 /
  max_target 96, 30 rounds, `calibration: shared`, `offset: false` (entry 20's best design).
- `sbatch/train_eval_chembl.sbatch`, `sbatch/eval_baselines_chembl.sbatch` -- as the Newsroom
  versions but `SCORER=clogp_residual`, `score_molecules.py` instead of `score_quality.py`,
  no judge stage, 96 new tokens.

**Verified before launching** (CPU, no GPU): the scorer reproduces the stored training labels
exactly (max abs diff 0.0 over 500 rows); scaffold retention on real targets is 1.00; garbage
text scores `nan`; prompt 95 tokens, full sequence median 134 / max 299 tokens.

**Gate G0 (job 11117359, `exp_log/launch/exp31_chembl_g0.sh`):** one client (CHEMBL240, hERG,
4,000 pairs), 20 rounds x 20 steps ~ 1 epoch. Pass: validity >= 0.90 and scaffold retention
>= 0.80 on dev. **PENDING on priority** -- the cluster is full and entry 28's two Qwen3-8B runs
hold GPUs with 3-day limits.

G1-G3 are not launched: G1 (prompting) needs a trained run's snapshots, and G2/G3 need the
federated and local pair. Launch scripts are ready; the method + local pair goes out as soon as
G0's round-10 dev evaluation shows workable validity.

## 32. ChEMBL attribute fix: the baseline is the requested core (2026-10-02)

**User-raised correctness issue, and it was a real hole.** Entry 31's scorer subtracted the
cLogP of the *generated* molecule's own Murcko scaffold. That makes the quantity being
subtracted something the model controls: swap the core and the baseline moves with it, so any
residual is reachable without decorating the requested core. The sharper statement is that the
attribute became **independent of the prompt**.

Measured on two real test prompts whose requested cores differ by ~17 logP units (CHEMBL236
core cLogP −7.60, CHEMBL243 core cLogP +9.35), scoring one fixed generation
(`CCCCCCCCc1ccc2ccccc2c1`, which contains neither core):

| scorer | vs. core A | vs. core B | difference |
|---|---|---|---|
| `clogp_residual` (requested core, **new**) | 13.34 | −3.60 | **16.95** |
| own Murcko core (**old, rejected**) | 2.90 | 2.90 | **0.00** |

The old definition gave the identical score for two completely different requests.

**Changes** (G0 job 11117359 was still PENDING, and `train_eval_chembl.sbatch` snapshots the
code at job start, so the fix is picked up with no resubmission):
- `fedsteer/molecules.py`: `clogp_residual(text, rec)` now uses `rec["scaffold"]` -- the core
  given in the prompt -- as the baseline, with a cache for core cLogP. The key is **required**,
  not defaulted, so a caller without a requested core fails loudly instead of silently
  reverting to the unsafe definition. Added `clogp_residual_strict` (nan unless the core
  survives) and `clogp_residual_self` (the old behaviour, kept only as a drift diagnostic).
- **Regression check:** on training targets the requested core *is* the target's Murcko
  scaffold by construction, so all three scorers still reproduce the stored labels (max abs
  diff 1.07e-14 for the requested-core path, 0 for the self path). The definitions diverge
  only on core-abandoning generations, which is the intended behaviour.
- `fedsteer/metrics.py`: registered the three scorers; added `unscorable_row_rate` to
  `SUMMARY_KEYS` / `LOWER_IS_BETTER` so the share of prompts that could not be scored at
  every alpha appears in every summary instead of being invisible.
- `scripts/rescore_eval.py` (new): recomputes all direction metrics from an eval file's saved
  generations under a different scorer, no regeneration. The chembl sbatch now runs it with
  `clogp_residual_strict` after the test evaluation, so every run reports how much of its
  steering result rests on generations that lost the core.
- Verified: an all-unscorable client yields `pct_calib_err: None`, `unscorable_row_rate: 1.0`
  and does not crash `summarize` or `summarize_sweep`.

**Decision on which scorer is primary.** Primary stays the lenient `clogp_residual` (requested
core, nan only if unparseable), with retention reported per alpha and the strict variant as a
robustness check. Reason: the strict scorer drops a whole prompt when any single alpha loses the
core, so at retention 0.85 it would discard ~56% of 5-alpha rows and bias the surviving set.
With the baseline fixed to the requested core the gaming vector is closed either way; core
abandonment is now a quality failure visible in retention, which is the same division of labour
the Newsroom task uses (density read next to a faithfulness metric).

Plan updated: `chembl-experiment-plan.md` §2.2 and the metrics table.

## 33. ChEMBL gate G0: FAILS on Qwen3-4B at 1 epoch (2026-10-02, job 11117359)

Run: `runs/exp31_chembl_g0_qwen3_4b_20261002-190601_j11117359`. One client (CHEMBL240, hERG,
4,000 pairs), 20 rounds x 20 steps. Confirmed the code snapshot carried entry 32's
requested-core scorer, so the steering numbers use the fixed attribute definition.

**Verdict: G0 fails both conditions.**

| quantity | gate | result |
|---|---|---|
| validity (mean over alpha) | >= 0.90 | **0.818** (worst alpha 0.74) |
| scaffold retention | >= 0.80 | **0.576** (worst alpha 0.324) |

Retention collapses as alpha rises -- **0.81 / 0.79 / 0.59 / 0.38 / 0.32** at alpha
0 / 0.25 / 0.5 / 0.75 / 1 -- which is exactly the failure entry 32's fix was designed to expose.

**The steering number is not trustworthy on this run.** Test metrics looked acceptable at face
value (pct_calib_err 0.217 vs the 0.300 constant-output baseline, in-support 0.193), but
`scripts/rescore_eval.py --scorer clogp_residual_strict` shows **94% of prompts lose the core at
some alpha** (vs. 48% unscorable from invalid SMILES alone), leaving 6 of 100 prompts intact.
Metrics on those 6 are better (pct_err 0.190, Spearman 0.851) but meaningless at that n.

**Mechanism, from the generations.** At low alpha the model keeps the core and makes small polar
edits (adds F), sometimes copying the core verbatim with no decoration at all (score 0.00). At
high alpha it **restructures the core itself** rather than decorating it -- e.g.
`O=C(Nc1ccccn1)N1CCCc2cccnc21` becomes `CC(C)(C)N(C(=O)N1CCCc2cccnc21)Cc1ccccn1`, the same atoms
reconnected, so the substructure match fails. Invalid outputs are mostly bad valences from the
same restructuring (`...C1=Oc1ccccc1`). So the model raises lipophilicity by rebuilding the
molecule, not by decorating the requested core. Adjacent-alpha ties are also high (0.36).

**Diagnosis: undertraining is the leading hypothesis, not the backbone.** Dev loss fell 0.651 ->
0.545 between rounds 10 and 20 with no sign of a plateau, and 400 steps x batch 8 is only ~0.8
epochs over 4k pairs. The plan's failure branch said "switch to SELFIES, then a bigger base",
but neither addresses the measured problem: SELFIES would fix validity (0.82) and do nothing for
retention (0.58), which is the larger gap.

**Launched: G0b (job 11118146, `exp_log/launch/exp33_chembl_g0b.sh`)** -- same client, **100
rounds x 20 steps (~4 epochs)**, snapshots every 20 rounds so the validity/retention trend over
training is visible. Dev evals save all generations, so `score_molecules.py` can measure
retention at each snapshot without regenerating.

**If G0b still fails**, retention is structural rather than a training-budget problem, and the
task framing needs changing rather than the model. Options, cheapest first: (a) make the target
the *decorations* instead of the whole molecule, so the core cannot be restructured; (b) accept
`clogp_residual_strict` as primary and report retention as a first-class result; (c) constrained
decoding that requires the core substructure. Option (a) is the most promising and keeps the
attribute definition intact.

**G1-G3 remain blocked** and were deliberately not launched: the gate is doing its job. Running
the federated/local pair now would produce a steering number whose apparent success comes from
core abandonment.

**Note on the sbatch:** SLURM copies the batch script at submission, so the `rescore_eval.py`
stage added to `train_eval_chembl.sbatch` after entry 31's submission did not run in job
11117359; it was run by hand. It is in the script for G0b.

## 34. Correction to entry 33: "G0" was not the plan's G0, and the alpha story is directional (2026-10-02)

**Two corrections, one of them to my own conclusion.**

**(1) The gate did not measure what it was defined to measure.** The plan's E0/G0 is *"SFT on one
client's 4k pairs, **no steering**. Measure validity, uniqueness, scaffold retention."* Job
11117359 instead ran the full steered configuration (`rank_shared: 16`, alpha-conditioned loss,
trained gain/warp) and evaluated on the alpha grid {0, .25, .5, .75, 1}. So it measured steering
quality, not backbone competence, and **cannot attribute** the retention failure between "this
backbone cannot decorate a core" and "the steering pressure breaks cores". The plain-SFT number
is the missing attribution baseline and was never taken.

Launched **G0-SFT (job 11119175)**: same client, `fed.lr_shared=0`. `rank_shared=0` is
unsupported (`scale_d = lora_alpha_shared / rank_shared` divides by zero), but `B_d` initializes
to **zeros**, so a zero shared LR leaves the direction exactly zero forever and the model is
alpha-independent -- plain SFT of the private adapter, no new code. Assertions to check in the
log: `direction_norm == 0` every round, and `no_effect_rate == 1.0` / `text_tie_rate == 1.0` on
dev (identical output at every alpha). 100 rounds, snapshots every 20.

**(2) The degradation is NOT an extrapolation artifact, and my entry-33 "undertraining" reading
is weaker than I claimed.** `scripts/score_molecules.py` now splits validity and retention by
whether alpha lies inside the client's own support, the way percentile error already was. For
job 11117359 (support [0.0503, 0.95]):

| | mean | in-support | out-of-support | worst alpha |
|---|---|---|---|---|
| validity | 0.818 | 0.840 | 0.785 | 0.740 |
| retention | 0.576 | **0.583** | **0.566** | 0.324 |

In- and out-of-support retention are **the same** (0.583 vs 0.566). Per alpha it is
0.81 / 0.79 / 0.59 / 0.38 / 0.32.

**Both endpoints are out-of-support** (alpha 0 < 0.0503 and alpha 1 > 0.95), yet alpha = 0 has
the *best* retention (0.81) and alpha = 1 the worst (0.32); the in-support alpha = 0.75 is
already down at 0.38. So out-of-support status does not predict the failure at all -- **the
direction of the requested edit does**. Decreasing lipophilicity is achievable with small local
edits that keep the core (add F, OH); increasing it needs bulky greasy substituents, and in SMILES
token space rewriting the whole string is apparently easier than grafting groups onto a fixed
core. The failure is asymmetric in alpha, not symmetric in distance-from-support.

**Consequence for the plan.** This is evidence for a structural fix rather than a bigger budget or
a different backbone: entry 33's option (a) -- make the target the *decorations* rather than the
whole molecule -- addresses exactly this asymmetry, because the core then cannot be rewritten at
all. G0b (job 11118146, steered, 100 rounds, now RUNNING) still tests the budget hypothesis, and
G0-SFT bounds what the backbone can do with no steering pressure. Those two plus entry 33 give a
clean attribution:

| run | steering | what it isolates |
|---|---|---|
| G0-SFT (11119175) | none | backbone competence: can it decorate a core at all? |
| entry 33 (11117359) | yes, 20 rounds | steering pressure at a short budget |
| G0b (11118146) | yes, 100 rounds | whether the budget was the problem |


## 35. Delta (delta-login2) set up as a second execution host (2026-10-02)

No experiment was run. This entry records the infrastructure so Delta results can be traced.

**Layout (Delta).** Code `/u/lucmon/rein` (no `.git`; one-way copy from cc-login via
`scripts/sync_to_delta.sh`, never edited on Delta). `runs` -> `/work/nvme/bhby/lucmon/rein`;
`data/{newsroom,chembl,newsroom_fed,chembl_fed,newsroom_stats}` ->
`/work/nvme/bhby/datasets/<name>` (3.2 GB; `toy_length.jsonl` stays in the code tree);
`HF_HOME=/work/nvme/bhby/lucmon/hf_home`. The 111 GB of existing cc runs were not copied.

**Environment.** New conda env `rein` = clone of Delta's `steer` + `pip install rdkit` (steer
itself untouched; it had no rdkit). Versions differ from cc `steer`: torch 2.5.1+cu124 (cc: cu118),
numpy 2.0.1 (cc: 1.26.3), scipy 1.14.1 (cc: 1.15.1), pyarrow 18.1.0 (cc: 19.0.0);
transformers 4.56.0, peft 0.14.0, trl 0.22.1, datasets 4.0.0, accelerate 1.10.1, sklearn 1.6.1,
rdkit 2026.3.6 are identical. **Cross-host numeric differences are possible**; compare a
Delta run to a cc run only after a same-seed check.

**Models cached on Delta** (offline load verified): Qwen3-4B-Instruct-2507, Qwen3-8B,
Llama-3.2-1B-Instruct. Not copied: Qwen2.5-7B-Instruct (one mention, unused by configs).

**SLURM on Delta** (user-specified): `--account=bhby-delta-gpu`,
`--partition=gpuA100x4,gpuA100x8,gpuH200x8`, `--gpus-per-node=1`, `--ntasks-per-node=16`, plus
`--mem=64G` added by me (Delta default is 1 GB/CPU). The sync script rewrites these into the Delta
copy of `sbatch/*.sbatch` and swaps `source activate steer` -> `rein`, adds `HF_HOME`; all 10
files checked. **8B caveat:** A100x4 GPUs are assumed to be 40 GB (unverified); an 8B run peaked
near 72 GB on cc, so submit 8B jobs with `--partition=gpuA100x8,gpuH200x8`.

**Not yet verified:** no GPU job has been submitted on Delta; the sbatch scripts are untested
end to end. Existing `exp_log/launch/*.sh` still hardcode cc settings (`RR=/u/lucmon/lucmon/rein_runs`
resolves to `/projects/bhby/lucmon/rein_runs` on Delta, not `/work`); Delta launch scripts must
set `RR=/u/lucmon/rein/runs` (or omit `out_dir` overrides) so output lands on `/work`.
**Log convention for Delta:** tag entries `[delta]` and keep Delta run dirs distinguishable; the
log lives on cc-login only.

Files added/changed: `scripts/sync_to_delta.sh` (new).

## 36. Math CoT (OpenR1-Math-220k): data downloaded, feasibility measured, plan drafted (2026-10-02)

No GPU job. Everything here ran on cc-login1 (CPU). Plan: `math-cot-experiment-plan.md`.

**Data.** `open-r1/OpenR1-Math-220k`, the default and extended splits (3.8 GB), downloaded to
`/projects/illinois/eng/cs/arindamb/lucmon/data/openr1_math` and symlinked as `data/openr1_math`.
Not yet on Delta (`sync_to_delta.sh` DATASETS must gain `openr1_math`).

**Scripts (new).**
- `scripts/openr1_stats.py`: flattens to one row per trace (`traces.parquet`), plus counts, length, accuracy by length decile, client candidates, and the R6 within-problem check (`stats.json`).
- `scripts/openr1_ladder_stats.py`: the target-ladder variant and the 12-client selection (`ladder_stats.json`).

**Key measurements.**
- 190,299 problems and 456,559 traces. Usable (complete and Math-Verify correct): 376,034 traces over 161,926 problems; median 3,293 Qwen3 tokens. 2.80 chars/token, Spearman(chars, tokens) 0.979.
- Accuracy falls with length: 0.90 in the shortest decile, 0.64 in the longest.
- Only 4 `source`s have ≥ 4k usable traces, so client = source × problem_type (21 groups with ≥ 1k problems; 12 selected, evenly spaced).
- **R6 fails for natural traces:** within-problem share of log-length variance 0.064 (0.085 over the 12 clients); median within-problem α range 0.085. The assumption in `dataset-candidates.md` was wrong (now annotated there).
- **Fix: a per-problem target ladder** (solution / R1 write-up / think cut at 25-50-75% / full). Within share 0.83, median within-problem α range 0.66. Cost: natural client skew shrinks (median spread 0.58 → 0.31, mean uncovered share 0.14). C1 therefore relies on a *holdings* regime (worked-solution vs. reasoning-trace clients), as specified in the plan.
- Eval cost: ≈ 12.7k generated tokens per problem over the 5-point α grid → ≈ 15M tokens per full test. vLLM is not installed on either host; the plan specifies an exact rank-32 LoRA export per (client, α).

Files: `math-cot-experiment-plan.md` (new), `scripts/openr1_stats.py` (new),
`scripts/openr1_ladder_stats.py` (new), `dataset-candidates.md` (R6 correction), `data/openr1_math` (symlink).

**Addendum (same day, after user discussion: is within-problem spread decisive?).** Inline analyses, no new files.
- **Length is only partly predictable from the problem text.** 12 selected clients, ≤ 6k problems each, 114,580 traces, 20% of problems held out. Client identity explains 0.175 of log-length variance, and the oracle problem identity explains 0.908. But a TF-IDF (1–2-gram) ridge with client one-hot and problem length gets held-out R² of only **0.41 overall**, and **0.29 within a client** beyond the client mean. From the model's side, most of the within-client α variation is therefore *not* given by x, so the α label carries signal for D. The 6.4% within-problem figure overstated the R6 problem.
- **Clients holding different solution sources give limited spectra with real targets only.** Roles alternate over the difficulty order. Clients holding only the NuminaMath `solution` field have supports of about [0.02, 0.40]; clients holding only R1 `full` traces about [0.45, 0.98]; "both" clients are broad. Mean uncovered share is **0.45** (natural full-only: 0.20).

**Addendum 2 (2026-10-03): α on the CoT only (user: drop the reference `solution` as a low-α source).** Inline analysis, usable traces ≤ 8k tokens.
- **Think-part tokens:** p5 / p50 / p95 = 885 / 2,406 / 6,535. The write-up after `</think>` is 230 / 454 / 692.
- **Rank agreement:** Spearman(think tokens, total tokens) = 0.997, so the choice barely changes α. Spearman(think tokens, paragraph steps) = 0.78; tokens per step p5 / p50 / p95 = 23 / 50 / 136.
- **CoT-only natural clients** (19 source × type groups with ≥ 1k problems): medians span 0.13 → 0.72, but supports are wide (width 0.69–0.89, mean 0.81).
- **Finer partition** (× question_type, 23 groups): medians span 0.15 → 0.78; widths are 0.57–0.88. The narrowest are hard banks missing the low end (cn_contest/Geometry [0.41, 0.98], aops_forum/Number Theory [0.29, 0.98]). Easy banks reach 0.81–0.95 at the top.
- **Retraction:** the 0.45 uncovered share in addendum 1 used role assignments that put every held-out client in the trace-only role, and it included `solution` targets. It is superseded.

**Addendum 3 (2026-10-03): short-CoT alternative, AI-MO/NuminaMath-CoT** (user: R1 traces are too long to learn).
Downloaded to `/projects/illinois/eng/cs/arindamb/lucmon/data/numinamath_cot` (1.2 GB), symlinked as `data/numinamath_cot`.
New script `scripts/numina_cot_stats.py` → `data/numinamath_cot/stats.json`.
- **Size and length:** 859,494 problems, one GPT-4o step-by-step solution each. 2.79 chars/token (Spearman 0.953). Solution tokens p5 / p50 / p95 / p99 = 118 / 356 / 914 / 1,158, about 7× shorter than R1 think parts (median 2,406).
- **Natural skew with limited spectra**, 8 sources with ≥ 5k rows, on the global scale:

  | source | median | support |
  |---|---|---|
  | synthetic_math | 0.22 | [0.03, 0.63] |
  | orca_math | 0.31 | [0.01, 0.72] |
  | gsm8k | 0.32 | [0.08, 0.65] |
  | cn_k12 | 0.34 | [0.02, 0.84] |
  | synthetic_amc | 0.53 | [0.20, 0.77] |
  | math | 0.55 | [0.10, 0.85] |
  | aops_forum | 0.87 | [0.59, 0.99] |
  | olympiads | 0.87 | [0.56, 0.98] |

  Median spread 0.64, mean support width **0.61** (R1 CoT-only: 0.81).
- **Learnability:** source explains 0.48 of log-length variance. A TF-IDF ridge reaches held-out R² 0.655 overall and **0.34 within a source**, so about two thirds of within-client variation is not predictable from the text.
- **Answer format:** `\boxed{}` present in ≥ 0.93 of solutions for every source except aops_forum (0.61).
- **Only 8 sources reach 5k** (amc_aime has 4,070), so 12 clients need the large sources split, e.g. by topic.

## 37. Results: C1 at 8B (Qwen3-8B, exp28); gain-clamp caveat for every local baseline (2026-10-03)

(Entry numbers 30–32 are used twice in this log: the ChEMBL entries from another session reuse them. This entry continues after 36.)

**Jobs:**
- Federated 11098308: done 10-02 22:03 (21.5 h).
- Local 11098309: training and test done; LLM judge still running at writing time.
- Both selected round 100 on dev.
- Report: `exp_log/reports/qwen3_8b_c1.txt`; style: `exp_log/reports/style_test_8b.txt`.

**Steering, test** (200 articles per client):

| Run | Pct error (worst) | In-support | Out-of-support (worst) | Reach | Spearman | Near-tie |
|---|---|---|---|---|---|---|
| **Fed 8B** | **0.156** (0.236) | **0.130** | **0.171** (0.261) | **0.36** | **0.917** | **0.20** |
| Local 8B | 0.177 (0.231) | 0.153 | 0.183 (0.281) | 0.32 | 0.880 | 0.32 |
| Fed 1B (entry 20) | 0.151 (0.199) | 0.135 | 0.161 (0.188) | 0.39 | 0.933 | 0.20 |
| Local 1B | 0.165 (0.205) | 0.148 | 0.165 (0.249) | 0.36 | 0.906 | 0.25 |

**Per client, fed 8B − local 8B** (paired bootstrap):
- **Overall better on 6/8** (theguardian.com −0.100, forbes.com, people.com, wsj.com, cbc.ca, aol.com), **worse on 1/8** (reuters.com +0.019); nypost.com a tie. At 1B: 4/8 better, 0 worse.
- Out-of-support: better on 3 (theguardian.com −0.137, people.com, aol.com), worse on 3 (forbes.com +0.051, cbc.ca, reuters.com). The same coverage trade-off as at 1B (entry 20).
- **C1 replicates on a second model family**, with a larger overall effect than at 1B.

**Quality, test:**
- Fed 8B AlignScore 0.797 / 0.763 (gap to the same-α reference −0.01 / +0.08); local 8B 0.826 / 0.725 (+0.02 / +0.04).
- BERTScore 0.903 / 0.893 vs. 0.904 / 0.891; length gap +1.6 / +2.3 vs. +4.5 / −0.2. **On par.**
- Judge, fed 8B: faithfulness 0.86 / 0.84, relevance gap +0.09 / +0.09; higher than fed 1B (0.82 / 0.79). Local 8B judge pending.

**House style** (entry 31 metric):
- Attribution fed 8B 0.559, local 8B 0.577 (1B: 0.552 / 0.553; real summaries 0.587).
- Feature gap 0.106 vs. 0.115.
- Attribution is flat across α (fed 8B 0.51–0.56). The private adapter keeps the house style at 8B too.

**8B is not better than 1B in absolute steering** (0.156 vs. 0.151). The difference is the copy-heavy clients:
- nypost.com 0.236 vs. 0.177; reuters.com 0.221 vs. 0.199.
- Their α = 0 outputs sit at percentile 0.27–0.28 (1B: 0.18–0.22): the 8B adapters absorb more of the clients' copying.

**Caveat found while analysing: the gain clamp binds for local runs.**
- The calibration gain s = exp(u) is clamped to [1/4, 4] (`lora.gain_max = 4`).
- Final local gains:
  - local 8B: 7/8 clients at 4.0 (nypost.com 2.66);
  - local 1B no-offset: 6/8 at 4.0 (reuters.com 2.86, nypost.com 2.23);
  - local 1B private calibration: 6/8 at 4.0.
- Federated shared gains are well inside the range: 1.93 (1B), 2.31 (8B).
- Local clients want a larger coefficient than the clamp allows. Their D can still grow to compensate (D → cD is equivalent to s → s·c), but more slowly, so **local baselines may be handicapped.** Example: local 8B theguardian.com saturates at percentile 0.51 at α = 1.
- **Needed before C1 is final:** local runs with a wider gain range (`lora.gain_max=16`), 1B first (~8.7 h), then 8B if the gap changes.

**Environment note:** `fedsteer/metrics.py` now imports `fedsteer/molecules.py` (ChEMBL work from another session), which needs `rdkit` at import time. Scripts that import `fedsteer` must run in the `steer` environment, not the system `python3`.

## 36. Attribution settled: the whole-molecule format is the problem, not steering (2026-10-03)

Jobs 11118146 (G0b: steered, 100 rounds) and 11119175 (G0-SFT: `fed.lr_shared=0`, no steering,
100 rounds) both completed. With entry 33 this gives the three-way attribution.

**G0-SFT's inertness assertion passed:** Spearman exactly 0.000 at every round, and validity and
retention identical at every alpha (0.840 / 0.536, in- and out-of-support equal). The direction
stayed at zero, so the model really was alpha-independent -- this is plain SFT. Its percentile
error 0.383 is worse than the 0.300 constant-output baseline, as it must be.

| run | steering | rounds | validity | retention |
|---|---|---|---|---|
| G0-SFT 11119175 | **none** | 100 | 0.840 | **0.536** |
| entry 33, 11117359 | yes | 20 | 0.818 | 0.576 |
| G0b 11118146 | yes | 100 | 0.770 | 0.498 |

**Both of my earlier hypotheses were wrong.**
- *Undertraining* (entry 33): wrong. 5x the training made validity and retention **worse**
  (0.818 -> 0.770, 0.576 -> 0.498). Train loss fell to 0.21 while dev loss plateaued at
  0.47-0.51 from round ~40, and steering did not improve either (pct err 0.216 at round 19 vs
  0.206 at round 99). It is overfitting, not underfitting.
- *Steering pressure breaks the core* (entry 34): wrong as the primary cause. **Plain SFT with no
  knob at all loses the core 46% of the time.** Core abandonment is a property of the
  whole-molecule output format, not of the knob.

**Correct decomposition.** There is a ~0.54 retention floor from the format alone; steering then
*redistributes* it across alpha (0.81 at alpha 0 down to 0.32 at alpha 1, entry 34) without
improving the mean. So the alpha-dependence I reported in entry 34 is real, but it sits on top of
a baseline incompetence that has nothing to do with alpha.

**Fix adopted: the decoration format** (plan §2.4), following Arús-Pous et al. 2020 and SAFE's
scaffold-decoration convention -- see the new plan §4b for the prior-art review. The core is given
in the prompt with numbered attachment points and the model emits only the decorations, so it can
never re-emit or renumber the core. SELFIES is explicitly *not* the fix: it addresses validity
(0.84) and leaves retention (0.54) untouched.

**New data `data/chembl_deco`** (`scripts/build_chembl_fed.py --format deco`), 36,687 records,
same clients and splits:
- cut bonds get the same dummy label on both sides, so `Chem.molzip` reassembly is unambiguous;
- only pairs whose split round-trips to the original molecule exactly are kept -- per-client yield
  0.71-0.96, train 1,396-4,000, test 142-198;
- **alpha labels are unchanged** (`clogp_residual_deco` reproduces the stored labels to 2.8e-14),
  so deco vs. smiles is a clean format comparison;
- retention measured at **1.000** when the true decorations are used (structural, as intended);
- targets are half as long: median 17 tokens vs 33.

New code: `fedsteer/molecules.py` (`split_core_decorations`, `rejoin_decorations`,
`clogp_residual_deco`, `assembled_descriptors`); `scripts/build_chembl_fed.py --format`;
`scripts/score_molecules.py` scores the assembled molecule for deco records;
`configs/chembl_deco.yaml`; `sbatch/train_eval_chembl.sbatch` gained `DATA` and `RESCORE`
(`RESCORE=none` for deco, where the strict scorer is meaningless).

**Launched (`exp_log/launch/exp35_chembl_deco.sh`):** 11124738 deco G0 (1 client, 100 rounds),
then **11124739 federated** and **11124740 local** (8 participants, rotation 0, 30 rounds), both
`--dependency=afterok` on the G0 job. Note this chains the long pair on a *clean exit*, not on the
gate passing, at the user's instruction to start them now.

**Environment note:** `pip install safe-mol` upgraded `fsspec` 2023.4.0 -> 2026.9.0 and
`protobuf` -> 6.33.6 in the shared `steer` env. torch 2.5.1, transformers 4.56.0 and datasets
4.0.0 were untouched and all still import; `datasets` prints a declared-constraint warning but
works, and no code in this repo imports it. The final implementation needs `safe-mol` only for
the prior-art review, not at runtime -- the decoration format uses RDKit alone.

## 37. Decoration format PASSES gate G0 decisively; G0 job timed out after its work was done (2026-10-03)

**Job 11124738 (deco G0, 1 client CHEMBL240, 100 rounds) hit its 6 h wall clock, but only after
finishing training *and* the test evaluation.** Timeline: start 04:07, training + all 5 dev evals
done 05:51 (1 h 44 m), test eval written 08:35 (2 h 44 m for 500 short generations), then
`score_molecules.py` was still running when SLURM killed the job at 10:07. The usable results were
already on disk; the two dependent jobs (11124739 fed, 11124740 local) died
`DependencyNeverSatisfied`, the exact failure mode flagged when they were chained with `afterok`.

**G0 passes, and the decoration format is better on every axis.** Test eval, dev-selected round 80:

| metric | whole-molecule, 20 rounds (entry 33) | whole-molecule, 100 rounds (G0b) | **decoration, 100 rounds** |
|---|---|---|---|
| validity | 0.818 | 0.770 | **1.000** |
| scaffold retention | 0.576 | 0.498 | **1.000** |
| unscorable row rate | 0.48 | — | **0.00** |
| pct_calib_err (constant = 0.300) | 0.217 | 0.206 | **0.165** |
| in-support / out-of-support | 0.193 / 0.254 | — | **0.150 / 0.189** |
| Spearman | 0.694 | 0.720 | **0.921** |
| concordance | 0.784 | — | **0.906** |
| endpoint increase rate | 0.904 | — | **1.000** |
| uniqueness / novelty | 0.995 / 0.99 | 1.0 / 0.979 | **1.000 / 1.000** |

Also unlike the whole-molecule runs, dev loss fell monotonically (0.543 -> 0.295) with **no
overfitting**, and the dev-selected checkpoint was round 80 of 100, i.e. still improving. Gate G2
(method < 0.25 vs the 0.300 constant baseline) is already met on a single client at 0.165.

⚠️ **Caveat to watch: the decorations are trivially small.** Sampled generations are
`[1*]C.[2*]C.[3*]O` -> `[1*]C.[2*]C.[3*]C`, i.e. the model steers by swapping a hydroxyl for a
methyl. That is directionally and chemically correct, and QED 0.603 / MW 425 / TPSA 68 are
reasonable, but `text_tie_rate` 0.28 and `adjacent_tie_rate` 0.285 say adjacent alphas often give
identical output. The knob may be exploiting a degenerate single-atom strategy rather than real
decoration chemistry. **To check on the 8-client runs:** decoration heavy-atom count per alpha,
and the gap to the same-alpha real reference molecules (`data/chembl_deco/refs.json`).

**Why the wall clock blew out, and the one real bug.** Generation was *not* the problem: outputs
stop at EOS (median 17 tokens, max 68; 0% reached the 96- or 192-token cap). Two causes:
1. **My bug in `scripts/score_molecules.py`:** the novelty precomputation called the full
   `assembled_descriptors` (molzip + sanitize + QED + TPSA + ...) on all ~37k training records
   when it only needs the assembled canonical SMILES. Fixed to use `rejoin_decorations` +
   `MolToSmiles`; the same scoring now runs in **26 s** instead of >1 h 32 m.
2. **Lustre was stalling for long stretches today** (plain `ls` and `squeue` hung for minutes from
   the login node around 10:00). That plausibly explains the 2 h 44 m test eval for 500 short
   generations, which should take minutes.

**Resubmitted the pair without a dependency** (`exp_log/launch/exp36_chembl_deco_pair.sh`):
**11135486 federated**, **11135487 local**, 8 participants of rotation 0, 30 rounds,
`--time=2-00:00:00`, `monitor.full_prompts` 40 -> 25 and dev `max_new_tokens` 192 -> 96 (observed
max 68), `TEST_PROMPTS=150`. These give G3 (federated vs local) and the multi-client G2.

**Held, then released (2026-10-03 ~14:30).** Both jobs landed on **ccc0284** and were requeued
after exactly 2 m 11 s each with `user_env_retrieval_failed_requeued_held`. Slurm's login-shell env
retrieval timed out before the batch script ran, so neither job wrote a log. The likely cause is an
NFS/`/u` stall on that node: `~/.bashrc` runs the conda hook from `/u/lucmon/lucmon/anaconda3`.
This is a guess and was not confirmed. On the login node, `bash -lc` takes 1.1 s, and other users'
jobs started normally on ccc0284 in the same hour. `~/.bashrc` (last changed 2026-05-28) and the
`steer` env (2025-09-21) were not modified. Fix: `scontrol release` on both jobs, same job ids.
(ccc0284 was briefly added to ExcNodeList, then removed: it is the only dali node.)

## 37. Math CoT on NuminaMath-CoT: data built, pipeline added, gates launched on Delta (2026-10-03) [delta]

User decision (10-03): switch the math task from OpenR1 (R1 traces, median 2.4k think tokens) to
**NuminaMath-CoT** (short step-by-step CoT, median ≈ 300 tokens). α = global percentile of the solution
length in ruler tokens; clients are sources, with the large sources split by problem type. Framing:
"CoT length control"; test-time compute only if accuracy-vs-α earns it. Plan rewritten:
`math-cot-experiment-plan.md`.

**Data (`data/math_fed/`, built on cc and synced to Delta `/work/nvme/bhby/datasets/math_fed`).**
- `scripts/build_math_fed.py` (new). NuminaMath-CoT, with NuminaMath-1.5 (downloaded to `data/numinamath_15`, 530 MB) used only for `problem_type`.
  - The 1.5 join works for cn_k12 (95%), orca_math (100%) and synthetic_math (99.8%), but not for olympiads (0.7%), aops_forum (19%), math (4%) or synthetic_amc (0%). So only the first three are split, and the gold answer is the reference solution's last `\boxed{}` for every client.
- Filtering: 45,121 exact-duplicate problems dropped, keeping the copy under the original benchmark (gsm8k > math > … > orca_math; otherwise gsm8k disappears into orca_math). 32,166 solutions without `\boxed{}` dropped. Targets ≤ 1,024 ruler tokens; prompts ≤ 480.
- **12 clients × 4,000 train / 50 dev / 100 test.** Median tokens: orca_math/Logic 138, cn_k12/Logic 170, gsm8k 244, synthetic_math/Algebra 271, orca_math/Algebra 283, cn_k12/Inequalities 287, synthetic_math/Geometry 304, cn_k12/Geometry 315, math 403, synthetic_amc 411, olympiads 679, aops_forum 726.
- Supports on the 12-client scale: olympiads **[0.58, 0.99]** and aops_forum **[0.60, 0.99]**; gsm8k [0.12, 0.65]; Logic & Puzzles [0.01, 0.68].
- Rotation 0 holds out orca_math/Logic, synthetic_math/Algebra, synthetic_math/Geometry and synthetic_amc. Rotation 1 holds out olympiads (the C2 coverage test).

**Code (new or changed).**
- `fedsteer/mathcot.py` (new): `cot_tokens` scorer (fixed ruler `Qwen/Qwen3-4B-Instruct-2507`, override with `FEDSTEER_RULER`), `extract_boxed`, `is_correct` (Math-Verify, plus normalized exact match for MCQ letters), `repetition_loop`, `gzip_ratio`.
- `fedsteer/metrics.py`: registered `cot_tokens`.
- `fedsteer/baselines.py`: B1 math template (`math_level_instruction`, `prompt_with_level_math`; auto-selected for records with `problem`).
- `scripts/score_math.py` (new): per-α accuracy / boxed / truncated / loop / tokens / gzip, split by in/out of support, written to `math_<eval>.json`.
- `scripts/eval_base_math.py` (new): E0a; the base model through the same generation path, with both B matrices at zero.
- `configs/math_fedavg.yaml` (new): method defaults, `max_target_tokens 1024`, monitor `cot_tokens` with a 1,280-token cap.
- `sbatch/train_eval_math.sbatch`, `sbatch/eval_base_math.sbatch`, `sbatch/eval_b1_math.sbatch` (new). The last finds a run by its training job id, so it can be chained with `--dependency`.
- `scripts/sync_to_delta.sh`: adds `math_fed` to DATASETS. It also rewrites `set -e` around env setup (see below).

**Checks (cc, CPU).**
- All 1,200 test references score correct against their own gold; a perturbed answer scores wrong.
- The `cot_tokens` scorer reproduces the stored labels exactly.
- A Llama-1B CPU smoke run (1 round, 2 steps, monitor with 2 dev prompts) went through train → dev eval → `score_math` → `summarize_sweep`.

**Delta infrastructure: two bugs found and fixed** (the first Delta GPU jobs; entry 35 had said "untested end to end").
1. **`rein` had CPU-only torch.** `conda create --clone steer` (10-02) resolved to `pytorch 2.5.1 cpu_mkl`. Entry 35's "torch 2.5.1+cu124" for `rein` was wrong. Fix: `conda remove --force pytorch libtorch torchaudio`, then `pip install torch==2.5.1 --index-url .../whl/cu124`. Now `2.5.1+cu124`, sympy back to 1.13.1, and a GPU node returns `torch.cuda.is_available() == True`; math-verify works.
2. **Delta's `/etc/bashrc` returns non-zero on compute nodes** (a `profile.d` script, line 79), which kills any script that runs `set -e` before `source ~/.bashrc`. Jobs died in about 12 s with an empty log. This affects every sbatch script synced to Delta, not just the math ones. Fix: the sync script now rewrites the Delta copies to `set +e; source ~/.bashrc; source activate rein; set -e`.
- First submission 22640626–28 (FAILED in setup, bug 2; bug 1 would have followed). Debug jobs 22640683, 22640731/37/50, 22640755.

**Jobs (Delta; launch file `exp_log/launch/exp37_math_gates.sh`; outputs in Delta `runs/` → `/work/nvme/bhby/lucmon/rein`; logs in Delta `sbatch/logs/`).**

| job | purpose | config / overrides | run dir prefix | status |
|---|---|---|---|---|
| 22640760 | E0a: base Qwen3-4B on 12 × 100 test problems | `eval_base_math.sbatch`, 1,280-token cap | `runs/exp37_math_e0a_base_qwen3_4b_20261003-115825_j22640760` | COMPLETED (2 h 40 min); **cap-limited**, see below |
| 22640761 | G0: plain SFT on `math`, D frozen at zero | `clients=[math] fed.lr_shared=0 fed.rounds=60 fed.save_every=10 monitor.full_every=10 monitor.full_prompts=50` | `runs/exp37_math_g0_sft_qwen3_4b` | PENDING |
| 22640762 | G2: single-client steering on `math` | same, steering on | `runs/exp37_math_g2_single_qwen3_4b` | PENDING |
| ~~22640765~~ | G1, first submission | — | — | CANCELLED (the base row needed a longer cap) |
| ~~22643461~~ | G1, second submission | — | — | CANCELLED (moved after the 120-round G2) |
| 22643676 | G0 resumed to **120 rounds** (same run dir) | `RESUME=<G0 run> OVERRIDES="fed.rounds=120 monitor.full_every=20"`, afterok 22640761 | G0 run | PENDING (dependency) |
| 22643677 | G2 resumed to **120 rounds** (same run dir) | same, afterok 22640762 | G2 run | PENDING (dependency) |
| 22643678 | G1: B1 prompting on the extended G2 (k = 0 / 3 client at 1,280; k = 3 base at 4,096) | `eval_b1_math.sbatch`, `RUN_JOB=22640762`, afterok 22643677 | G2 run | PENDING (dependency) |
| 22643457, 22643459, 22643460 | E0a-long: base model at a **4,096** cap, 3 × 4 clients, batch 16 | `eval_base_math.sbatch`, `MAX_NEW=4096 BATCH=16 CLIENTS=…` | `runs/exp37_math_e0a_long4096_qwen3_4b` | PENDING |

**E0a result (job 22640760): the 1,280 cap, not the model, sets base accuracy.** Mean over 12 clients:
accuracy 0.464, boxed 0.618, **truncated 0.416**, loop 0.128, mean length 833 tokens.

| client | acc | trunc | acc on untruncated | base mean tokens | reference median |
|---|---|---|---|---|---|
| gsm8k | 0.93 | 0.01 | 0.94 | 297 | 233 |
| orca_math/Logic | 0.82 | 0.10 | 0.91 | 377 | 124 |
| math | 0.64 | 0.38 | **0.98** | 830 | 418 |
| cn_k12/Geometry | 0.30 | 0.54 | 0.59 | 1,009 | 276 |
| olympiads | 0.18 | 0.79 | **0.81** | 1,172 | 662 |
| aops_forum | 0.09 | 0.89 | **0.73** | 1,238 | 678 |

- The base model writes 1.3–4× longer than the reference solutions.
- Most of its "errors" are truncations before `\boxed{}`. Accuracy on untruncated outputs is optimistic (it favours easy problems) but shows the cap dominates.

Consequences:
1. **G0's reference ("accuracy ≥ base − 5 points") is invalid at this cap.** It would pass trivially. → E0a-long at a 4,096 cap (jobs above). `eval_base_math.py` gained `--clients`; `eval_base_math.sbatch` gained `CLIENTS` and `BATCH`.
2. **G1's base-model row** would be truncated too, and truncated outputs score as maximally long, which flatters calibration at α = 1. → `eval_b1_math.sbatch` now uses `MAX_NEW_BASE=4096` (batch 16) for the base row only. The client-model rows stay at 1,280: fine-tuned outputs are short (G2 truncates ≤ 4% at any α).
3. The constant-output baseline (0.30) is unaffected: it is defined from the data alone.

**G0 / G2 dev curves so far** (math client, 50 dev problems × 5 α):

| round | G0 (D = 0) pct err / Spearman | G2 (steering) pct err / Spearman |
|---|---|---|
| 10 | 0.386 / 0.000 | 0.360 / 0.515 |
| 20 | 0.391 / 0.000 | 0.341 / 0.519 |
| 30 | 0.397 / 0.000 | 0.299 / 0.671 |
| 40 | 0.387 / 0.000 | **0.273 / 0.745** |

- G0 behaves as designed: identical output at every α, Spearman 0.
- G2 is still improving at round 40 and is below the 0.30 constant baseline from round 30.

**Training length: G0 and G2 extended from 60 to 120 rounds** (user question, 10-03).
- G2 at round 50: pct err 0.235, Spearman 0.824, still improving by about 0.04 per 10 rounds, with the direction norm still growing (25.0 / 29.3 / 33.0 at rounds 30 / 40 / 50).
- Cost: 60 rounds = 2.4 epochs over 4k pairs. After epoch 2 the private adapter starts memorizing: dev loss went from 0.307 (round 50) to 0.337 (round 58), with train loss at 0.185. Newsroom showed the same pattern (steering kept improving after dev loss rose; 100 rounds at 4k).
- Both runs are resumed in place (`train_fed.py --resume`), with dev evals every 20 rounds. Checkpoint selection runs over all dev evals of the run, so the extension can only add candidates. Watch dev accuracy (`score_math`) for the memorization cost.
- The 60-round jobs still finish their own test evaluation first, so the round-60 result is kept.
- `sbatch/train_eval_math.sbatch` gained `RESUME=<run>`. `sbatch/eval_b1_math.sbatch` now selects over all of a run's dev evals instead of matching the training job id.

**Time limits raised to 1 day** (user, 10-03).
- All pending jobs updated in place with `scontrol update TimeLimit=1-00:00:00`: 22643457/59/60, 22643676/77/78. Delta's GPU partitions allow up to 2 days.
- The running 60-round jobs (22640761/62) could not be raised ("Access/permission denied") and keep 8 h. They are expected to finish around 6.3 h.
- In case they time out in their test evaluation, the resume jobs' dependencies were changed from `afterok` to `afterany`; `state.pt` at round 60 is enough to resume. G1 stays `afterok` on the resumed G2.
- The `#SBATCH --time` default is now `1-00:00:00` in `train_eval_math.sbatch`, `eval_base_math.sbatch` and `eval_b1_math.sbatch`; the per-job `--time` flags were removed from `exp_log/launch/exp37_math_gates.sh`.

## 38. Deco pair: federated 8-client run done; local crashed on a config conflict (2026-10-03)

**11135487 (local) FAILED after 2 m 17 s, exit 1.** `configs/chembl_deco.yaml` bakes in
`fed.calibration: shared` (the method default from entry 20) and `fed.py` rejects shared
calibration in local mode -- nothing is aggregated there, so local training always has per-client
calibration. The launcher overrode `fed.mode=local` but not the calibration. The Newsroom launch
scripts avoid this by leaving calibration out of `$COMMON` and adding `=shared` only to the fed
run; baking it into the config broke that pattern. Resubmitted as **11143950** with
`fed.calibration=private` (`exp_log/launch/exp37_deco_local_fix.sh`).

**11135486 (federated, 8 participants, 30 rounds) COMPLETED in 3 h 26 m.**

| | mean | worst client |
|---|---|---|
| pct_calib_err (constant = 0.300) | **0.237** | **0.312** (CHEMBL243, HIV-1 protease -- *worse than constant*) |
| in-support / out-of-support | 0.204 / 0.282 | — |
| Spearman | 0.800 | 0.521 |
| adjacent tie rate | **0.503** | 0.65 |
| unscorable row rate | 0.01 | 0.03 |

Validity and retention hold up at 8 clients (unscorable 0.00-0.03), so the decoration format
transfers. But steering is much weaker than the single-client G0 (0.165): the mean clears gate G2
only marginally and the worst client does not clear it at all.

⚠️ **Undertrained, unambiguously.** Dev pctErr improved at every snapshot (0.259 -> 0.249 ->
0.237), dev loss fell throughout (0.368 -> 0.305 -> 0.273) and **the dev-selected checkpoint was
the final round**. Newsroom used 100 rounds; 30 is not comparable. Launched
**11143987 fed / 11143988 local at 100 rounds** (`exp_log/launch/exp38_deco_100r.sh`); the
30-round local (11143950) still runs so there is also a matched-budget fed-vs-local pair.

**On the entry-37 degeneracy worry: partly answered, and the answer is better than feared.**
Mean decoration size barely moves with alpha (6.0 -> 6.7 heavy atoms), but **individual clients
move in opposite directions**, which is chemically correct -- logP rises either by adding
lipophilic bulk or by removing polar groups:

| client | a=0 | a=0.25 | a=0.5 | a=0.75 | a=1 |
|---|---|---|---|---|---|
| CHEMBL325 (HDAC1) | 6.2 | 6.8 | 7.7 | 9.2 | **10.9** (adds bulk) |
| CHEMBL204 (thrombin) | 9.2 | 9.5 | 7.8 | 6.9 | **6.9** (drops polar amidines) |
| CHEMBL240 (hERG) | 3.9 | 3.9 | 4.7 | 5.6 | 6.3 |
| CHEMBL243 (HIV-1 protease) | 10.4 | 10.0 | 9.6 | 9.3 | 10.1 (flat -- the worst client) |
| CHEMBL2039 (MAO-B) | 2.2 | 2.6 | 3.0 | 3.3 | 3.6 |

So the `C.C.O -> C.C.C` single-atom pattern seen in G0 was CHEMBL240-specific, not the general
strategy. **But a real quality gap remains:** real training decorations average 7-12 heavy atoms
(12.3 in the lowest attribute decile, 7.0 mid, 10.0 in the top 30% -- a U-shape, since both
extremes need large groups), while the model emits 2-4 for several clients. It under-decorates.
Add decoration heavy-atom count vs the same-alpha reference to the reported metrics.

**Tie rate is the live concern:** 0.46-0.65 of adjacent-alpha pairs give identical output, far
above Newsroom's. If 100 rounds does not bring it down, the knob is too coarse on this task.

## 38. Math CoT gates, results: G0 fails on accuracy, G2 steers but misses calibration (2026-10-03) [delta]

Jobs 22640761 (G0, 6 h 28 min) and 22640762 (G2, 5 h 34 min) COMPLETED.
- Runs (Delta): `runs/exp37_math_g0_sft_qwen3_4b_20261003-115907_j22640761` (selected round 50) and `runs/exp37_math_g2_single_qwen3_4b_20261003-115948_j22640762` (selected round 60, the last).
- Report: `scripts/math_gate_report.py` (new; from saved evals only). Client `math`, 100 test problems, the same problems as E0a.

**G2 per α (test, round 60).** pct err **0.226** (in-support 0.208, out-of-support 0.253), Spearman **0.772**, concordance 0.842, adjacent-decrease rate 0.253, gain 2.83 (max 4).

| α | target tokens | generated median (IQR) | pct err | acc | boxed | trunc |
|---|---|---|---|---|---|---|
| 0 | 77 | 248 (204–306) | 0.201 | 0.53 | 0.99 | 0.01 |
| 0.25 | 292 | 353 (280–435) | 0.224 | 0.54 | 1.00 | 0.00 |
| 0.5 | 403 | 431 (365–512) | 0.193 | 0.55 | 0.98 | 0.02 |
| 0.75 | 530 | 454 (383–551) | 0.207 | 0.53 | 1.00 | 0.00 |
| 1 | 1,024 | 502 (416–616) | 0.306 | 0.57 | 0.98 | 0.02 |

- **The range is compressed at both ends:** generated medians run 248 → 502 against targets 77 → 1,024.
- The dev curve flattened between rounds 50 and 60 (0.235 → 0.232).
- An example problem: the α grid changes the solution's detail (one-paragraph sketch → numbered, bolded steps), with the same GPT-4o house format.

**Accuracy, the same 100 problems.**

| model | acc | boxed | trunc | median tokens |
|---|---|---|---|---|
| base (cap 1,280) | 0.64 | 0.65 | 0.38 | 740 |
| plain SFT (G0, round 50) | 0.57 | 0.98 | 0.02 | 408 |
| G2, α = 0 … 1 | 0.53–0.57 | ≥ 0.98 | ≤ 0.02 | 248–502 |

- SFT vs. base, paired: SFT right / base wrong 7; SFT wrong / base right 14.
- **On the 62 problems where the base finished: base 0.98, SFT 0.79.**
- The drop is already there at round 10 (G0 dev accuracy 0.60–0.64, flat over rounds 10–60). So it is a shift to the GPT-4o solution style, not overfitting. The student (Qwen3-4B-Instruct-2507) reasons better in its own long style than in the teacher's.
- **G2 costs no accuracy against SFT** (paired +13/−17 … +9/−9 per α, none significant).
- **Accuracy is flat in α** (0.53–0.57): length here is detail, not compute, as the plan's framing anticipated.

**Loop metric corrected.** The 10-word × 3 detector mostly fired on restated formulas (e.g. a `\cot(\sum \operatorname{arccot} z_k)` expression repeated). Rates by definition:

| | 10 × 3 | 20 × 3 | 10 × 5 |
|---|---|---|---|
| base | 0.110 | 0.020 | 0.030 |
| G0 | 0.090 | 0.020 | 0.030 |
| G2 | 0.054 | 0.002 | 0.016 |

The default is now 20 × 3 (`fedsteer/mathcot.py`; `scripts/score_math.py` docstring updated). Synced, so the queued jobs use it.

**Gate verdicts.**
- **G0: FAIL on accuracy.** Format passes: boxed 0.98, truncation 0.02, loops 0.02. Accuracy is 0.57 vs. a base reference of 0.64 even at the cap-limited 1,280 setting; E0a-long (4,096) will raise the reference and widen the gap.
- **G2: FAIL on calibration** (0.226, needs < 0.20). Passes Spearman (0.772 ≥ 0.7) and accuracy vs. SFT (in-support 0.54 vs. 0.57).

Pending: G0/G2 extended to 120 rounds (22643676/77), G1 (22643678), E0a-long (22643457/59/60). All PENDING (queue), none started yet.

## 39. Backbone screen for the math-CoT task; Qwen3-4B jobs abandoned (2026-10-03) [delta]

**Decision (user, 10-03): the accuracy loss from SFT is the important problem**, so the queued Qwen3-4B jobs were cancelled before they started:
- E0a-long (22643457/59/60);
- the 120-round G0/G2 extensions (22643676/77);
- G1 (22643678).

The completed 60-round G0/G2 (entry 38) stay as the record of the style-mismatch result.

**Literature (checked 10-03).** SFT on another model's solutions can lower a strong student's reasoning; the documented cause is a style or distribution mismatch.
- Huang et al. 2026, arXiv 2604.14164 (TESSY): Qwen3-8B fine-tuned on stronger teachers' data loses 3.25 / 10.02 points on LiveCodeBench-Pro / OJBench.
- Ren et al., EMNLP 2024 ("I Learn Better If You Speak My Language"): the target data's perplexity under the student predicts how well fine-tuning goes.

So the fix is a backbone for which GPT-4o's style is familiar and an upgrade. Self-generated targets were rejected: one generator would erase the per-client solution styles (87% client attribution with math masked; the PFL "local characteristics" side).

**Screen (no training): `scripts/backbone_screen.py` (new) + `sbatch/backbone_screen.sbatch` (new), launch file `exp_log/launch/exp39_backbone_screen.sh`.**
- *Familiarity:* mean target-token NLL / perplexity of 42 train solutions per client (504 in total), with the training chat formatting and the base model (LoRA with both B matrices at zero).
- *Headroom:* greedy base accuracy, boxed, truncation and length on 50 test problems from each of gsm8k, cn_k12/Geometry, math and olympiads, at a **4,096** cap, then `score_math.py`.
- Models staged on Delta's `HF_HOME`:
  - copied from the cc cache: Llama-3.1-8B-Instruct (gated) and Qwen2.5-7B-Instruct;
  - downloaded on Delta: Qwen2.5-3B-Instruct;
  - already present: Qwen3-4B-Instruct-2507, Qwen3-8B, Llama-3.2-1B-Instruct.
- All offline-load checked on Delta. Chat formatting checked for every model on cc; Qwen3-8B gets the empty `<think></think>` of non-thinking mode.

| job | backbone | run dir prefix | status |
|---|---|---|---|
| 3308052 [dtai] | Qwen3-4B-Instruct-2507 (reference: the G0 backbone) | `runs/exp39_screen_qwen3_4b_2507` | CANCELLED after 2 of 4 headroom clients |
| 3308053 [dtai] | Qwen3-8B, thinking off | `runs/exp39_screen_qwen3_8b_nothink` | CANCELLED after 2 of 4 headroom clients |
| 3308054 [dtai] | Qwen2.5-7B-Instruct | `runs/exp39_screen_qwen25_7b` | COMPLETED (35 min) |
| 3308056 [dtai] | Qwen2.5-3B-Instruct | `runs/exp39_screen_qwen25_3b` | COMPLETED |
| 3308058 [dtai] | Llama-3.1-8B-Instruct | `runs/exp39_screen_llama31_8b` | CANCELLED after 3 of 4 headroom clients |
| 3308063 [dtai] | Llama-3.2-1B-Instruct | `runs/exp39_screen_llama32_1b` | COMPLETED |

Next: G0 (plain SFT, D frozen at zero) on the one or two best candidates (lowest perplexity, with base accuracy below the data's level), then G2.

**Infrastructure note.** A remote command that also ran `huggingface-cli whoami` blocked the tool's shell for about 30 minutes, with no output at all. Remote calls now get `timeout` and `</dev/null`.

**Moved to DeltaAI (dtai-1), 10-04.** The Delta submissions 22647630–38 never started (Priority, no start estimate) and were cancelled; the screen was resubmitted on dtai, where 5 of 6 started at once.

DeltaAI setup (user-provided header: `--ntasks-per-node=16 --partition=ghx4 --gpus-per-node=1 --account=bhby-dtai-gh`):
- **Architecture:** dtai-1 = gh-login01, **aarch64**; ghx4 nodes have 4 × GH200 (120 GB).
- **Filesystems:** `/work/nvme/bhby` is shared with Delta, so `runs/`, datasets and `HF_HOME` are the same directories. **Home is not shared**: the code goes to dtai's own `/u/lucmon/rein`.
- **Env:** dtai has its own conda (`/work/nvme/bhby/lucmon/anaconda3_deltaai`). New env **`rein-gh`** = `conda create --clone steer` (dtai's steer) + scipy 1.17.1, scikit-learn 1.9.1, rdkit 2026.03.6, math-verify (all missing from steer).
  - torch **2.13.0+cu129** (pip; the clone kept CUDA, checked).
  - transformers 4.56.0 and peft 0.14.0, identical to Delta. numpy 2.4.6.
  - ⚠️ torch 2.13 vs. 2.5.1 on Delta and cc, plus a different architecture: **compare dtai runs only with dtai runs.** The whole screen runs on dtai for that reason.
- **SSH:** the `dtai-1` block in `~/.ssh/config` lacked `ControlMaster auto`, so `ssh -fN dtai-1` made no reusable socket. Added it (backup `~/.ssh/config.bak-20261004`).
- **Sync:** `scripts/sync_to_delta.sh --host dtai` (new option). It syncs the code to dtai's home, links `runs/` and `data/` to the shared `/work` directories, and rewrites the sbatch headers to account `bhby-dtai-gh`, partition `ghx4` and env `rein-gh` (plus `--mem=64G` and the `set +e` bashrc guard).
- **Smoke test:** job 3308023, Qwen2.5-3B with tiny settings, COMPLETED in 1 min 39 s on gh064. Load, familiarity, generation and `score_math` all worked.
- The launch file `exp_log/launch/exp39_backbone_screen.sh` takes `HOST=dtai-1`.

**Screen, familiarity results** (mean target-token NLL over 12 clients × 42 GPT-4o solutions; base models, training chat format; dtai):

| backbone | NLL | perplexity |
|---|---|---|
| Qwen2.5-3B-Instruct | **0.537** | **1.71** |
| Qwen2.5-7B-Instruct | **0.560** | **1.75** |
| Llama-3.1-8B-Instruct | 0.642 | 1.90 |
| Llama-3.2-1B-Instruct | 0.705 | 2.02 |
| Qwen3-8B (thinking off) | 1.075 | 2.93 |
| Qwen3-4B-Instruct-2507 (G0 backbone) | **1.292** | **3.64** |

The GPT-4o solutions are about 2.4× less surprising (in NLL) to Qwen2.5 than to Qwen3-4B-2507, which is the backbone that lost accuracy in G0. This is consistent with the style-mismatch account (Ren et al.). Headroom (base accuracy at 4,096) is still running.

**Screen, headroom results and the backbone decision** (dtai; 50 test problems per client, 4,096 cap; `runs/exp39_screen_*`).

| backbone | ppl | gsm8k acc / median tok | cn_k12/Geometry acc / trunc / loop / median tok | math acc / median tok | olympiads acc / median tok |
|---|---|---|---|---|---|
| reference median tokens | — | 240 | 261 | 413 | 665 |
| **Qwen2.5-7B-Instruct** | 1.75 | 0.98 / 273 | 0.52 / 0.00 / 0.02 / 570 | 0.72 / 570 | 0.48 / 641 |
| Qwen2.5-3B-Instruct | 1.71 | 0.92 / 295 | 0.52 / 0.00 / 0.04 / 535 | 0.68 / 513 | 0.36 / 739 (trunc 0.02) |
| Llama-3.1-8B-Instruct | 1.90 | 0.98 / 238 | **0.14 / 0.26 / 0.24** / 632 | 0.64 (trunc 0.18, loop 0.22) / 410 | cancelled |
| Llama-3.2-1B-Instruct | 2.02 | **0.02** (boxed 0.02) / 193 | 0.08 / 0.12 / 0.12 / 506 | 0.14 (boxed 0.44) / 356 | 0.00 (boxed 0.28, trunc 0.24) / 697 |
| Qwen3-8B, no thinking | 2.93 | 0.94 / 268 | 0.42 / 0.10 / 0.10 / 800 | cancelled | cancelled |
| Qwen3-4B-Instruct-2507 | 3.64 | 0.98 / 266 | 0.48 / 0.18 / 0.04 / **1,224** | cancelled | cancelled |

**Decision: Qwen2.5-7B-Instruct.**
- It is the most familiar with the data, tied with the 3B.
- Its format is clean: 100% boxed, no truncation, ≤ 2% loops.
- Its natural lengths are already near the references, so fine-tuning nudges its style instead of overriding it.
- Its accuracy is moderate (0.48–0.98), with room to change.

Rejected:
- Llama-3.1-8B degenerates on cn_k12/Geometry (26% truncated at 4,096, 24% loops).
- Llama-3.2-1B cannot follow the answer format.
- The Qwen3 models are unfamiliar and verbose (Qwen3-4B: median 1,224 tokens on Geometry against a reference of 261).

Qwen2.5-3B is the cheaper fallback. The three slow Qwen3 / Llama-8B jobs were cancelled once the decision no longer depended on them, since they kept drawing on the allocation while the new jobs queued.

**User direction (10-04):** focus on the CoT length; accuracy is reported, not gated. The backup if accuracy still degrades is the gold answer in the prompt: `scripts/make_answer_prompt_data.py` (new) built `data/math_fed_ans/` (same records, prompts end in "The final answer is $X$. Please reason step by step to reach it, …"). It is synced to the shared `/work` and added to the sync DATASETS.

## 38. Design of the calibration (gain) function: learnable vs. constant, shared vs. per-client (2026-10-03)

**Questions (user):**
1. Does the learnable, non-linear calibration g(α) = s·h(α) help compared with a constant one, g(α) = α?
2. Should it be shared by all clients or kept per client?

**Design:** Llama-3.2-1B, 4k per client, 100 rounds, federated, no offset, standard pipeline (dev selection → test → quality → judge). Each arm differs from the method run 11063910 only in g.

| Arm | g(α) | Overrides | Question | Job |
|---|---|---|---|---|
| Method (existing, 11063910) | s·h(α), shared, s and the warp h learned | `fed.calibration=shared` | reference | done (0.151, entry 20) |
| **const** | α (s = 1, h = identity) | `lora.warp=none fed.fix_gain=true` | Q1 | 11145307 |
| **linear** | s·α, shared, s learned | `lora.warp=none` | Q1: learned scale vs. learned non-linearity | 11145308 |
| **private** | s_i·h_i(α), per client | `fed.calibration=private` | Q2 | 11145309 |

**Notes:**
- Earlier evidence on Q2 *with* an offset: shared 0.158 vs. private 0.155 (11063906 vs. 11041282, entry 20). The private arm here completes the no-offset pair.
- New-client evidence (E2, entry 30): the shared calibration gives the same curve as one trained per client.
- Federated per-client gains stay inside the clamp (0.97–2.2 in 11041282), so `gain_max` stays 4. The clamp issue of entry 37 concerns local runs only.
- Run directories in project space (`/u/lucmon/lucmon/rein_runs/`, linked into `runs/`), because of the home quota.
- What to compare (with `compare_runs.py`, per client, paired bootstrap): error in/out of support, reach, Spearman, near-ties, quality, judge, house style. For Q2 also the per-client calibration curves (from the eval files' warp curves).

**Launch:** [exp38_gain_design.sh](launch/exp38_gain_design.sh). About 6.5 h each once running.

## 39. Gain cap kept at 4 everywhere; home file-count analysis and a correction (2026-10-03)

**Decision (user):** the per-client gain cap stays at 4 (`lora.gain_max = 4`) in every run, so all comparisons use the same calibration range. That includes the calibration-design arms of entry 38 and all earlier federated and local runs. The local rerun with a wider range (entry 37) is **not** submitted. The observation that most local gains end at the cap stays as a caveat for the write-up.

**Home file count** (`/u/lucmon`, 455k files; soft limit 490k, hard limit 500k):
- 16 VS Code / Cursor remote-server installations (`.vscode-server*`, `.cursor-server`): ≈ 386k files (85%).
- `.local`, `.cache`, `course`, `mlopt`, `.codex`, `.cargo`, others: ≈ 69k.
- `rein` (this project, including `.git`): 923 files, 3.1 GB (data ~350 MB, SLURM logs, code).
- By size: `.cache` 17.4 GB, `tensorflow_datasets` 15.6 GB, editor servers 1–4 GB each.
- **Our code is not the cause.**

**Correction to entry 28:** `rein/runs` has been a symlink into project space (`/projects/illinois/eng/cs/arindamb/lucmon/rein/runs`) since 09-29. So run directories, code snapshots and evaluation files never counted against the home quota. Moving the 8B runs to `/u/lucmon/lucmon/rein_runs/` (also project space) was unnecessary but harmless; they stay linked into `runs/`.

## 39. ChEMBL 100-round fed/local pair: G2 passes, **C1/G3 fails** (2026-10-04)

All four deco runs are in: fed-30 (11135486), local-30 (11143950), fed-100 (11143987),
local-100 (11143988). Test evals, mean over 8 clients, constant-output baseline 0.300:

| run | err | in-sup | out-sup | rho | ties | reach | range | unscor | worst |
|---|---|---|---|---|---|---|---|---|---|
| fed-30 | 0.235 | 0.204 | 0.282 | 0.789 | 0.505 | 0.119 | 0.437 | 0.006 | 0.312 |
| local-30 | 0.213 | 0.185 | 0.255 | 0.822 | 0.468 | 0.146 | 0.491 | 0.003 | 0.299 |
| **fed-100** | **0.194** | 0.177 | 0.219 | 0.856 | 0.390 | 0.200 | 0.562 | 0.008 | 0.271 |
| **local-100** | **0.184** | 0.166 | 0.212 | 0.868 | 0.367 | 0.203 | 0.576 | 0.010 | 0.270 |

**G2 passes:** both 100-round runs are well below the 0.300 constant baseline and below the 0.25
gate. The 30-round runs were undertrained exactly as diagnosed; 100 rounds improved every metric
(fed err 0.235 -> 0.194, ties 0.505 -> 0.390, reach 0.119 -> 0.200, range 0.437 -> 0.562), and
round 100 was *still* the dev-selected checkpoint.

**G3 / claim C1 fails, with statistical backing.** `scripts/compare_runs.py` paired bootstrap over
test inputs, fed-100 vs local-100: **federated significantly better on 0/8 clients, significantly
worse on 4/8** (CHEMBL325 +0.027*, CHEMBL4078 +0.022*, CHEMBL2835 +0.013*, CHEMBL2039 +0.011*).
Only CHEMBL240 is significantly better, and only out-of-support (-0.019*). Raw per-client counts:
federated better on 0/8 at 30 rounds, 1/8 at 100 rounds.

**Crucially, federation does not help out-of-support either** (0.219 vs 0.212), which is where the
coverage argument lives.

**Why -- and it is the reason predicted in plan §3.3, not a method failure.** ChEMBL clients have
**no coverage gaps to fill**: only **17.2%** of attribute variance is between clients (entry
analysis, 2026-10-03), each client's support spans ~[0.02, 0.95], and the mean uncovered share of
the alpha scale is 0.19. So "out-of-support" here means the thin 5% tails of an already-broad
support, not a genuine gap -- exactly the gap-vs-tail distinction `federated-steering-plan.md` §4
flags as needing separation. Federation therefore offers no coverage benefit while still paying
the cost of constraining one direction to serve 8 clients whose decoration chemistry differs
(decoration-vocabulary Jaccard 0.155). Local-only wins because it is strictly less constrained.

⚠️ **A mechanism I checked and must NOT claim.** The fed run's `client_delta_cos_mean` collapses
from 0.604 to ~0.000 over training, which looked like "the clients disagree about the direction".
**Newsroom does the same**: exp28 Qwen3-8B 0.739 -> -0.000, nr_global_fedavg 0.570 -> +0.000,
toy 0.861 -> +0.008. So the collapse is a generic property of this setup (once the direction is
learned, the residual per-client gradients are noise, which is near-orthogonal), **not** a
ChEMBL-specific pathology, and it does not explain the C1 failure.

**Decoration under-sizing partly resolved by longer training** (entry 38's concern). Mean heavy
atoms per alpha now 5.8 / 5.9 / 6.0 / 6.5 / 7.2 (was 6.0 -> 6.7), and CHEMBL325 reaches 12.6 at
alpha 1, matching the 10-12 of real molecules at the extremes. Still small for CHEMBL2039
(2.9-4.0) and CHEMBL4078 (4.2-4.9) against a real-reference 7-12.

**Dissociation worth noting:** dev *loss* bottomed around round 60 (0.266) and rose to 0.291 by
round 100, while steering error kept falling (0.216 -> 0.185). LM-likelihood overfitting and
control quality come apart, so dev loss is the wrong early-stopping signal for this task.

**What this means for the paper.** As constructed, **ChEMBL cannot test C1** -- there is nothing
for collaboration to contribute. Options: (a) run the controlled truncation experiment (remove
part of each client's alpha range and test outside it), which is the only way to create a real gap
here and is already in the plan; (b) keep ChEMBL purely as a C2/portability and generality task
and state plainly that C1 is tested on Newsroom; (c) drop it. Not yet decided.

## 40. Controlled coverage experiment on ChEMBL: complementary truncation of ALL clients (2026-10-04)

Entry 39 showed C1/G3 fails on ChEMBL because the clients have no coverage gaps. This experiment
manufactures them. **New script `scripts/truncate_chembl.py`, new data `data/chembl_deco_trunc`.**

**Why all 8 clients and not the 3-client subset in the plan.** The only real constraint is that
*someone must still hold the region the others are missing* -- truncating every client in the same
direction would leave no client with high-alpha data, the shared direction could never learn that
region, and the experiment would test nothing. That forbids truncating everyone the *same* way,
not truncating everyone. Complementary truncation satisfies the constraint and is strictly better:
8 clients with real gaps instead of 3, and it is literally the plan's motivating sentence ("each
client's data is skewed, and the skews differ"). The 3-client subset also leaves 5 donors with
unrealistically complete coverage.

**Design.** alpha is computed under the full (untruncated) equal-weight mixture of participant
CDFs -- the natural meaning of "the top 40% of the range". Clients below the pooled alpha median
keep only alpha <= 0.6; clients above keep only alpha >= 0.4. Dev and test are **untouched**, so
every client is tested in its own removed region against real molecules.

| client | side | alpha median | train full | kept | kept % | retained alpha |
|---|---|---|---|---|---|---|
| CHEMBL243 | lose_top | 0.27 | 2,912 | 2,068 | 71.0% | [0.00, 0.60] |
| CHEMBL204 | lose_top | 0.29 | 3,278 | 2,495 | 76.1% | [0.00, 0.60] |
| CHEMBL325 | lose_top | 0.40 | 3,188 | 1,979 | 62.1% | [0.00, 0.60] |
| CHEMBL2835 | lose_top | 0.45 | 2,215 | 1,459 | 65.9% | [0.00, 0.60] |
| CHEMBL4078 | lose_bottom | 0.55 | 2,376 | 1,689 | 71.1% | [0.40, 1.00] |
| CHEMBL2039 | lose_bottom | 0.56 | 1,396 | 1,040 | 74.5% | [0.40, 1.00] |
| CHEMBL240 | lose_bottom | 0.62 | 4,000 | 3,088 | 77.2% | [0.40, 1.00] |
| CHEMBL228 | lose_bottom | 0.64 | 2,359 | 1,848 | 78.3% | [0.40, 1.00] |

4 lose_top / 4 lose_bottom, overlap band [0.4, 0.6], union still covers [0, 1]. 15,666 participant
training rows (was 21,724).

**Launched:** **11152030 federated** / **11152031 local**, 100 rounds (entry 39: 30 is
undertrained), rotation 0, `data/chembl_deco_trunc`.

**Caveat to state when reporting:** truncation cuts each client's training data to 62-78%, so
absolute numbers are not comparable to the untruncated runs of entry 39. The valid comparison is
federated vs local *within* the truncated setting, and specifically the error in each client's
removed region.

**What the outcomes would mean.**
- Federated beats local in the removed regions -> C1 holds when a gap exists, and entry 39 is a
  property of ChEMBL's natural coverage, not of the method. ChEMBL then earns a place as a second
  task with this experiment as its C1 evidence.
- Federated still does not win -> the method's collaboration claim does not transfer off Newsroom
  even with gaps present, which is a substantive negative result and an argument for dropping
  ChEMBL rather than reporting it weakly.

## 41. Natural coverage skew replaces entry 40's hard cut (2026-10-04)

**Entry 40's truncation was artificial and the jobs were cancelled before starting**
(11152030/11152031). Three things were wrong with a hard `alpha <= 0.6` / `>= 0.4` cut:

1. **Wrong mechanism.** It censored at the molecule level. Real library skew comes from *which
   chemical programmes a company ran*, and a programme (a congeneric series sharing a scaffold)
   occupies a narrow property band. Nobody deletes all compounds above a logP threshold.
2. **Hard edges.** It left *exactly zero* mass beyond the cut. Newsroom's nypost.com support
   starts at 0.51, meaning it has *few* low-extractiveness summaries, not none.
3. **Too uniform -- this was the worst of the three.** It gave all 8 clients identical window
   widths, which is **more** artificial than the real data. Measured Newsroom supports: only
   **2 of 8 clients are strongly skewed** (theguardian.com 0.04-0.55 width 0.50, nypost.com
   0.51-0.95 width 0.44); the other six are broad (0.67-0.96). Forcing uniform skew would have
   made the ChEMBL setting less realistic than Newsroom, not more.

**New `scripts/skew_chembl.py`** keeps the one structural requirement -- some client must still
hold what another lacks, or transfer is untestable -- and otherwise reproduces the *kind* of skew
Newsroom has:
- **series-level selection:** whole scaffold series are kept or dropped,
  `p(keep) = (1 - beta_i) + beta_i * sigmoid(s_i (alpha_series - 0.5) / tau)`, tau = 0.12;
- **within-series thinning:** a weaker ramp (tau_mol = 0.22, strength 0.75 * beta_i) on molecules
  inside surviving series, because SAR exploration concentrates where the team was optimising.
  Series selection alone could not narrow a support below ~0.72, since a single ChEMBL series
  itself spans ~0.31 of the alpha scale;
- **heterogeneous strength:** beta_i from 0.95 down to 0.0, interleaved across the two sides so
  each side has one strongly skewed client. (A first pass put both strong betas on low-median
  clients, leaving nobody short of low alpha and half the transfer question untested.)

| client | beta | side | kept | support (5-95%) | width | tail mass | n in tail |
|---|---|---|---|---|---|---|---|
| CHEMBL243 | 0.95 | low | 54% | [0.01, 0.54] | 0.53 | 0.0121 | 19 |
| CHEMBL204 | 0.60 | low | 72% | [0.02, 0.78] | 0.75 | 0.0387 | 91 |
| CHEMBL2835 | 0.10 | low | 92% | [0.07, 0.91] | 0.83 | 0.1471 | 300 |
| CHEMBL325 | 0.30 | low | 80% | [0.14, 0.97] | 0.84 | 0.1745 | 444 |
| CHEMBL2039 | 0.25 | high | 88% | [0.16, 0.92] | 0.76 | 0.0679 | 83 |
| CHEMBL240 | 0.50 | high | 70% | [0.22, 0.97] | 0.75 | 0.0466 | 130 |
| CHEMBL228 | 0.90 | high | 56% | [0.37, 0.98] | 0.61 | 0.0038 | 5 |
| CHEMBL4078 | 0.00 | high | 100% | [0.01, 0.95] | 0.94 | 0.1351 | 321 |

Support widths **min 0.53, median 0.76, max 0.94** against Newsroom's **0.44 / 0.81 / 0.96**.
CHEMBL243 [0.01, 0.54] is almost exactly theguardian.com [0.04, 0.55]; CHEMBL228 [0.37, 0.98] is
the nypost.com analogue. **No tail is hard-censored** -- the two most skewed clients still hold 19
and 5 molecules in their disfavoured region. 16,207 participant train rows (was 21,724).

Dev and test remain untouched, so each client is still evaluated across the whole alpha grid
against real molecules, including in the region its training data barely covers.

**Launched: 11152110 federated / 11152111 local**, 100 rounds, rotation 0.

Caveat unchanged from entry 40: training data drops to 54-100% per client, so absolute numbers are
not comparable with entry 39's untruncated runs; the comparison is federated vs local *within*
this setting, read per client and split by support.

## 42. Natural skew v2: stronger high side, and middle-only clients (2026-10-04)

Entry 41's v1 (11152110/11152111) was cancelled before starting, on two objections.

**(1) The high side was under-skewed** -- [0.37, 0.98] against nypost.com's [0.51, 0.95].
**Cause, measured:** the clients' *natural* alpha medians only span 0.27-0.64, because the global
alpha scale is built from these same eight clients. CHEMBL228's natural 25th percentile is already
0.44. A propensity ramp centred at 0.5 therefore disfavours only half the range and cannot lift a
high-side 5th percentile to ~0.5. Fix: the ramp **centre is now a per-role parameter** (0.54 for
the high specialist, 0.46 for the low one) with a sharper tau (0.07) at beta 0.97.

**(2) Added middle-only clients -- a regime Newsroom does not contain.** Every Newsroom client is
one-sided (skewed low or high). A client holding only mid-range alpha must borrow **both** tails,
so the shared direction has to extrapolate in both directions for the same client. That is a
strictly harder test of C1 than one-sided skew, and realistic: a lead-optimisation group that
stays in the drug-like sweet spot. Propensity is a Gaussian bell on alpha rather than a sigmoid.
The percentile-based support definition handles these correctly (the support stays contiguous), and
`metrics.py` already splits out-of-support reach into below/above, so both tails are measured.
A *barbell* client (endpoints only, missing the middle) was considered and **rejected**: its
5th-95th percentile support would read as [0.05, 0.95] and hide the hole, so the current support
definition would mis-measure it. That would need a density-based support first.

Roles are assigned by natural alpha median: lowest -> low specialist, highest -> high specialist,
most central -> strong middle, then outward.

| client | role | kept | support (5-95%) | width | <0.25 | >0.75 |
|---|---|---|---|---|---|---|
| CHEMBL243 | low_specialist | 54% | [0.01, 0.41] | 0.40 | 0.830 | **0.010** |
| CHEMBL204 | untouched | 100% | [0.03, 0.91] | 0.88 | 0.461 | 0.145 |
| CHEMBL2039 | low_moderate | 51% | [0.12, 0.89] | 0.77 | 0.173 | 0.163 |
| CHEMBL2835 | middle_moderate | 67% | [0.13, 0.87] | 0.74 | 0.255 | 0.165 |
| CHEMBL240 | broad | 91% | [0.15, 0.96] | 0.81 | 0.121 | 0.347 |
| CHEMBL325 | high_moderate | 64% | [0.16, 0.98] | 0.82 | 0.149 | 0.364 |
| CHEMBL4078 | **middle_strong** | 48% | **[0.28, 0.81]** | 0.53 | **0.040** | **0.080** |
| CHEMBL228 | high_specialist | 55% | **[0.46, 0.98]** | 0.53 | **0.009** | 0.620 |

Widths **min 0.40, median 0.76, max 0.88** against Newsroom's **0.44 / 0.81 / 0.96** -- the
extreme is now slightly *more* skewed than Newsroom's, which was the point. Every edge stays soft:
the low specialist keeps ~16 molecules above alpha 0.75, the high specialist ~12 below 0.25, the
strong middle client ~45 below 0.25 and ~91 above 0.75. Clients covering each alpha: 0.1 -> 25%,
0.5 -> 88%, 0.9 -> 50%, so the ends are thin but genuinely held by someone.
15,151 participant train rows (was 21,724). ⚠️ CHEMBL2039 falls to 710 rows, the smallest client.

**Launched: 11152198 federated / 11152199 local**, 100 rounds, rotation 0.

**What to read first when they finish:** per-client error in each client's own missing region --
the low specialist above 0.75, the high specialist below 0.25, and **both** tails for
CHEMBL4078. If federation wins anywhere, it should win there.

## 40. Math CoT with Qwen2.5-7B-Instruct: gates + C1 launched on DeltaAI (2026-10-04) [dtai]

Launch file `exp_log/launch/exp40_math_qwen25_7b.sh`. All jobs run on dtai-1 (`ghx4`, env `rein-gh`); outputs go to the shared `runs/` (`/work/nvme/bhby/lucmon/rein`).
- 100 rounds × 20 steps × batch 8 (4 epochs).
- Dev evals every 20 rounds: 50 problems for the single-client runs, 20 per client for E1. Generation batch 64; test evals at 100 problems with the 1,280 cap.
- `sbatch/train_eval_math.sbatch` gained `EVAL_BATCH`.
- The smoke test guards the first training on dtai (torch 2.13, aarch64); every real run is `afterok` on it.

| job | purpose | overrides (beyond `model_name=Qwen/Qwen2.5-7B-Instruct`) | run dir prefix | status |
|---|---|---|---|---|
| 3308443 | smoke: 2 clients, 2 rounds × 5 steps, 4-prompt evals, 128-token cap | `clients=[math,olympiads] fed.rounds=2 fed.local_steps=5 …` | `runs/exp40_smoke_qwen25_7b` | PENDING (Priority) |
| 3308444 | G0: plain SFT on `math`, D = 0 | `clients=[math] fed.lr_shared=0 fed.rounds=100 …` | `runs/exp40_g0_sft_qwen25_7b` | PENDING (afterok smoke) |
| 3308445 | G2: single-client steering on `math` | `clients=[math] fed.rounds=100 …` | `runs/exp40_g2_single_qwen25_7b` | PENDING (afterok smoke) |
| 3308446 | G1: B1 prompting on G2's checkpoint | `eval_b1_math.sbatch RUN_JOB=3308445` | (into G2's run) | PENDING (afterok G2) |
| 3308447 | **E1 federated**, rotation 0 (8 participants), shared calibration, no offset | `fed.rounds=100 monitor.full_prompts=20 …`, 2-day limit | `runs/exp40_e1_fed_qwen25_7b` | PENDING (afterok smoke) |
| 3308448 | **E1 local-only** (B2) | `fed.mode=local fed.calibration=private`, otherwise identical | `runs/exp40_e1_local_qwen25_7b` | PENDING (afterok smoke) |

**Entry 40 update (math, 10-04):**
- The `ghx4` queue filled during the day: 165 running and 2,768 pending. The smoke test's estimated start became 2026-10-05 13:57.
- The `afterok:smoke` dependency of G0 / G2 / E1-fed / E1-local (3308444/45/47/48) was removed with `scontrol update Dependency=`, so they queue in parallel with the smoke test instead of after it. If the smoke test fails, cancel them before they start. G1 (3308446) stays `afterok` on G2.
- Delta (2,435 pending) and cc (182 pending) were no better.

**Log hygiene note.** A parallel session (ChEMBL) appends to this file with overlapping entry numbers. The math entries are 37, 38, 39 and 40, each titled "Math CoT …" (and tagged [delta] or [dtai]). From now on the math entries are only *appended* (no read-modify-write), so as not to overwrite the other session's appends.

## 40. Results: calibration design, question 1 (learned vs. constant gain) (2026-10-04)

Jobs from entry 38: const 11145307 and linear 11145308 done (10-04 ~12:50); both selected round 100. Report: `exp_log/reports/gain_design_const_linear.txt`. The private arm (11145309, question 2) is still running.

**Test** (200 articles per client):

| Arm | g(α) | Pct error (worst) | In-support | Out-of-support (worst) | Reach | Spearman | Near-tie |
|---|---|---|---|---|---|---|---|
| Method | s·h(α), learned (s = 1.93) | **0.151** (**0.199**) | 0.135 | **0.161** (**0.188**) | 0.39 | **0.933** | 0.20 |
| Const | α | 0.153 (0.217) | **0.134** | 0.165 (0.207) | **0.40** | 0.927 | 0.20 |
| Linear | s·α, learned (s = 1.94) | 0.153 (0.208) | 0.136 | 0.164 (0.200) | 0.39 | 0.931 | 0.22 |

**Per client** (paired bootstrap):
- Const − method: better on 1/8 (people.com −0.008), worse on 1/8 (reuters.com +0.018); the rest tie.
- Linear − method: better on 1/8, worse on 3/8 (cbc.ca, reuters.com, nypost.com; +0.005 to +0.011).
- Quality (AlignScore, BERTScore, length, judge) is the same across the three.

**Reading:**
- The learned calibration adds little: mean error 0.151 vs. 0.153.
- The visible benefit is on the copy-heavy clients and the worst case: reuters.com 0.199 vs. 0.217 for const; worst client 0.199 vs. 0.217.
- **Why so small:**
  - The method's learned warp is essentially the identity (h(0.25) = 0.251, h(0.5) = 0.501, h(0.75) = 0.751), so its g is linear, s·α with s = 1.93, the same as the linear arm learns (1.94).
  - With s fixed at 1, the direction D grows instead (D → cD is equivalent to s → s·c).
  - Under the global percentile scale the response is already close to linear in α, so a non-linear map has little to correct.
- **Caveat:** the warp is pulled toward the identity by a small penalty (`fed.warp_reg = 0.01`). An unpenalized warp might deviate more; not tested.
- **Implication for the paper:** the calibration can be presented as a learned scale with an optional warp. The essential parts of the design are the global α scale and the shared direction, not the non-linearity. A1 shows g(α) = α nearly matches on average and is somewhat worse on the copy-heavy clients and the worst client.

## 41. Results: calibration design, question 2 (shared vs. per-client) (2026-10-04)

Job 11145309 (private calibration, no offset) done 10-04 14:50; selected round 100. Compared with the method (shared, no offset, 11063910) and the earlier with-offset pair (shared 11063906, private 11041282). Report: `exp_log/reports/gain_design_shared_private.txt`.

**Test:**

| Calibration | Offset | Pct error (worst) | In-support | Out-of-support (worst) | Reach | Spearman | Endpoint near-tie |
|---|---|---|---|---|---|---|---|
| **Shared** (method) | no | **0.151 (0.199)** | 0.135 | **0.161 (0.188)** | 0.39 | **0.933** | 0.002 |
| Per-client | no | 0.168 (0.320) | 0.135 | 0.183 (0.365) | 0.37 | 0.915 | 0.046 |
| Shared | yes | 0.158 (0.229) | 0.141 | 0.167 (0.204) | 0.38 | 0.920 | 0.002 |
| Per-client | yes | 0.155 (0.229) | 0.134 | 0.165 (0.253) | 0.40 | 0.926 | 0.009 |

**Per client, per-client − shared (no offset):**
- Better on 3/8, by small amounts: theguardian.com −0.015, people.com −0.009, wsj.com −0.006.
- **Worse on 2/8:** reuters.com +0.026 and **nypost.com +0.143**; nypost.com's out-of-support error goes from 0.188 to 0.365.
- With an offset the two are close: per-client better on 4/8, worse on nypost.com (+0.042).

**Mechanism (nypost.com):**
- Its data covers only α ∈ [0.51, 0.95], so a per-client calibration is fitted only there. It learned a small gain (s = 0.68) and a strongly bent warp (h(0.5) = 0.28).
- Without an offset, g(0) = 0, and the adapter alone already produces copied leads. So the low α it never saw are unreachable.
- Output percentile at α = 0, .25, .5, .75, 1: per-client 0.64, 0.70, 0.76, 0.85, 0.90 (34% of articles give near-identical text at α = 0 and 1). Shared 0.18, 0.41, 0.71, 0.84, 0.90.
- reuters.com (also copy-heavy) learned s = 0.86 and starts at 0.29 instead of 0.22.
- An offset partly repairs this (o = −0.58 for nypost.com, entry 16), hence the closer with-offset pair.
- Clients with broad support learn near-linear curves (s 1.5–2.0, h(0.5) ≈ 0.45–0.56), close to the shared one.

**Answer to question 2: share it.**
- A per-client calibration can only be fitted where the client has data. It extrapolates badly for exactly the skewed clients federation is meant to help, and it brings at most small in-support gains for broad clients.
- Shared calibration has the best mean and worst client, and needs no per-client fitting for new clients (E2, entry 30: same curve as private).
- Together with question 1 (entry 40): **one shared, learned scale** is the robust choice. The warp is optional (it stays near the identity). No offset is needed once the calibration is shared.

## INFRA-ANVIL-1. Purdue Anvil surveyed and set up for the molecule workstream (2026-10-04)

Relayed from the user by another session: Anvil is available when cc (dali / IllinoisComputes-GPU)
or Delta are backed up. Namespaced `INFRA-ANVIL-` because it is shared infrastructure, not one
workstream's experiment (see the namespace convention: `NR-`, `MATH-`, `MOL-`).

**Cost, measured -- the number everyone needs before spending this.**
`scontrol show partition ai|gpu` gives `TRESBillingWeights=GRES/gpu=1.0`, i.e. **1 SU per
GPU-hour**, so the 500 SU allocation is **500 GPU-hours**. `mybalance`: 500.0 limit, 0.0 used.
A 100-round ChEMBL fed/local pair is ~12 GPU-hours = **~12 SU, 2.4% of the allocation**. Cheap.

**Capacity -- the reason to bother.** `sinfo -p ai,gpu`: 17 of ~21 `ai` nodes and 10 of 16 `gpu`
nodes were in `mix-` (partly free) while this session's cc jobs sat on `(Priority)` for hours.
4 GPUs per node, `--gpus-per-node=1`, 48 h max, `gpu-debug` for 30-minute tests.

⚠️ **Correction to the relayed note:** it said /home is 85% full. `myquota` for x-zchen17 reports
**home 34.4 KB of 25 GB (0.0%)**, scratch 0 of 100 TB, projects 0 of 5 TB. The 85% figure is
presumably the whole filesystem, not this account's quota. Space is not a constraint; even so the
env, HF cache and run outputs all go to scratch rather than home.

**State before this entry:** nothing existed -- empty scratch, no env, no code, no data, no models,
no jobs queued on `cis260796-ai`. Nobody else had claimed it.

**New `scripts/setup_anvil.sh`** (deliberately a separate script, *not* an edit to
`scripts/sync_to_delta.sh`, which the math-CoT workstream owns and is actively using).
Layout, all under `lucmon/` as instructed since the account is x-zchen17's, not the user's:

| path | contents |
|---|---|
| `/anvil/scratch/x-zchen17/lucmon/rein` | code (rsync from cc, `--delete`; cc stays the only place code is edited) |
| `/anvil/scratch/x-zchen17/lucmon/data/{chembl_deco,chembl_deco_skew}` | datasets, symlinked into `rein/data/` |
| `/anvil/scratch/x-zchen17/lucmon/runs` | run outputs, symlinked to `rein/runs` |
| `/anvil/scratch/x-zchen17/lucmon/envs/rein` | conda env (`module load conda`; no pytorch module exists on Anvil, so torch comes from pip) |
| `/anvil/scratch/x-zchen17/lucmon/hf_home` | `HF_HOME`; Qwen3-4B-Instruct-2507 (7.6 GB) |

✅ Code and both ChEMBL datasets synced (45 MB). 🔄 Conda env + model download running
(`scripts/setup_anvil.sh`, ~2.5 GB of wheels plus 7.6 GB of weights). Setup costs 0 SU -- it is
all login-node work.

⏳ **Still to write: an Anvil sbatch.** `sbatch/train_eval_chembl.sbatch` is cc-specific
(`--account=lucmon-ic`, `--partition=dali,IllinoisComputes-GPU`, `--gres=gpu:1`,
`source activate steer`, cc paths). The Anvil version needs `--account=cis260796-ai`,
`-p ai`, `--gpus-per-node=1`, `module load conda && source activate $ROOT/envs/rein`,
`HF_HOME=$ROOT/hf_home`.

**Not duplicating work:** the cc pair 11152198 / 11152199 (natural-skew fed/local, entry 42) stays
queued. Once Anvil is ready, whichever is still pending gets submitted there and the cc copy
cancelled -- running both would waste either cc GPU time or SU.

## MATH-1. dtai training path verified; smoke-test bug; Anvil option assessed (2026-10-04) [dtai]

First entry under the `MATH-` namespace (see the log convention). Continues entry "40. Math CoT with Qwen2.5-7B-Instruct … [dtai]".

**The first smoke test (3308443) FAILED in 39 s, from my bug, not dtai.**
- Its override `clients=[math,olympiads]` contains a comma, and `sbatch --export` splits on commas, so the YAML override was truncated (`ParserError: while parsing a flow sequence`).
- The real jobs (G0 3308444, G2 3308445, E1 3308447/48) have no comma in their overrides. Their `afterok` dependency on it had already been removed, so they were unaffected.
- **Rule:** no commas inside `OVERRIDES` passed through `--export`.

**Smoke test, second attempt (3309218): training WORKS on dtai** (torch 2.13 / aarch64 / GH200, env `rein-gh`).
- `--partition=test` was refused for account `bhby-dtai-gh`. **`--partition=ghx4-interactive` (2 h limit) accepted it and started immediately** (gh058), while `ghx4` estimated the next day.
- Config: the 8 rotation-0 clients, 1 round × 2 local steps, 4-prompt dev and test evals, 128-token cap; run `runs/exp40_smoke2_qwen25_7b_20261004-161059_j3309218`.
- Round 0 logged: mean loss 0.514, delta cos 0.218 (FedAvg over 8 clients), **36.2 s for 16 steps = 2.3 s/step (7B, batch 8)**, dev loss 0.496. The full dev eval and `score_math` ran. Peak GPU memory ≈ 40 of 98 GB.
- Implied E1 cost: 100 rounds × 8 clients × 20 steps × 2.3 s ≈ **10 h of training per run**, plus evals.

**Queue (dtai `ghx4`):** G0/G2/E1-fed/E1-local are PENDING with an estimated start of **2026-10-05 17:37**; G1 waits on G2.

**Anvil (shared infrastructure, see INFRA-ANVIL-1):** 1 SU per GPU-hour, 500 SU shared by all workstreams; capacity was free on 10-04. The math suite (G0 + G2 + G1 + the E1 pair) would cost roughly **50–60 SU (10–12% of the allocation)**. It would also need math-verify and Qwen2.5-7B (15 GB) added to the molecule session's shared env and cache. **Not used yet: pending a decision by the user**, since it is a borrowed, shared allocation. If used, move the whole suite, so that all math comparisons stay on one host and one torch version.

## INFRA-ANVIL-2. Anvil usable (ai only), plus three unused GPU partitions found on cc (2026-10-04)

Follows INFRA-ANVIL-1. All of this is now in the shared memory `reference-slurm.md` and a new
`feedback-cluster-failover.md`, and was messaged to the news-summarization, math-CoT and SLURM
sessions.

**Anvil: set up, with four corrections to the relayed note.**
1. **`-A cis260796-ai` can use `-p ai` ONLY** (`sacctmgr show assoc` -> `Partition=ai`). `-p gpu`
   and `-p gpu-debug` are rejected: *"You are submitting a job to a non-AI partition while using
   an AI allocation."* So there is no 30-minute debug queue for this allocation.
2. **Multi-partition is impossible there:** `-p ai,gpu` -> *"Multiple partition job request not
   supported when a partition is set in the association."*
3. **`/home` is not 85% full:** `myquota` gives home 34.4 KB / 25 GB (0.0%), scratch 0 / 100 TB,
   projects 0 / 5 TB. The 85% was presumably the whole filesystem.
4. **1 SU per GPU-hour** (`TRESBillingWeights=GRES/gpu=1.0`), so 500 SU = **500 GPU-hours** --
   generous, not tight. A 100-round fed/local pair is ~12 SU. Setup cost 0 SU.
Also: `wholenode`, `shared`, `standard`, `wide`, `highmem`, `debug`, `profiling` are **CPU-only**
(null GRES), so they cannot run these jobs regardless of permissions. Only `ai` (21 nodes), `gpu`
(16) and `gpu-debug` (16) carry `gpu:4`.

**Layout mirrors cc** (code in home, bulk on scratch via symlinks), per the user on 10-04:
code `/home/x-zchen17/lucmon/rein` (1.3 MB); `data/<n>` -> `$SCRATCH/lucmon/data/<n>`, `runs` ->
`$SCRATCH/lucmon/runs`, env `$SCRATCH/lucmon/envs/rein` (18 GB), `HF_HOME=$SCRATCH/lucmon/hf_home`
(7.6 GB, Qwen3-4B-Instruct-2507). Env **pinned to cc's `steer`**: torch 2.5.1+cu124, transformers
4.56.0, numpy 1.26.3, pyarrow 19.0.0 -- an unpinned install had given torch 2.14.1 / transformers
5.18.0, a major-version jump that risks the hand-rolled LoRA wrapping and would make Anvil numbers
non-comparable with the cc runs beside them. No pytorch module exists on Anvil, so torch is pip.
Scripts: `scripts/setup_anvil.sh`, `sbatch/train_eval_chembl_anvil.sbatch`. `sync_to_delta.sh` was
deliberately left alone (the math-CoT workstream owns it).

**cc: three GPU partitions were going unused.** `lucmon`'s associations are *not* partition-pinned,
so multi-partition submission works. `dali,IllinoisComputes-GPU` is only 5 GPU nodes. Deciding
number: **peak GPU memory 18.0 GiB** for the 4B ChEMBL run (`GPU_LOG=1`).

| partition | AllowAccounts | MaxTime | verdict (~6 h, 4B) |
|---|---|---|---|
| `scavenger` | **ALL** | 1 d | ✅ **added**, 19 GPU nodes; exclude `ccc0089,ccc0090` (16 GB V100s) |
| `secondary` | ALL | 4 h | ✗ too short (run takes ~6 h, nothing auto-resumes); ok for smoke tests |
| `ic-express` | acc-illinoiscomputes | 8 h | ✗ H100 MIG `1g.20gb`: 20 GB vs 18.0 GiB peak, no headroom |
| `eng-research-gpu` | acc-eng-research | 2 d | ✗ lucmon-ic not in that account |

`sbatch/train_eval_chembl.sbatch` now uses
`--partition=dali,IllinoisComputes-GPU,scavenger --exclude=ccc0387,ccc0089,ccc0090`.
⚠️ **Gotcha:** in a multi-partition request `--time` must fit the *smallest* MaxTime in the set.
With `scavenger` the cap is 24 h, so `--time=2-00:00:00` fails with *"Requested time limit is
invalid"*. `exp_log/launch/exp42_chembl_skew_roles.sh` now uses `--time=24:00:00`.
`scavenger` is preemptible; a requeued job restarts from round 0 with a new stamp, so recovery is
via the 20-round snapshots plus `train_fed.py --resume`.

⚠️ **My error, and the rule that came out of it.** To widen the partition list I cancelled
11152198 / 11152199 -- which had *just been allocated* ccc0388 and ccc0284. I killed running work,
and the resubmit then failed on the time-limit rule above, leaving nothing running. User
instruction: **when racing a job across clusters, cancel the other copies only once one is
actually RUNNING, never in order to resubmit with better settings.** Check
`squeue -j <id> -h -o "%T %M %R"` before any `scancel`. Resubmitted as **11156942 fed /
11156950 local** (24 h, wider pool). Anvil smoke test **21067708** is queued on `-p ai`.

## MATH-2. Qwen2.5-7B math suite raced on cc as well; automatic race watcher (2026-10-04) [cc+dtai]

**Why:** dtai's `ghx4` queue estimated a start of 2026-10-05 17:37 for the suite in entry "40. Math CoT with Qwen2.5-7B-Instruct [dtai]". The user's standing rule (memory `feedback-cluster-failover`): queueing is the main bottleneck, so submit to several clusters and cancel the other copies **only once one is RUNNING**. Anvil is not used yet: that awaits the user's answer on spending the shared SU (MATH-1).

**cc copies** (launch `exp_log/launch/MATH-2_qwen25_7b_cc_race.sh`; identical overrides to the dtai copies; partitions `dali,IllinoisComputes-GPU`, A100 80 GB / H200; env `steer`; run dirs under cc `runs/` → `/projects/illinois/eng/cs/arindamb/lucmon/rein/runs`):

| race | cc job | dtai job | note |
|---|---|---|---|
| G0 (plain SFT, `math`) | 11157615 | 3308444 | |
| G2 (steering, `math`) | 11157616 | 3308445 | |
| G1 (B1 prompting, afterok G2 on the same host) | 11157617 | 3308446 | cancelled together with its host's G2 copy |
| E1 federated | 11157618 (3-day limit) | 3308447 | group "E1": fed and local are kept on **one** host |
| E1 local | 11157619 (3-day limit) | 3308448 | group "E1" |

Partitions not used, per the molecule session's survey: `scavenger` and `secondary` (24 h / 4 h caps are too short for E1) and `ic-express` (20 GB MIG slices).

**math-verify on cc without touching the shared `steer` env:** `pip install --no-deps --target /u/lucmon/lucmon/pylib/mathverify math-verify==0.9.0 latex2sympy2_extended==1.11.0 antlr4-python3-runtime==4.13.2`.
- It uses the env's sympy 1.13.1; verified `verify(1/2, 0.5) = True`, `verify(3, 4) = False`.
- The math sbatch scripts (`train_eval_math`, `eval_b1_math`, `eval_base_math`, `backbone_screen`) add that directory to `PYTHONPATH` only if it exists. On Delta and dtai the envs already have math-verify.

**`scripts/race_watch.py` (new):** runs detached on cc-login1 (`nohup setsid`, pid 1269842, 5-minute polls, 96 h max); spec `exp_log/launch/MATH-2_race.json`; log **`sbatch/logs/race_MATH-2.log`**.
- A copy is cancelled only when another copy of its race (or its group) has started, and only if a fresh check right before `scancel` still says PENDING.
- Dependents (G1) of a cancelled G2 copy are cancelled with it. A host that fails to answer is left alone.
- A dry run (cancellation disabled) read all 10 job states correctly and cancelled nothing while all were pending.
- ⚠️ It reaches dtai through the `ssh -fN dtai-1` master on cc-login1. If that dies, the watcher stops cancelling dtai copies (it logs that) and both copies could end up running.

⚠️ **Cross-host comparability:** cc runs torch 2.5.1 on A100s and dtai runs torch 2.13 on GH200s. The single-client gates may finish on different hosts (each gate is read on its own); the E1 pair cannot, by construction.

## MATH-3. Third copies of the gates on cc `scavenger`; race watcher made preemption-aware (2026-10-04) [cc]

Follows MATH-2, triggered by the cc `scavenger` partition reported in INFRA-ANVIL-2. Anvil is still not used: it awaits the user's answer on spending the shared SU (MATH-1). A peer session's view that 500 SU is generous is not the user's approval.
- **Only the single-client gates go to `scavenger`**: about 4–5 h each, within its 24 h cap. The E1 pair needs well over 24 h on an A100, so it stays on `dali,IllinoisComputes-GPU` and dtai.
- **Nodes:** H100 / H200 / L40S only, via `--exclude`:
  - ccc0089/90 (V100 16 GB);
  - ccc0232–0236 (Quadro RTX 6000, 24 GB, too small for the 7B);
  - ccc0496–0499 (RTX6000B, Blackwell, which needs CUDA ≥ 12.8; cc's torch is 2.5.1+cu118).
- **Jobs:**
  - 11157678 (G0, scavenger) and 11157682 (G2, scavenger): `--time=12:00:00`, same overrides as the other copies;
  - 11157683 (G1, afterok the scavenger G2, on the *regular* cc partitions).

  Launch: `SCAV=1 bash exp_log/launch/MATH-2_qwen25_7b_cc_race.sh`.
- **`scripts/race_watch.py`: preemption-aware.** Hosts in the spec's `"preemptible"` list (here `ccscav`) win only by **COMPLETING**, not by starting: a scavenger copy can be preempted and requeued from round 0, and treating its start as a win would have cancelled the safe copies. Their *pending* copies are cancelled like any loser once a non-preemptible copy starts. A running scavenger copy is left alone (worst case a duplicate run on free cc time).
- **Watcher restarted** with the 3-host spec (`exp_log/launch/MATH-2_race.json`): old pid 1269842 had taken no action and was stopped; new pid **1301173**; same log `sbatch/logs/race_MATH-2.log`.

## MOL-13. exp42 skew pair duplicated onto Anvil to race the cc copies (2026-10-04)

Correcting my own framing in INFRA-ANVIL-2: I described the cc skew pair and the Anvil *smoke test*
as a race whose loser should be cancelled. They were **different tasks**, so that was wrong --
the failover rule only applies to **duplicate copies of one experiment** on different
allocations. The math-CoT workstream had already got this right (`math2_g0_sft_cc` alongside
`math2_g0_sft_scav`).

So the exp42 natural-skew experiment now has two copies of each job:

| job | cc (`lucmon-ic`, dali/IC/scavenger, 24 h) | Anvil (`cis260796-ai`, `-p ai`, 20 h) |
|---|---|---|
| federated, 100 rounds | 11156942 | 21068552 |
| local, 100 rounds | 11156950 | 21068553 |

Launch script `exp_log/launch/exp43_skew_anvil_duplicate.sh`. Identical data
(`data/chembl_deco_skew`), identical overrides, and the Anvil env is pinned to cc's versions
(torch 2.5.1+cu124, transformers 4.56.0), so whichever wins is comparable with the entry MOL-9
untruncated runs.

**The Anvil smoke test 21067708 was cancelled** rather than left to run: it had not started, and
it competed with these real jobs for the same `ai` allocation. The real run's own first rounds
validate the stack just as well, and a stack failure surfaces in minutes at ~0 SU.

**Cancellation policy for this race:** cancel the loser once the winner is `RUNNING` *and past
startup* (model loaded, round 1 in the log) -- not merely `RUNNING`, because the Anvil environment
is newly built and could still fail early. Budget impact if Anvil wins: ~12 SU of 500.

## MATH-4. Cluster rule: math-CoT on Delta + dtai only; cc copies withdrawn, Delta copies race dtai (2026-10-04) [delta+dtai]

**User rule (relayed by another session, recorded at the top of memory `reference-slurm.md`):** math-CoT → Delta and DeltaAI only; news-summarization and molecule-generation → cc and Anvil only. This supersedes the cc parts of MATH-2 / MATH-3; Anvil was never used for math.

**cc withdrawn.**
- The watcher (pid 1301173) was stopped; it had taken no action.
- All 8 cc math jobs were cancelled after a per-job check that each was still PENDING (none had started): 11157615/16/17/18/19 (regular partitions) and 11157678/82/83 (scavenger).
- They would otherwise have competed with the other workstreams' jobs on cc. No math job remains on cc.
- `/u/lucmon/lucmon/pylib/mathverify` stays on disk; it is unused unless math runs on cc again.

**Delta copies** (launch `exp_log/launch/MATH-4_qwen25_7b_delta_race.sh`; same overrides; `--partition=gpuA100x8,gpuH200x8`, 80 GB+ only, since `gpuA100x4` is 40 GB, about the 7B's ~40 GB peak measured on dtai; env `rein` torch 2.5.1+cu124; shared `/work` model cache and data):

| race | Delta | dtai |
|---|---|---|
| G0 | 22665955 | 3308444 |
| G2 | 22665956 | 3308445 |
| G1 (afterok G2, same host) | 22665957 | 3308446 |
| E1 federated (2-day limit) | 22665958 | 3308447 |
| E1 local (2-day limit) | 22665959 | 3308448 |

**Watcher:** new pid **1478227**, spec `exp_log/launch/MATH-4_race.json`, log **`sbatch/logs/race_MATH-4.log`**. Same rules: cancel only once a copy RUNS, with a fresh PENDING check before each `scancel`; the E1 pair stays on one host; G1 follows its host's G2. It needs both SSH masters on cc-login1 (`delta-login2`, `dtai-1`).

⚠️ Delta (torch 2.5.1, A100/H200) and dtai (torch 2.13, GH200) differ, so a gate may finish on either host; the E1 pair cannot split.

## MATH-5. Every Qwen2.5-7B run went NaN (SDPA backward on long batches); fixed with eager attention + NaN guard; suite resubmitted (2026-10-05) [dtai+delta]

**What happened to the MATH-4 race.**
- All four dtai copies started on 10-04 between 22:42 and 22:47, far earlier than the estimated 10-05 17:37: G0 3308444 (gh128), G2 3308445 (gh003), E1-fed 3308447 (gh086), E1-local 3308448 (gh128).
- `race_watch.py` cancelled the five Delta copies (22665955–59) correctly, each confirmed PENDING, then exited.

**But every run went NaN** (analysis on 10-04 at 23:58, about 75 min in):

| run | first NaN | evidence |
|---|---|---|
| G0 (`math`, D = 0) | after round 4 | train loss 0.39 → 0.33 → 0.31 → 0.33 → 0.33 (rounds 0–4); dev loss NaN from round 4, train loss NaN from round 5 |
| G2 (`math`) | round 4 | `direction_norm` 3.1 → 5.8 → 7.6 → 9.0 → NaN, the same step as G0 |
| E1-fed | round 0, client **olympiads** | other clients finite (aops_forum 0.319, cn_k12/Geometry 0.470, gsm8k 0.333, math 0.376, …); the server average turned the shared direction NaN, so every client was NaN from round 1 |
| E1-local | olympiads in round 0, aops_forum in round 2, math in round 3 | the other clients stayed finite (orca_math/Algebra 0.30 → 0.14) |

- Dev evals of all four (G0 / G2 at rounds 20 and 40): pct err 0.484, Spearman 0, i.e. degenerate output.
- **Deterministic per batch:** the same client failed at the same step in fed and local, and G0 and G2 failed together. The longest-solution clients failed first.
- All four were cancelled (states confirmed RUNNING; work unrecoverable, since the first snapshot, at round 10, came after the corruption), together with G1 3308446 (PENDING).

**Diagnosis** (dtai `ghx4-interactive`, 1 job per user; `clients=[olympiads]`, 2 rounds × 20 steps, same seed and batches):
- **`fedsteer/fed.py`: new NaN guard.** A step whose loss or gradient norm is non-finite is skipped (no optimizer update; `clip_grad_norm_` would otherwise turn an inf norm into NaN weights) and reported: sequence lengths, target tokens, α, and the parameters with non-finite gradients. The count goes into the round log (`skipped_steps`).
  - Default on: it changes behaviour only in the case that used to corrupt the weights.
  - Shared code (other workstreams use `fed.py` too). The normal path was unchanged in a cc CPU smoke run (Llama-1B).
- **3311333, baseline (SDPA):**
  - `[nan-guard] round 0 client olympiads step 5: non-finite grad; seq lens [929, 1088, 576, 778, 747, 710, 698, 531], target tokens [707, 972, …]; 588 params with non-finite grads` (every LoRA matrix down to layer 0).
  - **The forward loss was finite; the backward pass produced the non-finite gradients**, on the batch with the longest sequence so far (1,088 tokens).
  - The guard skipped it and training continued (dev loss 0.632 → 0.630).
- **3311345, `attn_implementation=eager`:** the **same 40 steps, including that batch: no non-finite value**. Dev loss 0.648 → 0.637; 33.8 vs. 30.8 s per round (~10% slower).
- **Conclusion:** SDPA's backward on long batches with this torch (2.13, aarch64 / GH200) gives non-finite gradients. The Qwen3-4B runs on Delta (torch 2.5.1, SDPA, 60 rounds) never showed it.
- The guard alone would not be an acceptable fix: it would systematically drop the longest batches, which hold the high-α examples the knob must learn.
- The third planned variant (no gradient checkpointing) was not run: the per-user submit limit, and eager already isolates the cause.

**Fix:** `configs/math_fedavg.yaml` now sets `attn_implementation: eager` (math-only config; comment cites this entry). The NaN guard stays as a safety net; any `skipped_steps > 0` must be reported with results.

**Resubmitted** (launch `exp_log/launch/MATH-5_qwen25_7b_eager.sh`; same overrides as entry 40; new run prefixes `runs/math5_*`, so the NaN runs (`exp40_*`) cannot be confused with these):

| race | dtai | Delta (`gpuA100x8,gpuH200x8`) |
|---|---|---|
| G0 | 3311353 | 22671278 |
| G2 | 3311354 | 22671279 |
| G1 (afterok G2, same host) | 3311355 | 22671280 |
| E1 federated (2-day limit) | 3311356 | 22671281 |
| E1 local (2-day limit) | 3311357 | 22671282 |

**`scripts/race_watch.py` changes:**
- `--min_running_s` (here 1200): a copy wins only after RUNNING for 20 min (past model load and round 1), per the updated memory rule that a fresh setup can crash in its first minutes.
- **Latent bug fixed:** a copy that FAILED early used to count as "started" and could have cancelled its healthy twin; now only RUNNING (past the minimum) or COMPLETED wins.
- Watcher pid **3398675**, spec `exp_log/launch/MATH-5_race.json`, log **`sbatch/logs/race_MATH-5.log`**.

## MOL-14. Skewed ChEMBL: fed still not better than local — but the comparison was confounded (2026-10-05)

Both Anvil copies of the natural-skew pair finished (21068552 fed 3 h 50 m, 21068553 local 3 h 41 m;
the cc duplicates 11156942/11156950 were cancelled once these were running and past startup).
Dev-selected: fed round 80, local round 100. Evals copied to `runs/anvil/` on cc.

**Test results (150 prompts/client, constant baseline 0.300):**

| run | err | in-sup | out-sup | rho | reach | range | ties |
|---|---|---|---|---|---|---|---|
| FED (r80) | 0.198 | 0.169 | 0.224 | 0.870 | 0.303 | 0.545 | 0.370 |
| LOCAL (r100) | **0.187** | **0.161** | **0.211** | 0.863 | **0.353** | **0.572** | 0.356 |

Paired bootstrap: **federated significantly better on 0/8 clients, significantly worse on 2/8**
(CHEMBL4078 +0.040*, CHEMBL2039 +0.014*); the other six are ties. So manufacturing coverage gaps
did **not** flip MOL-9's negative result.

**Two hypotheses tested and discarded.**
- *"The direction is non-unique, so sharing is a pure constraint."* The 8 independently learned
  local directions do end up nearly orthogonal (`client_direction_cos_mean` 0.599 -> 0.095), but
  **Newsroom is worse on this measure** (0.740 -> 0.025, 0.570 -> 0.021 across runs) and federation
  *helps* there. Discarded.
- *"Client chemistry is too distinctive."* Per-client fed-minus-local error does not track
  decoration-vocabulary distinctiveness: CHEMBL240 (40.8% unique types) is the one client where
  federation wins, while CHEMBL2039 and CHEMBL228 (33% and 31.9%, the least distinctive) are among
  the worst hit. Discarded.

**⚠️ The actual cause is a confound in my own design.** The federated run used
`fed.calibration=shared`, which pins **every client to gain 1.053**; the local run necessarily uses
private calibration, and every client chose a **much higher gain, 1.251-1.670 (1.19-1.59x the
shared value)**. A larger gain means a larger coefficient on the direction, hence more attribute
swing — which is exactly the observed pattern (local reaches further, 0.353 vs 0.303, and spans a
wider range, 0.572 vs 0.545). So the run compared *{shared direction + shared calibration}* against
*{local direction + private calibration}*: two factors at once, and the calibration factor alone
accounts for the result.

| client | role | local gain | x shared | FED-LOC err | out-of-sup | n_train |
|---|---|---|---|---|---|---|
| CHEMBL2039 | low_moderate | 1.670 | 1.59 | +0.016* | +0.010 | 710 |
| CHEMBL4078 | **middle_strong** | 1.592 | 1.51 | **+0.040*** | **+0.065** | 1,135 |
| CHEMBL228 | high_specialist | 1.504 | 1.43 | +0.016 | +0.015 | 1,303 |
| CHEMBL2835 | middle_moderate | 1.462 | 1.39 | +0.002 | +0016 | 1,495 |
| CHEMBL204 | untouched | 1.392 | 1.32 | +0.003 | +0.009 | 3,278 |
| CHEMBL240 | broad | 1.385 | 1.32 | **-0.004** | **-0.010** | 3,642 |
| CHEMBL243 | low_specialist | 1.292 | 1.23 | +0.007 | +0.001 | 1,560 |
| CHEMBL325 | high_moderate | 1.251 | 1.19 | +0.002 | -0.003 | 2,028 |

- **Spearman(local gain, FED-LOC out-of-support error) = +0.76**; overall error +0.67. The clients
  federation hurts are exactly those whose locally optimal gain is furthest above the shared one.
- The worst-hit client is **CHEMBL4078, the middle-only client** — the one case where coverage
  transfer should matter most, because it lacks *both* tails. It needs the steepest mapping to
  reach [0,1] from training data spanning [0.28, 0.82], locally picks gain 1.592, and shared
  calibration forces it to 1.053, so it under-reaches both ends. Mechanism and worst case agree.
- Spearman(n_train, FED-LOC error) = **-0.81**, i.e. federation hurts the *smallest* clients most,
  the opposite of the expected benefit — but n_train and the needed gain are strongly collinear
  here (the heavily skewed clients are also the ones that lost the most data), so with 8 clients
  these two explanations cannot be separated. The gain story covers both.

**Consequence: MOL-9 and MOL-14 do not yet test C1.** They test shared-direction-plus-shared-
calibration against local-direction-plus-private-calibration. Newsroom's entry 20 found shared
calibration best *there*, but Newsroom clients have broad supports of similar width, whereas after
skewing ChEMBL's support widths range 0.40-0.94, so clients genuinely need different gains. Working
hypothesis: **shared calibration is safe when support widths are similar and harmful when they
differ** — worth stating in the paper either way.

**Launched MOL-15: the matched comparison**, `fed.calibration=private` with everything else
unchanged, so only the direction varies: **Anvil 21092869** and **cc 11165980** (duplicates; cancel
the loser once one is RUNNING). `fed.calibration=private` is the only legal matched setting, since
`fed.py` rejects local+shared (nothing is aggregated in local mode), so a full 2x2 is unavailable.
