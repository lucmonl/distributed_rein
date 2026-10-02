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
