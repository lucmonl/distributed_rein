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
| 11060306 | 09-30 | exp11 cap 4k, regularized (resubmit) | `runs/exp11_cap4k_reg_20260930-231707_j11060306` | Running (started 09-30 23:17) |
| 11060307 | 09-30 | exp11 natural sizes, regularized (resubmit) | `runs/exp11_natural_reg_20260930-232106_j11060307` | Running (started 09-30 23:21) |
| 11061046 | 09-30 | Local-only at 4k (entry 16), fair counterpart of 11041282 | `runs/exp16_local_cap4k_20260930-232203_j11061046` | Running (started 09-30 23:22); also the local arm of exp17-B |
| 11063906 | 10-01 | exp17 B: federated, shared calibration, 4k (entry 17) | `runs/exp17_fed_calshared_cap4k_<stamp>_j11063906` | Queued |
| 11063910 | 10-01 | exp17 B, no offset: federated, shared calibration, 4k | `runs/exp17_fed_calshared_nooff_cap4k_<stamp>_j11063910` | Queued |
| 11063911 | 10-01 | exp17 local, no offset, 4k | `runs/exp17_local_nooff_cap4k_<stamp>_j11063911` | Queued |

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
