# Federated steering on math CoT: a solution-length knob (NuminaMath-CoT)

Experiment plan · drafted 2 October 2026 · rewritten 3 October for NuminaMath-CoT · **updated 5 October 2026** (backbone Qwen2.5-7B, eager attention, Delta/DeltaAI only, **answer-in-prompt as the main setting**; changelog at the end).
Companion to `federated-steering-plan.md`: method, metrics, baselines and claims C1/C2 are reused unless stated. Sister task: `chembl-experiment-plan.md`.
Evidence: `exp_log/EXPERIMENT_LOG.md`. The math entries are the "Math CoT …" / "Backbone screen …" ones numbered 36–40, then **MATH-1 … MATH-7** (namespaced, because several sessions write to the log).
Data: `data/math_fed/` (original prompts) and `data/math_fed_ans/` (answer in the prompt), built by `scripts/build_math_fed.py` and `scripts/make_answer_prompt_data.py`.
**Hosts (user rule, 10-04): math-CoT runs on Delta and DeltaAI (dtai) only**; cc and Anvil belong to the news and molecule workstreams. Code is edited on cc-login1 only and synced with `scripts/sync_to_delta.sh` (Delta) or `scripts/sync_to_delta.sh --host dtai`. Jobs are raced on both hosts, and `scripts/race_watch.py` cancels the loser once a copy has run 20 minutes. Launch files are in `exp_log/launch/`.

Status markers: ✅ measured / done · 🔄 running · ⏳ planned · ⚠️ known risk or open problem.

---

## 1. Why this task

The claim is unchanged: **clients each fine-tune their own model, jointly learn one steering direction, and can then reach attribute levels their own data barely covers.** The attribute here is **how long the chain of thought is**, from a terse solution with only the key steps to a fully spelled-out one.

| Criterion (`dataset-candidates.md`) | How it scores (measured) |
|---|---|
| R1 continuous, fine-grained | ✅ token count, ≈ 50 → 990 tokens across the 12 clients |
| R2 cheap deterministic scorer | ✅✅ one tokenizer call; accuracy (Math-Verify) is deterministic too; **no judge anywhere** |
| R3 clients with a limited spectrum | ✅ **natural, no constructed windows.** olympiads [0.58, 0.99] and aops_forum [0.60, 0.99] of the global scale; gsm8k [0.12, 0.65] and the Logic & Puzzles banks [0.01, 0.68] (§3.2) |
| R4 not summarization | ✅ |
| R5 prompting cannot reach it | ⚠️ the weakest point. "Be brief" / "be detailed" is verbalizable. Gate G1 is on calibration error, not Spearman |
| R6 attribute free given the input | ✅ a TF-IDF ridge on the problem text explains only **0.34** of log-length variance within a client, so the α label carries signal for D |

**Framing.** These are clean, linear step-by-step solutions (0% contain "wait" / "alternatively", vs. 97–99% of R1 traces), so length means *more steps and more explanation*, not *more search*. The knob is called **CoT length control**.

**Focus (user, 10-04 / 10-05): the length of the CoT trajectory, not accuracy.** Accuracy is reported, never gated. In the main setting the gold answer is given in the prompt (§2.1), so length cannot be explained by "searching for the answer".

---

## 2. Task definition

### 2.1 Input and output: two settings

| setting | data | prompt ends with | role |
|---|---|---|---|
| **answer-in-prompt (main, 10-05)** | `data/math_fed_ans/` | "The final answer is $X$. Please reason step by step to reach it, and put your final answer within \boxed{}." | the study of CoT length on its own: "explain this known answer at length α" |
| original | `data/math_fed/` | "Please reason step by step, and put your final answer within \boxed{}." | reference: "solve this problem at length α" (MATH-5 / MATH-6 runs) |

- Both settings have the **same records, splits, targets and length labels**, so results pair problem by problem. Only the prompt differs.
- **y** = a GPT-4o step-by-step solution ending in `\boxed{answer}` (NuminaMath-CoT; one per problem; ≤ 1,024 tokens).
- Prompt cap: 512 tokens for the original setting; **576** for answer-in-prompt (7 of 49,800 such prompts exceed 512 chat-formatted tokens, max 549; the formatter would cut the start of the problem).
- Backbone chat template: Qwen2.5-7B-Instruct.

### 2.2 The attribute: what α is

\[
a(y)=\#\text{tokens}(y)\ \text{under a fixed ruler tokenizer (Qwen3-4B-Instruct-2507)},\qquad \alpha(y)=F\big(a(y)\big)
\]

- F is the equal-weight mixture of the participating clients' empirical CDFs of a on their training solutions (`alpha_mode: global`). α = 0.5 means the same length on every client.
- **Fixed ruler**, independent of the backbone (`fedsteer/mathcot.py`), so every run is scored on the same scale. Training labels and the `cot_tokens` scorer use the same function (verified identical).
- Everything the model generates counts. The solution *is* the CoT.
- Steps (paragraphs) are a secondary readout ⏳.

### 2.3 Accuracy: reported, not gated

- Gold answer = the content of the reference solution's last `\boxed{}`. Correct = Math-Verify equivalence, with normalized exact match for MCQ letters. All 1,200 test references score correct against their own gold.
- **Answer-in-prompt:** accuracy becomes a consistency check (does the solution reach the given answer?), expected near 1.
- **Original prompts:** accuracy is the model actually solving the problem. It is a side readout: G2 showed it falling with α (MATH-6).

---

## 3. Data ✅ built

### 3.1 Provenance and filtering

- `AI-MO/NuminaMath-CoT`: 859,494 problems, one GPT-4o CoT solution each. NuminaMath-1.5 supplies only `problem_type`, used to split the large sources.
- 45,121 exact-duplicate problems dropped (kept under the original benchmark source: gsm8k > math > amc_aime > aops_forum > olympiads > synthetic_amc > cn_k12 > synthetic_math > orca_math).
- Solutions without `\boxed{}` dropped (32,166); targets > 1,024 ruler tokens dropped.
- `data/math_fed_ans/` = `data/math_fed/` with rewritten prompts only (`scripts/make_answer_prompt_data.py`). Both are synced to the shared `/work/nvme/bhby/datasets/`, which Delta and dtai share.

### 3.2 Clients ✅

12 clients spread over the median length, ≤ 3 per source family; cn_k12, orca_math and synthetic_math are split by `problem_type`. Each has 4,000 train / 50 dev / 100 test problems. Supports are the 5th–95th training percentiles on the 12-client global scale.

| client | median tokens | p5–p95 | support | role, rotation 0 |
|---|---|---|---|---|
| orca_math / Logic and Puzzles | 138 | 49–440 | [0.01, 0.68] | **held-out** |
| cn_k12 / Logic and Puzzles | 170 | 55–427 | [0.01, 0.67] | participant |
| gsm8k | 244 | 140–415 | [0.12, 0.65] | participant |
| synthetic_math / Algebra | 271 | 125–582 | [0.10, 0.82] | **held-out** |
| orca_math / Algebra | 283 | 94–554 | [0.06, 0.80] | participant |
| cn_k12 / Inequalities | 287 | 97–652 | [0.06, 0.87] | participant |
| synthetic_math / Geometry | 304 | 149–584 | [0.14, 0.83] | **held-out** |
| cn_k12 / Geometry | 315 | 91–763 | [0.05, 0.93] | participant |
| math | 403 | 180–729 | [0.20, 0.91] | participant |
| synthetic_amc | 411 | 241–645 | [0.33, 0.87] | **held-out** |
| olympiads | 679 | 371–972 | **[0.58, 0.99]** | participant |
| aops_forum | 726 | 381–987 | **[0.60, 0.99]** | participant |

- **Limited spectra, naturally:** the competition banks never see the bottom ~58% of the scale; gsm8k and the Logic & Puzzles banks never see the top ~33%.
- **Rotation 1 holds out olympiads** (the C2 coverage test).
- **Local characteristics (the PFL tension):** reference solutions identify their client 87% of the time even with the math masked out, flat across length. These are per-source writing conventions (e.g. gsm8k "Therefore" in 100%, olympiads headers in 49%, cn_k12 boxing choice letters). The private adapter should preserve them while D shares the range. Metric ⏳ (§5).

### 3.3 Splits

Train 4,000 / dev 50 / test 100 per client, disjoint by problem. One reference per input. No drift split.

---

## 4. Model, configuration and compute

- **Method: unchanged** (`configs/math_fedavg.yaml`): `W_0 + B_i^P A_i^P + g(α) B^D A^D`, shared frozen `A^D`, FedAvg on `B^D`, `calibration: shared`, `offset: false`, `adapter: private`, `alpha_mode: global`, E = 20 local steps × batch 8, rank 16, `max_target_tokens 1024`.
- **Backbone: Qwen2.5-7B-Instruct** (selected 10-04 from a six-model screen). It is the most familiar with the data (target perplexity 1.75 vs. 3.64 for Qwen3-4B-2507), its format is clean, and its natural lengths are near the references. **Plain SFT keeps its accuracy** (G0, MATH-6), unlike Qwen3-4B, which lost 0.98 → 0.79 to the teacher-style mismatch. Qwen2.5-3B is the cheaper fallback.
- **`attn_implementation: eager`** ⚠️ required on dtai. SDPA's backward gave non-finite gradients on long (> 1k-token) batches with torch 2.13 / GH200, and every run went NaN within 5 rounds (MATH-5). Eager trains the same batches cleanly at ~10% more time per round.
- **NaN guard** (`fedsteer/fed.py`): a step with a non-finite loss or gradient is skipped and reported, and `skipped_steps` is logged per client. Every result must state that count (0 so far).
- **Rounds:** 100 × 20 steps (4 epochs over 4k pairs); dev eval every 20 rounds (50 problems for single-client runs, 20 per client for E1); generation batch 64.
- **Fits a 40 GB GPU** (MATH-11; verified under a 39.5 GiB cap): each step's 8 examples run as 4 micro-batches of 2 (`fed.batch_size 2`, `fed.grad_accum 4`); dev-loss batch 4 (`monitor.batch_size`); **generation uses SDPA** while training uses eager (no backward pass, so the dtai NaN does not apply); B1 3-shot / base rows at batch 8. Peaks: training 21.9 GiB, test eval 19.9 GiB, B1 23.6 GiB. Cost: training rounds ~2× slower than with batch 8.
- **Generation:** greedy, `max_new_tokens 1280`; outputs at the cap count as truncated. The base-model references use a 4,096 cap (at 1,280, Qwen3-4B truncated 42% of base outputs).
- **Compute, measured (dtai GH200):** single-client gate ≈ 2.9 h; E1 ≈ 4–4.5 min per round (8 clients), i.e. ≈ 7 h of training plus evals.
- **Hosts:** dtai (`ghx4`, GH200 96 GB, env `rein-gh`, torch 2.13) and Delta (all GPU partitions `gpuA100x4,gpuA100x8,gpuH200x8`; the A100s are 40 GB, env `rein`, torch 2.5.1). ⚠️ Not numerically identical: an E1 fed/local pair is always kept on one host; single-client gates may finish on either.

---

## 5. Metrics

As in the parent plan, with these substitutions (`scripts/score_math.py` and `scripts/math_gate_report.py`, computed from saved generations):

| Dimension | Math CoT |
|---|---|
| **Calibration (primary)** | percentile error \|F(a(ŷ)) − α\| on ruler tokens; constant-output baseline 0.30 on the 5-point grid. ⏳ also report the interior grid (α ∈ [0.1, 0.9]): α = 0 and 1 target the dataset's shortest and longest solutions (77 / 1,024 tokens), which inflates even the in-support error |
| **Coverage (C1)** | in/out-of-support error and reach rate; the natural gaps of §3.2 |
| **Steerability** | per-problem Spearman across α; concordance |
| **Format (gate G0)** | `\boxed{}` present; truncated at the cap; repetition loop (a 20-word n-gram ≥ 3 times); gzip ratio |
| **Accuracy (reported)** | per α, in/out of support (§2.3) |
| **House style (PFL tension) ⏳** | client attribution of generated solutions by a classifier trained on reference solutions with the math masked: own-client share in and out of support, leakage toward the clients that cover that α, answer-convention compliance. Compares federated (private adapters) vs. A2 (shared adapter) vs. local |
| **Specificity ⏳** | steps vs. tokens |

Reporting: mean and worst client, per-client paired bootstrap over test problems, checkpoint selected on dev (lowest percentile error).

---

## 6. Baselines

Same IDs as the parent plan. B1 and B4 run on each client's own model with B^D removed.

| ID | Baseline | Note for this task |
|---|---|---|
| **B1** | Prompting: a 0–100 length level in the prompt, k ∈ {0, 3} nearest-α examples; also on the base model (4,096 cap) | The gate (G1). The prompt reuses each record's own closing instruction, so the answer-in-prompt setting keeps its answer line (fixed 10-05). ⏳ second template with an explicit token budget |
| **B2** | Local-only: each client learns its own D | The C1 comparison |
| **B3** | One-shot merged direction | Is iterative federation needed? |
| **B4** | Federated activation steering (CAA) | Watch the loop and boxed rates |
| **B5** | Pooled reference | Cost of decentralization |
| **A2** | Shared adapter (non-personalized FedAvg) | "Isn't this just PFL?"; also the house-style comparison |

---

## 7. Experiments and status

| Step | What | Status |
|---|---|---|
| Backbone screen | 6 backbones: familiarity (target perplexity) and headroom (base accuracy, 4,096 cap) | ✅ Qwen2.5-7B selected (entry "Backbone screen …", 10-04) |
| History, Qwen3-4B | E0a / G0 / G2 on Delta | ✅ G0 lost accuracy (style mismatch); G2 pct err 0.226, range 248 → 502 tokens (entries 37–38). Superseded |
| **G0, original prompts** | plain SFT on `math`, D = 0 (dtai 3311353) | ✅ **PASS** (MATH-6): format perfect; accuracy 0.68 vs. base 0.72 on 50 shared problems (paired 4 / 6, noise); 0 skipped steps |
| **G2, original prompts** | steering on `math` (dtai 3311354) | ⚠️ **misses calibration** (MATH-6): test pct err **0.263** (< 0.20 needed), Spearman 0.712 ✅. **Range compressed: 279 → 511 tokens vs. targets 77 → 1,024**, nearly the same as Qwen3-4B, so not backbone-specific. Accuracy falls with α (0.62 → 0.50). Dev plateau from round 60; memorization (train loss 0.07, dev loss 0.30 → 0.43); gain near the clamp (≈ 3.4–3.8 of 4) while ‖D‖ keeps growing |
| E1-fed, original prompts | 8 participants, federated (dtai 3311356) | 🔄 **reference run**, kept by the user; round-20 dev pct err 0.339, Spearman 0.50. ⚠️ Its local partner was cancelled (not started), so C1 on the original prompts needs a rerun |
| **Answer-in-prompt suite (main)** | G0, G2, G1, E1-fed, E1-local + base reference (MATH-7) | 🔄 queued on dtai (3316647–52) and Delta (22685329–33), raced |
| E2 | held-out clients join with n ∈ {16, 64, 256, 1024}; rotation 0, then rotation 1 (olympiads held out) | ⏳ |
| E3 | drift | skipped |

---

## 8. Decision gates

| Gate | Test | Pass condition | Status / if it fails |
|---|---|---|---|
| **G0** | plain SFT vs. base on the same client | `\boxed{}` ≥ 95%; loops ≤ 5%; truncation ≤ 5%. Accuracy reported, not gated | ✅ original prompts (Qwen2.5-7B). 🔄 answer-in-prompt |
| **G1** | B1 vs. G2 | B1 pct err ≥ 0.20, or the method better by ≥ 0.05 | 🔄 answer-in-prompt (the original-prompts G1 was cancelled). If it fails: the knob is promptable; keep the task for C1/C2 only |
| **G2** | single-client steering | pct err < 0.20; Spearman ≥ 0.7 (the accuracy condition is dropped: accuracy is not gated) | ⚠️ fails on original prompts (0.263). 🔄 answer-in-prompt. If it fails again: see §10, the range investigation |
| **G3** | E1, method vs. B2 | better on most gapped clients at their out-of-support α | ⏳ answer-in-prompt pair (MATH-7) |

---

## 9. Risks

1. ⚠️ **Compressed range (open problem).** On two backbones the knob spans only ≈ 280 → 510 tokens. Candidate causes: the gain clamp (`gain_max = 4`), memorization after ~60 rounds (4 epochs), endpoint targets at the data's extremes. See §10.
2. ⚠️ **Prompting (R5).** Length is verbalizable; G1 is on calibration.
3. ⚠️ **Length gaming** (padding, restating, looping): caught by the loop and gzip metrics.
4. **Platform numerics.** SDPA on torch 2.13 / GH200 gave NaN gradients (fixed with eager). Delta and dtai differ in torch and GPU, so the E1 pair never splits across hosts.
5. **Infra.** cc-login1 rebooted on 10-05, killing the SSH masters and the race watcher; after a reboot the race is resolved by hand. The SSH masters need the user's Duo.
6. **Source style.** Length differences partly reflect GPT-4o's per-source style; that is part of each client's data (and the house-style signal of §3.2).

**Why not OpenR1-Math (the 10-02 draft).** R1 thinking traces have a median of 2.4k tokens, 50–80× Newsroom's targets; natural traces of the same problem barely differ in length; CoT-only clients had broad supports. Kept as a possible later "long reasoning" extension.

---

## 10. Next steps

1. 🔄 Answer-in-prompt suite (MATH-7): read G0 (format), G2 (calibration; is the range still compressed?), G1 (prompting), then E1 fed vs. local (G3).
2. 🔄 **Per-layer calibration** (MATH-14): `lora.warp_scope=module` on single-client G2, with and without `lora.gain_max=16`; carry into E1 if it helps.
3. ⏳ **Range investigation** (single-client, ≈ 3 h each, can run in parallel): (a) `lora.gain_max=16`; (b) stop at ~60 rounds or use more data per client (the source pools are larger than 4k); (c) score the interior α grid on existing outputs (no training; changes how the metric is reported, so agree with the user first).
4. ⏳ House-style metric (§5) for E1: federated vs. A2 vs. local.
5. ⏳ E2 on rotation 0, then a rotation-1 federated run.
6. ⏳ Optionally rerun E1-local on the original prompts to pair the kept reference E1-fed.

---

## Changelog

| Date | Change | Evidence |
|---|---|---|
| 10-02 | OpenR1-Math draft | entry 36 (math) |
| 10-03 | Rewritten for NuminaMath-CoT; data built; gates launched on Delta (Qwen3-4B) | entry 37 (math) |
| 10-03 | Qwen3-4B: G0 fails on accuracy (style mismatch), G2 0.226 with a compressed range | entry 38 (math) |
| 10-04 | Backbone screen → Qwen2.5-7B; accuracy reported, not gated; answer-in-prompt data built as a backup | entry 39 (math) |
| 10-04 | Runs move to DeltaAI; cluster rule: math on Delta + dtai only | entry 40 (math), MATH-1 … MATH-4 |
| 10-05 | Every Qwen2.5-7B run NaN (SDPA backward on long batches) → eager attention + NaN guard; suite resubmitted | MATH-5 |
| 10-05 | Qwen2.5-7B: G0 PASS; G2 0.263, range still compressed | MATH-6 |
| 10-05 | **Answer-in-prompt becomes the main setting**; the running original-prompts E1-fed is kept as the reference | MATH-7 |
| 10-07 | Diagnosis: per-client response levels set by the private adapter; the calibration [0, s] cannot express negative or above-clamp coefficients (MATH-13). Per-layer warps launched (MATH-14) | MATH-13, MATH-14 |
| 10-06 | Delta copies OOMed on 40 GB A100s; runs made to fit 40 GB (micro-batching, small loss batch, SDPA generation), verified under a 39.5 GiB cap; Delta back in the race | MATH-9 … MATH-11 |

## References (task-specific)

- NuminaMath-CoT: https://huggingface.co/datasets/AI-MO/NuminaMath-CoT · NuminaMath-1.5: https://huggingface.co/datasets/AI-MO/NuminaMath-1.5
- Math-Verify: https://github.com/huggingface/Math-Verify
- Huang et al. *How to Fine-Tune a Reasoning Model? A Teacher–Student Cooperation Framework to Synthesize Student-Consistent SFT Data* (TESSY). 2026. arXiv 2604.14164
- Ren et al. *I Learn Better If You Speak My Language.* EMNLP 2024. https://aclanthology.org/2024.emnlp-main.571/
- Han et al. *Token-Budget-Aware LLM Reasoning.* Findings of ACL 2025. https://aclanthology.org/2025.findings-acl.1274/
- Aggarwal and Welleck. *L1: Controlling How Long a Reasoning Model Thinks with RL.* 2025. https://arxiv.org/abs/2503.04697
