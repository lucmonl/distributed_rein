# Federated steering on math CoT: a solution-length knob (NuminaMath-CoT)

Experiment plan · drafted 2 October 2026 · **rewritten 3 October 2026 for NuminaMath-CoT** (the OpenR1 draft is superseded; why in §9).
Companion to `federated-steering-plan.md`: method, metrics, baselines and claims C1/C2 are reused unless stated. Sister task: `chembl-experiment-plan.md`.
Evidence: `exp_log/EXPERIMENT_LOG.md` entries 36 (with its addenda) and 37. Data: `data/math_fed/` (built by `scripts/build_math_fed.py`). Statistics: `data/numinamath_cot/stats.json` (`scripts/numina_cot_stats.py`).
**Hosts:** code is edited on cc-login1 only. Every GPU job runs on Delta (delta-login2) from the synced copy (`scripts/sync_to_delta.sh --data`); launch files in `exp_log/launch/`.

Status markers: ✅ measured / done · 🔄 running · ⏳ planned · ⚠️ known risk.

---

## 1. Why this task

The claim is unchanged: **clients each fine-tune their own model, jointly learn one steering direction, and can then reach attribute levels their own data barely covers.** The attribute here is **how long the chain of thought is**, from a terse solution with only the key steps to a fully spelled-out one.

| Criterion (`dataset-candidates.md`) | How it scores (measured) |
|---|---|
| R1 continuous, fine-grained | ✅ token count, ≈ 50 → 990 tokens across the 12 clients |
| R2 cheap deterministic scorer | ✅✅ one tokenizer call. Utility (answer accuracy) is deterministic too (Math-Verify against the gold answer); **no judge anywhere** |
| R3 clients with a limited spectrum | ✅ **natural, no constructed windows.** olympiads and aops_forum cover only [0.58, 0.99] and [0.60, 0.99] of the global scale; gsm8k [0.12, 0.65] and the two Logic & Puzzles banks [0.01, 0.68]. Newsroom-like disjoint supports in both directions (§3.2) |
| R4 not summarization | ✅ |
| R5 prompting cannot reach it | ⚠️ the weakest point. "Be brief" / "be detailed" is verbalizable, so prompting will move length in the right direction. What it is expected to miss is a *calibrated* level. Gate G1 is on calibration error, not Spearman |
| R6 attribute free given the input | ✅ for what matters, a model-side test: a TF-IDF ridge on the problem text explains only **0.34** of log-length variance within a client (0.655 overall, most of it the client itself). About two thirds of within-client variation is not given by x, so the α label carries signal for D |

What it adds to Newsroom and ChEMBL: an attribute everyone already wants to control (CoT length), a **verifiable utility** (accuracy at each α), and natural limited-spectrum clients on *both* ends of the scale.

**Framing (decided 10-03).** These are clean, linear step-by-step solutions. In a sample, 0% contain "wait" or "alternatively" and 2% contain a check, against 97–99% in R1 thinking traces. So length here means *more steps and more explanation*, not *more search*. The paper calls the knob **CoT length control**. The test-time-compute framing is claimed only if the accuracy-vs-α curve earns it (accuracy rising with α, above all for the hard clients).

---

## 2. Task definition

### 2.1 Input and output

- **x** = the problem + `Please reason step by step, and put your final answer within \boxed{}.` (prompt ≤ 480 tokens).
- **y** = a GPT-4o step-by-step solution ending in `\boxed{answer}` (NuminaMath-CoT; one per problem; ≤ 1,024 tokens).
- Backbone: Qwen3-4B-Instruct-2507 with its chat template (`enable_thinking: false`).

### 2.2 The attribute: what α is

\[
a(y)=\#\text{tokens}(y)\ \text{under a fixed ruler tokenizer (Qwen3-4B-Instruct-2507)},\qquad \alpha(y)=F\big(a(y)\big)
\]

- F is the equal-weight mixture of the participating clients' empirical CDFs of a on their training solutions (`alpha_mode: global`), exactly as in the parent plan. α = 0.5 means the same length on every client.
- **Fixed ruler.** The tokenizer is independent of the backbone (`fedsteer/mathcot.py`, `FEDSTEER_RULER`), so a Llama smoke run and a Qwen run are scored on the same scale. The build computes training labels with the same function the `cot_tokens` scorer applies to generations (verified identical).
- **Not steps.** Steps (paragraphs) are a formatting artifact and agree only loosely with tokens. They are a secondary readout ⏳.
- Everything the model generates counts. The solution *is* the CoT; there is no separate answer section.

### 2.3 Utility: accuracy at each α

- The **gold answer** is the content of the reference solution's last `\boxed{}` (`extract_boxed`). Using it for every client is uniform: the NuminaMath-1.5 `answer` field does not join for olympiads (0.7%) or aops_forum (19%).
- **Correct** = Math-Verify equivalence of the generation's last `\boxed{}` with the gold answer, with normalized exact match as a fallback for MCQ letters. Sanity check: all 1,200 test references score correct against their own gold, and a perturbed answer scores wrong.
- Reported per α and in/out of support, next to the base model's accuracy at its natural length (E0a).

---

## 3. Data ✅ built (`data/math_fed/`)

### 3.1 Provenance and filtering

- `AI-MO/NuminaMath-CoT`: 859,494 problems, one GPT-4o CoT solution each (1.2 GB, `data/numinamath_cot`). NuminaMath-1.5 (`data/numinamath_15`) supplies only `problem_type`, used to split the large sources.
- Exact duplicate problems after whitespace normalization: **45,121 dropped**, keeping the copy under the original benchmark source (gsm8k > math > amc_aime > aops_forum > olympiads > synthetic_amc > cn_k12 > synthetic_math > orca_math). Without this priority, gsm8k vanished into orca_math, which contains its problems.
- Solutions with no `\boxed{}` dropped (32,166 in the selected sources), so every target has a gold answer and a consistent format.
- Targets > 1,024 ruler tokens dropped. Length percentiles over all solutions: p5 118, p50 356, p95 914, p99 1,158 (chars/token estimate; exact counts are used in the build).

### 3.2 Clients ✅

The `source` field gives only 8 sources with ≥ 5k problems. The three large sources whose problems join to NuminaMath-1.5 (cn_k12 95%, orca_math 100%, synthetic_math 99.8%) are split by `problem_type`. 12 clients are spread over the median length, with ≤ 3 per source family. Each has 4,000 train / 50 dev / 100 test problems.

Supports are the 5th–95th training percentiles on the 12-client global scale (each run recomputes the scale from its own participants):

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

- **Limited spectra, naturally.** The two competition banks never see the bottom ~58% of the scale; gsm8k and the Logic & Puzzles banks never see the top ~33%. The rest are broad. Both Newsroom regimes (gapped and broad) appear in one task.
- **Rotations** (held-out every third along the median order, as for Newsroom and ChEMBL): rotation 0 holds out four low-to-mid clients, so the narrow top clients are *participants*. **Rotation 1 holds out olympiads** (plus cn_k12/Logic, orca_math/Algebra, cn_k12/Geometry), which is the C2 coverage test: a high-only client joining and being asked for short solutions.
- The heterogeneity is problem difficulty plus source style (the synthetic sources are GPT-4o-written problems with uniform solution style; olympiad solutions are long because they need more steps).

### 3.3 Splits

- Per client: **train 4,000 / dev 50 / test 100**, disjoint by problem (one solution per problem, so splitting by record is splitting by problem).
- One reference per input, as on Newsroom: no within-problem references at other α.
- No drift split (E3 stays optional).

---

## 4. Model, configuration and compute

- **Method: unchanged** (`configs/math_fedavg.yaml`): `W_0 + B_i^P A_i^P + g(α) B^D A^D`, shared frozen `A^D`, FedAvg on `B^D`, `calibration: shared`, `offset: false`, `adapter: private`, `alpha_mode: global`, E = 20 local steps × batch 8, rank 16, `max_prompt_tokens 512`, `max_target_tokens 1024`.
- **Rounds.** Single-client gates: 60 rounds (2.4 epochs over 4k pairs), dev evaluation every 10. ChEMBL's G0 was undertrained at 0.8 epochs (entry 33). Federated runs start at 30 rounds; the gates' dev curves decide whether that is enough.
- **Backbone:** Qwen3-4B-Instruct-2507 (cached on Delta). Llama-3.2-1B only for the CPU smoke test. Qwen3-8B to confirm the top rows. ⚠️ Qwen3 has very likely seen NuminaMath: this affects absolute accuracy, not comparisons.
- **Generation:** greedy, `max_new_tokens 1280` (25% above the longest target). Outputs hitting the cap count as truncated. The current HF loop is enough: a 5-α test eval is ≈ 3M generated tokens over 12 clients. vLLM is not needed.
- **Compute (estimate):** ≈ 1.5–2 h per single-client gate run (training ≈ 20 min, dev evaluations dominate). The 8-participant runs are expected at ≈ 6–8 h. To be replaced by measured times from entry 37's jobs.

---

## 5. Metrics

As in the parent plan, with these substitutions (`scripts/score_math.py` computes the math-specific rows from saved generations, with no regeneration):

| Dimension | Math CoT |
|---|---|
| **Calibration (primary)** | percentile error \|F(a(ŷ)) − α\| on ruler tokens; constant-output baseline 0.30 on the 5-point grid |
| **Coverage (C1)** | in/out-of-support error and reach rate; the natural gaps of §3.2 |
| **Steerability** | per-problem Spearman across α; concordance |
| **Ties** | identical text across α (greedy) |
| **Format (reported first; gate G0)** | `\boxed{}` present; truncated at the cap; repetition loop (a 10-word n-gram ≥ 3 times); gzip ratio |
| **Utility** | **accuracy** per α, in/out of support; accuracy vs. tokens per client; base-model accuracy (E0a) as the reference line |
| **Specificity** | steps vs. tokens (more steps, or wordier steps?) ⏳ |

Reporting: mean and worst client, per-client paired bootstrap over test problems, checkpoint selected on dev (lowest percentile error). Generation is cheap enough for the 11-point α grid on test and 3 seeds for the main table.

---

## 6. Baselines

Same IDs as the parent plan. B1 and B4 run on each client's own model with B^D removed.

| ID | Baseline | Note for this task |
|---|---|---|
| **B1** | Prompting: a 0–100 length level in the prompt (`prompt_with_level_math`), k ∈ {0, 3} nearest-α examples; also on the base model | The gate (G1). Expected: ordered but miscalibrated. ⏳ second template with an explicit token budget F⁻¹(α) |
| **B2** | Local-only: each client learns its own D | The C1 comparison; the decisive clients are olympiads / aops_forum (low α) and gsm8k / Logic & Puzzles (high α) |
| **B3** | One-shot merged direction | Is iterative federation needed? |
| **B4** | Federated activation steering (CAA) | Watch the loop and boxed rates |
| **B5** | Pooled reference | Cost of decentralization |
| **A2** | Shared adapter (non-personalized FedAvg) | "Isn't this just PFL?" |

Budget forcing (s1) is dropped from the OpenR1 draft: it truncates *thinking* before a separate answer, and these solutions have no such split.

---

## 7. Experiments, in gate order

The order applies the ChEMBL lesson (entries 33–34): measure the backbone with no steering first, so a later failure can be attributed.

| Step | What | Status |
|---|---|---|
| **E0a** | Base Qwen3-4B on all 12 clients' test problems: accuracy, length, format. The reference lines | ✅ at a 1,280 cap (job 22640760): **cap-limited**, 42% truncated, base outputs 1.3–4× longer than the references. 🔄 rerun at 4,096 (22643457/59/60) |
| **G0** | Plain SFT on one client (`math`, broad support [0.20, 0.91]), `fed.lr_shared=0` so D stays exactly zero | 🔄 Delta job 22640761 |
| **G2** | The same client with steering on: calibration, per-problem Spearman, accuracy per α | 🔄 Delta job 22640762 (round 40: pct err 0.273, Spearman 0.745) |
| **G1** | B1 (k = 0 and 3, client and base model) on G2's client model; the base row at a 4,096 cap | ⏳ job 22643461, chained after G2 |
| **E1** | 8 participants (rotation 0), federated vs. local (B2), then B3, A2, B4 | ⏳ after G0–G2 |
| **E2** | Held-out clients join with n ∈ {16, 64, 256, 1024}; `frozen_D` vs. `local_D` (+ `prompt`). Rotation 0 (broad held-out clients), then **rotation 1 (olympiads held out)** for coverage | ⏳ |
| **E3** | Drift | skipped (optional in the parent plan) |

---

## 8. Decision gates

| Gate | Test | Pass condition | If it fails |
|---|---|---|---|
| **G0** | plain SFT vs. base on the same client (E0a-long: base at a 4,096-token cap, since at 1,280 it truncates 42% of outputs) | `\boxed{}` ≥ 95%; loops ≤ 5%; truncation ≤ 5%; accuracy ≥ base − 5 points | Lower LR / fewer rounds (SFT on GPT-4o solutions can pull a strong 4B model down); check the chat template |
| **G1** | B1 vs. G2 | B1 percentile error ≥ 0.20, or the method better by ≥ 0.05 | The knob is promptable on this task; keep it only for C1/C2 and say so |
| **G2** | single-client steering | percentile error < 0.20 (constant 0.30); per-problem Spearman ≥ 0.7; accuracy at in-support α ≥ G0 − 5 points | Train longer; if D is ignored (no-effect rate high), the length is too predictable from x for this backbone (§1, R6) |
| **G3** | E1, method vs. B2 | better on most gapped clients at their out-of-support α, accuracy no worse | C1 does not transfer to this task; report it as a negative |

---

## 9. Risks, and why not OpenR1

1. ⚠️ **Prompting (R5).** Length is verbalizable. G1 is on calibration; if prompting calibrates well, the task supports C1/C2 only.
2. ⚠️ **Length gaming.** Padding, restating, looping. Caught by the loop and gzip metrics and by accuracy, reported at every α.
3. ⚠️ **Accuracy vs. α confound.** Short solutions on hard problems will lose accuracy by design. Report the curve; never pair the best α of one metric with the best α of another.
4. **SFT pulling the backbone down.** Fine-tuning a strong instruct model on GPT-4o solutions can lower accuracy. G0 measures it against E0a.
5. **Source style.** Length differences partly reflect GPT-4o's per-source style (synthetic vs. scraped). That is part of each client's data, the same as house style on Newsroom.
6. **Delta:** first GPU jobs on Delta (entry 35 noted the sbatch scripts were untested end to end). `rein` now has sympy 1.14 (math-verify) while torch 2.5.1 pins 1.13.1; harmless without `torch.compile`, but watch for import errors.

**Why not OpenR1-Math (the 10-02 draft).** R1 thinking traces have a median of 2.4k tokens (≈ 3k with the write-up): 50–80× Newsroom's targets, needing vLLM and slow iteration, with length control over thousands of tokens that SFT is known to do poorly (L1 needed RL). Natural traces of the same problem barely differ in length (entry 36), and CoT-only clients had broad supports (mean width 0.81 vs. 0.61 here). Kept as a possible later "long reasoning" extension.

---

## 10. Next steps

1. 🔄 E0a, G0, G2 on Delta (entry 37). Read G0 against E0a, then G2.
2. ⏳ G1: `eval_baselines.py --baseline prompt` with `--scorer cot_tokens --max_new_tokens 1280` on G2's run (k = 0, k = 3 client; k = 3 base), then `score_math.py`.
3. ⏳ If G0–G2 pass: E1 rotation 0, federated and local (`fed.mode=local`), 30 rounds; then B3/A2/B4.
4. ⏳ E2 on rotation 0 and a rotation-1 federated run.
5. ⏳ Steps-vs-tokens readout; the 11-point α grid; 3 seeds for the main table.

## References (task-specific)

- NuminaMath-CoT: https://huggingface.co/datasets/AI-MO/NuminaMath-CoT · NuminaMath-1.5: https://huggingface.co/datasets/AI-MO/NuminaMath-1.5
- Math-Verify: https://github.com/huggingface/Math-Verify
- Han et al. *Token-Budget-Aware LLM Reasoning.* Findings of ACL 2025. https://aclanthology.org/2025.findings-acl.1274/
- Aggarwal and Welleck. *L1: Controlling How Long a Reasoning Model Thinks with RL.* 2025. https://arxiv.org/abs/2503.04697
