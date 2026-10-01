# Federated Learning of Steering Directions

Research plan · created 29 September 2026 · **updated 1 October 2026** (see the changelog at the end and `exp_log/EXPERIMENT_LOG.md` for evidence). Supersedes the task and baseline sections of `shared-steering-research-plan.md`.

Status markers: ✅ done · 🔄 running / partial · ⏳ planned, not started.
Sizes, thresholds and schedules are working settings, not results.

---

## 1. Thesis and scope

**Question.** Can clients that each fine-tune their own copy of a model *jointly learn one steering direction* by federated training, when **each client's data is skewed toward a different part of the attribute's range**? Can a client then produce attribute levels its own data barely contains? Does the direction stay usable on clients that never took part in training, and after further private fine-tuning?

**What is new.** The object learned federatedly is a *control direction*, and the paper's main contribution is a protocol for judging the *quality of that direction*. Personalized federated learning (PFL) shares weights to improve per-client task accuracy at a fixed operating point. Here the shared weights exist to provide a continuous knob, and success is measured with control and quality metrics.

**Why federate at all** (vs. each client fine-tuning locally): the motivation is **coverage, not data volume**. A client can have thousands of examples and still almost none at the attribute levels it wants (e.g. a publication whose summaries are 87% copied leads has few rewritten ones). Other clients have plenty there. A shared α scale plus a shared direction lets that knowledge transfer. Data scarcity is a secondary story (data-budget curve).

**The three claims the paper makes (and nothing else):**

| # | Claim | Decisive comparison | Status |
|---|---|---|---|
| C1 | **Collaboration.** A federated direction steers better than one each client learns alone, above all in the parts of the α scale a client's own data does not cover. | Federated vs. local-only, per client, in-support vs. out-of-support α, at equal data | 🔄 2k: federated better on 6/8 clients, worse on the 2 most copy-heavy (exp-log entry 15). 4k local arm running |
| C2 | **Portability.** The direction (and, with shared calibration, the α→coefficient mapping) works on a *new* client's independently fine-tuned model with zero or a few labelled examples. | Frozen federated D (+ shared calibration) vs. a local direction trained on the same k examples | ⏳ |
| C3 | **Stability.** The direction keeps working while clients continue private fine-tuning. | Control metrics before and after drift | ⏳ |

**Explicitly out of scope.** Multi-attribute composition, formal privacy/DP, heterogeneous architectures, hypernetworks/routers, theory beyond the identifiability remarks, and safety/sycophancy/creativity tasks.

---

## 2. Model: one loss, no phases ✅

Each client i shares the frozen base W_0 and has a private adapter P_i. The server holds the shared direction D. For every adapted weight matrix:

\[
h \mapsto W_0h + B_i^{P}A_i^{P}h + g(\alpha)\,B^{D}A^{D}h,\qquad g(\alpha)=o+s\,h(\alpha),\; s=\exp(u)\in[\tfrac14,4],\;\alpha\in[0,1].
\]

- **A^D is a shared random matrix, generated from a common seed and frozen; only B^D is trained** (FFA-LoRA style [5]). Server averaging is then *exact*: \(\tfrac1M\sum_i B_i^D A^D = (\tfrac1M\sum_i B_i^D)A^D\). B^D starts at zero. The coefficient multiplies the finished product once.
- **Calibration g(α) = o + s·h(α)** maps α to a coefficient on D:
  - gain s;
  - optional offset o (`lora.offset`);
  - warp h: an increasing map with h(0) = 0, h(1) = 1, starting as the identity. Options (`lora.warp`): `none` | `kumaraswamy` | **`kumaraswamy_mix`** (h = (1−w)α + w[1 − (1 − α^p)^q]) | `step`.
- **Calibration mode** (`fed.calibration`): `private` (one g per client, fitted on its own data) or **`shared`** (one g for all clients, averaged by the server every round like D). Under the global α scale (below) there is no principled reason for per-client mappings, and the offset's original job (starting a client partway along D) is done by the scale itself. Shared calibration also lets a new client steer with **zero** labelled examples. 🔄 Shared calibration, with and without the offset, is being tested against private calibration (exp-log entry 17).
- **Adapter mode** (`fed.adapter`): `private` (default) | `shared` | `none`.
- Warm-up: the gain and offset train from round 2, the warp from round 5 (with a penalty toward the identity).
- **Per-component regularizers** (`reg:` section, all off by default): private-adapter weight decay (low-rank penalty), LoRA dropout, decorrelation of Pᵢ from D, FedProx on D, weight decay on D, priors on the gain and offset.
  - The first untuned bundle **hurt** steering and memorized more (cosine LR + priors pinning the gain near 1).
  - The decorrelation term was inactive (Pᵢ and D already nearly orthogonal).
  - Regularization is **not** part of the method until an ablation shows a benefit (entry 11).

**Control coordinate: global percentile** (`alpha_mode: global`, the method; `local` kept as an ablation). Let a(y) be the attribute score of a target text and F_i the empirical CDF of a(·) on client i's training data. The shared reference is the **equal-weight mixture** F = (1/M) Σ_i F_i of the participants' CDFs, and every training example is labelled

\[
\alpha(y)=F\big(a(y)\big).
\]

- α = 0.5 means the same behaviour on every client. Each client's data covers only part of [0, 1], its **support** (5th–95th percentile of its own data on this scale).
- The server can build F by averaging per-client normalized histograms, so no examples are shared. Runs save it as `alpha_reference.json`, and held-out clients reuse it.
- A client wanting its own local semantics converts at inference: α_global = F(F_i⁻¹(α_local)).
- *Why not local percentiles (the original plan):* the same local α meant different behaviour per client (nypost's local α 0.13–1 is all "copy the lead"), which gave the shared direction conflicting signals.

**Single local objective** (task adaptation and control are the same loss):

\[
\mathcal L_i(P_i,\theta_g,B^D)=\mathbb E_{(x,y)\sim S_i}\big[-\log p_{W_i(\alpha(y))}(y\mid x)\big].
\]

**Identifiability, revised.**
- α varies within each client, so Pᵢ cannot absorb *all* of the attribute signal.
- But Pᵢ **can** absorb a client's typical level or format. Example: nypost.com's adapter learns its fixed 200-character truncated lead (entry 12). Its α = 0 then requires a coefficient (−0.58) no other client trains, and extrapolation fails (see the discussion around entry 16).
- Countermeasures under test: shared calibration (exp17); then shared/no adapter. A support-aware calibration prior is the fallback.

**Federated protocol (FedAvg).**
1. The server broadcasts B^D (and the calibration, if shared).
2. Each client runs E local steps on its private parameters and local copies of the shared ones.
3. The server takes a **uniform** average, so large clients don't dominate the direction.
4. Private states and optimizer states persist across rounds; shared parameters' optimizer state is reset each round.

Defaults: E = 20 steps × batch 8, 8 clients per round (full participation), rank 16 for Pᵢ and D, adapters on all attention and MLP projections.

**Scale remark.** D → cD, s → s/c leaves the model unchanged. Normalize ‖B^D A^D‖_F per layer before comparing directions. A client seen at a single α cannot identify D.

---

## 3. Tasks: fine-grained labels and deterministic or off-the-shelf scorers

### 3.1 Flagship: summary extractiveness on Newsroom [1] ✅

- **Attribute:** Grusky et al.'s fragment density, reimplemented (regex tokenizer; Spearman 0.99 with the dataset's own values) and computed on the truncated article. It is used for both the training labels and evaluation.
- **Clients:** 12 publications with ≥ 5k usable pairs, evenly spaced by median density (telegraph.co.uk 1.3 → nypost.com 32): 8 participants, 4 held out, 3 stratified rotations (`data/newsroom_fed/clients.json`). All runs so far use rotation 0.
- **Splits per client:**
  - train pool up to 5k pairs (reuters.com 4,023, bbc.com 4,228, theguardian.com 4,705, others 5,000);
  - dev 100; test 200 (official test split);
  - **drift 600** (3 stages of 200) from the latest year(s), with train/dev/test strictly earlier. That holds for 11/12 clients; aol.com falls back to a same-year split.
- **Articles** are truncated to 400 words, and pairs whose summary depends on the removed part are dropped.
- **Data budgets used:** 2k, **4k (current best)**, natural (all of the pool). The planned 64 / 256 / 2k curve is ⏳.
- **Raw-data facts to report:**
  - nypost.com's summaries are machine-truncated article leads (≈ 200 characters + "…"; 92% in the raw data). They are **kept**, since that's the dataset, and treated as an extreme *format convention*.
  - Smaller shares of article-prefix summaries elsewhere (latimes.com 16%, reuters.com 14%, cbc.ca 13%).
  - Density partly tracks summary length (Spearman 0.64).
- **Quality (must not reward copying; ROUGE not used):** AlignScore [7] (factual consistency), BERTScore F1 vs. the reference, and an LLM judge (FineSurE-style faithfulness [17], relevance, coherence), **each compared with real summaries at the same global α** (matched extractiveness, Ladhak et al. [18]). Real summaries in the most rewritten α range score AlignScore 0.48 vs. 0.84–0.87 when extractive, so a fixed bar would be misleading. ✅ A blinded human audit of 200 samples remains ⏳.
- **Off-target:** length ✅ (reported per α); readability (FKGL) ⏳.

### 3.2 Secondary: review rating on Amazon Reviews 2023 [2] ⏳

- **Why:** continuity with the REIN draft's sentiment task, fine-grained 1–5 labels, and product categories as natural clients.
- **Clients:** 8 + 4 categories. Input: product title and description; target: the review. α: global percentile of the star rating (ties spread uniformly within each star level).
- **Scorer:** expected star rating from a 5-class review classifier trained on *categories not used as clients*, plus an off-the-shelf rating model for robustness.
- **Utility:** product relevance and fluency. Off-target: length.

Only Table 1's key rows run on this task. It shows generality; it doesn't carry the paper.

**Dropped:** creativity, sycophancy, refusal, toxicity (Perspective API shuts down on 31 Dec 2026).

---

## 4. Evaluating the quality of a learned steering direction

A direction has no standalone score: it's always evaluated as an intervention on specific client models. **Checkpoints are selected on dev and reported on test.**

**Protocol.**
- **During training:** a full dev evaluation every 10 rounds (100 articles × α ∈ {0, .25, .5, .75, 1}, greedy) with dev loss. ✅
- **After training:**
  - select the checkpoint on dev (lowest mean percentile error); test evaluation (200 articles) ✅;
  - quality metrics on test ✅; LLM judge on a 25-article subsample ✅.
- **Every generated text is saved,** so metrics can be recomputed without regenerating.
- ⏳ **Still to add:**
  - an 11-point α grid for final figures;
  - 4 samples at T = 0.7 on a subset (also measures the distribution shift that greedy ties hide);
  - quality-aware checkpoint selection (best steering among checkpoints whose quality stays within ε of the same-α reference).

| Dimension | Metric | Definition | Status |
|---|---|---|---|
| **Calibration (primary)** | Percentile error | \(|F(a(\hat y_\alpha)) - \alpha|\) averaged over α. Bounded and scale-free. A constant output scores 0.30 on the 5-point grid | ✅ |
| **Coverage (key for C1)** | In-/out-of-support error; reach rate | Percentile error split by whether α lies inside the client's support. Reach rate = share of out-of-support outputs that leave the client's own range in the requested direction | ✅ |
| **Steerability** | Concordance; per-article Spearman (no-effect articles count as 0); order rate | Pairwise ordering of outputs across α | ✅ |
| **Ties** | Exact and **near-tie** rates; concordance excluding near-ties | Near-identical = same text after normalization (quotes, whitespace, trailing ellipsis and cut word) up to a small token edit (ratio ≥ 0.95) | ✅ |
| **Range** | Percentile range; IQR-normalized range | Output spread from α = 0 to α = 1 | ✅ |
| **Quality / utility** | AlignScore, BERTScore F1, LLM-judge faithfulness / relevance / coherence; gaps to the same-α reference | Per α, in-/out-of-support | ✅ (valid-α-region summary ⏳) |
| **Specificity** | Off-target ratio | Change in length and FKGL per unit on-target change | 🔄 (length only) |
| **Agreement** (diagnostic) | Slope-sign agreement; slope dispersion; product cosine to local directions | Per-client slope of score on α; cos(B^D A^D, B_i^D A^D) | ⏳ |

**Reporting:**
- mean and **worst client** ✅; median ⏳;
- **per-client paired bootstrap over articles** for method comparisons (same test articles, same α scale) 🔄, done ad hoc so far and to be moved into a script;
- **3 training seeds** for the main table ⏳ (single seed so far, so intervals cover article sampling only).
- Show steering and quality at the same α; never pair the best α of one metric with the best α of another.

**Three evaluation settings, one per claim:**

- **E1: Participants (C1).** 🔄 Each client's co-trained model. Compare federated vs. local per client, in-support and out-of-support, at data budgets 2k / 4k / natural ✅/🔄 and later the low-budget curve 64 / 256 ⏳.
  - **Controlled coverage experiment ⏳:** remove the top or bottom 40% of α from balanced clients (forbes.com, wsj.com, aol.com), retrain, and test at the removed α against their real held-out summaries. This measures extrapolation directly instead of relying on nypost.com alone.
- **E2: Held-out clients (C2).** ⏳ The new client fine-tunes Pⱼ by plain SFT with no D and no α. Then attach the frozen D:
  - **shared calibration:** k = 0, use the shared g as is;
  - **private calibration:** fit the gain/offset/warp on k ∈ {16, 64} labelled examples.

  Compare with a local direction and with few-shot prompting at the same k on the same Pⱼ; 3 rotations.
- **E3: Drift (C3).** ⏳ After federation, each participant keeps fine-tuning Pᵢ on its later-year drift stream with plain, label-free SFT and D frozen. Evaluate after 1, 2 and 3 drift stages: as is, and (private calibration only) after refitting on k = 16 examples.

---

## 5. Baselines and ablations

All methods share the backbone, the client data, the α labels, the adapter placement and the total trainable rank per client.

| ID | Baseline | Claim it tests | Status |
|---|---|---|---|
| **B1** | **Prompting** on the client's fine-tuned model: numeric-level instruction plus k few-shot examples at target percentiles | Is a learned knob needed at all? (gate G1) | ⏳ **next; premise of the paper** |
| **B2** | **Local-only**: same model and loss, each client trains its own D | C1, C2 | ✅ 2k; 🔄 4k (exp16); 🔄 4k without offset (exp17) |
| **B3** | **One-shot merged direction**: average of B2's local directions | Is iterative federated training needed? (gate G3) | 🔄 Saved by every local run; not yet evaluated on Newsroom |
| **B4** | **Federated activation steering** (CAA [4]): per-client mean-difference vectors, averaged, with a fitted gain | Weight vs. activation space | ⏳ |
| **B5** | **Pooled reference**: all clients' data centralized | Cost of decentralization | ⏳ |

**Ablations:**
- **A1:** gain fixed at 1 (`fed.fix_gain`) ⏳.
- **A2:** shared adapter (`fed.adapter: shared`) or no adapter (`none`). ⏳ Now also a test of whether the private adapter blocks extrapolation for copy-heavy clients.
- **A3:** PFL-structured conditional SFT: shared and private LoRA with α as a *text control token* (FedDPA / FedSA-LoRA structure [8, 9]). ⏳ This answers "isn't this just PFL?".
- **Calibration** (new): private vs. **shared** vs. shared without offset 🔄 (exp17). `none` (g = α) is not planned for now.
- **α protocol** (new): global (method) vs. local ✅ (entries 7–9).
- **Regularizers** (new): bundle tested ✅ (it hurt); a proper ablation is ⏳: no gain/offset priors, no cosine decay, stronger weight decay and dropout.
- **Endpoint-only** (optional): train on each client's bottom and top quartiles, test intermediate α. This links to the original REIN claim.

**Not included, with the reason (for the rebuttal):** full PFL algorithms (no control coordinate; structure covered by A3); CWS [3] and task arithmetic (post-hoc extraction is covered by B3 and B4); hypernetwork or contextual steering (a different claim).

---

## 6. Experiments and expected figures

| Table/Figure | Content | Status |
|---|---|---|
| **Table 1 (main)** | E1: federated vs. B1–B5, steering and quality, mean / worst client; key Amazon rows | 🔄 (B2 only so far) |
| **Fig. 1** | Control curves: output percentile vs. α for clients with different supports, the support band shaded | 🔄 (data exists) |
| **Fig. 2 (C1, coverage)** | Per client: federated − local difference with paired CIs, in- vs. out-of-support; plus the controlled coverage experiment | 🔄 / ⏳ |
| **Fig. 3 (C1, data)** | Steering vs. data budget, federated vs. local | 🔄 (2k, 4k) |
| **Fig. 4 (C2)** | Held-out clients: metrics vs. k (0 with shared calibration) for frozen D, local direction, prompting | ⏳ |
| **Fig. 5 (C3)** | Retention across drift stages | ⏳ |
| **Table 2** | Ablations: calibration, adapter, α protocol, A1, A3, regularization | 🔄 |
| **Fig. 6** | Quality vs. α for generated vs. real summaries (AlignScore, judge); length off-target | ✅ (data exists) |
| **Fig. 7 (diagnostic)** | Learned calibration curves g(α) per client; coefficient ranges used by each client; agreement diagnostics | 🔄 |

**Backbone:**
- Development and all runs so far use **Llama-3.2-1B-Instruct**: cached, and about 6–7 h per 100-round run including evaluation.
- **Final tables:** Qwen3-4B-Instruct-2507 [10] (not yet downloaded; budget about 4× the time). Repeat Table 1's top rows on one 7–8B model only if the main result holds.

**Cost:**
- One 100-round 1B run at 4k ≈ 3 h training + 2 h in-training dev evaluation + ~1 h test, quality and judge.
- 3 GPUs on `dali` (shared with other users).
- Every job runs from its own code snapshot (`runs/_code/<stamp>/`).

---

## 7. Gates and next steps

| Gate | Criterion | Status |
|---|---|---|
| **G0** | Publications differ materially in density range | ✅ Passed (median density 1.1 → 22.6) |
| **G1** | The method beats prompting (B1) on calibration at matched quality | ⏳ **Untested; highest priority.** If prompting is as good, the premise fails |
| **G2** | The method beats local-only (B2) | 🔄 2k: better on 6/8 clients, worse on cbc.ca and nypost.com (copy-heavy). 4k comparison running |
| **G3** | One-shot merging (B3) does not already match the method | ⏳ (evaluation only; files exist) |
| **G4** | Frozen D (+ calibration) beats a local direction at k ≤ 64 on held-out clients | ⏳ |

**If a gate fails:**
- G1 fails: the paper's premise fails; stop.
- G2 fails broadly but G4 holds: reframe around portability.
- G3 fails: simplify to one-shot merging; make the paper about the evaluation protocol and portability.
- G2 holds only for some clients: report it as a coverage result (federation helps where a client's support is missing), and use the calibration/adapter ablations to explain the exceptions.

**Next steps, in order:**
1. B1 prompting and B3 merged: evaluation only, can run while training jobs queue.
2. Analyse exp16/exp17 (local 4k; shared calibration ± offset) and pick the calibration design.
3. E2 held-out clients (zero-shot with shared calibration).
4. Controlled coverage experiment and adapter ablation (A2).
5. E3 drift.
6. FKGL / specificity and agreement diagnostics; the bootstrap script; quality-aware selection.
7. Seeds (3) and the Qwen3-4B backbone for the final tables; Amazon key rows; human audit.

---

## 8. Positioning

- **vs. PFL** (FedRep [11], FedDPA [8], FedSA-LoRA [9]): same shared/private split, but a different shared object (a control direction, not task features) and a different success criterion (control and quality metrics, not per-client accuracy). The structural overlap is tested directly by A3.
- **vs. weight/activation steering** (CWS [3], CAA [4]): those learn a direction on one model. We learn it across many privately fine-tuned models with skewed data and test whether it transfers. Directions transferred between models [12] and steering undone by fine-tuning [13] are the closest findings to C2 and C3.
- **vs. personalized steering** (BiPO [14], SteerX [15]): those personalize the vector per user. We share the direction and, with shared calibration, the α scale; only house style (the adapter) is private.
- **Prior federated steering:** a published patent application covers aggregating activation steering vectors [16]. Do not claim to be the first federated steering method. Claim the evaluation protocol and the three findings.

**Working title:** *Learning Steering Directions Across Clients: Federated Training and a Protocol for Direction Quality*

**Claim sentence (only if the results support it):** "Clients whose data covers different parts of an attribute's range can learn a single steering direction by federated training. It lets each client reach attribute levels its own data barely contains, with summary quality on par with real examples at those levels; it transfers to new clients without labelled examples; and it survives continued private fine-tuning."

---

## Changelog

| Date | Change | Evidence (`exp_log/EXPERIMENT_LOG.md`) |
|---|---|---|
| 09-29 | Plan created (local-percentile α, private gain, 2k pairs) | — |
| 09-29 | Gate G0 passed; Newsroom data built (12 clients, 3 rotations, temporal drift split) | entries 2–3 |
| 09-29 | Calibration metric → percentile error; Spearman counts no-effect articles; order rate de-emphasized (ties) | entries 4–5 |
| 09-29 | Memorization found; in-training dev monitor and checkpoint sweep (about 30–60 rounds suffice at 2k) | entry 6 |
| 09-29 | Private α warp (A/B/C) and post-hoc remap (F); offset; adapter modes | entry 7 |
| 09-29 | **α protocol: local → global** (equal-weight mixture CDF); motivation reframed to **coverage** ("skews differ"); in-/out-of-support metrics, reach rate | entries 7, 9 |
| 09-30 | Federated vs. local (2k): the Guardian reaches copying it lacks locally; nypost.com doesn't reach rewriting | entries 9, 15 |
| 09-30 | Quality metrics (AlignScore, BERTScore, LLM judge, same-α reference); full dev eval inside training; per-component regularizers | entry 10 |
| 09-30 | Data size × regularization: 4k halves the dev-loss rise and improves steering; the regularizer bundle hurts | entry 11 |
| 09-30 | nypost.com summaries are truncated leads in the raw data (kept; treated as a format convention) | entry 12 |
| 09-30 | Near-tie metric | entry 13 |
| 09-30 | Jobs run from code snapshots | entry 14 |
| 09-30 | Per-client paired comparisons; 4k effect; local-4k baseline queued | entries 15–16 |
| 10-01 | **Calibration shared vs. private**; offset optional (exp17 running) | entry 17 |
| 10-01 | Plan updated to the above | — |

---

## References

1. Grusky, Naaman, Artzi. *Newsroom: A Dataset of 1.3 Million Summaries with Diverse Extractive Strategies.* NAACL 2018. https://aclanthology.org/N18-1065/
2. Hou et al. *Bridging Language and Items for Retrieval and Recommendation* (Amazon Reviews 2023). 2024. https://arxiv.org/abs/2403.03952
3. Fierro and Roger. *Steering Language Models with Weight Arithmetic* (CWS). ICLR 2026. https://arxiv.org/abs/2511.05408
4. Rimsky et al. *Steering Llama 2 via Contrastive Activation Addition.* ACL 2024. https://aclanthology.org/2024.acl-long.828/
5. Sun et al. *Improving LoRA in Privacy-preserving Federated Learning* (FFA-LoRA). ICLR 2024. https://arxiv.org/abs/2403.12313
6. Wu et al. *AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders.* 2025. https://arxiv.org/abs/2501.17148
7. Zha et al. *AlignScore: Evaluating Factual Consistency with a Unified Alignment Function.* ACL 2023. https://arxiv.org/abs/2305.16739
8. Yang et al. *Dual-Personalizing Adapter for Federated Foundation Models* (FedDPA). NeurIPS 2024. https://github.com/Lydia-yang/FedDPA
9. Guo et al. *Selective Aggregation for Low-Rank Adaptation in Federated Learning* (FedSA-LoRA). ICLR 2025. https://github.com/Pengxin-Guo/FedSA-LoRA
10. Qwen3-4B-Instruct-2507 model card. https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
11. Collins et al. *Exploiting Shared Representations for Personalized Federated Learning* (FedRep). ICML 2021. https://arxiv.org/abs/2102.07078
12. *Activation Space Interventions Can Be Transferred Between Large Language Models.* 2025. https://arxiv.org/abs/2503.04429
13. *Does Fine-Tuning Undo Activation Steering? Behavioural Recovery Without Weight-Edit Reversal.* 2026. https://arxiv.org/abs/2608.24988
14. Cao et al. *Personalized Steering of LLMs: Versatile Steering Vectors Through Bi-directional Preference Optimization* (BiPO). NeurIPS 2024. https://arxiv.org/abs/2406.00045
15. *SteerX: Disentangled Steering for LLM Personalization.* 2025. https://arxiv.org/abs/2510.22256
16. *Federated Learning with Steering Vectors.* US20260195607A1. https://patents.justia.com/patent/20260195607
17. Song et al. *FineSurE: Fine-grained Summarization Evaluation using LLMs.* ACL 2024. https://aclanthology.org/2024.acl-long.51/
18. Ladhak et al. *Faithful or Extractive? On Mitigating the Faithfulness-Abstractiveness Trade-off in Abstractive Summarization.* ACL 2022. https://aclanthology.org/2022.acl-long.100/

Verify before submission: Newsroom access terms, the arXiv IDs of refs 5, 7 and 11 (cited from memory), and a final targeted search for federated steering-direction work published after September 2026.
