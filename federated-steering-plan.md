# Federated Learning of Steering Directions

Research plan · 29 September 2026 · supersedes the task and baseline sections of `shared-steering-research-plan.md`

All sizes, thresholds and schedules below are starting configurations to validate in the pilot. They are not results.

---

## 1. Thesis and scope

**Question.** Can clients that each fine-tune their own copy of a model *jointly learn one steering direction* by federated training, even though each defines the attribute's range differently? Does that direction stay usable on clients that never took part in training and after further private fine-tuning?

**What is new.** The object learned federatedly is a *control direction*, and the paper's main contribution is a protocol for judging the *quality of that direction*. Personalized federated learning (PFL) shares weights in order to improve per-client task accuracy at a fixed operating point. Here the shared weights exist to provide a continuous knob, and success is measured with control metrics.

**The three claims the paper makes (and nothing else):**

| # | Claim | Decisive comparison |
|---|---|---|
| C1 | **Collaboration.** A federated direction steers better than one each client learns alone, especially for clients with little data. | Federated D vs. local-only D_i, at several data budgets |
| C2 | **Portability.** The direction plugs into a *new* client's independently fine-tuned model, with only one scalar gain calibrated. | Frozen federated D + fitted gain vs. a local direction trained on the same k examples |
| C3 | **Stability.** The direction keeps working while clients continue private fine-tuning, and refitting the gain recovers calibration. | Control metrics before and after drift, with and without a gain refit |

**Explicitly out of scope.** Multi-attribute composition, formal privacy/DP, heterogeneous architectures, hypernetworks/routers, theory beyond one identifiability remark, and safety/sycophancy/creativity tasks.

---

## 2. Model: one loss, no phases

Each client i shares the frozen base W_0 and has a private adapter P_i and a private gain s_i. The server holds the shared direction D. For every adapted weight matrix:

\[
h \mapsto W_0h + B_i^{P}A_i^{P}h + s_i\,\alpha\,B^{D}A^{D}h, \qquad \alpha\in[0,1],\; s_i=\exp(u_i)\in[\tfrac14,4].
\]

- **A^D is a shared random matrix, generated from a common seed and frozen; only B^D is trained** (FFA-LoRA style [5]). This makes server averaging *exact*: \(\tfrac1M\sum_i B_i^D A^D = (\tfrac1M\sum_i B_i^D)A^D\). It removes the LoRA factor-averaging problem without any extra machinery. B^D starts at zero.
- α multiplies the finished product once. Scaling both factors by α would produce α² behavior.
- u_i is held at 0 for the first few rounds, until D is nonzero. This is an implementation detail, not a separate phase.

**Control coordinate = the client's local percentile.** Let a(y) be the attribute score of a target text, and F_i the empirical CDF of a(·) on client i's training data. Every training example is labeled with

\[
\alpha_i(y)=F_i\big(a(y)\big).
\]

So α = 0.9 means "as extractive as this client's 90th-percentile summary". Clients share the *direction* of the attribute, but α maps to different absolute levels on each client. That is exactly the heterogeneity s_i and P_i are meant to absorb, and it is **real rather than simulated** (see §3).

**Single local objective** (task adaptation and control are the same loss):

\[
\mathcal L_i(P_i,u_i,B^D)=\mathbb E_{(x,y)\sim S_i}\big[-\log p_{W_i(\alpha_i(y))}(y\mid x)\big].
\]

α varies within each client, so P_i cannot absorb the control signal. This is the identifiability condition from the previous plan, now satisfied automatically.

**Federated protocol (FedAvg on B^D):**
1. The server broadcasts B^D.
2. Each sampled client runs E local steps on (P_i, u_i, local copy of B^D).
3. The client uploads B_i^D. The server takes a uniform average (uniform weighting, so large clients don't dominate the direction).
4. Private states persist across rounds.

Defaults: E = 20, 8 clients per round (full participation), rank 16 for P_i and rank 16 for D, adapters on attention and MLP projections. Uplink per round is B^D only.

**Scale remark.** D → cD, s_i → s_i/c leaves the model unchanged. Before comparing directions, normalize ‖B^D A^D‖_F per layer and rescale the gains. This is the only theory the paper needs, plus the one-line observation that a client seen at a single α cannot identify D.

---

## 3. Tasks: fine-grained labels and deterministic or off-the-shelf scorers

### 3.1 Flagship: summary extractiveness on Newsroom [1]

- **Why:** 1.3M article–summary pairs written by the newsrooms of 38 publications. Each publication has its own mix of extractive and abstractive summaries, so clients genuinely disagree on what "extractive" means numerically. Extractiveness is scored **deterministically** with Grusky et al.'s fragment density (primary) and coverage.
- **Clients:** 12 publications with ≥ 5k pairs and clearly different density distributions. Use 8 as participants and 4 held out, with 3 rotations of the held-out set. Choose them in week 1 from the per-publication density quantiles.
- **Temporal split (for C3):** train on articles up to year Y, and use later years as the drift stream. Choose Y per client so that both parts are large enough.
- **Per client:** up to 2k training pairs (fewer in the data-budget study), 200 test articles, 200 drift-stream pairs.
- **Utility (must not reward the controlled attribute):** factual consistency with AlignScore [7], length-normalized. ROUGE is *not* a utility metric here, because it rewards copying. On top of that, a blinded audit of 200 samples judges relevance and fluency.
- **Off-target attributes:** summary length/compression and readability (FKGL). Both are deterministic.
- **Access risk:** Newsroom requires a download agreement. Secure access in week 1. If it is unavailable, the fallback is the same design on MACSum/CNN-DM (one domain, with simulated clients), which is weaker.

### 3.2 Secondary: review rating on Amazon Reviews 2023 [2]

- **Why:** continuity with the REIN draft's sentiment task, fine-grained 1–5 labels, and product categories as natural clients.
- **Clients:** 8 + 4 categories. Input: product title and description. Target: the review. α_i: the local percentile of the star rating (ties are spread uniformly within each star level).
- **Scorer:** expected star rating from a 5-class review classifier trained on *categories not used as clients* (avoids leakage). Also report an off-the-shelf rating model for robustness.
- **Utility:** product relevance (embedding similarity between review and description) and fluency. Off-target: length.

Only the key rows (§6, table 1) are run on this task. It shows generality; it does not carry the paper.

**Dropped from the previous plan:** creativity (no cheap reliable scorer outside AUT), sycophancy and refusal (no graded targets or desirable middle settings), and toxicity (Perspective API shuts down on 31 Dec 2026).

---

## 4. Evaluating the quality of a learned steering direction

This section is the core of the paper. A direction has no standalone score, so it is always evaluated as an intervention on specific client models. Evaluation uses test articles only. Gains may be fitted only on the stated calibration examples.

**Protocol.** For each client model and test input, generate at α ∈ {0, .25, .5, .75, 1} (11 values for final figures), using greedy decoding plus 4 samples at T = 0.7 on a subset. Score every output.

| Dimension | Metric | Definition |
|---|---|---|
| **Steerability** | Order rate; mean per-input Spearman ρ | Fraction of inputs whose scores increase monotonically across α (the draft's Correct-Order); rank correlation between α and score per input |
| **Range** | Normalized range | [a(α=1) − a(α=0)] ÷ the client's own interquartile range of a(·) in *training* data (a model-independent anchor) |
| **Calibration** | Target MAE | \(|a(\hat y_\alpha) - F_i^{-1}(\alpha)|\), i.e. distance to the client's α-quantile, averaged over α. The percentile definition of α provides this cardinal target at no extra cost. |
| **Utility retention** | Valid α-region | α values at which AlignScore ≥ (the client's α-free fine-tuned model) − ε, with ε fixed before testing |
| **Specificity** | Off-target ratio | Change in length and in FKGL per unit of on-target change, averaged over α. Lower is better. |
| **Agreement** (diagnostic) | Slope-sign agreement; slope dispersion; product cosine | Per-client slope g_i from a linear fit of score on α: the fraction of clients with g_i > 0 and the coefficient of variation of g_i. Also cos(B^D A^D, B_i^D A^D) against each client's local-only direction; this is well defined because it compares products, not factors. |

**Reporting:** the mean, median and **worst client**, with bootstrap intervals clustered by client and article; 3 training seeds for the main table. Show steerability and utility side by side at the same α. Never report the best α of one metric next to the best α of another.

**Three evaluation settings, one per claim:**

- **E1: Participants (C1).** Each client's co-trained (P_i, s_i, D). Repeat at local data budgets of 64 / 256 / 2k pairs per client, plus a mixed-budget setting in which half the clients have only 64 pairs.
- **E2: Held-out clients (C2).** The new client fine-tunes P_j by plain SFT with no D and no α. Then attach the frozen D and fit only u_j on k ∈ {0, 16, 64} labeled examples (k = 0 means s_j = 1). Compare with a local direction and with few-shot prompting, both using the same k examples on the same P_j.
- **E3: Drift (C3).** After federation ends, each participant keeps fine-tuning P_i on its later-year drift stream using **plain, label-free SFT** (the realistic case, where no one re-labels attributes), with D frozen. Evaluate after 1, 2 and 3 drift stages: (a) as is, (b) after refitting only u_i on k = 16 examples.

---

## 5. Baselines: five comparisons and three ablations

All methods share the backbone, the client data, the α labels, the adapter placement and the total trainable rank per client.

| ID | Baseline | Claim it tests | Notes |
|---|---|---|---|
| **B1** | **Prompting** on the client's fine-tuned model: numeric-level instruction plus k few-shot examples at the client's quantiles | Is a learned knob needed at all? | Tune templates on validation data. AxBench shows prompting can beat steering [6], so treat this as a real competitor. |
| **B2** | **Local-only direction**: the same model and loss, but B_i^D trained without federation | C1, C2 | The key comparison for collaboration |
| **B3** | **One-shot merged direction**: B2's local directions averaged once by the server (TIES as a variant) | Is iterative federated training needed, or is merging enough? | Run in the pilot. If it ties the full method, simplify the method. |
| **B4** | **Federated activation steering**: each client computes a CAA mean-difference vector [4] on its current model between its top and bottom quartiles; the server averages; each client fits a scalar gain | Weight-space vs. activation-space direction | Cheap. Recomputed after drift in E3. |
| **B5** | **Pooled reference**: the same model with all clients' data centralized | Cost of decentralization | A reference, not a competitor |

**Ablations (one panel):**
- **A1** s_i fixed at 1 (is the private gain needed?)
- **A2** No private P_i: a fully shared adapter plus D (is private adaptation needed?)
- **A3** **PFL-structured conditional SFT**: shared and private LoRA with the α value given as a *text control token* instead of the α-scaled branch, i.e. the FedDPA / FedSA-LoRA structure [8, 9] with conditioning. This single row answers "isn't this just PFL?". It shows whether the weight-space direction matters, compared with PFL's usual shared/private split conditioned on a prompt.

Optional, only if time remains: an **endpoint-only** variant that trains on each client's bottom and top quartiles only and tests intermediate α. This links back to the original REIN claim of steering without intermediate supervision.

**Not included, with the reason (for the rebuttal):** full PFL algorithms (they have no control coordinate, so their structure is covered by A3); CWS [3] and task arithmetic (post-hoc extraction is covered by B3 and B4); hypernetwork or contextual steering (a different claim).

---

## 6. Experiments and expected figures

| Table/Figure | Content | Runs |
|---|---|---|
| **Table 1 (main)** | E1 metrics for Full vs. B1–B5 on Newsroom, plus key rows on Amazon | 3 seeds |
| **Fig. 1** | Control curves: achieved score vs. α for three clients with different ranges, with each client's target quantiles overlaid | Main model |
| **Fig. 2 (C1)** | Steerability and calibration vs. local data budget: federated vs. local-only; the mixed-budget setting highlights clients with little data | E1 budgets |
| **Fig. 3 (C2)** | Held-out clients: metrics vs. k for frozen D + gain, local direction, prompting | 3 rotations |
| **Fig. 4 (C3)** | Retention across drift stages, as is vs. gain-refit, compared with B4 recomputed | E3 |
| **Table 2** | Ablations A1–A3 | 1–2 seeds |
| **Fig. 5 (diagnostic)** | Per-client slope agreement and product cosine to local directions; shows *why* sharing works or fails | From E1 |

**Backbone:** Qwen3-4B-Instruct-2507 [10] for everything. Repeat Table 1's top rows on one 7–8B model only if the main result holds.

**Cost check:** Newsroom summaries are short (tens of tokens). 12 clients × 200 articles × 5 α values ≈ 12k short generations per method per seed, and scoring is deterministic. Training is 8 clients × rank-16 adapters on a 4B model. The main compute is repeated seeds and rotations, not generation.

---

## 7. Schedule and go/no-go gates

| Week | Work | Gate |
|---|---|---|
| 1 | Newsroom access; per-publication density quantiles; choose 12 clients; implement scorers, AlignScore and FKGL | **G0:** publications differ materially in density range. If not, the heterogeneity story is weak; switch the flagship to Amazon. |
| 2 | Implement the model (shared frozen A^D, FedAvg on B^D) and B1/B2/B3; 4-client pilot at 256 pairs | **G1:** the full method beats prompting (B1) on calibration MAE at matched utility. **G2:** the full method beats local-only (B2) in the low-budget setting. **G3:** merging (B3) does *not* already match it. |
| 3–4 | Main E1 runs, 3 seeds; B4, B5; Amazon key rows | — |
| 5 | E2 portability (3 rotations) and E3 drift | **G4:** frozen D + gain beats a local direction at k ≤ 64 |
| 6 | Ablations, diagnostics, audit, writing | — |

**If a gate fails:**
- G1 fails (prompting is as good): the paper's premise fails; stop.
- G2 fails but G4 holds: reframe the paper around portability, i.e. a direction learned once transfers to new clients.
- G3 fails (merging ties): simplify the method to one-shot merging and make the paper about the evaluation protocol and portability.

---

## 8. Positioning

- **vs. PFL** (FedRep [11], FedDPA [8], FedSA-LoRA [9]): same shared/private split, but a different shared object (a control direction, not task features) and a different success criterion (control metrics, not per-client accuracy). The structural overlap is tested directly by A3.
- **vs. weight/activation steering** (CWS [3], CAA [4]): those learn a direction on one model. We learn it across many privately fine-tuned models and test whether it transfers. Directions transferred between models [12] and steering undone by fine-tuning [13] are the closest findings to C2 and C3.
- **vs. personalized steering** (BiPO [14], SteerX [15]): those personalize the vector per user. We share a direction and personalize only the gain and the base adapter.
- **Prior federated steering:** a published patent application covers aggregating activation steering vectors [16]. Do not claim to be the first federated steering method. Claim the evaluation protocol and the three findings.

**Working title:** *Learning Steering Directions Across Clients: Federated Training and a Protocol for Direction Quality*

**Claim sentence (only if the results support it):** "Clients that disagree on an attribute's scale can learn a single steering direction by federated training. It steers participants better than local training, transfers to new clients with one calibrated scalar, and survives continued private fine-tuning."

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

Verify before submission: Newsroom access terms, the arXiv IDs of refs 5, 7 and 11 (cited from memory), and a final targeted search for federated steering-direction work published after September 2026.
