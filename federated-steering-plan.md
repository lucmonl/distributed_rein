# Federated Learning of Steering Directions

Research plan · created 29 September 2026 · **updated 1 October 2026, evening** (after exp-log entries 19–23; see the changelog at the end and `exp_log/EXPERIMENT_LOG.md` for evidence). Supersedes the task and baseline sections of `shared-steering-research-plan.md`.

Status markers: ✅ done · 🔄 running / partial · ⏳ planned, not started · ⚠️ result currently against the claim.
Sizes, thresholds and schedules are working settings, not results.

---

## 1. Thesis and scope

**Question.** Can clients that each fine-tune their own copy of a model *jointly learn one steering direction* by federated training, when **each client's data is skewed toward a different part of the attribute's range**? Can a client then produce attribute levels its own data barely contains? Does the direction stay usable on clients that never took part in training? (Optional: and after further private fine-tuning?)

**What is new.** The object learned federatedly is a *control direction*, and the paper's main contribution is a protocol for judging the *quality of that direction*. Personalized federated learning (PFL) shares weights to improve per-client task accuracy at a fixed operating point. Here the shared weights exist to provide a continuous knob, and success is measured with control and quality metrics.

**Why federate at all** (vs. each client fine-tuning locally): the motivation is **coverage, not data volume**. A client can have thousands of examples and still almost none at the attribute levels it wants (e.g. a publication whose summaries are 87% copied leads has few rewritten ones). Other clients have plenty there. A shared α scale plus a shared direction lets that knowledge transfer. Data scarcity is the second story: a client that **joins** with the frozen direction needs far less data than one that learns its own (claim C2, measured as a curve over n).

**The claims the paper makes (and nothing else): C1 and C2 are the core; C3 is optional** (decided 10-01: the paper does not depend on it; run only if time allows, otherwise discuss as a limitation):

| # | Claim | Decisive comparison | Status |
|---|---|---|---|
| C1 | **Collaboration.** A federated direction steers better than one each client learns alone, above all in the parts of the α scale a client's own data does not cover. | Federated vs. local-only, per client, in-support vs. out-of-support α, at equal data | 🔄 **Supported as a coverage trade-off** (4k, entry 20). Federated (shared calibration, no offset) vs. local: overall better on 4/8, worse on 0/8; in-support better on 6/8. Out-of-support: much better for the clients with large gaps (theguardian −0.10, nypost −0.04), slightly worse at the extreme α for clients that already cover ≈ [0.05, 0.95] (4/8, +0.02–0.03). Beats one-shot merging (B3) on 7/8 **Replicates at 8B (Qwen3-8B, entry 37):** fed 0.156 vs. local 0.177, better on 6/8, worse on 1/8; same out-of-support trade-off. ⚠️ Local gains sit at the clamp (s ≤ 4) for 6–7/8 clients: rerun local with `lora.gain_max=16` before claiming C1 |
| C2 | **Portability.** A *new* client that joins with the frozen direction (and, with shared calibration, the frozen α→coefficient mapping), training only its private adapter, steers better from little data than if it had to learn its own direction. | `frozen_D` vs. `local_D` on held-out clients, same n pairs and steps, n ∈ {16, 64, 256, 1024, all} | ✅ **Supported on rotation 0** (entries 26, 30): frozen_D beats local_D on all 4 held-out clients at n = 16–1024 (0.179 / 0.170 / 0.146 / 0.139 vs. 0.295 / 0.260 / 0.225 / 0.192); the gap closes at full data (0.135 vs. 0.151). Equal quality; shared calibration costs nothing (same curve as private). ⏳ Coverage on new clients (α windows / rotation 2) |
| C3 *(optional)* | **Stability.** The direction keeps working while clients continue private fine-tuning. | Control metrics before and after drift; federated vs. local | ⚠️ **Fails under steering-off drift** for both federated and local (error 0.15 → 0.31 after one stage; entry 26): retraining the adapter without α moves the α = 0 anchor to the client's mean. Federated keeps more of the ordering (Spearman 0.70 vs. 0.47). ⏳ The realistic protocol (drift with steering on, α per pair) is untested |

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
- **Calibration mode** (`fed.calibration`): `private` (one g per client, fitted on its own data) or **`shared`** (one g for all clients, averaged by the server every round like D). Under the global α scale (below) there is no principled reason for per-client mappings, and the offset's original job (starting a client partway along D) is done by the scale itself. Shared calibration also lets a new client steer without fitting anything but its adapter.
  - ✅ **Result (exp17, entry 20): shared calibration without offset is the best federated design.** It is best on mean error (0.151 vs. 0.155 private), worst client (0.199 vs. 0.229), out-of-support error, Spearman and near-ties.
    - Shared calibration fixes nypost.com (−0.042).
    - Dropping the offset helps people.com, reuters.com and nypost.com and hurts nobody.
  - **Proposed default for the method: `calibration: shared`, `offset: false`** (the config default is still private + offset so that earlier runs stay reproducible; to switch once confirmed).
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
- Countermeasures: shared calibration ✅ (fixes nypost.com, entry 20); shared/no adapter (A2) ⏳. A support-aware calibration prior is the fallback.

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
  - **Caveat for E2:** rotation 0's held-out clients cover most of the global α scale (support: telegraph.co.uk 0.02–0.99, latimes.com 0.11–0.98, bbc.com 0.03–0.79, mashable.com 0.08–0.79). The narrow, skewed clients (theguardian.com 0.04–0.55, nypost.com 0.51–0.95) are participants; they are held out only in **rotation 2**. Rotation-0 E2 therefore tests data efficiency, but hardly coverage (see §4, E2).
- **Splits per client:**
  - train pool up to 5k pairs (reuters.com 4,023, bbc.com 4,228, theguardian.com 4,705, others 5,000);
  - dev 100; test 200 (official test split);
  - **drift 600** (3 stages of 200) from the latest year(s), with train/dev/test strictly earlier. That holds for 11/12 clients; aol.com falls back to a same-year split.
- **Articles** are truncated to 400 words, and pairs whose summary depends on the removed part are dropped.
- **Data budgets used:** 2k, **4k (current; all of entries 16–23)**. The natural-size run was only tried with the regularizer bundle and was cancelled (entry 19). The low-budget curve for *participants* (64 / 256) is ⏳; for *new clients* it is part of E2 (n = 16 … all).
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
| **House style** (private adapter) | Extractiveness-controlled publication attribution; style-feature gap | Classifier trained on density-matched real summaries; share of outputs attributed to the own publication (in-support α). 11 interpretable features vs. the client's real summaries at the same α (`scripts/style_eval.py`) | ✅ |
| **Agreement** (diagnostic) | Slope-sign agreement; slope dispersion; product cosine to local directions | Per-client slope of score on α; cos(B^D A^D, B_i^D A^D) | ⏳ |

**Reporting:**
- mean and **worst client** ✅; median ⏳;
- **per-client paired bootstrap over articles** for method comparisons (same test articles, same α scale) ✅ `scripts/compare_runs.py` (steering, ties, quality, judge, paired CIs; `name=run::file_pattern` selects baseline and setting files);
- ⏳ split out-of-support into **gap** regions (the client has essentially no data there) and **tail** regions (the 5% tails of a near-full support). The coverage claim is about gaps (entry 20);
- **3 training seeds** for the main table ⏳ (single seed so far, so intervals cover article sampling only).
- Show steering and quality at the same α; never pair the best α of one metric with the best α of another.

**Three evaluation settings, one per claim:**

- **E1: Participants (C1).** 🔄 Each client's co-trained model. Compare federated vs. local per client, in-support and out-of-support, at data budgets 2k / 4k / natural ✅/🔄 and later the low-budget curve 64 / 256 ⏳.
  - **Controlled coverage experiment ⏳:** remove the top or bottom 40% of α from balanced clients (forbes.com, wsj.com, aol.com), retrain, and test at the removed α against their real held-out summaries. This measures extrapolation directly instead of relying on nypost.com alone.
- **E2: Held-out clients (C2).** 🔄 Redesigned 10-01 (entry 23; `e2_heldout.py`).
  - **Setup:** a new client joins with its first n training pairs (nested subsets; n ∈ {16, 64, 256, 1024, all}). α comes from the run's global reference, which is free to compute, so the scarce resource is **data, not labels**.
  - **Settings:**
    - **`frozen_D` (main):** the participants' objective (α per pair, steering on) with D frozen. Trains Pⱼ from scratch, plus the calibration only if the run's calibration is private.
    - **`local_D` (baseline, no federation):** the same objective and steps, but the client trains its own D from zero.
    - **`plugin` (ablation):** Pⱼ by plain SFT with steering off, then D attached. This was the first design, and it gave error 0.31–0.34 at full data. Participants' adapters co-adapt with D; an adapter that never saw D does not.
    - **`prompt` (baseline):** level in the prompt plus 3 shots from the n pairs.
  - **Steps:** 4 epochs (the participants' budget), clipped to [200, 2000].
  - Run on the method run (all settings) and on the private-calibration run (`frozen_D`, `plugin`).
  - ⏳ **Coverage version:** rotation 0's held-out clients are broad (§3.1). Two options:
    - `--alpha_window LO HI` restricts a held-out client's pairs to part of the scale (e.g. α ≤ 0.4 or ≥ 0.6) and tests outside it. This is cheap and needs no new training.
    - A **rotation-2** federated run holds out theguardian.com, wsj.com, cbc.ca and nypost.com (natural skew; about 6.5 h of training).
- **E3: Drift (C3), optional** (not on the critical path; entry 26 already shows the failure mode, which the paper can report as a limitation). After federation, each participant keeps fine-tuning Pᵢ on its later-year drift stream (3 stages × 200 pairs, 100 steps each) with plain, label-free SFT and **steering off**. This is deliberate: the client fine-tunes without knowing about the knob.
  - Evaluate after each stage (100 test articles): as is, and after a k = 16 calibration refit (a branch; the drift chain continues un-refit).
  - Run on the method run (done) and its local counterpart (running). The comparison is federated vs. local retention.
  - Result ⚠️ (entry 26): steering-off drift breaks the knob for both runs (anchor shift).
  - If time allows ⏳: drift fine-tuning with D attached and α on (a client that keeps training under the joining protocol).

---

## 5. Baselines and ablations

All methods share the backbone, the client data, the α labels, the adapter placement and the total trainable rank per client.

**Baseline definition (decided 10-02):** the control baselines replace **only the knob**.
- B1 and B4 run on each client's own model (frozen base plus that client's private adapter from the federated run, B^D removed). Only the control mechanism differs from the method: prompt, or activation vector.
- So they keep the client's task adaptation and house style.
- B1 on the plain base model (`b1_prompt_k3_base`) is reported as a reference row, not as the baseline.

| ID | Baseline | Claim it tests | Status |
|---|---|---|---|
| **B1** | **Prompting** on the client's fine-tuned model (direction removed): numeric-level instruction, k ∈ {0, 3} few-shot examples nearest to the target α; also k = 3 on the base model | Is a learned knob needed at all? (gate G1) | ✅ **Fails to steer** (entry 22): error 0.34–0.38, Spearman ≤ 0.08; with k = 0, 70% of articles give near-identical text at α = 0 and 1. Method better on 8/8. ⏳ second template and Qwen3-4B check |
| **B2** | **Local-only**: same model and loss, each client trains its own D | C1, C2 | ✅ 2k, 4k, 4k without offset (entries 15, 20) |
| **B3** | **One-shot merged direction**: uniform average of B2's local directions; each client keeps its own adapter and calibration | Is iterative federated training needed? (gate G3) | ✅ **Worse than both local and federated** (entry 22): 4k error 0.184 vs. local 0.161 and method 0.151; method better on 7/8 (2k: 5/8, worse on none). ⏳ variant with the calibration refit to the merged direction (fairer) |
| **B4** | **Federated activation steering** (CAA [4]): per-client mean-difference vectors at the middle layer, averaged, with a per-client gain fitted on dev (training-free) | Weight vs. activation space | 🔄 Steers partly (error 0.263, Spearman 0.75; method better on 7/8, reuters.com a tie) but **hurts quality**: out-of-support AlignScore 0.21 below the same-α reference, +11 tokens. **Under-tuned**: 6/8 clients chose the largest gain (0.8) with dev error still falling. ⏳ wider gain grid, layer choice |
| **B4′** *(proposed)* | **Federated *learned* activation steering:** the method with B^D replaced by learned activation vectors (one per layer, or a low-rank ReFT-style intervention) scaled by the shared g(α); same objective, data, FedAvg and private adapter | Weight vs. activation space at equal training (CAA is the training-free version) | Not planned (user, 10-02): the existing B4 (CAA) stays |
| **B5** | **Pooled reference**: all clients' data centralized | Cost of decentralization | ⏳ |

**Ablations:**
- **A1 (calibration design, entries 38, 40, 41):** ✅ on 1B.
  - **Learned vs. constant:** g(α) = α gives 0.153 vs. 0.151 for the method; it is worse only on reuters.com (+0.018) and the worst client (0.217 vs. 0.199). The learned warp stays near the identity, so the method effectively learns a shared scale (s = 1.93); the scale-only arm learns the same (0.153).
  - **Shared vs. per-client (no offset):** shared 0.151 (worst 0.199) vs. per-client 0.168 (worst 0.320). Per-client gains are small for broad clients (−0.006 to −0.015) but fail for skewed ones (nypost.com +0.143, reuters.com +0.026), because a per-client calibration is fitted only on the client's own support.
  - **Design kept:** shared calibration, no offset; warp optional.
- **A2:** shared adapter (`fed.adapter: shared`): one global FedAvg model, nothing personalized; the **non-personalized FL baseline**. ✅ (entry 30) **A trade-off, not a loss:** overall 0.159 vs. 0.151 (worse in-support, 0.177 vs. 0.135), but better out-of-support on 6/8 clients (0.136 vs. 0.161), reach 0.50 vs. 0.39, best worst client (0.176). Private adapters give **no** dev-NLL benefit at their best round (1.081 vs. 1.078) and memorize after round ~50. ✅ **The private adapter is kept: it carries house style** (entry 31). Extractiveness-controlled publication attribution is 0.55 for the method (real summaries 0.59, A2 0.41) and the style-feature gap is 0.11 (A2 0.19). Flat across α, so orthogonal to the steered attribute. Federated ≈ local, so sharing D keeps the style. No adapter (`none`) is not planned: it confounds capacity with personalization.
- **A3 (optional):** PFL-structured conditional SFT: shared and private LoRA with α as a *text control token* (FedDPA / FedSA-LoRA structure [8, 9]). ⏳ It would strengthen the answer to "isn't this just PFL?", but A2 (non-personalized FedAvg) and the positioning argument carry that answer without it.
- **Calibration:** private vs. shared vs. **shared without offset** ✅ (entry 20; shared without offset is best). `none` (g = α) is not planned for now.
- **α protocol** (new): global (method) vs. local ✅ (entries 7–9).
- **Regularizers** (new): bundle tested ✅ (it hurt); a proper ablation is ⏳: no gain/offset priors, no cosine decay, stronger weight decay and dropout.
- **Endpoint-only** (optional): train on each client's bottom and top quartiles, test intermediate α. This links to the original REIN claim.

**Not included, with the reason (for the rebuttal):** full PFL algorithms (no control coordinate; the personalization question is covered by A2, optionally A3); CWS [3] and task arithmetic (post-hoc extraction is covered by B3 and B4); hypernetwork or contextual steering (a different claim).

---

## 6. Experiments and expected figures

| Table/Figure | Content | Status |
|---|---|---|
| **Table 1 (main)** | E1: federated vs. B1–B5, steering and quality, mean / worst client; key Amazon rows | 🔄 B1, B2, B3, B4 done at 4k (entry 22); B5 and a tuned B4 ⏳ |
| **Fig. 1** | Control curves: output percentile vs. α for clients with different supports, the support band shaded | 🔄 (data exists) |
| **Fig. 2 (C1, coverage)** | Per client: federated − local difference with paired CIs, in- vs. out-of-support; plus the controlled coverage experiment | 🔄 / ⏳ |
| **Fig. 3 (C1, data)** | Steering vs. data budget, federated vs. local | 🔄 (2k, 4k) |
| **Fig. 4 (C2)** | Held-out clients: error (in-/out-of-support) and quality vs. n for frozen_D, local_D, prompt, plugin; plus the coverage version | 🔄 jobs running |
| *Fig. 5 (C3, optional; appendix)* | Retention across drift stages, federated vs. local | ⚠️ / optional |
| **Table 2** | Ablations: calibration ✅, α protocol ✅, adapter (A2) 🔄, A1 ⏳, A3 (optional) ⏳, regularization (bundle only) | 🔄 |
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
| **G1** | The method beats prompting (B1) on calibration at matched quality | ✅ **Passed, provisionally**: prompting barely moves the output (Spearman ≤ 0.08). To confirm with a second template and on Qwen3-4B; the stronger input-side baseline A3 (level as a control token in training) is optional |
| **G2** | The method beats local-only (B2) | 🔄 **Partly**: at 4k, better on 4/8 overall and worse on none; a coverage trade-off out-of-support (entry 20). Report it per client with the gap/tail split |
| **G3** | One-shot merging (B3) does not already match the method | ✅ **Passed**: B3 is worse than local on 6/8 and than the method on 7/8 (entry 22) |
| **G4** | On held-out clients, `frozen_D` beats `local_D` at small n (≤ 256 pairs) | ✅ **Passed** on rotation 0, all 4 clients, every n (entry 26). ⏳ coverage version, step-sweep fairness check |

**If a gate fails:**
- G1 fails: the paper's premise fails; stop.
- G2 fails broadly but G4 holds: reframe around portability.
- G3 fails: simplify to one-shot merging; make the paper about the evaluation protocol and portability.
- G2 holds only for some clients: report it as a coverage result (federation helps where a client's support is missing), and use the calibration/adapter ablations to explain the exceptions.

**Next steps, in order (core):**
1. **Finish E2:** results at n = 1024 and all, the private-calibration run, and the LLM judge on `frozen_D` / `local_D` (entry 27).
2. **E2 coverage and fairness:** α-window runs on held-out clients (`frozen_D` vs. `local_D`); a step sweep for both at n = 64 / 256.
3. **A2 (shared adapter)** analysis; confirm the method design (shared calibration, no offset) and switch the config default.
4. **Rotation 2** federated run, then E1 per-client comparison and E2 on it (natural skew; second rotation).
5. **Baseline fairness:** B4 wider gain grid (and layer); B3 with a refit calibration; a second B1 template.
6. Longer training (150–200 rounds); the gap/tail split of out-of-support error; FKGL / specificity; B5 pooled.
7. Seeds (3) and the Qwen3-4B backbone for the final tables; Amazon key rows; human audit.

**Optional (only if time allows):** E3 with steering-on drift (C3); A3 (PFL-structured conditional SFT).

---

## 8. Positioning

- **vs. PFL** (FedRep [11], FedDPA [8], FedSA-LoRA [9]): same shared/private split, but a different shared object (a control direction, not task features) and a different success criterion (control and quality metrics, not per-client accuracy). The structural overlap is tested by A2 (one non-personalized FedAvg model) and, optionally, A3.
- **vs. weight/activation steering** (CWS [3], CAA [4]): those learn a direction on one model. We learn it across many privately fine-tuned models with skewed data and test whether it transfers. Directions transferred between models [12] and steering undone by fine-tuning [13] are the closest findings to C2 and (optional) C3.
- **vs. personalized steering** (BiPO [14], SteerX [15]): those personalize the vector per user. We share the direction and, with shared calibration, the α scale; only house style (the adapter) is private.
- **Prior federated steering:** a published patent application covers aggregating activation steering vectors [16]. Do not claim to be the first federated steering method. Claim the evaluation protocol and the three findings.

**Working title:** *Learning Steering Directions Across Clients: Federated Training and a Protocol for Direction Quality*

**Claim sentence (only if the results support it):** "Clients whose data covers different parts of an attribute's range can learn a single steering direction by federated training. It lets each client reach attribute levels its own data barely contains, with summary quality on par with real examples at those levels; a new client that joins with the frozen direction steers well from far less data than it would need to learn its own." (Add "and it survives continued private fine-tuning" only if the optional E3 supports it.)

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
| 10-01 | Plan updated to the above | entry 18 |
| 10-01 | Baselines implemented: B1 prompting, B3 one-shot merge, B4 federated CAA | entry 19 |
| 10-01 | **exp17 results:** shared calibration without offset is the best federated design; vs. local a coverage trade-off (C1 per client) | entry 20 |
| 10-01 | E2 / E3 implemented; `compare_runs.py` (paired bootstrap) | entries 20–21 |
| 10-01 | **Baseline results:** G1 passes provisionally (prompting fails to steer); G3 passes (merge worse than local and federated); B4 steers partly with a quality cost and needs tuning | entry 22 |
| 10-01 | A2 (shared adapter, non-personalized baseline) submitted | entry 25 |
| 10-04 | **Calibration design:** learned gain ≈ constant on average (helps the worst client); shared ≫ per-client for skewed clients | entries 40, 41 |
| 10-03 | Calibration-design ablation launched (constant / linear / private, 1B) | entry 38 |
| 10-03 | **C1 replicates at 8B** (Qwen3-8B, 6/8 better); local gains hit the clamp, so a wider-range local rerun is needed | entry 37 |
| 10-02 | **House-style metric:** private adapters carry publication style orthogonal to extractiveness; A2 loses it; private adapter kept | entry 31 |
| 10-02 | **E2 curve complete** (frozen D with 16 pairs beats local D with 1024); **A2: precision–coverage trade-off**, private adapters bring no NLL benefit; Qwen3-8B runs follow the 1B pattern so far | entry 30 |
| 10-01 | **E3 (C3) and A3 made optional**; LLM judge added to E2 | entry 27 |
| 10-01 | **E2 results:** frozen D gives large data-efficiency gains (G4 passed). **E3 results:** steering-off drift breaks the knob for both runs (anchor shift); E3 to be rerun with steering-on drift | entry 26 |
| 10-01 | **E2 redesigned:** the new client trains its adapter with D frozen and α on; data size n replaces "k labelled examples"; plugin and prompt kept as ablation and baseline; the held-out clients of rotation 0 are broad, so coverage needs α windows or rotation 2 | entry 23 |

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
