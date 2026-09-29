# Shared Behavioral Control Across Personalized Language Models

Research plan · Literature checked September 29, 2026

**Recommended starting point:** jointly learn a shared weight direction and private client adapters, test creativity control and resistance to sycophancy first, and use selective refusal as a third, targeted robustness experiment. The central contribution should be transferable control under local calibration differences and continued personalization. Linear interpolation is the starting parameterization, not the novelty claim.

All experiment sizes, thresholds, and schedules below are proposed starting configurations, not established optimal settings. Results from prior papers motivate the hypotheses; they do not establish that this method will outperform its baselines.

**1. Motivation and intended contribution**

Organizations increasingly adapt the same open-weight model to their own work. A creative agency, educational publisher, and product-design team might each train on local examples, conventions, and feedback. They can agree on the direction “more original” while disagreeing about how original an acceptable response should be. They may also lack enough local examples to learn a reliable control mechanism independently.

This creates three coupled problems:

- Pooling supervision can obscure differences in what local endpoint labels mean.
- Independent control training cannot exploit evidence from other clients.
- Further local training changes the model on which a previously learned intervention acts.

The proposed question is: **Can clients jointly learn a reusable direction of behavioral change while retaining private, trainable models and locally meaningful control scales?**

Fine-tuning is part of the chosen application pipeline: clients repeatedly learn domain conventions, accepted examples, or new task skills. Do not claim that all personalization requires weight updates. Demonstrate that the selected local adaptation improves a useful task metric over the starting model and include a strong prompting alternative.

The relevant literature already contains contrastive weight steering, global/private adapters, shared low-rank factors, and federated activation-vector aggregation [R1, R6–R10, R16]. The plausible gap is their intersection: continuous shared behavioral control, heterogeneous local supervision, evolving private models, and transfer to unseen clients. A novelty claim still needs a final targeted literature check before submission.

The paper should establish:

1. A benchmark that distinguishes domain heterogeneity from differences in attribute scales and meanings.
2. A simple jointly trained shared/private control model, with a measured boundary of applicability.
3. A benefit from collaboration: better data efficiency or transferable control, beyond private fine-tuning and scalar calibration.
4. Retention or recovery of control during further local adaptation.
5. One credible two-attribute experiment.

Do not promise a universally portable direction, guaranteed monotonicity, formal privacy, or guaranteed improvement over prompting.

**2. Formal problem and assumptions**

There are M clients. They start from the same pretrained checkpoint W_0, architecture, tokenizer, and parameter ordering. Each client i has:

- Private task-adaptation examples A_i.
- Private control supervision C_i, initially examples at two local endpoints.
- A private update P_i and positive steering gain s_i.
- A local attribute rubric r_i(x,y) and a separate task-quality measure u_i(x,y).

The deployed family is

\[
W_i(\alpha)=W_0+P_i+s_i\alpha D,\qquad \alpha\in[0,1],\quad s_i>0.
\]

W_0 is frozen in the practical implementation. P_i changes effective model weights through a private LoRA adapter; the shared D is a separate trainable low-rank weight update. This is persistent weight adaptation even though the original checkpoint tensor stays frozen.

The server stores and updates D. Clients retain P_i, s_i, data, and optimizer states for their private parameters. Only shared-parameter updates or gradients are communicated. This describes data-local training; it does not imply differential privacy.

**Interpretation of alpha.** Initially, alpha is a relative control coordinate between the client's locally defined endpoints. Equal alpha need not imply equal absolute property levels across clients. Order, numerical calibration, and absolute cross-client equality are different claims. Endpoint-only training primarily supports testing order and interpolation; it does not supply arbitrary numerical calibration.

If an externally anchored score exists, define a desired local target

\[
\tau_i(\alpha)=a_i+(b_i-a_i)\alpha.
\]

The endpoint anchors a_i,b_i come from a fixed rubric or independent calibration data, never from the evaluated model's own minimum and maximum. Otherwise, a method with a collapsed range could look perfectly normalized.

**Scope of heterogeneity.** Start with heterogeneous data and private updates under a common initialization. Sharing a raw direction across unrelated checkpoints or architectures is a separate alignment problem.

**A necessary data condition.** Every training client should have both endpoints, preferably paired on the same input. If a client is observed only at alpha_i, then its contribution s_i alpha_i D can be absorbed into P_i. Without within-client variation or additional restrictions, the intended decomposition is not identified. “Each client provides only one polar class” is a substantially harder setting and should not be the initial benchmark.

**Scale ambiguity.** The transformation D -> cD, s_i -> s_i/c preserves the model. Use bounded positive gains and regularization during optimization, then a common norm convention when reporting or comparing directions. Do not interpret an unnormalized learned s_i as an intrinsic measure of client preference.

**3. Hypotheses and what would falsify them**

| Hypothesis | Critical comparison | Evidence against it |
|---|---|---|
| Shared control reduces local supervision needs | Shared D versus independent D_i at several local data budgets | No benefit for clients with little data |
| Private bases and gains handle scale heterogeneity | Full method versus no private adaptation and versus s_i=1 | Gains do not improve calibration; private adaptation alone explains everything |
| Joint learning matters | Jointly updated D versus frozen initialization and periodic local re-extraction | Cheap re-extraction performs equally well at comparable total cost |
| Shared control transfers | Held-out clients with frozen D | Requires effectively relearning full local directions |
| One direction is often sufficient | One direction versus a small shared basis and local directions | Strong negative transfer even under simple scale-only heterogeneity |
| The method supports compositional control | Joint two-attribute model versus independently learned axes | One knob controls both properties or combinations collapse |

The single-direction hypothesis may fail while a low-dimensional shared-subspace hypothesis succeeds. Decide this empirically before introducing a large hypernetwork or router.

**4. Tasks and datasets**

The task portfolio should combine a meaningful controllable trade-off with established behavioral robustness tests. New publication dates alone do not make a benchmark compelling; the supervision, adaptation, and evaluation must expose the proposed problem.

**4.1 Recommended flagship: controllable originality under local adaptation**

Recent weight-steering evidence exists: CreativityNeuro (July 2026; ICML workshop, not an ICML main-track paper) studies divergent thinking and reports transfer across DAT, AUT, and Task Task, as well as reduced output repetition. It modifies selected weights using contrastive prompt-derived importance scores rather than the endpoint-fine-tuning recipe. Its reported gains in novelty also come with utility trade-offs [R2].

This is a particularly relevant setting because different teams can reasonably want different degrees and kinds of novelty. A routine educational activity and an experimental entertainment concept should not require the same creative distance from established practice.

Use the following resources:

| Resource | Role | Proposed use and limitation |
|---|---|---|
| Infinity-Chat / Artificial Hivemind, NeurIPS 2025 [R3] | Main prompt pool and open-ended evaluation | Select ideation and creative-generation prompts, partition by verified domain/task tags, and evaluate quality and semantic repetition. Its human ratings are not automatically attribute-intensity labels. |
| Alternative Uses Test, as evaluated in CreativityNeuro [R2] | Direct comparability with weight-steering work | Reproduce the published small prompt set, then add a separately labeled object-held-out extension. Do not call an expanded set the original benchmark. |
| Task Task, as evaluated in CreativityNeuro [R2] | Open-ended challenge generation | Reproduce the published protocol, then create held-out constraints and prompt families for the client experiment. |
| Divergent Association Task [R2] | Cheap diagnostic only | Useful for reproduction and initial checks, but insufficient as the main evidence of useful generation. |

Infinity-Chat contains approximately 26K open-ended queries and a human-annotated subset [R3]. Inspect the release before assigning annotators as clients: stable identities and enough repeated judgments per annotator are required. Otherwise use transparent, simulated client rubrics.

**Proposed client construction.** Start with four manually verified groups such as product ideation, educational activities, narrative concepts, and entertainment challenges. These are proposed study categories, not a claim that they exactly match released dataset labels. For each group define three locally anchored novelty ranges: incremental, moderate, and exploratory. All ranges must preserve a common minimum of relevance, coherence, and constraint satisfaction.

The controlled property is originality; usefulness remains a separately reported quality constraint. A response can be unusual because it is nonsensical, so novelty alone is not success.

**Training-data construction.**

1. Split underlying prompts and semantic families before generating responses.
2. Generate a fixed candidate pool per training prompt using multiple instruction variants and sampling settings.
3. Independently rate relevance, feasibility/utility, and originality.
4. Within the quality-qualified pool, select lower- and higher-originality responses according to each client's rubric.
5. Match response length and presentation where feasible, so the direction cannot exploit verbosity alone.
6. Reserve a blind human audit; do not use the same automatic judge as both the sole selector and final evaluator.

Candidate generation is a one-time data cost supplied equally to training baselines. The complete client-labeled dataset is a new benchmark construction; existing resources do not provide it ready-made.

**Natural local training.** P_i learns newly accepted examples, domain vocabulary, output conventions, and task-specific constraints from time-ordered batches. Preserve the definition of the control endpoints in the main temporal experiment; change the rubric only in a separate drift experiment.

**Primary outcomes.** Originality at matched usefulness, useful control range, local target attainment, within-model repetition, and preservation of client-specific task competence. Measure diversity at matched temperature and output count. Temperature/top-p tuning is mandatory as a comparator for this task.

**Risk.** This is the best fit to useful continuous control, but the recent direct weight-steering evidence is preliminary and human evaluation is costly. Reproduce a meaningful quality–originality trade-off early. Do not commit the whole paper to an automatic creativity score.

**4.2 Second core task: selective resistance to sycophancy**

Contrastive Weight Steering (CWS), ICLR 2026, provides a direct behavioral weight-steering precedent and accompanying code/data [R1]. Use its released factual sycophancy evaluation for comparability, and SycoBench-600 (ACL Findings 2026) for a recent external test [R4].

SycoBench-600 measures reactions to social pressure and distinguishes accepting correct corrections from following incorrect suggestions. Its 600 instances share 272 normalized stems [R4]. Treat stems, their paraphrases, and all conversational variants as one split group.

| Resource | Role |
|---|---|
| cfierro/sycophancy_eval_answer [R1-code] | Factual answer-changing under user influence; direct CWS reproduction |
| SycoBench-600 [R4] | Locked external test of correction selectivity |
| Independently constructed domain QA pairs | Main client training pool; include neutral prompts, true suggestions, false suggestions, and evidence-backed corrections |
| CWS GCD example [R1-code] | Optional controlled mechanism check; not the flagship task |

**Control definition.** Sweep resistance to unsupported user pressure and measure whether genuine corrections are still accepted. Do not describe factual incorrectness or an intermediate amount of sycophancy as a desirable personalized target. Here, the control family is useful for finding a robust operating point and testing behavioral generalization.

**Heterogeneity.** Vary topic, social-pressure wording, reliability of user suggestions, and amount of local control data. Ground-truth correctness remains global. Client-specific labels must not redefine a false answer as true.

**Persistent adaptation.** Clients learn domain QA and revisions in sequential batches. Use ordinary adaptation as the main condition. A deliberately biased stream where user suggestions are always correct can be a separate stress test; label it synthetic.

**Metrics.** Neutral accuracy, accuracy after misleading pressure, harmful answer-change rate, correct-correction acceptance, incorrect-correction acceptance, and utility at each alpha. A model that refuses to change any answer should not score well.

**Interpretive limit.** This task provides strong behavioral evidence but a weaker story for “all intermediate settings are desirable.” Keep that distinction explicit.

**4.3 Third task: selective refusal after benign specialization**

Use a harmful/benign evaluation pair so the model cannot win by refusing everything.

| Resource | Role |
|---|---|
| GSM-Danger and DirectHarm4 from the CWS repository [R1-code] | Direct reproduction tests after task-specific adaptation |
| OR-Bench / OR-Bench-Hard and its harmful comparison set, ICML 2025 [R5] | Measure unnecessary refusal alongside harmful compliance |
| Disjoint benign task-adaptation data | Produce realistic local specialization and a utility metric |
| Separately authored ambiguous requests with explicit rubrics | Optional client-specific caution range; new data, not existing ground truth |

The original CWS work evaluates restoration of refusals after task fine-tuning [R1]. Your extension tests whether a shared control remains useful across differently specialized clients.

Define local heterogeneity in handling genuinely ambiguous or context-sensitive requests. Keep clearly harmful and clearly benign audit sets common to all clients. A blanket refusal rate is not a quality metric.

Report harmful compliance versus benign completion, task performance after specialization, clarification frequency on ambiguous requests, and the usable operating region. Different local caution levels should not silently become different definitions of correctness on the common audit.

**Do not make induced “evilness” or emergent misalignment generation a main task.** They are possible diagnostic evaluations in prior work, but add a less natural continuous-control motivation and substantial evaluation complexity.

**4.4 Minimal task commitment**

Run creativity and sycophancy as the two main families. Add refusal once the training and evaluation pipeline works, using the same architecture and optimizer. If creativity cannot be evaluated credibly or its effect disappears under strong sampling controls, use sycophancy plus selective refusal and frame the paper around transferable behavioral calibration rather than claiming universally useful intermediate attributes.

**5. Worker heterogeneity: controlled design**

Heterogeneity is part of the scientific contribution. Manipulate factors separately before combining them.

| Setting | What changes | What stays fixed | Purpose |
|---|---|---|---|
| H0: approximately IID | Random client allocation | Rubrics, endpoint ranges, adaptation amount | Sanity check |
| H1: domain shift | Topics/task distributions | Attribute rubric and endpoint anchors | Ordinary non-IID generalization |
| H2: scale/range shift | Local low/high intensity ranges | Input distribution and attribute meaning | Direct test of s_i and P_i |
| H3: domain plus scale | H1 and H2 together | Common attribute orientation | Main realistic setting |
| H4: interpretation shift | Clients emphasize different components of an attribute | Broad attribute label | Stress test for a shared subspace |
| H5: data imbalance | Number of control pairs and adaptation examples | Other factors | Benefit to clients with little data |
| H6: temporal weight drift | Sequential local adaptation | Control rubric initially fixed | Stability and recovery |
| H7: participation variation | Some clients skip rounds | Data and model architecture | Small systems robustness appendix |

For creativity, H4 could contrast conceptual novelty with unusual wording or recombination. These are not necessarily the same attribute; disagreement is an expected failure mode for rank-one sharing. For sycophancy, varying suggestion reliability is distinct from varying endpoint intensity.

**Concrete starting split.** Four domains times three endpoint-range settings gives 12 client profiles. Train on two range profiles per domain (8 clients), reserve the third per domain (4 clients) for held-out-client transfer, and rotate which range is held out. Run a separate leave-one-domain-out test; do not conflate a new range with a new domain.

Proposed endpoint data budgets per client: 32, 128, and 512 paired inputs, with both endpoints for each. A realistic imbalance condition can mix these budgets while holding the total fixed. Add roughly 1K–2K disjoint local adaptation examples per client if the underlying pool supports it. Reduce sizes rather than duplicate prompts across train and test.

Construct low/moderate/high heterogeneity through documented rubric ranges and verified domain allocations. A Dirichlet label split can be a supplementary comparison, but cannot substitute for explicit differences in endpoint meaning.

**Missing endpoints.** Add only as an optional difficult condition after the paired setting works. Include a small local anchor set or shared restrictions and acknowledge the identifiability issue.

**6. Proposed algorithm and initialization**

**6.1 Starting objective**

For paired endpoint examples (x,y_i^0,y_i^1), use supervised control training:

\[
\mathcal L_{\mathrm{ctl},i}
=\tfrac12\mathbb E[
-\log p_{W_i(0)}(y_i^0\mid x)
-\log p_{W_i(1)}(y_i^1\mid x)].
\]

Add local task adaptation at a predeclared reference control alpha_ref:

\[
\mathcal L_{\mathrm{task},i}
=\mathbb E_{(x,y)\sim A_i}
[-\log p_{W_i(\alpha_{\mathrm{ref}})}(y\mid x)].
\]

The initial optimization is

\[
\min_{D,\{P_i,u_i\}}\sum_i p_i
[\mathcal L_{\mathrm{ctl},i}
+\lambda\mathcal L_{\mathrm{task},i}]
+\beta_D\|D\|_F^2+\beta_s\sum_i u_i^2,
\qquad s_i=\exp(u_i).
\]

Start with alpha_ref=0.5 and uniformly weighted clients, p_i=1/M. Clamp gains to a declared finite range such as [1/4,4] as an engineering starting point and report sensitivity. The losses do not guarantee intermediate behavior; that is evaluated explicitly.

Use task examples compatible with the reference setting. If utility is checked or trained across multiple alpha values, use control-invariant targets or answer-token masks. Requiring the exact same full stylistic response at every alpha would suppress the intended control.

Begin with SFT because it isolates the representation and sharing question. If reliable preference pairs are available, add a DPO variant later with the same conditional model and reference-policy definition. Do not change architecture, training loss, and data simultaneously.

**6.2 Obtain the initial controllable model without requiring fine-grained labels**

1. Start every client from the same W_0 and the same adapter layout.
2. Train a small private P_i on the client's task data, with D=0. Retain this checkpoint as the independent-personalization reference.
3. Initialize shared low-rank D with one random factor and one zero factor. Set s_i=1.
4. Warm up D on paired endpoint data for a few rounds, initially fixing P_i and s_i.
5. Unfreeze private P_i and s_i, and alternate their adaptation with shared-direction updates.

Both LoRA factors must not be initialized to zero. Hold gains fixed while D is zero or extremely small; otherwise gain learning is uninformative.

This is a practical initialization recipe, not a claim that a separate Phase I is theoretically necessary. Include a from-scratch joint-training ablation. An optional public-data CWS initialization can be tested, but every relevant baseline must get the same public data and its cost must be counted.

**6.3 Communication protocol**

Use a simple synchronous protocol first:

1. The server broadcasts the current shared factors defining D.
2. Each selected client updates P_i and s_i for E local minibatches while D is fixed.
3. Clients compute gradients of the control/task objective with respect to the same broadcast shared factors, at their current private states.
4. The server averages these shared-factor gradients with the declared client weighting and takes an optimizer step.
5. Clients retain their private states for the next round.

This is alternating personalized training with a shared gradient update. It is preferable for the first scientific study because it avoids ambiguity from averaging independently evolved LoRA factorizations. Test E=1 and a moderate E such as 5 or 10; keep processed tokens and communication visible.

When implementing a weight matrix, use two additive branches:

\[
h\mapsto W_0h+B_i^P A_i^P h+s_i\alpha B^D A^D h.
\]

The scalar multiplies the completed shared update once. Multiplying both A^D and B^D by alpha produces alpha-squared behavior.

If later using multi-step local optimization of D, distinguish factor averaging from averaging effective weight updates:

\[
(\sum_i p_iB_i)(\sum_i p_iA_i)
\ne \sum_i p_i B_iA_i.
\]

Use an explicitly specified product-space aggregation/compression rule or synchronized-factor protocol. Do not attribute aggregation artifacts to failure of the steering hypothesis.

**6.4 Normalization and evaluation**

Bounds and regularization discourage gain/direction blow-up but do not establish a unique semantic decomposition. For comparisons, normalize D to a chosen Frobenius norm and inversely rescale gains so predictions remain unchanged. If renormalizing during training, transport gains and optimizer states consistently, including clients that missed rounds. Avoid adding this complexity to the first implementation.

**6.5 Escalation if one direction fails**

Use a small shared basis:

\[
W_i(\alpha)=W_0+P_i+\alpha\sum_{r=1}^{R}c_{ir}D_r,
\qquad R\in\{2,4\}.
\]

Private c_i permits different combinations of shared components. This relaxes the common-direction assumption; it is not evidence of independent semantic factors. Compare against increasing the LoRA rank of a single D so gains are not merely extra capacity.

A scalar nonlinear calibration g_i(alpha) is appropriate only for a monotone but distorted control curve. It cannot repair an incorrect direction, and fitting it needs calibration information beyond two endpoint labels unless its shape is otherwise constrained.

**7. Baselines: complete direct comparison map**

The table separates direct controls, existing algorithms, and adapted versions. An adapted method must be named as such. Not every method belongs in every experiment.

| ID | Baseline | Exact comparison and priority |
|---|---|---|
| B0 | Personalized model without a control mechanism | P_i trained on the same task stream; report its operating point. Mandatory sanity check. |
| B1 | Personalized prompting and few-shot control | Same task-personalized model; local rubric, numerical target or descriptive instruction, and the same calibration examples. Tune templates on validation. Mandatory. |
| B2 | Control-conditioned SFT | Learn a numeric/text control token in a local adapter from exactly the same endpoint data. Tests whether weight interpolation is needed. Mandatory. |
| B3 | Independent local weight directions | W_0+P_i+s_i alpha D_i, same loss and total local rank. Tests sharing. Mandatory. |
| B4 | Post-hoc contrastive weight steering | Independently fit endpoint updates, form their difference, and apply it after local adaptation. Use local and aggregated variants. Direct CWS comparison [R1]. |
| B5 | Frozen shared direction | Learn initial D, freeze it, continue training private P_i,s_i. REIN-inspired comparison isolating shared direction updates; do not label it an exact REIN reproduction. Mandatory. |
| B6 | Periodically refreshed local directions | Refit/re-extract after each temporal stage using the same available control data. Charge its additional training cost. Important if claiming persistent adaptation benefits. |
| B7 | Fully shared control model | Shared base adapter plus shared D, no private P_i; with and without private gain. Tests whether private weights matter. Mandatory ablation. |
| B8 | FedDPA with control conditioning | Use global/private LoRA structure, retain its prescribed mixing, and add the same control instruction to training/inference. Its original mixing coefficient is not an attribute coordinate [R6]. Closest named structural baseline. |
| B9 | FedSA-LoRA with control conditioning | Share A and retain B_i; provide the same control input. Additional explicit steering variant W_0+P_i+s_i alpha B_iA must be labeled our adaptation [R7]. Closest shared-subspace competitor. |
| B10 | Activation steering, extracted | Local CAA and a shared aggregation of local contrastive directions with client calibration. Report extraction on each current local model [R8]. Cheap and relevant. |
| B11 | Activation steering, jointly trained | Shared v_l plus private P_i,s_i, trained with the same endpoint loss. Stronger fair comparison than extracted vectors alone. Mandatory for a weight-versus-activation claim. |
| B12 | Centralized pooled-data reference | Same conditional architecture and client identities with pooled training access. Reference for decentralization cost, not a guaranteed upper bound. |
| B13 | Independent endpoint models and interpolation | W_i(alpha)=(1-alpha)W_i^-+alpha W_i^+. Distinguish independent endpoint training from B3 joint-path training. Useful direct baseline; overlaps with B4 if origins are matched. |

Do not multiply near-identical variants unnecessarily. If two implementations are algebraically and procedurally identical, report one and explain the equivalence.

**Task-specific additions**

- Creativity: reproduce CreativityNeuro [R2], both as a direction derived from W_0 and as a locally recomputed intervention after adaptation. Its mask-based update need not be low rank, so report its actual storage and communication separately. Also compare validation-tuned temperature/top-p; add best-of-N selection only if the proposed system uses comparable generation/evaluator budgets or makes a stronger efficiency claim.
- Sycophancy/refusal: use CWS and prompting with task-appropriate rubrics. Include joint task-plus-behavior SFT as a fixed-operating-point reference; it cannot be credited with a continuous control curve unless made conditional.
- Preference-learning version: FedPDPO (2026 preprint) and FedPrism (IJCAI 2026) become direct competitors when the main task is federated preference optimization [R10,R11]. They are related work rather than mandatory full reproductions for a strictly paired-SFT control study.
- Multi-objective version: Panacea and, if using objective-weight preferences, COS-DPO are relevant conditional controls [R12,R13]. HoE is a stronger expert-routing comparator if the contribution expands to general multi-objective alignment [R14].
- Broad natural-language or context-conditioned steering: CLAS or HyperSteer is relevant if claiming those capabilities [R18,R19]. Neither is necessary merely to make a scalar-control baseline look modern.

**Priority for the first full table:** B1, B2, B3, B4, B5, B8, B9, B11, and the proposed method. B0, B7, B10, and B12 can share ablation/reference panels. For creativity, include the task-specific sampling and CreativityNeuro comparisons.

This is more than a two-baseline study, but most core comparisons reuse the same adapter training code. Avoid a claim of superiority over all federated alignment or all steering methods; that would unnecessarily expand the baseline burden.

**8. Evaluation protocols for the global direction**

D has no standalone behavioral score. Evaluate it as an intervention on a defined population of local models.

**E1: Participating clients.** Evaluate each jointly trained P_i,s_i,D combination. This measures end-to-end success including co-adaptation.

**E2: Held-out client, independent base.** Build P_j with task adaptation that never used D, freeze it, and allow only a small calibration of s_j. Compare directions on the exact same P_j. This is the strongest plug-in transfer test.

**E3: Held-out client, limited co-adaptation.** Freeze D but train P_j,s_j with 16/64/256 endpoint pairs. Compare against learning D_j from scratch and conditional prompting/SFT at the same budget. This measures whether D is a reusable training prior even if E2 fails.

**E4: Continued local training.** Freeze D, apply three disjoint local adaptation stages, and evaluate before any repair, after scalar-only recalibration, and after the next shared training round. These distinguish retained control, gain correction, and full recovery.

**E5: Common public diagnostic panel.** Send the same non-training prompts to all clients and collect behavior scores. The evaluator can aggregate metrics without accessing private P_i. Public prompts used for loss or model selection must not later be called a held-out test.

**E6: Bare-base probe.** W_0+alpha D is optional. Failure there does not refute a method trained for personalized bases. Avoid using it as the sole assessment of global quality.

**Control metrics**

1. Sweep alpha at 0, 0.25, 0.5, 0.75, 1 in the pilot; use 11 values for final control curves if useful.
2. Report property level and task quality together, not separate best-case numbers from different coefficients.
3. Report reversal magnitude or pairwise ordering violations, and require nontrivial endpoint separation. A constant-output model must not count as successful monotone control.
4. Report useful range: the portion of independently anchored property levels reachable while satisfying a predeclared quality constraint.
5. Use target MAE only for cardinal targets supported by a rubric. For ordinal labels use ordering and target-bin attainment.
6. Inspect per-prompt distributions. A mean intermediate score may conceal a mixture of extreme outputs.
7. Report mean, median, and bottom-quartile client results; record negative transfer versus each client's local-only baseline.
8. For transfer, report data needed to reach a fixed quality-qualified target and the complete learning curve, not only the best final point.

Define a quality-qualified control region:

\[
\mathcal A_i^{\rm valid}
=\{\alpha:\ U_i(\alpha)\ge U_i^{\rm ref}-\epsilon,\
\text{task-specific constraints hold}\}.
\]

Choose U_i^ref and epsilon before test evaluation, using a common reference independent of the compared method. For noncontiguous valid regions report attained bins or the full curve; do not imply every value between the minimum and maximum is usable.

For creativity, measure repetition over multiple samples of the same prompt and cross-client similarity on shared prompts. Different outputs caused only by different topics are not evidence that the intervention mitigates mode collapse.

**9. Multi-attribute experiment**

Begin with two named axes:

\[
W_i(\boldsymbol\alpha)
=W_0+P_i+s_{i1}\alpha_1D_1+s_{i2}\alpha_2D_2.
\]

A directly grounded robustness pair is resistance to unsupported pressure plus refusal conservatism. Both have weight-steering precedent, but useful operating regions should preserve truthful correction and benign assistance. Do not assume every point in the two-dimensional rectangle is desirable.

Use two tests:

- A union of separate task probes, to test whether changing one knob damages the other skill.
- A smaller, explicitly constructed set of joint prompts where social pressure and ambiguity both occur. A union alone does not establish compositional control on the same input.

Provide some jointly labeled combinations during training in the main experiment. A harder compositional split can hold out selected combinations; label it as such. Endpoint-axis examples alone do not justify a claim that all corners or interior combinations are learnable.

Sweep a 5-by-5 grid on one model and a reduced client set. Measure own-axis response, cross-axis effects, joint target attainment, and task quality. Compare:

1. Joint two-axis training.
2. Combining two independently trained directions.
3. A two-control conditional prompt/SFT model.
4. A shared basis if the single-axis sharing hypothesis failed.

Orthogonality of D_1 and D_2 in weight space does not establish independent behavior. The relevant measurement is the functional response matrix, estimated by finite differences of each attribute score with respect to each knob.

An application-facing extension could combine creative originality with critical evaluation of a user's suggestion. Treat this as new task construction with weaker direct evidence, not a requirement for the first paper.

**10. Fairness, leakage control, and statistical design**

**Same information.** Baselines get the same task data, endpoint pairs, rubric information, and calibration examples. Prompting should operate on a comparably personalized model. A natural-language rubric unavailable to one method but supplied to another changes the information regime and must be reported.

**Two validation regimes.** Main endpoint-only study selects checkpoints using endpoint performance and independent utility validation. A separately labeled calibration-assisted regime can use a small intermediate-level validation set, supplied equally to baselines. Do not silently use intermediate supervision to support an endpoint-only claim.

**Capacity and cost.** Report local and shared trainable parameter counts separately. Compare both equal local capacity and matched communication budgets when ranks differ. Count actual uplink/downlink bytes, model broadcasts, generation and judge calls, and all re-extraction costs. Show control quality against training cost and communication, rather than forcing every method to match every resource simultaneously.

**Seeds and splits.** Use at least three training seeds for final core comparisons and multiple held-out-client assignments. Start with one or two seeds during screening. Bootstrap by client and underlying prompt/stem, retaining repeated alpha evaluations and repeated samples within their cluster. Generated completions of the same prompt are not independent training/test units.

**Human validation.** Start with about 200–300 blinded response pairs across methods, control levels, and clients, with three independent ratings per pair. Use pilot disagreement and effect variance to set the final sample size. Ask separately about target property, usefulness, and constraint satisfaction; randomize order and hide method identity. Do not use a small human study merely to endorse an unrelated automatic score.

**Metric robustness.** Use an independent evaluator and a second scoring approach on a subset. For creativity, test whether ranking changes when controlling length, temperature, and formatting. For sycophancy, exact correctness and suggestion polarity should dominate stylistic judgments.

**Dataset boundaries.** Respect official splits when available. For prompt pools without suitable splits, publish a new grouped split. Keep near-duplicates, paraphrases, source documents, objects/challenge families, and conversational variants together. Existing pretrained-model exposure is not eliminated by a new split; new held-out prompt variants help but do not prove zero contamination.

**11. Models and concrete experiment budget**

Use Qwen3-4B-Instruct-2507 for the initial practical study; the official card identifies it as a non-thinking 4B model [R20]. Use Qwen2.5-7B-Instruct as a direct bridge to existing weight-steering experiments [R21]. These are explicit reproducibility choices, not a claim that they are the newest or strongest models available.

Run a separate federation for each backbone. Do not share D between model families. If resources permit, add a second-family checkpoint only for the strongest comparisons after verifying its availability and behavior.

Suggested adapter starting ranks: private rank 8 or 16, shared rank 8 or 16, applied to a fixed set of attention/MLP projections. Match the effective total rank for local-only controls. Low-rank results should be described as such; they do not automatically extend to arbitrarily divergent full fine-tunes.

| Stage | Scope | Decision |
|---|---|---|
| Pilot | One backbone, four clients, one main task, 128 pairs/client, five alpha values | Does useful controllability exist beyond prompting/sampling? |
| Core | Eight train clients plus four held-out profiles, two tasks, main baselines | Is collaboration useful under domain-plus-scale heterogeneity? |
| Mechanism | Scale-only, no gain, frozen D, private-only, shared-basis comparisons | What explains the benefit? |
| Temporal | Three adaptation stages, before/after repair | Does sharing help with evolving local weights? |
| Confirmation | Second backbone, strongest 3–4 baselines | Is the result specific to one model? |
| Extension | Refusal and a small two-axis grid | Does the finding extend beyond the flagship? |

Avoid the full Cartesian product of every dataset, model, heterogeneity setting, seed, and baseline. Select core configurations before seeing final test outcomes.

For example, 8 clients times 50 test prompts times 5 control values times 4 samples already yields 8,000 generations per method for one creative evaluation. Control inference and annotation costs by using a small diagnostic panel during training and reserving the full evaluation for selected checkpoints.

**12. Figures and tables that should carry the paper**

1. Main control curves: target attribute versus quality, with client-level uncertainty.
2. Heterogeneity plot: performance under domain-only, scale-only, and combined shifts.
3. Held-out-client learning curves: shared D versus learning a direction locally.
4. Temporal plot: control before adaptation, after drift, after gain calibration, and after shared updates.
5. Two-attribute response heatmap and cross-interference matrix.
6. Cost table: private/shared parameters, bytes, training tokens, and inference/judge cost.
7. Failure examples: incompatible local meanings and cases where a shared subspace helps or still fails.

A useful mechanism diagnostic is alignment of client gradients with respect to the shared effective update, together with behavioral transfer between clients. Raw cosine similarity between arbitrary LoRA factors is not meaningful because factorization has symmetries.

**13. Optional theory that matches the actual method**

Prioritize small, interpretable statements over a generic nonconvex convergence theorem.

- Prove the one-control-value confounding observation from Section 2.
- State the first-order compatibility condition: if q_i(W) is a differentiable expected property score, then the local steering slope is s_i times the inner product of its weight gradient with D. A common improving direction must lie in the intersection of the clients' positive-slope half-spaces. Positive gains cannot repair a sign conflict.
- Under an explicitly local smoothness assumption, relate changes in P_i to changes in that slope. This clarifies why preserving D does not guarantee preserving behavior.
- For a shared basis, show how the feasible local directions expand; do not claim that larger expressivity guarantees better generalization.

These observations provide testable diagnostics: gradient conflict, slope reversals, and recovery under a shared basis. They do not establish the full nonlinear generation behavior globally.

**14. Go/no-go criteria and scope decisions**

Continue with the main hypothesis if the pilot shows a usable control range and the core study shows at least one practically meaningful collaboration benefit: less local data, better held-out-client adaptation, or faster recovery after local drift, at matched quality.

Reconsider the claim when:

- Strong personalized prompting matches the method throughout its useful range.
- Benefits disappear after matching local parameter counts or supervision.
- Only participants improve and held-out clients require a new direction anyway.
- A scalar calibrated post-hoc direction performs as well as joint training.
- Creativity gains are explained by longer outputs, temperature, or lower usefulness.
- Single-direction sharing fails even when only attribute intensity varies.

If a small shared basis fixes only interpretation heterogeneity, the paper can explain a meaningful boundary: scale differences admit shared directions; meaning differences require shared subspaces. If neither direction nor basis sharing helps, do not add a large router simply to preserve the original hypothesis.

**15. Suggested execution sequence**

1. Reproduce one creativity-steering result and one sycophancy control curve; audit data and evaluator behavior.
2. Construct the grouped client benchmark, including fixed external rubrics and held-out client profiles.
3. Implement shared/private additive LoRA, local-only control, conditioned SFT, and frozen-D variants.
4. Run the four-client pilot with strong prompting and task-specific cheap controls.
5. Add the named personalized baselines and learned activation comparison.
6. Run the core heterogeneity and transfer experiments.
7. Run temporal adaptation, one two-attribute experiment, and confirm the strongest finding on the second backbone.
8. Complete blinded human evaluation, cost accounting, and failure analysis.

A practical initial allocation is two weeks for reproduction/data/implementation, two to three weeks for core experiments, and one to two weeks for confirmation and evaluation, subject to available GPUs and annotators. Measure actual throughput before committing to that schedule.

**16. Recommended paper framing**

Working title: **Learning Shared Behavioral Control Across Personalized Language Models**

Potential claim, only if supported: “Clients can learn a shared weight-space control mechanism from locally defined endpoint supervision, retaining private task adaptation and improving the data efficiency of control on new clients.”

The strongest story has three linked results: the shared component is useful, local calibration is necessary, and ongoing private training creates a measurable challenge that joint learning or efficient recovery addresses. The method can remain simple if those results are convincing.

**References and implementation starting points**

The sources below support the literature statements. Client constructions, algorithm choices, evaluation protocols, and experiment sizes above are proposals.

- **[R1]** Fierro and Roger. *Steering Language Models with Weight Arithmetic*. ICLR 2026. [Paper](https://arxiv.org/abs/2511.05408); [official proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/df59090e951681e0f98d40f131c4b628-Abstract-Conference.html).
- **[R1-code]** [Author code and dataset links](https://github.com/safety-research/weight-steering); [factual sycophancy data](https://huggingface.co/datasets/cfierro/sycophancy_eval_answer); [GSM-Danger](https://huggingface.co/datasets/vfleaking/GSM-Danger); [DirectHarm4](https://huggingface.co/datasets/vfleaking/DirectHarm4).
- **[R2]** Schapiro et al. *CreativityNeuro: Steering Language Model Weights to Improve Divergent Thinking and Reduce Mode Collapse*. July 2026, ICML Workshop on Creativity & Generative AI per arXiv metadata. [Paper](https://arxiv.org/abs/2607.01433). Reproduction code availability was not verified in this review; the paper provides algorithm and task prompts.
- **[R3]** Jiang et al. *Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)*. NeurIPS 2025 Datasets and Benchmarks. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/754d5a526a5ee5a47220664a0eb92751-Abstract-Datasets_and_Benchmarks_Track.html).
- **[R4]** Sinha. *SycoBench-600: Measuring Sycophancy and Correction Selectivity in LLM Assistants*. ACL Findings 2026. [Paper](https://aclanthology.org/2026.findings-acl.1759/); [data](https://huggingface.co/datasets/dsinha/sycobench-600); [code](https://github.com/debu-sinha/sycobench-600).
- **[R5]** Cui et al. *OR-Bench: An Over-Refusal Benchmark for Large Language Models*. ICML 2025. [Paper](https://proceedings.mlr.press/v267/cui25a.html); [code and release links](https://github.com/justincui03/or-bench).
- **[R6]** Yang et al. *Dual-Personalizing Adapter for Federated Foundation Models*. NeurIPS 2024. [Paper](https://proceedings.nips.cc/paper_files/paper/2024/hash/45a30141c6719e9cfedfb51f1c665a37-Abstract-Conference.html); [code](https://github.com/Lydia-yang/FedDPA).
- **[R7]** Guo et al. *Selective Aggregation for Low-Rank Adaptation in Federated Learning*. ICLR 2025. [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f53a37f820d5be5930415d964f4a0187-Abstract-Conference.html); [code](https://github.com/Pengxin-Guo/FedSA-LoRA).
- **[R8]** Rimsky et al. *Steering Llama 2 via Contrastive Activation Addition*. ACL 2024. [Paper](https://aclanthology.org/2024.acl-long.828/).
- **[R9]** Yi et al. *pFedLoRA: Model-Heterogeneous Personalized Federated Learning with LoRA Tuning*. 2023 preprint. [Paper](https://arxiv.org/abs/2310.13283). Architectural predecessor; its original experiments are not an LLM continuous-steering benchmark.
- **[R10]** Zhu et al. *FedPDPO: Federated Personalized Direct Preference Optimization for Large Language Model Alignment*. March 2026 preprint, under review in retrieved metadata. [Paper](https://arxiv.org/abs/2603.19741).
- **[R11]** Wang et al. *Beyond Client Clustering: Fine-Grained Preference Alignment in Federated RLHF via Self-Evolving Routing*. IJCAI 2026. [Paper](https://www.ijcai.org/proceedings/2026/552).
- **[R12]** Zhong et al. *Panacea: Pareto Alignment via Preference Adaptation for LLMs*. [Paper](https://arxiv.org/abs/2402.02030).
- **[R13]** Ren et al. *COS-DPO: Conditioned One-Shot Multi-Objective Fine-Tuning Framework*. UAI 2025. [Paper](https://proceedings.mlr.press/v286/ren25a.html).
- **[R14]** Li et al. *Multi-objective Large Language Model Alignment with Hierarchical Experts*. ICLR 2026. [Paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d191ba4c8923ed8fd8935b7c98658b5f-Abstract-Conference.html).
- **[R15]** *Letting Tutor Personas Speak Up for LLMs: Learning Steering Vectors from Dialogue via Preference Optimization*. BEA 2026. [Paper](https://aclanthology.org/2026.bea-1.7/). Relevant shared activation direction and personal-strength precedent; distinct from evolving private weight bases.
- **[R16]** *Federated Learning with Steering Vectors*, published application US20260195607A1. [Disclosure text](https://patents.justia.com/patent/20260195607). Technical prior on aggregating activation-derived steering information, not an empirical comparison. Avoid a “first federated steering” claim.
- **[R17]** *Does Fine-Tuning Undo Activation Steering? Behavioural Recovery Without Weight-Edit Reversal*. August 2026. [Paper](https://arxiv.org/abs/2608.24988). Relevant to preservation of behavioral effects under further training.
- **[R18]** Hsu et al. *Contextual Linear Activation Steering of Language Models*. April 2026 preprint. [Paper](https://arxiv.org/abs/2604.24693).
- **[R19]** Sun et al. *HyperSteer: Activation Steering at Scale with Hypernetworks*. June 2025 preprint. [Paper](https://arxiv.org/abs/2506.03292).
- **[R20]** [Official Qwen3-4B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
- **[R21]** [Official Qwen2.5-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

**Evidence boundary:** No retrieved source establishes the exact proposed combination or guarantees its performance. In particular, existing weight-steering results do not prove that one direction can satisfy heterogeneous endpoint definitions after private adaptation. That is the experiment this project must answer.
