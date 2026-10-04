**Introduction outline — learning shared control from distributed behavioral coverage**

Drafted against `federated-steering-plan.md` and `exp_log/EXPERIMENT_LOG.md`, including the October 3 results and corrections. This is an argument outline with candidate topic sentences, not finished introduction prose. The drafting notes at the end are internal and should not appear verbatim in the paper.

**Recommended central argument**

Organizations adapting a common language model can have substantial local datasets while lacking examples of particular behaviors they want to produce. Their datasets may collectively cover a much broader behavioral range. The research question is whether that complementary coverage can be distilled into a shared, continuously adjustable control direction that works with private client adapters and can be reused by new clients with little local data.

Make three properties central: reaching underrepresented attribute levels, retaining client-specific generation characteristics, and reusing the learned control mechanism. Frame federation as the setting that permits collaboration when examples remain local. Newsroom supplies a simulation with natural publication partitions; it does not establish that the participating publications actually require federated deployment.

**Paragraph 1 — Practical goal: locally adapted models need adjustable behavior.**

Candidate opening: “An organization that adapts a language model to its own data inherits the behavioral distribution of that data, even when its deployment requires a wider range of outputs.”

Introduce a concrete publishing scenario. A publisher wants summaries that retain its writing conventions while allowing editors to adjust how closely the wording follows the source: direct extraction for one use, more rewriting for another. A model that reproduces the publisher's typical summary does not necessarily provide that range on demand. The goal is an inference-time control value with predictable effects, while preserving useful task performance and client-specific characteristics.

Keep the opening centered on controllable language generation. Summary extractiveness is the running example. Mathematical solution length can become a second example if its accuracy-versus-control results support it; it is currently an extension under evaluation.

**Paragraph 2 — Why collaboration: local data can be abundant but behaviorally incomplete.**

Candidate topic sentence: “The difficulty is that a large local dataset can still provide little supervision for the attribute levels a client needs.”

Explain the difference between data quantity and attribute coverage. A publication may have thousands of examples concentrated around one extractive strategy. More samples from that same distribution need not supply the missing examples. Other clients may provide precisely those levels, creating an opportunity for collaboration across complementary distributions.

Use Newsroom as evidence of this premise: on the shared percentile scale, the Guardian's central training range is approximately [0.04, 0.55], whereas the New York Post's is approximately [0.51, 0.95]. These are 5th–95th percentile ranges, not hard support boundaries. Describe the latter as a format convention involving truncated article leads, rather than inferring an editorial preference.

End with the motivating question: “Can clients turn their complementary examples into a common control mechanism that expands each client's behavioral range while keeping their training examples and adaptation parameters local?”

**Paragraph 3 — Central tension: local specialization versus global behavioral coverage.**

Candidate topic sentence: “Local training preserves a client's generation conventions but limits supervision to its own behavioral range, whereas training a single global model draws on broader coverage but can weaken client-specific characteristics.”

Develop this tension as the reason for personalized federated training. Learning independently gives each client control over its adaptation, yet leaves it to learn underrepresented behaviors from little evidence. A single shared model can learn from the collective range, but its shared parameters must accommodate different writing and format conventions. Neither objective alone captures the desired outcome: broader behavioral control within a model that remains useful to its particular client. Examples remain local in the deployment setting; global training here can be implemented through federation and need not mean pooling raw data.

Personalized federated learning provides the organizing principle: share what clients can learn together while retaining what must specialize locally. Cite FedRep and FedSA-LoRA as precedents for this division. Then sharpen the research gap: the shared component in this work must support adjustable behavior across personalized models, including attribute levels poorly represented at an individual client. A shared/private architecture alone does not establish that capability. [FedRep](https://arxiv.org/abs/2102.07078); [FedSA-LoRA](https://arxiv.org/abs/2410.01463).

Suggested transition: “This tension motivates personalized federated learning of steering directions: clients collaborate to learn how behavior can be adjusted, while retaining private adaptation for their own generation requirements.”

Keep the steering literature concise here. Existing activation and weight steering motivate using a behavioral intervention as the shared object; personalized steering already studies intensity control and transfer across models/LoRAs. The question is how to learn such control collaboratively under uneven local coverage and establish its usefulness beyond the training clients. [CAA](https://aclanthology.org/2024.acl-long.828/); [contrastive weight steering](https://arxiv.org/abs/2511.05408); [BiPO](https://arxiv.org/abs/2406.00045).

**Paragraph 4 — Technical motivation: three requirements for useful shared steering.**

Candidate lead: “Realizing this form of collaboration requires more than dividing a model into shared and private components.”

| Challenge | Why it is nontrivial | Response developed in the introduction |
|---|---|---|
| **1. Expand behavioral coverage while preserving local specialization.** | The shared direction must learn from complementary client data and remain effective in each privately adapted model. Local adaptation can absorb a client's typical attribute level, while sharing all adaptation can weaken its generation conventions. Independently learned directions may also become incompatible with local models when merged. | Jointly learn the shared steering mechanism and private adaptation, so control is trained in the client models where it will be used. Assess both underrepresented behaviors and retention of client-specific characteristics. |
| **2. Translate requested attribute levels into effective intervention strengths.** | A direction identifies how to change behavior, but does not by itself specify how strongly to intervene at each requested level. The model's behavioral response need not vary linearly with intervention strength; a fixed scaling rule may allocate too little or too much intervention to parts of the requested range. | Learn a monotone nonlinear scaling function jointly with the direction. Treat its benefit over fixed and learned linear scaling as a hypothesis requiring direct ablation. |
| **3. Learn control that remains useful beyond the participating clients.** | Success on co-trained clients may depend on the particular private adaptations seen during federation. It does not establish that the shared mechanism captures reusable control, or that a new client can acquire it with little data. | Make usefulness on held-out clients a requirement of the work. Test whether freezing the learned control mechanism reduces the local data needed to obtain effective steering relative to learning control locally. |

The progression is compatibility during collaboration, effective control across requested levels, and usefulness beyond the training federation. Paragraphs 5 and 6 explain the design responses to the first two. Paragraph 7 introduces the third as an evaluation question and empirical contribution. Quality measurements support these claims; they are not a separate technical challenge.

**Paragraph 5 — Design protocol: jointly learn shared steering and private adaptation.**

Candidate topic sentence: “We address the local–global tension by learning a shared steering direction jointly with each client's private adaptation.”

Explain the division of responsibilities at the conceptual level. The shared direction draws on complementary behavioral examples across clients. Private adaptation retains client-specific generation characteristics. During training, each example is associated with its observed attribute level, and the shared direction is active as the client learns to generate that example. Clients periodically combine updates to the shared control mechanism while keeping their private adaptation local.

The key design choice is joint learning: the control mechanism is optimized in the presence of the personalized models it must influence. This gives a reason to compare against independently learning and subsequently merging directions, as well as against a single globally adapted model. Present preservation of specialization and expansion of coverage as goals whose trade-off is measured empirically.

Keep the introduction at this protocol level. Reserve the model equation, LoRA factorization, frozen basis, exact aggregation identity, and optimizer details for the methodology section. The percentile reference is part of the task and label definition there; alignment of local and global alpha semantics is not an introduction claim or contribution.

**Paragraph 6 — Design protocol: learn how strongly to steer.**

Candidate topic sentence: “Alongside the direction, we learn a nonlinear scaling function that maps a requested attribute level to an intervention strength.”

Explain why this is distinct from learning a direction. The direction specifies a way to change the model; the scaling function determines how far to move along it for different requests. A trainable monotone function permits uneven changes in intervention strength across the requested range while preserving their order. Its monotonicity constrains the intervention schedule, but does not guarantee monotone or accurately calibrated generated behavior.

The scaling function is learned with the shared direction and private adaptations, drawing on observations from the participating clients. Present this as a mechanism designed to accommodate nonlinear behavioral response. Do not claim it improves calibration over a trivial scaling rule until the ablation supports that claim. The completed shared-versus-private calibration comparisons answer a different question from whether nonlinear scaling is necessary.

Leave the function family, gain parameterization, offsets, bounds, and training schedules to the methodology section. The introduction needs the functional role and the reason for learning it.

**Paragraph 7 — Beyond participants: test whether collaboration produces reusable control.**

Candidate topic sentence: “A further test of the learned direction is whether it reduces the data a new client needs to acquire behavioral control.”

Return explicitly to challenge 3. The intended contribution extends beyond steering the models that participated in training: a held-out client should benefit from the learned mechanism without having to relearn that mechanism from its own limited examples. This distinguishes reusable control from a direction whose usefulness is confined to its co-trained clients.

Introduce the comparison at a high level: a held-out client reuses the frozen direction and scaling function while adapting to its own data, versus learning its control mechanism locally with the same local data and training budget. Measure the advantage over a range of local data sizes. Frame this paragraph as the portability question and evaluation contribution, with the answer previewed in paragraph 9 and summarized in the contributions.

Keep the detailed adaptation procedure in methodology or experimental setup. State only the necessary scope here: transfer allows private adaptation by the new client; it does not claim immediate compatibility with an arbitrary previously fine-tuned model.

**Paragraph 8 — Evaluation overview: test coverage, specialization, and control together.**

Candidate topic sentence: “We evaluate a steering direction by the behaviors it enables across client models and requested attribute levels.”

Introduce two principal settings: participating clients, which test the benefit of collaboration, and held-out clients with nested data budgets, which test portability and data efficiency. Evaluate outputs along the common alpha scale and distinguish levels inside and outside each client's central training range.

Preview the criteria that make the study more informative than one aggregate control score: percentile calibration error, ordering and near-ties, range and reach, summary quality relative to references at comparable attribute levels, and preservation of publication-specific characteristics. Motivation for matching extractiveness in quality evaluation has direct precedent. [Faithful or Extractive?](https://aclanthology.org/2022.acl-long.100/).

Keep this paragraph subordinate to the three challenges: these measurements establish whether the desired behavior was obtained while retaining useful generation and client characteristics. Do not elevate routine quality checks into an independent technical challenge or claim novelty for individual metrics. Keep severe coverage gaps conceptually separate from small tail regions; the formal gap/tail split is still pending.

**Paragraph 9 — Results preview: lead with the strongest current evidence.**

Use a short paragraph with two or three findings, not a catalog of every experiment:

- **New-client data efficiency:** across four held-out Newsroom clients, reusing the frozen direction with 16 local pairs gives mean percentile error 0.179, versus 0.192 when learning a local direction with 1,024 pairs. At equal budgets of 16–1,024 pairs, the frozen-direction condition has lower overall error for all four clients. These are results under the evaluated optimization settings; the participant training data is additional upstream supervision.
- **Participant collaboration:** current mean errors are 0.151 versus 0.165 for Llama-3.2-1B and 0.156 versus 0.177 for Qwen3-8B. Benefits are heterogeneous, with particularly large gains for some clients missing substantial portions of the attribute range. Treat these comparisons as provisional until the local gain-bound check is completed.
- **Preserved client characteristics:** extractiveness-controlled publication attribution is 0.552 for the private-adapter method, 0.553 for local training, and 0.409 for the shared-adapter ablation. This supports preservation of measured publication characteristics; it does not prove complete separation of style, content, and extractiveness.

Describe quality as broadly comparable on the measured automatic metrics and small judge sample. Do not turn that observation into an unconditional “no quality cost” claim. Small-data transfer produces longer summaries, and density itself correlates with length.

Evidence: experiment-log entries 20, 26, the October 2 “full E2 curve” entry 30, the “House style” entry 31, and the October 3 “C1 at 8B” entry 37. Some entry numbers are duplicated, so identify them by title when tracing results.

**Paragraph 10 — Contributions: four concrete deliverables.**

Candidate contribution wording:

1. **Problem formulation.** We study the tension between local specialization and collective behavioral coverage in controllable language generation, requiring a learned control mechanism to serve both participating and previously unseen clients.
2. **Personalized federated steering.** We develop a protocol that jointly learns shared behavioral control and private client adaptation, with a trainable nonlinear scaling function that maps requested attribute levels to intervention strengths. The benefit of nonlinear scaling over simpler rules remains an ablation question.
3. **Usefulness beyond participants.** We demonstrate substantial new-client data-efficiency gains under the evaluated settings when clients reuse the frozen control mechanism, establishing its value beyond the models that participated in federation.
4. **Empirical characterization.** On publication-partitioned summarization, we characterize the trade-offs among behavioral coverage, control accuracy, generation quality, and preservation of client-specific characteristics. Participant comparisons across two model families provide additional evidence, subject to the outstanding baseline check.

For final submission prose, resolve the outstanding checks and replace provisional language with the resulting supported claims. A contribution should state the delivered result; it should not promise the result of an unfinished run.

**Internal drafting notes — boundaries on the current claims**

- Nonlinear scaling needs a direct comparison among a fixed-gain linear rule, a learned constant-gain linear rule, and the learned nonlinear function, with the remaining design and evaluation held comparable. “Constant gain” still permits the intervention to change with the requested level; a coefficient fixed for every request would disable steering and would not be the relevant baseline. Since the direction's magnitude can compensate for a scalar gain, separate the evidence for learning the gain from evidence for learning the function's nonlinear shape. No such improvement is claimed yet, and no new experiment is launched by this outline revision.
- Move percentile-label construction and the LoRA implementation to methodology. Credit the fixed-basis construction to [FFA-LoRA](https://arxiv.org/abs/2403.12313) there. The introduction motivates personalized federated control without presenting cross-client label alignment as a research contribution.
- Avoid claiming the first federated steering method: a prior patent application explicitly describes federated learning with steering vectors. [Prior application](https://patents.justia.com/patent/20260195607).
- The latest log identifies a binding gain cap in most participant local baselines. Wider-range reruns are needed before asserting a settled collaboration advantage. The held-out comparisons also merit the planned optimization-budget check; do not infer optimization adequacy solely from near-zero training loss.
- The 16-versus-1,024 comparison is an observed crossing of tested data budgets. Avoid claiming a universal “64x sample-efficiency improvement,” a total-data saving, or equal total training compute.
- Most held-out clients in the completed rotation have broad attribute coverage. That experiment establishes a data-efficiency result under the tested setting; it does not yet establish transfer into severe missing-support regions for unseen clients.
- The shared-adapter ablation has better out-of-support control on several clients. Present private adaptation as a measured trade-off involving client characteristics and calibration, not an across-the-board improvement.
- Publication attribution and style-feature gaps are useful diagnostics, not guarantees of stylistic disentanglement. The current style evidence is strongest within client support.
- Continued fine-tuning with steering disabled breaks calibration. Do not claim robustness to arbitrary private fine-tuning or plug-and-play compatibility.
- One training seed and one held-out rotation underpin the main completed results. Article-bootstrap intervals do not cover seed or partition variation.
- Prompting and CAA results concern the tested implementations and tuning ranges. Avoid general statements that prompting cannot control the attribute or that weight steering universally beats activation steering.
- Math CoT is currently a feasibility extension with single-client development results. ChEMBL decoration generation passes a single-client gate, but the multi-client comparison is unfinished. Neither supports a completed cross-task generality claim yet. Amazon ratings were dropped in the later experiment log despite remaining in parts of the older plan.
- “Training examples remain local” describes the protocol. No formal privacy guarantee is established.

**Suggested first figure**

Show two participating clients with complementary shaded training ranges feeding a shared direction and calibration; retain one private adapter per client. Add a held-out client that freezes the shared components and learns its adapter from a small dataset. Beside this schematic, show real control curves against the common target scale, with local coverage shaded. This connects the practical motivation, the shared/private design, and the evaluation question in one figure. Label planned or illustrative curves as such; use measured curves for any result claim.
