**Introduction outline — learning shared control from distributed behavioral coverage**

Drafted against `federated-steering-plan.md` and `exp_log/EXPERIMENT_LOG.md`, including the October 3 results and corrections. This is an argument outline with candidate topic sentences, not finished introduction prose. The drafting notes at the end are internal and should not appear verbatim in the paper.

**Recommended central argument**

Organizations adapting a common language model can have substantial local datasets while lacking examples of particular behaviors they want to produce. Their datasets may collectively cover a much broader behavioral range. The research question is whether that complementary coverage can be distilled into a shared, continuously adjustable control direction that works with private client adapters and can be reused by new clients with little local data.

Make three properties central: reaching underrepresented attribute levels, retaining client-specific generation characteristics, and reusing the learned control mechanism. Frame federation as the setting that permits collaboration when examples remain local. Newsroom supplies a simulation with natural publication partitions; it does not establish that the participating publications actually require federated deployment.

**Paragraph 1 — From useful generation to adjustable behavior: why control and steering matter.**

Begin with a familiar need: the same input can call for different valid outputs depending on the user's purpose. Introduce controlled generation as the ability to choose an output property and its degree while retaining task quality. Then introduce steering as one way to provide that control, before bringing in locally adapted models. This lets the reader understand the value of the control mechanism before encountering the difficulty of learning it collaboratively.

Candidate paragraph:

“A language model may need to produce different kinds of outputs from the same input as users' needs change. An editor, for example, may want a summary that closely preserves an article's wording for one publishing format and a more extensively rewritten version for another. Both should convey the relevant information faithfully, and useful choices may lie between these endpoints. Controlled generation aims to give users this ability to specify how an output should vary while maintaining its usefulness. Steering offers one approach: a learned direction of behavioral change is applied with adjustable strength during generation, allowing a model to support a range of outputs without retraining it for each setting. The goal is a predictable relationship between the requested level and the generated behavior. For organizations that adapt models to their own data, this control must also work alongside local writing and format conventions. Learning to reproduce those conventions does not by itself teach a model how to vary its behavior across the range users may request.”

Attach steering citations to the sentence introducing the mechanism in the manuscript: [CAA](https://aclanthology.org/2024.acl-long.828/); [contrastive weight steering](https://arxiv.org/abs/2511.05408). These motivate adjustable interventions; they do not imply a guarantee of precise control. Keep prompting as a legitimate alternative to assess experimentally, without claiming in the opening that it cannot provide control.

Use summary extractiveness as the concrete example, explaining it first as the extent to which a summary reuses source wording. The application determines the desired level; higher or lower extractiveness is not inherently better. Reserve direction geometry, activation-versus-weight implementations, and training objectives for later sections. Mathematical solution length can become a second example if its accuracy-versus-control results support it; it is currently an extension under evaluation.

**Paragraph 2 — Why collaboration: local data can be abundant but behaviorally incomplete.**

Candidate topic sentence: “Learning such control is difficult when an organization's examples cover only a narrow portion of the desired behavioral range, even if its local dataset is large.”

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

The progression is compatibility during collaboration, effective control across requested levels, and usefulness beyond the training federation. Paragraphs 5 and 6 explain the design responses to the first two. Paragraph 7 summarizes the contributions, including usefulness beyond participants. The experiments addressing these requirements are organized by research question in [the evaluation outline](evaluation-outline.md). Quality measurements support these claims; they are not a separate technical challenge.

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

**Paragraph 7 — Contributions: four concrete deliverables.**

Candidate contribution wording:

1. **Problem formulation.** We study the tension between local specialization and collective behavioral coverage in controllable language generation, requiring a learned control mechanism to serve both participating and previously unseen clients.
2. **Personalized federated steering.** We develop a protocol that jointly learns shared behavioral control and private client adaptation, with a trainable nonlinear scaling function that maps requested attribute levels to intervention strengths. The benefit of nonlinear scaling over simpler rules remains an ablation question.
3. **Usefulness beyond participants.** We demonstrate substantial new-client data-efficiency gains under the evaluated settings when clients reuse the frozen control mechanism, establishing its value beyond the models that participated in federation.
4. **Empirical characterization.** On publication-partitioned summarization, we characterize the trade-offs among behavioral coverage, control accuracy, generation quality, and preservation of client-specific characteristics. The measured publication characteristics remain close to those of locally trained models; participant comparisons across two model families provide additional evidence for collaboration, subject to the outstanding baseline check.

Keep this as the introduction's brief summary of empirical value. The protocols, numerical findings, and qualifications previously in paragraphs 7–9 now appear under the research questions in [the evaluation outline](evaluation-outline.md). For final submission prose, resolve the outstanding checks and replace provisional language with the resulting supported claims. A contribution should state the delivered result; it should not promise the result of an unfinished run.

**Internal drafting notes — boundaries on the current claims**

- Nonlinear scaling remains a design hypothesis until the ablations in evaluation RQ4 establish its benefit. No new experiment is launched by this outline revision.
- Move percentile-label construction and the LoRA implementation to methodology. Credit the fixed-basis construction to [FFA-LoRA](https://arxiv.org/abs/2403.12313) there. The introduction motivates personalized federated control without presenting cross-client label alignment as a research contribution.
- Avoid claiming the first federated steering method: a prior patent application explicitly describes federated learning with steering vectors. [Prior application](https://patents.justia.com/patent/20260195607).
- Numerical evidence and experimental limitations are maintained in [the evaluation outline](evaluation-outline.md). In particular, participant superiority remains provisional, transfer permits private adaptation, and preservation of publication characteristics does not establish complete disentanglement. The introduction should not claim robustness to arbitrary fine-tuning, universal superiority to prompting, or completed cross-task generality.
- “Training examples remain local” describes the protocol. No formal privacy guarantee is established.

**Suggested first figure**

Show two participating clients with complementary shaded training ranges feeding a shared direction and calibration; retain one private adapter per client. Add a held-out client that freezes the shared components and learns its adapter from a small dataset. Beside this schematic, show real control curves against the common target scale, with local coverage shaded. This connects the practical motivation, the shared/private design, and the evaluation question in one figure. Label planned or illustrative curves as such; use measured curves for any result claim.
