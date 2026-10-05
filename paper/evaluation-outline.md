**Evaluation outline — experiments organized by research question**

Companion to [the introduction outline](introduction-outline.md). This file relocates the former introduction paragraphs 7–9 and organizes their evaluation rationale and findings around four research questions. Results below retain the evidence snapshot used in that outline: the October 1–3 entries in `exp_log/EXPERIMENT_LOG.md` and `federated-steering-plan.md`. They are not a fresh audit of running jobs. Distinguish completed evidence from proposed checks when drafting the manuscript.

**Opening and organization**

Candidate opening: “We evaluate whether personalized federated steering expands the behaviors available to individual clients, preserves their generation characteristics, and provides reusable control for clients outside the training federation. We further examine which components of the control mechanism account for its effectiveness.”

| Research question | Link to the introduction | Principal comparison |
|---|---|---|
| **RQ1. Does collaboration improve behavioral control, particularly where local examples are scarce?** | Challenge 1: expand local behavioral coverage. | Federated versus local steering at equal local data; prompting, merged directions, and activation steering as additional baselines. |
| **RQ2. Does private adaptation preserve client-specific characteristics while supporting shared control?** | Challenge 1: retain local specialization. | Private versus shared adaptation, with local models as a reference. |
| **RQ3. Does the learned control mechanism reduce the data needed by a new client?** | Challenge 3: usefulness beyond participants. | Frozen shared control versus locally learned control across held-out-client data budgets. |
| **RQ4. Does learning a nonlinear scaling function improve control over simpler scaling rules?** | Challenge 2: effective intervention strengths. | Fixed-gain linear, learned-gain linear, and learned nonlinear scaling. |

Introduce the shared experimental setup once, then give each RQ its own comparison, results, and interpretation. Quality and off-target effects accompany each relevant result rather than becoming a separate technical challenge. Backbone replication belongs within the questions it tests. Optional drift analysis follows the core questions as a boundary-condition study.

**Shared experimental setup**

Describe the task, client construction, data partitions, and control labels before presenting results. The completed flagship is summary extractiveness on Newsroom, using publications as natural clients, with eight participants and four held-out clients in the evaluated rotation. Explain fragment density in plain language, then define it precisely. The publication partitions simulate clients; do not imply that this is an actual deployment involving the publishers.

Describe the two evaluation settings: participants test collaborative learning, while held-out clients test reuse and local data efficiency. The participant comparisons include Llama-3.2-1B and Qwen3-8B; the completed held-out curves summarized here use the 1B model. Do not imply that every experiment was replicated on both backbones.

Define the shared percentile reference and requested levels here or refer to their methodology definition. This is part of making the experiment reproducible, not a research question about aligning client semantics. Evaluate on the same articles and control requests across methods. Define each client's central training range by its 5th–95th attribute percentiles; values outside this range are not necessarily completely absent from training.

Select checkpoints on development data and report test results. State local data and optimization budgets, decoding settings, client weighting, baseline tuning, and the upstream participant data available to transferred directions. Document that the main completed results use one seed and one held-out rotation. Paired bootstrap intervals over articles quantify article-sampling uncertainty, not training-seed or client-partition uncertainty.

**Measures used across the questions**

| Aspect | Measures and interpretation |
|---|---|
| Control accuracy | Percentile error between requested and generated attribute levels; report mean and worst-client results. |
| Behavioral coverage | Error inside and outside the client's central training range; output range and reach into requested underrepresented regions. Separate large gaps from small tail regions when that analysis is available. |
| Meaningful response | Ordering across requests, per-article Spearman, and exact/near-tie rates, so unchanged or minimally edited outputs do not appear strongly steerable. |
| Task quality | AlignScore, BERTScore, and judge faithfulness, relevance, and coherence; interpret quality alongside references at comparable extractiveness. |
| Off-target changes | Output length across control levels, including its relationship to fragment density. |
| Client characteristics | Extractiveness-controlled publication attribution and interpretable style-feature gaps, especially within client support. |

Motivation for evaluating summary quality at comparable extractiveness has precedent in [Faithful or Extractive?](https://aclanthology.org/2022.acl-long.100/). Do not claim novelty for individual metrics. Read control and quality at the same requested levels. Broadly comparable automatic scores and a small judge sample do not establish an unconditional absence of quality costs.

**RQ1 — Does collaboration improve control where a client's data provides limited coverage?**

Purpose: test the premise that complementary client data helps an individual client acquire a wider behavioral range. The main contrast is federated versus local steering with the same local examples and comparable training opportunities. Add prompting, one-shot direction merging, and federated CAA to assess whether simpler control or collaboration mechanisms suffice.

Present an overall steering-and-quality table, then per-client results split by coverage. The per-client analysis is central: a small aggregate advantage can conceal both substantial gains in large coverage gaps and losses at the extremes of broadly covered distributions. Show requested-versus-generated control curves with each client's training range shaded.

Current participant results, mean percentile error (lower is better):

| Backbone | Federated | Local |
|---|---|---|
| Llama-3.2-1B | 0.151 | 0.165 |
| Qwen3-8B | 0.156 | 0.177 |

These results suggest a collaboration benefit under the evaluated settings, with heterogeneous effects. In the logged comparisons, large gains occur for some clients missing substantial parts of the range, while some broadly covered clients have better local performance in their small tail regions. This supports investigating coverage as the explanation; it does not establish that every client benefits.

Necessary qualifications and follow-up evidence:

- The October 3 analysis identifies a binding gain cap in most participant local baselines. Wider-range local reruns are needed before treating their deficit as settled evidence for collaboration.
- Natural client comparisons entangle coverage with publication characteristics. The planned experiment removing high- or low-attribute training examples from otherwise broad clients would more directly test the coverage explanation. Keep it labeled as proposed until run.
- One-shot merging retains calibration fitted to the original local directions in the current baseline. A calibration-refit variant would make that comparison stronger.
- Prompting uses limited templates, and CAA needs a wider gain search and potentially layer tuning. Conclude about the evaluated baselines, not universal limitations of prompting or activation steering.

Evidence: log entries “Results: shared vs. private calibration…” (20), “Results: baselines B1…” (22), and “Results: C1 at 8B…” (October 3, numbered 37).

**RQ2 — Does private adaptation preserve client characteristics while supporting shared steering?**

Purpose: test the local–global tension directly. Compare the method's private adapters with the shared-adapter ablation, which trains one globally adapted model, and use locally trained models to assess whether collaboration alters the characteristics retained by private adaptation.

Current publication-attribution results, controlled for extractiveness:

| Model | Attribution to the client's publication |
|---|---|
| Real test summaries, reference | 0.587 |
| Federated steering with private adaptation | 0.552 |
| Local steering | 0.553 |
| Federated steering with shared adaptation | 0.409 |

The corresponding style-feature gap is approximately 0.108 for the method and 0.190 for shared adaptation, with lower values indicating closer agreement with the client's real summaries. These findings support preservation of measured publication characteristics through private adaptation. Attribution is a diagnostic and does not prove complete separation of style, topic, and extractiveness; the evidence is strongest within client support.

Report the control trade-off in the same subsection. The shared-adapter ablation has worse in-support error (0.177 versus 0.135) but better out-of-support error (0.136 versus 0.161), and improves out-of-support control on several clients. Its better extrapolation prevents a claim that private adaptation is uniformly superior. The scientific finding is the trade-off between preserving client characteristics and broadening control.

Pair the attribution/style table with in-/out-of-support control and summary quality. Do not use development likelihood alone as proof of style preservation; the best development likelihoods were similar despite differences in attribution.

Evidence: the October 2 “full E2 curve… A2 (shared adapter)” entry 30 and “House style…” entry 31. The 8B entry provides additional style measurements but does not replace the shared-adapter comparison.

**RQ3 — Does the learned control mechanism reduce the data needed by a new client?**

Purpose: establish meaningfulness beyond the models co-trained during federation. Participant performance alone cannot show whether the direction captures reusable control. The held-out client is allowed to adapt privately; this is not a claim of zero-shot compatibility with arbitrary fine-tuned models.

Use held-out clients and nested local data budgets of 16, 64, 256, 1,024, and all available training pairs. Compare:

- **Frozen shared control:** retain the participants' direction and scaling function, and train the new client's private adapter with steering active at each example's attribute level.
- **Locally learned control:** train the private adapter and a new direction/calibration from the same local examples with the same step budget.
- **Plug-in ablation:** train the private adapter with steering disabled, then attach the direction. This tests the importance of adapting in the presence of the reused control mechanism.
- **Prompting baseline:** use the available local examples to support the prompt-based control condition.

Current mean percentile errors across four held-out clients:

| Local pairs | Frozen shared direction | Locally learned direction |
|---|---|---|
| 16 | 0.179 | 0.295 |
| 64 | 0.170 | 0.260 |
| 256 | 0.146 | 0.225 |
| 1,024 | 0.139 | 0.192 |
| All, approximately 4–5k | 0.135 | 0.151 |

The frozen direction has lower overall error for all four clients at each tested budget from 16 to 1,024. With 16 pairs, it already has lower mean error than local direction learning with 1,024 pairs. At full data, the gap narrows and two clients tie statistically in the logged paired comparisons. This is the main evidence for data-efficient reuse under the tested conditions.

Present a curve of error versus local data, accompanied by quality and length. Automatic quality measurements and the small judge sample are broadly comparable between the two principal conditions, but the frozen direction at 16 pairs produces longer summaries: approximately 14 tokens above same-alpha references out of support. Preserve this qualification alongside the control gain.

The 16-versus-1,024 result is an observed crossing of tested budgets. It is not a universal “64x sample-efficiency” guarantee, a saving in total training examples, or equal total compute: the transferred mechanism has benefited from participant data. Planned optimization-budget checks remain useful; near-zero training loss alone does not establish that the local baseline is optimally trained.

The completed held-out rotation has relatively broad attribute coverage. It tests data efficiency more strongly than transfer into severe coverage gaps. Alpha-window restrictions or a rotation holding out naturally narrow clients are proposed coverage tests, not completed results. Keep these two interpretations separate.

Use the plug-in failure to delimit the result: a direction can be reusable through compatible private adaptation without working when attached after unrelated fine-tuning. The shared-versus-private calibration transfer curves can be a secondary analysis here, with calibration-design interpretation under RQ4.

Evidence: log entries 23 and 26, and the October 2 “full E2 curve” entry 30. Some log numbers are duplicated; use titles and dates when tracing evidence.

**RQ4 — Does learning nonlinear scaling improve control over simpler rules?**

Purpose: directly test the introduction's proposed response to nonlinear behavioral sensitivity. A direction's usefulness does not establish that its trainable scaling function is needed. Shared-versus-private calibration results do not answer whether nonlinear scaling is better than a simpler rule.

The central ablation should compare these three conditions while holding the remaining design and evaluation comparable:

| Condition | What it isolates |
|---|---|
| Fixed-gain linear scaling | A simple mapping from requested level to intervention strength. |
| Learned-gain linear scaling | Whether learning a single overall scale helps. |
| Learned monotone nonlinear scaling | Whether changing the mapping's shape adds value beyond learning its overall scale. |

Here “constant gain” means a constant multiplier of the requested control value. It does not mean applying the same intervention at every request, which would remove control. Also, the direction's magnitude can compensate for a scalar gain; interpret the gain ablation separately from the nonlinear-shape ablation.

Train the compared variants under comparable budgets, select on development data, and report test calibration error, ordering, range, quality, and performance across requested levels. Plot both learned scaling functions and generated behavioral response curves. A curved function alone is not evidence that the extra flexibility improves control. Assess effects across clients rather than showing only a favorable example.

Current evidence is partial: shared calibration without an offset performs best among the tested federated calibration configurations (entry 20), and frozen shared calibration performs comparably to refitted private calibration in the held-out curves (entry 30). The direct fixed/learned-linear/nonlinear comparison is still needed. Its result must remain open in the outline. If simpler scaling performs similarly, simplify the claim and reconsider the prominence of nonlinear scaling in the introduction.

Suggested display: one ablation table with the three scaling conditions, plus a compact response-curve figure. Treat sharing and offsets as separate secondary factors so their effects are not attributed to nonlinearity.

**Optional boundary-condition analysis — What happens after further private fine-tuning?**

This is outside the four core RQs. The current stress test fine-tunes private adapters with steering disabled and shows calibration degradation for both federated and local directions. Discuss it as a limitation of compatibility: the private model can move away from the operating conditions under which control was learned. It does not support robustness to arbitrary continued fine-tuning.

A continuation protocol that keeps steering active would address a different question and remains untested in the evidence snapshot. Include the current result briefly in analysis or the appendix, with the main limitation stated clearly. Evidence: entry 26, E3.

**Scope and manuscript presentation**

Use one setup section followed by the four RQs. Each RQ should open with its comparison and end with an answer bounded by the evidence. Integrate quality with the relevant control results. Keep the introduction's contributions concise; detailed protocols, numbers, and qualifications belong here.

Math CoT and ChEMBL are extensions under evaluation in this snapshot. Single-client feasibility does not establish federated gains or cross-task generality. Once completed evidence is available, add task-specific utility measures and results under the same RQs where applicable. Amazon ratings were dropped in the later log despite remaining in parts of the older plan. No new jobs or code changes are authorized or launched by creating this outline.

Across all RQs, retain the distinction between keeping training examples local and providing a formal privacy guarantee; the latter is not established by these experiments.
