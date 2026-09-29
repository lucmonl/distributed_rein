# REIN for Controlled Curriculum Self-Improvement

Research plan, 28 September 2026. This is a proposed study, not a report of established results. Numerical settings below are pilot defaults to validate before the main experiments.

**Recommended thesis.** A task proposer should improve at generating useful training problems while retaining the ability to target different skill combinations. Test whether preserving that control during proposer updates produces more effective solver learning under a fixed total compute budget.

The hypothesis is narrower than preventing all self-training collapse. It concerns loss of conditional task coverage: a proposer trained on the solver's current weaknesses may become less able to generate tasks for previously supported or newly requested combinations. Establish that failure empirically before presenting REIN as its remedy.

**1. REIN applies to the proposer, whose continuing training is already part of self-play.**

Use separate models, or separate adapters on a frozen backbone, for the proposer P and solver S. The proposer writes candidate problems; a trusted executable environment validates them and supplies answers; the solver attempts and learns from them. The proposer is also trained using feedback on its task quality and suitability for the current solver. R-Zero and Absolute Zero provide precedents for this learning loop [1,2].

| Original REIN | Proposed setting |
|---|---|
| Steerable text generator | Controllable task proposer |
| Attribute intensity | Verified structural properties of a reasoning task |
| Broad Phase I supervision | Tasks covering the initial control space |
| Skewed Phase II examples | Repeated proposer updates concentrated on current curriculum needs |
| Preserve behavior at unobserved intensities | Preserve ability to request temporarily neglected and composed task types |
| Generation quality | Valid task yield, curriculum responsiveness, and downstream solver learning |

Phase II is natural because the solver changes: the proposer must learn to produce new useful problems for that changing solver. A frozen generator can also be a valid solution; include it as a baseline. Do not claim all self-improvement systems must update their proposers.

The new central distinction is between broad unconditional diversity and conditional accessibility. A generator may produce many different problems but fail to supply the particular skill combination currently requested. Conversely, perfectly following control requests is insufficient if the generated problems do not improve the solver.

Use the layerwise parameterization

\[
W_t(z)=W_0+C_t+z_LD_L+z_BD_B,
\qquad z=(z_L,z_B).
\]

Phase I trains C, D_L and D_B jointly. Phase II freezes the two D matrices and updates C. The original one-dimensional REIN is recovered by retaining a single D. Represent each matrix as its own low-rank product and add the products; do not interpolate LoRA factors and claim the resulting full-weight path is linear.

Freezing parameter directions is only an inductive bias. Changing C changes the network's response to the fixed D matrices. Therefore measure behavioral control after every update round. If Phase II only uses one fixed z, this construction can be equivalent to adapting the active model and reapplying its original offsets under matched parameterizations; retain that equivalence as a sanity check, not a separate claimed advantage.

Do not transfer the original “without intermediate guidance” claim unchanged. Here structural labels can be computed cheaply, including during Phase II. The challenge is preserving conditional coverage when solver feedback and training concentrate unevenly across tasks. Replay and control rewards must be allowed as serious competitors.

**2. Use executable program reasoning as the main task domain, with a small procedural pilot first.**

Begin with bounded program-execution prediction: the solver receives a small deterministic program and an input, and must predict the output without execution access. The proposer emits a structured task specification; the evaluator executes it and computes the correct answer. This makes correctness independent of the proposing and solving models.

Initially restrict the program language to a documented Python subset with bounded loops, scalar/list state, arithmetic and conditionals. Use an independently implemented validator, a fixed execution harness, time and memory limits, and no external I/O. The proposer cannot modify the checker, reward computation or reference interpreter. Keep the task statement a faithful rendering of the verified program and input.

Start with two controls:

| Control | Operational definition | Intended capability |
|---|---|---|
| Loop-dependent state tracking, z_L | A bounded measure of loop-carried operations in the executed dependency slice of the requested output | Tracking changing state across iterations |
| Conditional dependence, z_B | A bounded measure of branch decisions that control values in that output's dependency slice | Reasoning about conditions and paths |

Implement these as reproducible structural descriptors. They are proxies for the intended reasoning demands, not ground truth about a model's internal skills. Static syntax counts alone are insufficient: dead loops, unused branches and redundant operations should not satisfy control requirements. On the restricted language, combine tracing and dependency slicing; audit selected cases by changing the relevant condition/state and checking whether the output can change. Audit descriptor validity before training.

Use low, medium and high bands on each axis, creating nine feasible cells. Determine numerical band limits using a small pilot so neither axis is always trivial or impossible. Control coordinates interpolate requested structural levels or distributions over adjacent levels; do not promise arbitrary exact real-valued counts.

An illustrative request is “substantial loop-state tracking with little conditional dependence,” followed later by “substantial loop-state tracking combined with substantial conditional dependence.” The latter must combine requirements inside a problem. Sampling separate loop-only and branch-only questions does not demonstrate compositional generation.

Keep structural controls separate from empirical difficulty. The same task can become easy after solver training, although its structural descriptors remain unchanged. Estimate current difficulty from solver success, and let the scheduler choose reachable control cells and task variants near the current learning frontier. Report infeasible requests instead of assuming every combination and difficulty is reachable.

Reasoning Gym provides procedural, verifiable environments useful for validating infrastructure and for a strong procedural curriculum baseline [3]. It does not automatically provide the proposed program-slicing annotations; those need implementation. A procedural baseline using the same program grammar is mandatory and is more directly comparable than unrelated Reasoning Gym tasks.

After the pilot, add a second task type such as input synthesis: given a program and required output, produce any valid input satisfying the program. Verification must accept all valid answers rather than compare with one hidden witness. Expand to code synthesis or corpus-grounded reasoning only after the proposer mechanism works. Avoid starting with unrestricted natural-language math generation: judging validity and answers would introduce a second research problem.

Evaluate on separately generated program templates and held-out compositions, not only new random seeds. Add an external code-reasoning benchmark whose input/output format matches the trained solver. If testing code generation, explicitly teach and evaluate that output format; do not infer code-generation transfer from output-prediction accuracy alone.

**3. Compare both proposer mechanisms and complete learning systems.**

All mechanism baselines should receive the same Phase I tasks, verified descriptors, solver feedback budget, requested curriculum schedule, and access to allowed replay. Match total adapter capacity and report Phase II trainable capacity separately. Start each solver from the same checkpoint and use the same solver optimizer.

| Baseline | Implementation | Question answered |
|---|---|---|
| Procedural generator with adaptive selection | Generate programs directly from the same grammar; use the same skill/difficulty scheduler | Is a learned proposer needed at all? |
| Frozen Phase I proposer | Change requests and filter outputs but never update proposer weights | Is proposer Phase II useful? |
| Prompt-conditioned adaptive proposer | Standard LoRA model receives the explicit control specification in its prompt; update on the same rewards | Can prompting plus ordinary learning solve the problem? |
| Free control-adapter updates | Same REIN architecture and Phase I checkpoint; update C and all D matrices | Does freezing control directions help? |
| Matched-capacity ordinary update | Tune an ordinary adapter with comparable trainable parameters | Is an apparent benefit just fewer trainable parameters? |
| Replay or distillation | Ordinary/free updates plus balanced Phase I replay, or reference-policy KL | Does familiar forgetting mitigation suffice? |
| Generate then select | Adaptive unconstrained/prompted proposer plus descriptor-based filtering and difficulty selection | Is native control better than rejection sampling at equal total cost? |
| Fixed verified archive plus scheduler | Select tasks from a large, diverse seed archive; account for archive construction | Is adaptive generation better than task selection? |
| REIN without difficulty recalibration | Freeze directions and update C, but use stale difficulty estimates | Does the proposed calibration component matter? |

Add full-loop comparisons with R-Zero-style self-play, an Absolute-Zero-style executable proposer, and R-Diverse-style skill-aware novelty plus memory [1,2,4]. When adapting their objectives to one common environment, label them as adaptations; do not present the result as an exact reproduction of their reported benchmark performance.

R-Few is a relevant grounding/replay comparison because it uses examples and mixed training to stabilize self-evolution [5]. Equalize access to seed examples; this project is not zero-data. R-Diverse's repository was located, but a repository link is not evidence of a complete reproducible training release; verify available components before estimating reproduction effort.

WIST already studies domain-targeted, controllable self-play, and SPADE learns executable training environments [6,7]. Both require explicit related-work discussion. They are not mandatory full reproductions for a small program-reasoning pilot, but become important comparisons if the scope expands to web-grounded curricula or generated environments. Do not claim the first controlled curriculum or the first adaptive executable task generator.

Run the procedural, frozen, prompt-conditioned, free-update and REIN comparisons first. Add the strongest replay and diversity competitors before committing to the full paper. DExperts and sentiment-era steering baselines are low priority for this setting.

**4. Obtain the Phase I proposer from automatically verified task specifications.**

Use an open instruction model as the proposer backbone. Qwen3-8B is one concrete initial candidate with an official model card [8]; it is a reproducibility choice rather than a claim that it is the newest or best model. Use separate proposer and solver adapters to avoid conflating proposer control preservation with direct solver preservation. A smaller solver can reduce feedback cost, but calibrate it so a meaningful fraction of tasks is learnable.

Build the initial data in four steps.

1. Generate a diverse seed pool from the restricted grammar. Supplement it with proposals from a prompted model or teacher, subject to the same independent checks. Record all teacher costs and provide the same accepted pool to competitors.
2. Execute each task, compute its answer and descriptors, reject invalid or trivial instances, and remove duplicates after canonicalizing variable names and literal-only changes where appropriate.
3. Balance accepted tasks across the nine control cells. A starting scale is 9,000 accepted tasks, roughly 1,000 per cell. Generate multiple templates, inputs and operator combinations per cell. These counts are planning defaults, not a sample-complexity claim.
4. Split by program template and structural family into training, development, monitoring and final evaluation pools. Keep final evaluation entirely out of scheduler decisions and reward tuning. For a separate composition test, withhold selected cells from Phase I; do not confuse that test with preserving cells that Phase I explicitly taught.

The training example is

\[
(x_i,z_i,y_i),
\]

where x_i contains the task-generation instruction and optional permitted seed context, z_i is the requested descriptor level, and y_i is the validated task specification. The proposer does not need to generate the authoritative answer: the interpreter supplies it.

Train all proposer adapters with conditional SFT:

\[
\mathcal L_{\rm I}
=-\mathbb E_{(x,z,y)}
\log P_{W_0+C+z_LD_L+z_BD_B}(y\mid x,z).
\]

Use ordinary natural-language or structured control specifications in x for both REIN and prompt baselines. The extra weight conditioning is the experimental treatment. Test whether removing or shuffling the weight controls affects performance; if not, the learned D matrices are unnecessary.

A feasible initial allocation is rank 16 for C and rank 8 for each D, for 32 total rank per adapted layer. Tune rank on development data and compare a rank-32 ordinary LoRA. Add both attention and FFN targets if the chosen framework supports them; select one placement in the pilot, then hold it fixed across comparable models.

SFT learns the map from requested coordinates to task structure. If task validity or control remains weak, add a short Phase I policy-optimization stage using automatic validity and control rewards, sampled uniformly across cells. No solver model is required for those structural rewards. Do not introduce self-play before validating the control surface.

Phase I acceptance gate: held-out template generation has useful validity and simultaneous control accuracy, and reasonable structural diversity within each cell. Compare against prompt-only and procedural generation immediately. A suggested engineering target is 90% validity and 80% simultaneous band compliance, revised if descriptor feasibility makes those thresholds inappropriate. These are project gates, not predicted results.

**5. Use a calibrated proposer-solver loop for Phase II.**

Maintain four state components: proposer P_t, solver S_t, a task archive, and a lightweight current-solver difficulty estimator. The estimator predicts success from the verified descriptors and additional task features. It need not be neural; start with a table over cells and coarse size bins, refreshed using actual solver attempts.

Each round:

1. Select control requests with a common scheduler. Start with a fixed rotation to isolate the proposer. Then use a simple adaptive scheduler that prioritizes undercovered, learnable cells and retains uniform exploration (for example 20%). The scheduler is shared across proposer baselines.
2. Generate candidate task specifications and independently validate them. Record costs of every rejected candidate.
3. Estimate solver success on valid tasks. Start with four attempts per task and allocate additional attempts only for uncertain decisions. Such estimates are noisy; use smoothing or uncertainty intervals rather than treating one outcome as a calibrated probability.
4. Score each proposed task using validity, structural compliance, a learnability proxy and novelty. Choose tasks near a broad success band, such as 0.2–0.8, for solver training, while retaining some easier review tasks.
5. Update the solver using execution-grounded RL. A fixed GRPO-style recipe is a reasonable starting point. Keep the optimizer, generated-token budget and replay allocation identical across proposer methods. In a cheaper diagnostic, rejection fine-tuning on correct solver traces is acceptable if used consistently.
6. Update the proposer by policy optimization, freezing D_L and D_B and training C. Evaluate difficulty against a fixed solver snapshot within each proposer update block to avoid changing the reward target during that block.
7. Refresh difficulty estimates against the new solver, archive task structures, and audit all control cells on fresh monitoring prompts.

One candidate proposer reward is

\[
R_P(q,z;S_t)=
\begin{cases}
-1, & q\text{ invalid},\\
w_\ell\,4\hat p_t(q)(1-\hat p_t(q))
+w_c\exp[-E_{\rm ctrl}(q,z)/\sigma^2]
+w_n\,N(q;\mathcal A_t), & q\text{ valid},
\end{cases}
\]

with nonnegative weights summing to one, control error computed from verified descriptors, and novelty N bounded in [0,1]. The valid reward is then nonnegative. Tune weights on development runs and hold them fixed in the main comparison. The success-variance term is a proxy for learnability, not measured learning progress. Whether it actually produces improvement must be assessed through held-out solver learning.

Use structural novelty or skill-pattern novelty with a persistent archive; raw wording diversity is insufficient. Monitor whether a novelty bonus induces meaningless complexity or superficial operator substitution. Retain an ablation without novelty, and give the same novelty mechanism to a non-REIN baseline.

Frozen directions are the minimal method. If they fail to preserve behavior, the first stronger variant is a small, balanced control-replay term on fixed Phase I anchors:

\[
\mathcal L_{P,t}
=\mathcal L_{\rm RL,current}
+\lambda_{\rm replay}\mathcal L_{\rm SFT,anchors}.
\]

Report REIN alone, replay alone and their combination. A 5% replay-token fraction is a starting setting; sweep it against compute. This is explicit additional supervision and must not be described as guidance-free.

If a significant structural-preservation gap remains and the main loop already works, investigate episodic Phase I training: adapt C on one subset of control cells and optimize post-update likelihood on a disjoint balanced query set. This may teach control directions that withstand skewed updates. It is an optional second-stage research contribution, not a prerequisite for the first pilot. Use the same episodic training on the prompt-conditioned baseline to separate meta-training effects from REIN.

Difficulty is recalibrated; structural meaning is preserved. Never use “z=0.8 remains 80% hard” as a preservation target. Structural axes need not be orthogonal, and some combinations may be infeasible. Request joint control only within the documented feasible region.

**6. Demonstrate the causal chain in three experiments.**

**Experiment A isolates proposer control preservation.** Use a fixed sequence of solver snapshots and request schedules for every method. Concentrate proposer updates on one region, switch to another, and then revisit the first and the combined region. The controlled schedule is a diagnostic stress test, not the claimed natural data distribution. Compare conditional compliance, valid-task yield and novelty before and after each update.

Include a cache-based matched-data diagnostic where feasible: all methods receive the same verified proposal pool and feedback, then are evaluated on new task generations. This distinguishes an update-geometry effect from different online experience collection.

**Experiment B tests actual self-improvement.** Let each solver learn from its own proposer's outputs under a common adaptive scheduler and a fixed total compute budget. Run at least three seeds for the main comparisons. The key outcome is held-out solver performance over cumulative compute, including proposer generation, rejected tasks, execution, solver probes, training, calibration and replay.

Cross two proposer update methods (ordinary and REIN) with two schedulers (uniform and adaptive). This four-way study checks whether any benefit comes from the scheduler alone and whether the two components interact.

**Experiment C tests whether the retained controls are useful.** Request a neglected skill combination or shift the curriculum priorities, using only training-side monitoring data. Measure how many candidates and feedback calls are needed to produce a fixed number of valid, compliant, suitably difficult tasks, and how quickly the solver improves on independent tasks from that region. Include a generated-task archive selector and a prompt-repair baseline.

In an additional mediation study, train identical solver copies on curricula matched for size, validity, difficulty and broad diversity. Then test whether targeted composition or conditional task availability explains any remaining gain. Do not mistake a correlation between proposer compliance and solver accuracy for a causal result.

**7. Measure the result at the proposer, curriculum and solver levels.**

| Level | Primary measurements |
|---|---|
| Proposer | Joint band-compliance rate; normalized descriptor error; valid yield; retained coverage after updates; diversity within a requested cell |
| Curriculum | Usable tasks per generated token and per solver probe; current difficulty calibration; skill-combination coverage; revisit latency; task duplication across rounds |
| Solver | Held-out accuracy; worst-cell accuracy; held-out composition accuracy; learning-curve area against total compute; external transfer where task format is compatible |

Define a usable task before the main run: it must be valid, satisfy the requested structural bands, pass the duplication policy, and fall within the measured learnability band to the precision supported by the probing budget. Report these components separately so one component cannot hide another.

Use task-template clusters as the unit of bootstrap resampling where many tasks share a template. Report seed variability. Do not declare failure solely because an easy cell no longer produces much training gain; mastery is expected. The failure of interest is inability to supply a valid requested task when that request is feasible.

Parameter-count matching does not equal compute matching. Report both matched solver-training token comparisons and matched total-compute comparisons. Charge rejection sampling, teacher seed generation and archived data construction. Separate one-time Phase I costs from recurring Phase II costs and show the amortization assumption.

**8. Use staged milestones rather than launching a full recursive-training project immediately.**

| Stage | Deliverable | Decision |
|---|---|---|
| Environment pilot | Restricted grammar, trusted checker, descriptor audit, nontrivial solver success range | Stop and repair if labels can be gamed or tasks are overwhelmingly trivial/impossible |
| Phase I | Nine-cell controllable proposer and strong prompting/procedural comparisons | Continue only if a useful control surface exists |
| Preservation pilot | Skewed update sequence with frozen/free/prompt/replay comparisons | If no control degradation occurs, drop the preservation premise or identify a justified harder regime |
| Learning pilot | Three rounds of self-play with a small solver and matched budget | Continue only if preserved control improves usable data or downstream learning |
| Full study | Five to ten rounds, main baselines, three seeds, held-out compositions and external transfer | Assess whether gains persist after total-cost accounting |

A starting pilot size is 9,000 accepted Phase I tasks, 100 fresh audit generations per cell at each audit, and 500–1,000 valid candidate tasks per self-play round. With 1,000 tasks, four solver probes and a 2,048-token probe cap, probing alone can consume roughly 8.2 million generated tokens per round, before proposer and update costs. Profile this before scaling. Use the same accounted feedback for solver learning where methodologically appropriate, and disclose reuse. No wall-clock or hardware estimate is reliable until sequence lengths and throughput are measured.

**What would justify a strong paper.** Ordinary proposer learning loses access to useful task combinations; REIN or its explicitly tested extension preserves that access; the preserved access yields better held-out solver learning than prompt-conditioned training, replay, diversity regularization, and procedural/adaptive selection at comparable total cost.

**What would not justify the claim.** Better descriptor accuracy with no solver improvement; higher lexical diversity; gains caused only by extra teacher data or solver queries; a learned proposer dominated by the procedural baseline; or a benchmark whose control labels are merely surface syntax.

The closest novelty boundary is preserving a compositional task-generation interface throughout learning and showing that it improves curriculum responsiveness and solver progress. Controlled curricula, diversity-aware self-play and learned environment generation already exist. Do not claim to solve general recursive self-improvement or model collapse.

**Primary references and implementation starting points.**

The user-provided REIN.pdf, especially Sections 3.1–3.2, supplies the frozen adapter-difference mechanism.

1. [R-Zero: Self-Evolving Reasoning LLM from Zero Data](https://arxiv.org/abs/2508.05004). [Official code](https://github.com/Chengsong-Huang/R-Zero). Challenger–solver co-evolution and a natural continuing proposer update.
2. [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://papers.nips.cc/paper_files/paper/2025/hash/9837dc00ff67d176373268ed48042d49-Abstract-Conference.html). [Official code](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner). Executable grounding of self-generated reasoning tasks.
3. [Reasoning Gym](https://github.com/open-thought/reasoning-gym). Procedural verifiable reasoning environments; use as infrastructure and a strong generator-selection baseline, not as evidence that the proposed descriptors are already implemented.
4. [R-Diverse: Mitigating Diversity Illusion in Self-Play LLM Training](https://arxiv.org/abs/2602.13103). [Repository](https://github.com/Gengsheng-Li/R-Diverse). Skill-aware diversity and memory-based protection against recurring patterns; check release completeness.
5. [Guided Self-Evolving LLMs with Minimal Human Supervision (R-Few)](https://arxiv.org/abs/2512.02472). Grounding and mixed training for stabilizing self-evolution.
6. [WIST: Web-Grounded Iterative Self-Play Tree for Domain-Targeted Reasoning Improvement](https://arxiv.org/abs/2603.22352). Prior domain-steerable self-play with adaptive exploration.
7. [SPADE: Self-Play in Adaptive Synthetic Executable Environments](https://arxiv.org/abs/2608.19197). Recent learned executable environments and adaptive self-play; broader and more expensive than the initial program-reasoning scope.
8. [Qwen3-8B official model card](https://huggingface.co/Qwen/Qwen3-8B). One reproducible initial backbone candidate; freeze a specific revision and report model provenance.

Sources checked through 28 September 2026. Literature positioning is based on targeted primary-source review, not an exhaustive novelty certification. Experimental settings, rewards, architecture choices and gates above are research proposals.
