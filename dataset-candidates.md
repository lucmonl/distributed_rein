# Candidate second/flagship tasks beyond summarization

Dataset scouting · 1 October 2026 · companion to `federated-steering-plan.md` §3.
Status: research only, nothing built. Supersedes §3.2 (Amazon review rating) as the recommended secondary task.

---

## 1. The screen

Five requirements, from the plan and from the 10-01 discussion:

| # | Requirement | Why it bites |
|---|---|---|
| R1 | **Continuous, fine-grained** attribute (not 2–5 discrete levels) | α ∈ [0,1] needs a dense scale; a 3-level corpus (Newsela, OneStopEnglish) can't give a percentile CDF |
| R2 | **Attribute intensity cheap and deterministic to score** | the scorer is called on every training target *and* every generation; no Perspective API (shuts down 31 Dec 2026), no paid judge in the loop |
| R3 | **Attribute-level heterogeneity across natural clients** | the whole C1 claim is "each client's data covers a different part of the range"; needs a natural partition field, ≥12 clients with ≥5k pairs, and *disjoint supports* |
| R4 | **Not summarization** | the flagship already is |
| R5 | **Prompting must not reach it** (B1 is the gate) | if an instruction can hit the level, no knob is needed and the paper has no gate G1 |

**R5 is the binding constraint, and it has a crisp test.** Prompting fails exactly when the model cannot *measure* the attribute on its own output and the attribute has no natural verbalization at intermediate levels. That kills every "semantic" attribute (sentiment, star rating, formality, politeness, technicality, persona traits) — instruction-tuned models track those well from words alone. It favours attributes that are **corpus statistics**: copy-fragment density (the current flagship), exact token/step counts, edit distance to the input, RDKit descriptors, syllable counts.

**A sixth criterion, not on the user's list but decisive** — call it R6: **the attribute must be free given the input.** If the input determines the attribute, α cannot vary within a problem and "steering" collides with correctness. This is what disqualifies the two most obvious ideas (code complexity, SQL complexity): the spec dictates the answer's complexity, so asking for α = 0.2 on a hard function asks for a wrong answer. Newsroom passes R6 because any article can be summarized extractively or abstractively.

---

## 2. Ranked candidates

| Rank | Task | Attribute | Scorer | Clients | R1 | R2 | R3 | R5 | R6 | Compute |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | Math CoT reasoning | trace length / step count | token count (exact) | problem source | ✅ | ✅✅ | ✅ | ✅ | ⚠️→✅ | ⚠️ high |
| **2** | Molecule generation | logP / TPSA / MW / QED | RDKit (exact) | target family | ✅ | ✅✅ | ✅✅ | ✅✅ | ✅ | ✅✅ low |
| **3** | Paraphrase generation | lexical / syntactic divergence | edit distance, tree distance | source corpus | ✅ | ✅ | ✅ | ✅ | ✅✅ | ✅ low |
| 4 | Table-to-text | cell coverage | string match on cells | table domain | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| 5 | Science QA verbosity | answer length | word count | source × subject | ✅ | ✅✅ | ✅ | ✅ | ⚠️ | ✅ |
| ✗ | Code generation | cyclomatic complexity | radon (exact) | repository | ✅ | ✅✅ | ✅ | ⚠️ | ❌ | ✅ |

---

### Candidate 1 — Reasoning-effort control on math CoT (**recommended flagship-grade second task**)

- **Data:** `open-r1/OpenR1-Math-220k` (94k default / 131k extended), built on `AI-MO/NuminaMath-1.5`. **2–4 independent DeepSeek-R1 traces per problem**, 16k-token cap, each problem has ≥1 verified-correct trace (Math Verify, plus Llama-3.3-70B judge on 12%).
- **Why the multi-trace structure matters:** it is the **R6 fix**. Different traces of the *same* problem have very different lengths, so α varies *within* a problem, not just across problems. That removes the length↔difficulty confound and buys an evaluation Newsroom cannot support: **same input, two real targets at two different α**, i.e. ground truth at both ends.
- **Attribute:** generated-trace token count → global percentile. Completely deterministic, zero scorer cost, no judge, no API.
- **Clients:** NuminaMath 1.5's `source` field (`olympiads`, `cn_k12`, `aops_forum`, `cn_contest`, `amc_aime`, `gsm8k`, `math`, synthetic variants) — "each institution has its own problem bank and its own solution-style convention". Natural skew is large: gsm8k solutions are short, olympiad solutions long. Second axis if sources are too few: `source × problem_type` (Algebra / Geometry / Number Theory / Combinatorics).
  - ⚠️ **Check first:** per-source counts in the *220k* subset, not in NuminaMath-CoT. cn_k12 and olympiads dominate (276k / 150k in CoT); amc_aime had only 4k. Needs ≥12 groups at ≥5k.
- **Why prompting fails (R5) — this is the strongest literature case of any candidate.** Prompt-level token budgets are known to break down: models revert to longer reasoning and abandon the constraint when the budget is tight, and budget-following is only non-trivial inside an empirically narrow "ideal window". The whole point of L1/LCPO, BudgetThinker and SelfBudgeter is that **weight training (RL or control tokens) is required** to make length controllable. So B1 fails for a documented reason, and G1 is defensible to reviewers without hand-waving.
- **Why fine-grained control is already desirable (the plan's criterion 1):** this *is* the test-time-compute knob. It is the one controllable-generation attribute a 2026 reviewer will accept as useful without argument.
- **Utility metric: answer accuracy, verifiable.** Far stronger than AlignScore/BERTScore — no judge, no "is this faithful" ambiguity. And the accuracy-vs-α curve is itself a result (the accuracy/compute trade-off per client).
- **Federated story:** credible. Each client has its own problem bank and its own trace-length convention; a client whose data is all short grade-school solutions cannot produce long olympiad-style traces, and vice versa. That is the coverage argument, verbatim.
- **Risks:**
  1. **Compute.** This is the real cost. 12 clients × 200 test × 5 α × ~1.5k tokens ≈ 18M generated tokens per full eval. The current HF `generate` greedy loop will not take it — needs vLLM, or a hard 1k-token cap and short-CoT data (which weakens the "effort" framing).
  2. A 1B base model produces weak traces; accuracy may floor out and hide the accuracy/α effect. Qwen3-4B or a distilled 1.5B R1 would be the honest choice.
  3. "Length is a trivial attribute" is a likely reviewer jab. Mitigation: report **step count** (deterministic: enumerated steps / equation lines) alongside tokens, and lead with accuracy-at-budget.

### Candidate 2 — Property-conditioned molecule generation (**recommended if compute is the constraint**)

- **Data:** ChEMBL (≈2M molecules; TDC packages it as `TDC.ChEMBL`, and MOSES/ZINC as cleaner alternatives). Instruction-formatted variants exist (Mol-Instructions, 331k instructions).
- **Attribute:** an RDKit descriptor — cLogP, TPSA, molecular weight, QED, or SA score → global percentile. **Exact, instant, dependency-light, and unimpeachable**: nobody argues about whether the scorer is right. This is the single best R2 of any candidate.
- **Clients:** target family / assay (kinases, GPCRs, ion channels, nuclear receptors, CNS vs. peripheral). **Heterogeneity is documented and mechanistic**, not incidental: kinase inhibitors sit at high MW/TPSA, CNS-directed compounds are deliberately confined to low MW/logP, and ADMET-vs-potency pressure pushes families to opposite ends of logP. These are genuinely disjoint supports — stronger than Newsroom's.
- **Why prompting fails (R5):** a general LM cannot compute logP for a SMILES string it is emitting. SmileyLlama is the evidence: property control over ChEMBL needed **SFT on weights, and SFT alone was "quite poor" without DPO on top**. Prompt-only control of the base model is not a contender.
- **Why fine-grained control is already desirable:** lead optimization is literally "same scaffold, dial the property". No framing work needed.
- **Why the federated framing is *better* here than anywhere else:** this is the one domain with a **real, funded, non-hypothetical federation** — MELLODDY, where pharma companies could not pool compounds for legal reasons and each firm's library is skewed by its own historical programs. The coverage motivation stops being a thought experiment.
- **Utility metrics:** validity, uniqueness, novelty, scaffold similarity, SA score, plus a QSAR activity predictor for the client's target. All deterministic.
- **Compute: by far the cheapest.** Outputs are ~40 tokens. A full 12-client × 5-α eval is minutes, not hours. This alone may decide it.
- **Risks:**
  1. **Base-model competence.** Llama-3.2-1B after 4–5k molecules of SFT may emit invalid SMILES at a high rate, and validity would then confound every steering metric. Mitigations: SELFIES instead of SMILES (invalid-by-construction is impossible), scaffold-constrained generation (x = scaffold, y = full molecule), or a chemistry-pretrained base.
  2. Reviewer-fit: a federated-LLM-steering paper with a chemistry experiment reads as unusual at NeurIPS — but as a *second* task next to Newsroom it reads as range, not as a detour.

### Candidate 3 — Quality-controlled paraphrase generation

- **Data:** the QCPG (ACL 2022) setup — MSCOCO captions, WikiAnswers question clusters, ParaBank 2.0. Public, large, sentence-level.
- **Attribute:** QCPG's three dimensions; the relevant one is **lexical divergence = normalized character-level edit distance between bags of words** (or 1 − self-BLEU). Deterministic and continuous. Syntactic divergence (tree edit distance) is a second, composable coordinate — relevant if multi-attribute ever comes back in scope.
- **Clients:** source corpus × domain. Captions, Q&A-site questions and back-translated ParaBank sentences have very different divergence distributions — though this is **provenance heterogeneity**, which is a weaker story than Candidate 2's mechanistic skew (see §4).
- **R6: the best of any candidate.** Divergence is entirely free given the input — any sentence admits a near-copy paraphrase or a complete rewrite. No confound with correctness at all.
- **Why prompting fails:** "paraphrase with lexical distance 0.4" is not a thing a model can hit; it cannot compute edit distance against its own output.
- **Utility:** semantic preservation (NLI entailment or BERTScore) — cheap, and the obvious failure mode (high divergence by changing the meaning) is exactly what that metric catches.
- **Compute:** trivial (~30-token outputs).
- **Risk:** paraphrase generation looks dated for a 2026 submission, and "is fine-grained paraphrase control useful?" is a weaker answer than test-time compute or lead optimization. Best positioned as a cheap third task that stresses composition, not as the second flagship.

### Candidate 4 — Table-to-text coverage (ToTTo)

120k+ examples, Wikipedia tables with annotator-highlighted cells and faithful targets. Attribute = **fraction of highlighted cells actually verbalized**, checked by string match: deterministic, continuous, free given the input, and unreachable by prompting ("mention 60% of these cells" is not instruction-followable at a percentile). Weak point is **R3**: no natural client field with the right skew — Wikipedia table *category* is a partition, but there is no reason for categories to differ systematically in coverage, so the heterogeneity would be semi-synthetic. Keep as a fallback.

### Candidate 5 — Science-QA answer verbosity (**the zero-cost pilot — data already on disk**)

`MegaScience` is already in the local HF cache (3.5 GB, 8 shards, `question` / `answer` / `subject` / `source`). I measured the verbosity heterogeneity directly (`answer` word count, groups ≥2k examples):

| source / subject | n | median's global percentile | support (p5→p95, global percentile) |
|---|---|---|---|
| natural_reasoning | 429,019 | 0.80 | [0.43, 0.98] |
| nemotro_science | 173,086 | 0.61 | [0.35, 0.88] |
| textbook_reasoning / cs | 17,737 | 0.32 | [0.04, 0.68] |
| textbook_reasoning / physics | 41,410 | 0.30 | [0.04, 0.72] |
| textbook_reasoning / math | 424,044 | 0.28 | [0.03, 0.69] |
| textbook_reasoning / chemistry | 32,150 | 0.21 | [0.02, 0.62] |
| textbook_reasoning / medicine | 81,638 | 0.18 | [0.02, 0.71] |
| textbook_reasoning / biology | 52,850 | 0.18 | [0.01, 0.70] |

Global length percentiles: p5 = 61, p25 = 157, p50 = 302, p75 = 482, p95 = 982 words.

This is **the Newsroom structure reproduced on a second task**: `natural_reasoning` lives in [0.43, 0.98] and never sees the bottom half of the scale; the textbook subjects live in [0.02, 0.70] and never see the top. Exactly the disjoint-support setting C1 is about, and with no download.

Catch: only 8 usable groups, and the split that carries the heterogeneity is **provenance** (`source`), not institution. Within `textbook_reasoning`, the subject-level clients are nearly homogeneous (medians 0.18–0.32) — so a "each university department is a client" framing would have almost no skew to exploit. Use it as a **fast sanity run** that the method transfers off summarization, not as a paper task.

---

## 3. Rejected, with the reason (keep for the rebuttal)

| Rejected | Why |
|---|---|
| **Newsela / OneStopEnglish** readability levels | 3–5 discrete levels, ~1.8k articles / 567 texts — fails R1 and far too small. Newsela also needs a signed data agreement |
| **FKGL / readability as the attribute** | fails R2 on *validity*, not cost: FKGL "is not a text simplification evaluation metric" (ACL GEM 2021) and is trivially gamed by sentence splitting. A reviewer kills the paper with one citation |
| **Sentiment / star rating** (current plan §3.2, Amazon Reviews 2023) | fails **R5**. Instruction models hit "write a 2-star review" easily, so B1 does not fail and gate G1 is lost. This is the main reason to replace §3.2 |
| **Toxicity** | Perspective API shuts down 31 Dec 2026 (already dropped) |
| **Formality control (CoCoA-MT)** | binary formal/informal — fails R1 |
| **Code: cyclomatic complexity per repo** | fails **R6** (the spec dictates complexity, so low-α = wrong code) and **R5** is shaky: one-shot prompting already reduces AvgCyclomatic/MaxCyclomatic measurably. Also, steering is known to corrupt code syntax, confounding the method with a generic failure |
| **Text-to-SQL complexity (Spider/BIRD)** | same R6 failure: the question determines the join/nesting count |
| **Technicality / jargon density** | partially promptable ("explain like I'm five") — R5 risk |
| **Persona / trait intensity, creativity, sycophancy, refusal** | already out of scope; all fail R5 and most fail R2 |

---

## 4. The one thing to decide first

Both top candidates beat Newsroom on scorer quality and on the federated story. They differ on which risk you'd rather carry:

- **Candidate 1 (reasoning length)** has the best *reviewer* story — test-time compute is the attribute everyone already agrees is worth controlling, the "prompting fails" gate is backed by three papers, and accuracy is a real utility metric. Its risk is **compute**: long generations, needing vLLM and probably a 4B base.
- **Candidate 2 (molecules)** has the best *scientific* story — mechanistic, documented client skew; a real-world federation (MELLODDY) that makes the privacy motivation genuine rather than stipulated; and the cheapest evaluation by an order of magnitude. Its risk is **base-model competence** on SMILES at 1B scale.

A cheap way to resolve it before committing: a one-day feasibility probe on each.
1. **Candidate 1:** count per-source examples in OpenR1-Math-220k's 94k default split and the within-problem length spread across its 2–4 traces. If ≥12 sources clear 5k pairs and within-problem spread is wide, R6 is solved and the task is viable.
2. **Candidate 2:** take 5k ChEMBL molecules, SFT Llama-3.2-1B for an hour, and measure SMILES validity. If validity clears ~90%, the cheap task is live; if not, switch to SELFIES and re-measure.

Candidate 5 needs no probe at all and could run this week on data already on disk — worth doing regardless, purely to show the method is not Newsroom-specific.

---

## 5. Sources

- Newsroom / extractiveness: Grusky et al., *Newsroom* (NAACL 2018)
- [L1: Controlling How Long A Reasoning Model Thinks With RL](https://arxiv.org/pdf/2503.04697) · [project page](https://cmu-l3.github.io/l1/)
- [Token-Budget-Aware LLM Reasoning](https://aclanthology.org/2025.findings-acl.1274.pdf) · [BudgetThinker](https://arxiv.org/pdf/2508.17196) · [SelfBudgeter](https://arxiv.org/html/2505.11274v5)
- [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) · [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) · [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) · [NuminaMath report](http://faculty.bicmr.pku.edu.cn/~dongbin/Publications/numina_dataset.pdf)
- [SmileyLlama: modifying LLMs for directed chemical space exploration](https://www.nature.com/articles/s43588-026-00986-y) · [arXiv](https://arxiv.org/pdf/2409.02231)
- [Therapeutics Data Commons](https://arxiv.org/pdf/2102.09548) · [Mol-Instructions](https://arxiv.org/pdf/2306.08018)
- [Federated learning of molecular properties with GNNs in a heterogeneous setting](https://www.cell.com/patterns/fulltext/S2666-3899(22)00118-0) · [Federated learning from molecules to processes](https://arxiv.org/pdf/2506.18525) (MELLODDY)
- [A physicochemical descriptor-based scoring scheme for kinase-like chemical space](https://link.springer.com/article/10.1186/1758-2946-4-4) · [Considerations for Target Selection in CNS Drug Discovery](https://www.sfn.org/~/media/SfN/Documents/Short%20Courses/2011%20Short%20Course%20III/2011_SC3_Hitchcock.ashx)
- [Quality Controlled Paraphrase Generation](https://aclanthology.org/2022.acl-long.45.pdf) · [code](https://github.com/IBM/quality-controlled-paraphrase-generation)
- [ToTTo: A Controlled Table-To-Text Generation Dataset](https://aclanthology.org/2020.emnlp-main.89.pdf)
- [Flesch-Kincaid is Not a Text Simplification Evaluation Metric](https://aclanthology.org/2021.gem-1.1.pdf) · [OneStopEnglish corpus](https://aclanthology.org/W18-0535.pdf)
- [Refactoring Programs Using LLMs with Few-Shot Examples](https://arxiv.org/pdf/2311.11690) · [CodeSearchNet](https://github.com/github/CodeSearchNet)
- MegaScience statistics: measured locally from the HF cache, 1 Oct 2026
