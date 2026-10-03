# Candidate second/flagship tasks beyond summarization

Dataset scouting · 1 October 2026 · companion to `federated-steering-plan.md` §3.
Status: research only, nothing built. Supersedes §3.2 (Amazon review rating) as the recommended secondary task.

---

## 1. The screen

Five requirements, from the plan and from the 10-01 discussion:

| # | Requirement | Why it bites |
|---|---|---|
| R1 | **Continuous, fine-grained** attribute (not 2–5 discrete levels) | $\alpha \in [0,1]$ needs a dense scale; a 3-level corpus (Newsela, OneStopEnglish) can't give a percentile CDF |
| R2 | **Attribute intensity cheap and deterministic to score** | the scorer is called on every training target *and* every generation; no Perspective API (shuts down 31 Dec 2026), no paid judge in the loop |
| R3 | **Attribute-level heterogeneity across natural clients** | the whole C1 claim is "each client's data covers a different part of the range"; needs a natural partition field, $\ge 12$ clients with $\ge 5\text{k}$ pairs, and *disjoint supports* |
| R4 | **Not summarization** | the flagship already is |
| R5 | **Prompting must not reach it** (B1 is the gate) | if an instruction can hit the level, no knob is needed and the paper has no gate G1 |

**R5 is the binding constraint, and it has a crisp test.** Prompting fails exactly when the model cannot *measure* the attribute on its own output and the attribute has no natural verbalization at intermediate levels. That kills every "semantic" attribute (sentiment, star rating, formality, politeness, technicality, persona traits) — instruction-tuned models track those well from words alone. It favours attributes that are **corpus statistics**: copy-fragment density (the current flagship), exact token/step counts, edit distance to the input, syllable counts.

**A sixth criterion, not on the user's list but decisive** — call it R6: **the attribute must be free given the input.** If the input determines the attribute, $\alpha$ cannot vary within a problem and "steering" collides with correctness. This is what disqualifies the two most obvious ideas (code complexity, SQL complexity): the spec dictates the answer's complexity, so asking for $\alpha = 0.2$ on a hard function asks for a wrong answer. Newsroom passes R6 because any article can be summarized extractively or abstractively.

---

## 2. Ranked candidates

| Rank | Task | Attribute | Scorer | Clients | R1 | R2 | R3 | R5 | R6 | Compute |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | Math CoT reasoning | trace length / step count | token count (exact) | problem source | ✅ | ✅✅ | ✅ | ✅ | ⚠️→✅ | ⚠️ high |
| **2** | Paraphrase generation | lexical / syntactic divergence | edit distance, tree distance | source corpus | ✅ | ✅ | ✅ | ✅ | ✅✅ | ✅ low |
| 3 | Table-to-text | cell coverage | string match on cells | table domain | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| 4 | Science QA verbosity | answer length | word count | source $\times$ subject | ✅ | ✅✅ | ✅ | ✅ | ⚠️ | ✅ |
| ✗ | Code generation | cyclomatic complexity | radon (exact) | repository | ✅ | ✅✅ | ✅ | ⚠️ | ❌ | ✅ |

---

### Candidate 1 — Reasoning-effort control on math CoT (**recommended flagship-grade second task**)

- **Data:** `open-r1/OpenR1-Math-220k` (94k default / 131k extended), built on `AI-MO/NuminaMath-1.5`. **2–4 independent DeepSeek-R1 traces per problem**, 16k-token cap, each problem has $\ge 1$ verified-correct trace (Math Verify, plus Llama-3.3-70B judge on 12%).
- **Why the multi-trace structure matters:** it is the **R6 fix**. Different traces of the *same* problem have very different lengths, so $\alpha$ varies *within* a problem, not just across problems. That removes the length$\leftrightarrow$difficulty confound and buys an evaluation Newsroom cannot support: **same input, two real targets at two different $\alpha$**, i.e. ground truth at both ends.
  - ⚠️ **Measured 10-02, and it does not hold** (`scripts/openr1_stats.py`): only 6.4% of log-length variance is within a problem, and the median longest/shortest trace ratio is 1.27. The R6 fix is instead a per-problem *target ladder* (reference solution, R1 write-up, truncated thinking, full trace), with 83% of the variance within a problem. See `math-cot-experiment-plan.md` §2.3.
- **Attribute:** generated-trace token count $\rightarrow$ global percentile. Completely deterministic, zero scorer cost, no judge, no API.
- **Clients:** NuminaMath 1.5's `source` field (`olympiads`, `cn_k12`, `aops_forum`, `cn_contest`, `amc_aime`, `gsm8k`, `math`, synthetic variants) — "each institution has its own problem bank and its own solution-style convention". Natural skew is large: gsm8k solutions are short, olympiad solutions long. Second axis if sources are too few: `source` $\times$ `problem_type` (Algebra / Geometry / Number Theory / Combinatorics).
  - ⚠️ **Check first:** per-source counts in the *220k* subset, not in NuminaMath-CoT. cn_k12 and olympiads dominate (276k / 150k in CoT); amc_aime had only 4k. Needs $\ge 12$ groups at $\ge 5\text{k}$.
- **Why prompting fails (R5) — this is the strongest literature case of any candidate.** Prompt-level token budgets are known to break down: models revert to longer reasoning and abandon the constraint when the budget is tight, and budget-following is only non-trivial inside an empirically narrow "ideal window". The whole point of L1/LCPO, BudgetThinker and SelfBudgeter is that **weight training (RL or control tokens) is required** to make length controllable. So B1 fails for a documented reason, and G1 is defensible to reviewers without hand-waving.
- **Why fine-grained control is already desirable (the plan's criterion 1):** this *is* the test-time-compute knob. It is the one controllable-generation attribute a 2026 reviewer will accept as useful without argument.
- **Utility metric: answer accuracy, verifiable.** Far stronger than AlignScore/BERTScore — no judge, no "is this faithful" ambiguity. And the accuracy-vs-$\alpha$ curve is itself a result (the accuracy/compute trade-off per client).
- **Federated story:** credible. Each client has its own problem bank and its own trace-length convention; a client whose data is all short grade-school solutions cannot produce long olympiad-style traces, and vice versa. That is the coverage argument, verbatim.
- **Risks:**
  1. **Compute.** This is the real cost. $12\text{ clients} \times 200\text{ test} \times 5\alpha \times \approx 1.5\text{k tokens} \approx 18\text{M}$ generated tokens per full eval. The current HF `generate` greedy loop will not take it — needs vLLM, or a hard 1k-token cap and short-CoT data (which weakens the "effort" framing).
  2. A 1B base model produces weak traces; accuracy may floor out and hide the accuracy/$\alpha$ effect. Qwen3-4B or a distilled 1.5B R1 would be the honest choice.
  3. "Length is a trivial attribute" is a likely reviewer jab. Mitigation: report **step count** (deterministic: enumerated steps / equation lines) alongside tokens, and lead with accuracy-at-budget.

### Candidate 2 — Quality-controlled paraphrase generation

- **Data:** the QCPG (ACL 2022) setup — MSCOCO captions, WikiAnswers question clusters, ParaBank 2.0. Public, large, sentence-level.
- **Attribute:** QCPG's three dimensions; the relevant one is **lexical divergence = normalized character-level edit distance between bags of words** (or $1 - \text{self-BLEU}$). Deterministic and continuous. Syntactic divergence (tree edit distance) is a second, composable coordinate — relevant if multi-attribute ever comes back in scope.
- **Clients:** source corpus $\times$ domain. Captions, Q&A-site questions and back-translated ParaBank sentences have very different divergence distributions — though this is **provenance heterogeneity**.
- **R6: the best of any candidate.** Divergence is entirely free given the input — any sentence admits a near-copy paraphrase or a complete rewrite. No confound with correctness at all.
- **Why prompting fails:** "paraphrase with lexical distance 0.4" is not a thing a model can hit; it cannot compute edit distance against its own output.
- **Utility:** semantic preservation (NLI entailment or BERTScore) — cheap, and the obvious failure mode (high divergence by changing the meaning) is exactly what that metric catches.
- **Compute:** trivial ($\approx 30$-token outputs).
- **Risk:** paraphrase generation looks dated for a 2026 submission, and "is fine-grained paraphrase control useful?" is a weaker answer than test-time compute. Best positioned as a cheap candidate that stresses composition, not as the main second flagship.

### Candidate 3 — Table-to-text coverage (ToTTo)

120k+ examples, Wikipedia tables with annotator-highlighted cells and faithful targets. Attribute = **fraction of highlighted cells actually verbalized**, checked by string match: deterministic, continuous, free given the input, and unreachable by prompting ("mention 60% of these cells" is not instruction-followable at a percentile). Weak point is **R3**: no natural client field with the right skew — Wikipedia table *category* is a partition, but there is no reason for categories to differ systematically in coverage, so the heterogeneity would be semi-synthetic. Keep as a fallback.

### Candidate 4 — Science-QA answer verbosity (**the zero-cost pilot — data already on disk**)

`MegaScience` is already in the local HF cache (3.5 GB, 8 shards, `question` / `answer` / `subject` / `source`). I measured the verbosity heterogeneity directly (`answer` word count, groups $\ge 2\text{k}$ examples):

| source / subject | n | median's global percentile | support (p5$\rightarrow$p95, global percentile) |
|---|---|---|---|
| natural_reasoning | 429,019 | 0.80 | [0.43, 0.98] |
| nemotro_science | 173,086 | 0.61 | [0.35, 0.88] |
| textbook_reasoning / cs | 17,737 | 0.32 | [0.04, 0.68] |
| textbook_reasoning / physics | 41,410 | 0.30 | [0.04, 0.72] |
| textbook_reasoning / math | 424,044 | 0.28 | [0.03, 0.69] |
| textbook_reasoning / chemistry | 32,150 | 0.21 | [0.02, 0.62] |

Global length percentiles: p5 = 61, p25 = 157, p50 = 302, p75 = 482, p95 = 982 words.

This is **the Newsroom structure reproduced on a second task**: `natural_reasoning` lives in [0.43, 0.98] and never sees the bottom half of the scale; the textbook subjects live in [0.02, 0.70] and never see the top. Exactly the disjoint-support setting C1 is about, and with no download.

Catch: only 8 usable groups, and the split that carries the heterogeneity is **provenance** (`source`), not institution. Within `textbook_reasoning`, the subject-level clients are nearly homogeneous (medians 0.18–0.32) — so a "each university department is a client" framing would have almost no skew to exploit. Use it as a **fast sanity run** that the method transfers off summarization, not as a paper task.

---

## 3. Rejected, with the reason (keep for the rebuttal)

| Rejected | Why |
|---|---|
| **Newsela / OneStopEnglish** readability levels | 3–5 discrete levels, $\approx 1.8\text{k}$ articles / 567 texts — fails R1 and far too small. Newsela also needs a signed data agreement |
| **FKGL / readability as the attribute** | fails R2 on *validity*, not cost: FKGL "is not a text simplification evaluation metric" (ACL GEM 2021) and is trivially gamed by sentence splitting. A reviewer kills the paper with one citation |
| **Sentiment / star rating** (current plan §3.2, Amazon Reviews 2023) | fails **R5**. Instruction models hit "write a 2-star review" easily, so B1 does not fail and gate G1 is lost. This is the main reason to replace §3.2 |
| **Toxicity** | Perspective API shuts down 31 Dec 2026 (already dropped) |
| **Formality control (CoCoA-MT)** | binary formal/informal — fails R1 |
| **Code: cyclomatic complexity per repo** | fails **R6** (the spec dictates complexity, so low-$\alpha$ = wrong code) and **R5** is shaky: one-shot prompting already reduces AvgCyclomatic/MaxCyclomatic measurably. Also, steering is known to corrupt code syntax, confounding the method with a generic failure |
| **Text-to-SQL complexity (Spider/BIRD)** | same R6 failure: the question determines the join/nesting count |
| **Technicality / jargon density** | partially promptable ("explain like I'm five") — R5 risk |
| **Persona / trait intensity, creativity, sycophancy, refusal** | already out of scope; all fail R5 and most fail R2 |

---

## 4. The one thing to decide first

Candidate 1 (reasoning length) beats Newsroom on scorer quality and on the federated story. Test-time compute is the attribute everyone already agrees is worth controlling, the "prompting fails" gate is backed by three papers, and accuracy is a real utility metric. Its main risk is **compute**: long generations, needing vLLM and probably a 4B base.

A cheap feasibility probe before committing:
1. **Candidate 1:** count per-source examples in OpenR1-Math-220k's 94k default split and the within-problem length spread across its 2–4 traces. If $\ge 12$ sources clear $5\text{k}$ pairs and within-problem spread is wide, R6 is solved and the task is viable.

Candidate 4 needs no probe at all and could run this week on data already on disk — worth doing regardless, purely to show the method is not Newsroom-specific.

---

## 5. Sources

- Newsroom / extractiveness: Grusky et al., *Newsroom* (NAACL 2018)
- [L1: Controlling How Long A Reasoning Model Thinks With RL](https://arxiv.org/pdf/2503.04697) · [project page](https://cmu-l3.github.io/l1/)
- [Token-Budget-Aware LLM Reasoning](https://aclanthology.org/2025.findings-acl.1274.pdf) · [BudgetThinker](https://arxiv.org/pdf/2508.17196) · [SelfBudgeter](https://arxiv.org/html/2505.11274v5)
- [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) · [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) · [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) · [NuminaMath report](http://faculty.bicmr.pku.edu.cn/~dongbin/Publications/numina_dataset.pdf)
- [Quality Controlled Paraphrase Generation](https://aclanthology.org/2022.acl-long.45.pdf) · [code](https://github.com/IBM/quality-controlled-paraphrase-generation)
- [ToTTo: A Controlled Table-To-Text Generation Dataset](https://aclanthology.org/2020.emnlp-main.89.pdf)
- [Flesch-Kincaid is Not a Text Simplification Evaluation Metric](https://aclanthology.org/2021.gem-1.1.pdf) · [OneStopEnglish corpus](https://aclanthology.org/W18-0535.pdf)
- [Refactoring Programs Using LLMs with Few-Shot Examples](https://arxiv.org/pdf/2311.11690) · [CodeSearchNet](https://github.com/github/CodeSearchNet)
- MegaScience statistics: measured locally from the HF cache, 1 Oct 2026