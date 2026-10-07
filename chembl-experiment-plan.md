# Federated steering on ChEMBL: scaffold decoration with a lipophilicity knob

Experiment plan · drafted 2 October 2026 · original proposal; implemented changes and results are recorded in §11.
Companion to `federated-steering-plan.md` (method, metrics and baselines are reused unchanged unless stated).
Dataset scouting that led here: `dataset-candidates.md`. Measured statistics: `data/chembl/` + §3 below.

Status markers: ✅ measured / done · ⏳ planned · ⚠️ known risk.

Maintenance: record substantial ChEMBL code, dataset, metric, and experiment changes in this file’s dated change log (§11), as requested on 2026-10-05. Historical proposal sections remain context; dated entries specify implemented changes.

---

## 1. Why this task

The method claim is unchanged: **clients each fine-tune their own model, jointly learn one steering direction, and can then reach attribute levels their own data barely covers.** This task is a second instance of it, chosen against the five criteria in `dataset-candidates.md`:

| Criterion | How ChEMBL scores |
|---|---|
| Continuous, fine-grained | ✅ cLogP is a real-valued RDKit descriptor; no discrete level set |
| Cheap deterministic scorer | ✅✅ RDKit, microseconds per molecule, no API, no judge, no model |
| Client heterogeneity | ✅ client medians span 0.08 → 0.90 of the global scale (wider than Newsroom's). ⚠️ but supports are *broad*, see §3.3 |
| Not summarization | ✅ |
| Prompting cannot reach it | ✅✅ a general LM cannot compute cLogP for the SMILES it is emitting; SmileyLlama needed SFT **and** DPO on weights, and SFT alone was "quite poor" |
| Attribute free given the input | ✅ by construction — see §2.2, this is what the residual attribute is for |

Two things this task adds that Newsroom cannot:
1. **A real federation.** MELLODDY is an actual consortium of pharma companies that could not pool compounds for legal reasons, each holding a library skewed by its own historical programs. The privacy motivation stops being stipulated.
2. **A ~40× cheaper evaluation.** Generated molecules are a median of **36 Llama-3.2 tokens** (p95 67) against Newsroom's 400-word articles and ~60-token summaries, with a scorer that costs nothing. A full 12-client × 200-test × 5-α sweep is ≈ 480k generated tokens — minutes, not hours. The 11-point α grid, multi-sample decoding at T = 0.7 and 3 seeds — all ⏳ on Newsroom because of cost — become affordable here.

---

## 2. Task definition

### 2.1 Input and output

- **x** = the target's preferred name + a **Murcko scaffold** (the molecule's ring system with linkers, stripped of substituents).
- **y** = the full molecule, as a canonical SMILES string.
- In words: *"decorate this core into a ligand for this target."* This is the lead-optimization step, and it mirrors the Newsroom structure one-to-one — varied input, one reference output per input, per-input paired comparisons, and a real reference at the same α.

### 2.2 The attribute: cLogP residual ✅ measured

\[
a(y) \;=\; \text{cLogP}(y) - \text{cLogP}(\text{core given in the prompt})
\]

— how lipophilic the decorations are. **α(y) = F(a(y))**, with F the equal-weight mixture CDF over participating clients, exactly as in `federated-steering-plan.md` §2 (`alpha_mode: global`).

**The baseline is the *requested* core, never the generation's own Murcko scaffold.** ✅ verified
With a self-referential baseline the model controls the quantity being subtracted: swap the core and the residual moves with it, so any α is reachable without decorating what was asked. Worse, the attribute becomes **independent of the prompt**. Measured on two real test prompts whose requested cores differ by 17 logP units (cLogP −7.60 and +9.35), one fixed generation scores:

| scorer | vs. core A | vs. core B | difference |
|---|---|---|---|
| `clogp_residual` (requested core) | 13.34 | −3.60 | **16.95** |
| own Murcko core (rejected) | 2.90 | 2.90 | **0.00** |

On training targets the two definitions coincide exactly — the prompt's core *is* that target's Murcko scaffold by construction — so the labels are unchanged; they diverge only where a generation abandons the core. Both scorers reproduce the stored labels to 1e-14.

**Scaffold preservation is therefore verified, not assumed.** `keeps_scaffold` (RDKit substructure match of the requested core in the generation) is reported per client and per α by `scripts/score_molecules.py`, and it is a **G0 pass condition**. Abandoning the core stays a *quality* failure rather than a definitional loophole — the same division of labour as the Newsroom task, where density is read next to a faithfulness metric instead of redefining density.

**Robustness check.** `clogp_residual_strict` returns `nan` unless the core survives, and `scripts/rescore_eval.py` recomputes every metric from the saved generations under it with no regeneration, so the main table can state how much of the steering result rests on core-abandoning outputs. It is not the primary metric because it drops a whole prompt when any single α loses the core. Unscorable prompts are dropped from row-wise metrics and reported as `unscorable_row_rate`.

**Why the residual and not raw cLogP.** The input must not determine the attribute (criterion R6), or there is nothing to steer. A Murcko scaffold is a subgraph of its molecule, so every raw descriptor is largely fixed by the input. Measured, over 8,261 scaffolds with ≥ 5 molecules, as the median within-scaffold p5–p95 spread on the global α scale:

| attribute | within-scaffold α spread (free given x) | client-median spread (heterogeneity) | corr(scaffold, molecule) |
|---|---|---|---|
| **cLogP residual** | **0.31** | **0.81** | — |
| mw residual | 0.30 | 0.51 | — |
| heavy_atoms residual | 0.26 | 0.50 | — |
| rotb residual | 0.24 | 0.50 | — |
| raw clogp | 0.18 | 0.73 | +0.76 |
| raw mw | 0.18 | 0.79 | +0.88 |
| raw tpsa | 0.15 | 0.79 | +0.85 |
| tpsa residual | 0.17 | 0.61 | — |
| raw qed | 0.15 | 0.72 | +0.69 |
| raw arom_rings | — | 0.55 | **+1.00** (fully determined — unusable) |
| hbd residual | 0.07 | 0.41 | — |

**cLogP residual wins on both axes at once**: the most freedom given the input (0.31, nearly double raw cLogP's 0.18) *and* the widest spread of client medians (0.81). Raw descriptors are 0.15–0.18 — i.e. ~85% of the attribute would be dictated by the input, which would make the direction unidentifiable. TPSA is additionally **discrete** (mass piles up at 9.2, 18.5, 26.0 …), which would wreck a percentile scale; cLogP is continuous.

⚠️ **Tie mass:** 3.7% of rows have residual exactly 0 — the molecule *is* its own scaffold (no decorations). Drop these at build time; they are not a decoration task.

### 2.3 Why prompting (B1) must fail here

The gate is stronger than on Newsroom. To comply with "decorate this core so that the added groups land at the 20th percentile of lipophilicity", a model would have to evaluate Crippen cLogP on a SMILES string it has not finished writing, and on its scaffold, and subtract. There is no lexical proxy. SmileyLlama is the published evidence that property control over ChEMBL requires weight training, with SFT alone insufficient.

### 2.4 Output format: decorations only ✅ adopted 2026-10-03

Gates G0 / G0b / G0-SFT (exp log 33-36) killed the whole-molecule format. **Plain SFT with no steering at all** keeps the requested core on only **0.536** of generations, and 100 rounds of steered training made it *worse* (0.498) than 20 rounds (0.576). The core is lost because the model must re-emit and renumber it, not because of the knob.

So the task follows [Arús-Pous et al. 2020](https://link.springer.com/article/10.1186/s13321-020-00441-8) and SAFE's scaffold-decoration convention (§4b):

- **x** = target name + the core carrying **numbered attachment points**, e.g. `[1*]c1cc([2*])c(Sc2ccccc2)cc1[3*]`
- **y** = the **decorations only**, as `[n*]`-labelled fragments joined by `.`, e.g. `[1*]S(N)(=O)=O.[2*]C(=O)NC1CCCCC1.[3*]Cl`
- scoring reassembles with RDKit `molzip` (matched isotope labels on both sides of every cut), then applies the same attribute.

Why this is the right fix rather than SELFIES: SELFIES guarantees *parseability* and does nothing for *retention*, which is the larger gap. Here **the core comes from the prompt and is never generated, so retention is structural** — verified at 1.000 when the true decorations are used.

Properties of the rebuilt data (`data/chembl_deco`, same clients, same splits, 36,687 records):
- the attribute and therefore every **α label is unchanged** (scorer reproduces the stored labels to 2.8e-14), so the two formats are directly comparable;
- per-client yield 0.71–0.96 (pairs whose core/decoration split round-trips exactly are kept), train 1,396–4,000, test 142–198;
- targets are **half as long**: median 17 tokens against 33 for whole molecules, so there is also less to get wrong.

Scorer: `clogp_residual_deco`. A generation that does not zip onto the core scores `nan` and is counted, not crashed on.

---

## 3. Data ✅ built

Everything below is measured, not estimated. Scripts: `scripts/chembl_targets.py`, `chembl_target_meta.py`, `chembl_stats.py`, `chembl_report.py`, `chembl_select_clients.py`. Data: `data/chembl/` → `/projects/illinois/eng/cs/arindamb/lucmon/data/chembl`.

### 3.1 Provenance and filtering

- **Activities:** `martinakaduc/ChEMBL_activities` (20.3M rows). Filtered to **assay_type = B** (binding), **confidence_score ≥ 8**, **standard_relation = "="**, **units = nM**, **standard_type ∈ {Ki, Kd, IC50, EC50}**, value > 0 → **1,337,996 activities** over **5,995 targets**.
- **Structures:** ChEMBL 37 `chembl_37_chemreps.txt.gz` from the EBI FTP (authoritative); **99.8%** of molecule IDs resolved.
- **Target metadata:** ChEMBL REST API — preferred name, organism, target type, and protein class (L1–L3) for the top 120 targets.
- Target sizes: **17** targets have ≥ 5,000 distinct molecules, **40** have ≥ 4,000, **116** have ≥ 2,000, **247** have ≥ 1,000. The Newsroom budget (4k pairs/client, 12 clients) fits comfortably.

### 3.2 Clients ✅ `data/chembl/clients.json`

One client = one protein target. 12 clients, selected to spread over the α scale, one per protein family, pairwise molecule overlap ≤ 0.15. Held-out clients are every third, stratified — the Newsroom rotation scheme.

| target | name | family | n | raw med | α(med) | support [α5, α95] | role |
|---|---|---|---|---|---|---|---|
| CHEMBL205 | Carbonic anhydrase 2 | lyase | 7,409 | −1.21 | 0.12 | [0.01, 0.75] | participant |
| CHEMBL204 | Prothrombin (thrombin) | protease | 5,148 | −0.37 | 0.31 | [0.04, 0.94] | **held-out** |
| CHEMBL243 | HIV-1 protease | protease | 4,646 | −0.31 | 0.32 | [0.03, 0.96] | participant |
| CHEMBL236 | Delta opioid receptor | membrane receptor | 4,100 | −0.16 | 0.37 | [0.03, 0.93] | participant |
| CHEMBL2835 | Tyrosine kinase JAK1 | kinase | 4,554 | 0.00 | 0.46 | [0.11, 0.92] | **held-out** |
| CHEMBL325 | Histone deacetylase 1 | epigenetic | 5,674 | 0.01 | 0.48 | [0.18, 0.99] | participant |
| CHEMBL4078 | Acetylcholinesterase | hydrolase | 5,363 | 0.14 | 0.54 | [0.02, 0.96] | participant |
| CHEMBL4005 | PI3K (p110-alpha) | transferase | 6,102 | 0.15 | 0.55 | [0.16, 0.90] | **held-out** |
| CHEMBL2039 | Monoamine oxidase B | reductase | 4,116 | 0.31 | 0.60 | [0.23, 0.92] | participant |
| CHEMBL4409 | PDE4 | phosphodiesterase | 4,559 | 0.33 | 0.61 | [0.23, 0.92] | participant |
| CHEMBL240 | hERG (KCNH2) | ion channel | 8,613 | 0.43 | 0.65 | [0.17, 0.96] | **held-out** |
| CHEMBL228 | Serotonin transporter | transporter | 4,659 | 0.54 | 0.68 | [0.22, 0.98] | participant |

- **64,943 (target, molecule) rows; 63,581 distinct molecules** — i.e. essentially no cross-client duplication (pairwise molecule overlap mean **0.005**, max 0.121).
- The heterogeneity is **mechanistic, not incidental**: carbonic anhydrase ligands are small polar sulfonamides, thrombin and HIV-protease inhibitors are large and polar, while hERG binders and SERT ligands are the lipophilic end of chemical space. These are the documented property biases of the families, not a quirk of the sample.
- A second, skew-maximizing selection is in `data/chembl/clients_skewed.json` (`--mode skewed`, ≤ 2 per family): overlap drops to mean 0.001 but coverage barely improves (see below), so the **even** set is the default.

### 3.3 ⚠️ The honest weakness: supports are broad

Newsroom's clients have genuinely disjoint supports (nypost.com 0.51–0.95 vs. theguardian.com 0.04–0.55). ChEMBL's do not. Each target has thousands of structurally diverse ligands spanning most of the lipophilicity range, so:

- mean uncovered share of the α scale per client: **0.19** (even set) / 0.23 (skewed set);
- share of clients covering each α: α = 0.3–0.7 is covered by **100%** of clients; only the extremes thin out (α = 0.1: 42%, α = 0.9: 83%).

**Consequence for claim C1.** Natural coverage gaps are *weaker* here than on Newsroom, so the collaboration claim cannot lean on natural skew. It must come from the **controlled coverage experiment** — truncate each client's training data to part of the α scale and test outside it. That experiment is already in `federated-steering-plan.md` §4 (E1 "controlled coverage", and `--alpha_window` for E2); here it becomes the *primary* evidence rather than a supplement. The flip side: this is a different and arguably more realistic heterogeneity regime — **different centres, overlapping tails** — and having both regimes in one paper is a strength, provided the paper says which is which.

### 3.4 Splits ⏳

Per client, from its ≥ 4,000 rows:

- **train** up to 4,000 (x, y) pairs — the Newsroom budget, so results are comparable;
- **dev** 100, **test** 200.
- **Split by scaffold, never by molecule.** A scaffold appearing in train and test leaks the answer. Measured: 25,126 distinct scaffolds over the 12-client set, median 1 molecule per scaffold, max 1,667.
- **Test set restricted to scaffolds with ≥ 5 molecules** (2,586 scaffolds, 52% of rows), so every test input has several real molecules at different α — this is what gives a **same-α reference** for the quality metrics, and it enables an eval Newsroom cannot do: *same input, real targets at two different α*.
- Drop rows with residual exactly 0 (3.7%).
- No temporal drift split for now; E3 is optional in the parent plan. If wanted later, ChEMBL document years give a natural drift axis.

---

## 4. Model and configuration

**Unchanged from the parent plan** — this is the point of running a second task. `W_0 + B_i^P A_i^P + g(α) B^D A^D`, shared frozen `A^D`, FedAvg on `B^D` only, `calibration: shared`, `offset: false`, `adapter: private`, `alpha_mode: global`, uniform server averaging, E = 20 local steps × batch 8, rank 16, ~30 rounds.

What needs new code:
1. **A dataset adapter** in `fedsteer/data.py`: read `data/chembl/molecules.parquet`, build prompts, score with RDKit, emit the same record shape the Newsroom loader emits.
2. **An RDKit scorer** (`fedsteer/molecules.py`) ✅ done: `a(y) = MolLogP(y) − MolLogP(requested core)`, `nan` for unparseable output so invalid generations are counted rather than crashed on, plus `clogp_residual_strict` (core must survive) and `clogp_residual_self` (drift diagnostic only).
3. **Prompt format**, fixed for all methods:
   `Target: {pref_name}\nCore: {scaffold_smiles}\nLigand:` → ` {smiles}`.
4. Everything else — training loop, α reference, snapshots, eval harness, `compare_runs.py` — is reused.

**Base model.** Start with Llama-3.2-1B-Instruct for the smoke test (it is what the Newsroom pilot used), but the headline runs should use whichever of Llama-3.2-1B / Qwen3-4B clears gate G0 below. ⚠️ Neither is chemistry-pretrained; validity is the main risk and G0 exists to catch it early.

---

## 4b. Prior art: how the SMILES generation problem is normally solved

Added 3 October 2026, after gates G0 / G0b / G0-SFT (exp log 33-36). This belongs in the paper's related work, and it changed the task's output format.

**Three places prior work intervenes.**

**1. The output format — the standard fix for losing the core.**
- [Arús-Pous et al. 2020, *SMILES-based deep generative scaffold decorator*](https://link.springer.com/article/10.1186/s13321-020-00441-8) (J. Cheminformatics) extends SMILES with a `[*]` attachment-point token and generates **only the decorations**. Trained on DRD2 actives and RECAP-sliced ChEMBL.
- [SAFE, *Gotta be SAFE*](https://arxiv.org/abs/2310.10773) (Noutahi et al., Digital Discovery 2024) re-notates molecules as a sequence of fragment blocks that stay RDKit-parseable, turning scaffold decoration into prefix continuation. An 87M GPT-2 on 1.1B SAFE strings beats SMILES-based models on scaffold decoration and linker design.
- [PromptSMILES](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00866-5) (Thomas et al. 2024) needs no retraining: it rearranges the scaffold SMILES so the attachment point is the final atom and lets the model continue the string.

**2. The tokenizer.** Atom-level regex tokenization (Schwaller; DeepChem/ChemBERTa); [SMILES Pair Encoding](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c01127) (Li & Fourches, JCIM 2021) — BPE initialized from atom tokens, ~3–5k chemically readable substrings, matching or beating atom-level tokenization on 24 QSAR benchmarks; Smirk and Atom-in-SMILES for "atomically complete" vocabularies, since bracket atoms encode isotope/chirality/charge/H-count and SPE-style vocabularies still emit unknown tokens (~19% on MoleculeNet, ~50% on tmQM); and [AtomDisc](https://arxiv.org/abs/2512.03080) (2025), which vector-quantizes each atom's local environment into new vocabulary entries.

⚠️ **This route is closed to us.** Every one of them replaces the tokenizer, hence the embedding matrix, hence the pretrained model. The method needs a frozen general backbone with LoRA on top, so we accept the tokenization penalty — measured on our data as **1.46 characters per token** (English is ~4), with ring-closure digits always standalone tokens and the frequent merges (`)(`, `)c`, `(N`, `)=`, `(F`) cutting across chemical boundaries.

**3. Dedicated chemical models vs. general LLMs.** Dedicated, trained from scratch: MolGPT, Chemformer, MolT5, ChemBERTa, SAFE-GPT, Chem42. General LLM + SFT keeping the general tokenizer (our regime): SmileyLlama, [Mol-Instructions](https://arxiv.org/pdf/2306.08018) (which converts everything to SELFIES), ChemLLM.

**Positioning consequence, to state plainly in the paper.** [MolGPT](https://pubs.acs.org/jcisd8/article/62/9/2064/883937/MolGPT-Molecular-Generation-Using-a-Transformer) already conditions generation on **scaffold SMILES together with continuous property values, including logP, TPSA, SAS and QED**. That is this task minus the federation. So the ChEMBL task must not be presented as novel controllable generation — the contribution is the *federated shared direction*, and this task is here to show the method transfers off summarization.

**What it does not change.** PromptSMILES enforces the *scaffold constraint* by string surgery, not by instruction, and says nothing about hitting a property percentile. Gate G1 (prompting fails to calibrate) is unaffected.

---

## 5. Metrics

Reused from the parent plan, with task-specific substitutions:

| Dimension | Newsroom | ChEMBL |
|---|---|---|
| **Calibration (primary)** | percentile error \|F(a(ŷ)) − α\| | same, on cLogP residual. **Constant-output baseline = 0.33** ✅ measured (always emit the client's median; 5-point grid). Newsroom's was 0.30 |
| **Coverage (key for C1)** | in/out-of-support error, reach rate | same, but out-of-support comes mostly from the **controlled truncation** (§3.3) |
| **Steerability** | per-article Spearman, concordance | per-scaffold Spearman across α |
| **Ties** | near-identical text | **exact SMILES identity** after canonicalization — cleaner than Newsroom's fuzzy near-tie rule |
| **Validity (new, gating)** | — | share of outputs RDKit can parse. **Every other metric is conditional on validity, so it is reported first, always** |
| **Faithfulness to the input (new)** | AlignScore | **scaffold retention**: does the generated molecule still contain the requested core (RDKit substructure match)? Deterministic, reported per α, and a G0 pass condition — the attribute's baseline is the requested core, so retention is what keeps the two definitions aligned |
| **Quality / utility** | AlignScore, BERTScore, LLM judge | validity, uniqueness, novelty vs. train, SA score, QED; **plus a QSAR activity predictor** for the client's own target, trained on *held-out* targets' data or on the client's own actives/decoys. All deterministic, **no LLM judge anywhere** |
| **Specificity (off-target)** | length, FKGL | MW and TPSA drift per unit cLogP-residual change — a real chemistry concern (don't just bolt on alkyl chains) |

Reporting follows the parent plan: mean and worst client, per-client paired bootstrap over test inputs, checkpoint selected on dev. Because generation is cheap, **do here what is ⏳ there**: 11-point α grid, 4 samples at T = 0.7, and 3 training seeds for the main table.

---

## 6. Baselines

Identical IDs to the parent plan, so the two tasks produce the same table shape.

| ID | Baseline | Note for this task |
|---|---|---|
| **B1** | Prompting on the client's fine-tuned model | **The gate.** Numeric level in the prompt, k ∈ {0, 3} nearest-α few-shot molecules. Expected to fail badly |
| **B2** | Local-only: each client learns its own D | The decisive C1 comparison |
| **B3** | One-shot merged direction | Tests whether iterative federation is needed |
| **B4** | Federated activation steering (CAA) | Weight vs. activation space. ⚠️ watch validity — activation steering is known to corrupt structured output |
| **B5** | Pooled reference (all data centralized) | Cost of decentralization. Cheap here, so run it |
| **A2** | Shared adapter (non-personalized FedAvg) | Answers "isn't this just PFL?" |

---

## 7. Experiments

- **E0 — feasibility (the gate).** SFT Llama-3.2-1B on one client's 4k pairs, no steering. Measure validity, uniqueness, scaffold retention. ~1 GPU-hour.
- **E1 — participants (C1).** 8 participants, federated vs. local, in/out of support, 4k budget. Plus the **controlled coverage experiment**, which is primary here: truncate 3 balanced clients to α ≤ 0.6 or ≥ 0.4, retrain, test at the removed α against their real held-out molecules.
- **E2 — held-out clients (C2).** The 4 held-out targets join with n ∈ {16, 64, 256, 1024, all}; `frozen_D` vs. `local_D` vs. `plugin` vs. `prompt`. Reuse `e2_heldout.py` unchanged apart from the data adapter.
- **E3 — drift.** Skip initially; optional in the parent plan. ChEMBL document years are the natural drift axis if it comes back.

---

## 8. Decision gates

Run in order; each one can stop the task cheaply.

| Gate | Test | Pass condition | If it fails |
|---|---|---|---|
| **G0** | E0 validity + retention after plain SFT | **≥ 90%** valid, **≥ 80%** retention | ⚠️ **FAILED** (exp log 33-36): whole-molecule SMILES gives validity 0.84 and retention **0.54 even with no steering at all**, and more training makes it worse. Fix adopted: the **decoration format** (§2.4) -- the core is fixed in the prompt and the model emits only the decorations, so retention is structural. SELFIES is *not* the fix: it would address validity and leave retention untouched |
| **G1** | B1 prompting | percentile error **≥ 0.30** (i.e. no better than constant) | The knob is not needed; drop the task |
| **G2** | Method vs. constant baseline | error **< 0.25** (vs. 0.33 constant) | The attribute is not learnable at this scale; try raw cLogP, accept the weaker R6 |
| **G3** | Method vs. B2 local | better on a majority of clients, above all in the truncated regions | C1 does not transfer off Newsroom — an interesting negative, report it |

---

## 9. Risks

1. ⚠️ **Validity (highest).** A general 1B LM has no chemistry pretraining. Mitigations in G0's order: SELFIES → bigger base → chemistry base. SELFIES costs interpretability of the prompt but guarantees parseable output.
2. ⚠️ **Broad supports** (§3.3) — mitigated by making controlled truncation the primary coverage evidence, and by stating the two heterogeneity regimes explicitly.
3. ⚠️ **Lipophilicity is easy to game**: the model can hit high α by appending alkyl chains. This is exactly what the specificity (MW/TPSA drift) and quality (SA score, QED, QSAR activity) metrics exist to catch. Report them at every α, never only at the best one.
4. **Scaffold leakage** between train and test — addressed by splitting on scaffold (§3.4).
5. **Activity labels are noisy** across assays and labs; the QSAR utility metric inherits that noise. Treat it as a secondary metric, with validity/retention/SA as the primary quality gate.
6. **Reviewer fit.** A chemistry experiment in an FL-for-LLMs paper reads as range if Newsroom stays the flagship, and as a detour if it displaces it. Keep Newsroom first.

---

## 10. Next steps

1. ⏳ `scripts/build_chembl_fed.py` — splits by scaffold, α reference, drop residual-0 rows, write `data/chembl_fed/` in the same layout as `data/newsroom_fed/`.
2. ⏳ `fedsteer` dataset adapter + RDKit scorer + prompt format.
3. ⏳ **E0 / gate G0** on one client (~1 GPU-hour). Decide SMILES vs. SELFIES here.
4. ⏳ **B1 / gate G1** — cheap, and it decides whether the task is worth any training at all.
5. ⏳ 12-client smoke run (3 rounds), then the 30-round method + local pair.

Steps 3 and 4 together are under half a day of GPU time and settle whether this task is viable, before anything long is launched.


---

## 11. Dated implementation and experiment changes

### 2026-10-05 — Parent-ligand metrics and a separately pruned dataset

**Status: implemented and dataset built; no new training jobs launched by this change.**
This entry records the output audit, the supplementary failure-penalty metric, and the new
parent-ligand scoring/data contract. It supersedes the proposal's assumption that successful
RDKit parsing/assembly alone establishes valid scaffold decoration. Legacy scores remain available.

**Finding.** Exp43 outputs can change cLogP without changing the attached ligand by adding
unattached `Cl` or `[Br-]` components. These patterns also occur in training targets; they are
parseable multicomponent records, not necessarily invalid under the original data representation.
The mismatch is with the intended *single parent ligand* attribute. See
[the concrete output audit](exp_log/reports/exp43_output_audit.md). This does not prove a model
intentionally exploits the scorer or that shared calibration alone explains the run difference.

**Definitions, applied identically to federated and local outputs:**

- `clogp_residual_deco`: existing whole-assembly scorer, retained for historical comparisons.
- `clogp_residual_deco_pruned`: remove unlabelled extra components, then score only the attached
  parent ligand relative to the requested scaffold. Extras cannot change the descriptor and
  do not themselves incur a failure penalty. Missing/duplicate/invalid attachment labels and
  malformed final ligands remain unscorable.
- `clogp_residual_deco_strict`: reject extras rather than remove them. Every positive attachment
  label must occur exactly once on each side and be bonded to its fragment; assembly must leave
  one connected ligand and no dummy atoms, retaining the requested scaffold. Charged parents
  remain allowed; this is not charge neutralization or a full chemical standardization pipeline.
- `pct_calib_err_penalized`: average every generated prompt × alpha cell, using ordinary absolute
  percentile error for a finite score and **1** for an unscorable cell. One is the global maximum
  error, although the maximum valid error at fixed alpha is `max(alpha, 1-alpha)`. This is a
  supplementary failure-cost convention; report validity and conditional metrics alongside it.

Full evaluations with the legacy decoration scorer now also save `pct_calib_err_pruned`,
`pct_calib_err_strict`, `pct_calib_err_penalized_pruned`, `pct_calib_err_penalized_strict`,
plus each variant's out-of-support error and unscorable-sweep rate. All are summarized across
clients with lower-is-better worst-client handling. The pruned and strict scorers can also be
selected explicitly in `eval_direction.py` and `scripts/rescore_eval.py`. Checkpoint selection
supports the new error metrics with `--select`; its default is unchanged.

**Post-hoc Exp43 sensitivity result.** Same dev-selected checkpoints (FED r80, LOCAL r100),
identical saved generations and original saved run CDFs. Scores are recomputed from the texts
at full precision. Previous four-decimal-grid audits differ slightly because the empirical CDF
has jumps. No checkpoints or prompts were reselected for these comparisons.

| All-cell error, penalty 1 (lower is better) | FED 21068552 | LOCAL 21068553 |
|---|---:|---:|
| Legacy whole assembly | 0.200600 | **0.194659** |
| **Pruned parent ligand** | 0.204054 | **0.200185** |
| **Strict parent-ligand contract** | **0.215583** | 0.229812 |

Pruning alone does **not** reverse the ranking; strict rejection plus the failure penalty does.
This is a task-definition sensitivity analysis, not evidence that the cleaned-data experiment
has succeeded. Both definitions are reported rather than choosing only the favorable one.
Machine-readable per-client results: [Exp43 ligand scoring reports](exp_log/reports/exp43_ligand_scoring/).
Historical rescoring uses the original CDF to keep the requested alpha scale fixed; cleaned-data
training uses a new CDF. Scores across those different references are not directly comparable.

**New dataset: `data/chembl_deco_skew_pruned/`.** Built from `data/chembl_deco_skew/` by
`scripts/prune_chembl.py`. The rule uses record structure only, not model performance or alpha.
Apply the same filtering to **all 12 clients and every split**; drop affected records rather than
silently rewriting their targets. Retained prompts, targets, record IDs, client assignments and
split assignments are preserved. Scores are recomputed from the accepted assembly. The builder
checks per-client scaffold disjointness and refuses to overwrite an existing output directory.

- Total retained: **28,732 / 30,114**; dropped: **1,382**, all for unattached fragments.
- Train: **26,200 / 27,457**; dev: **513 / 542**; test: **2,019 / 2,115**.
- Fresh local quantiles and the rotation-0 equal-client global CDF use **retained train records
  only**. Training independently rebuilds the reference for its configured participants.
- Recomputing scores changes 8 retained labels by more than `1e-8` (maximum absolute change
  0.08168); this is recorded separately from dropping samples, rather than assuming all stored
  source labels exactly equal the current assembly scorer. All eight are CHEMBL4005 training
  records (a held-out client in rotation 0). The current legacy and strict scorers agree on their
  recomputed values, so these discrepancies predate the new strict/pruned distinction.
- Original source data is preserved. Dataset artifacts include `rejected.jsonl` (IDs and reasons),
  `pruning_manifest.json` (source hashes, RDKit version, counts, support ranges and score changes),
  and `source_clients.json` (the original metadata). Old skew/median statistics are not reused as
  statistics of the new dataset.

| Participating client | Original train | Pruned train |
|---|---:|---:|
| CHEMBL243 | 1560 | 1554 |
| CHEMBL204 | 3278 | 2876 |
| CHEMBL325 | 2028 | 2020 |
| CHEMBL2835 | 1495 | 1493 |
| CHEMBL4078 | 1135 | 1073 |
| CHEMBL2039 | 710 | 643 |
| CHEMBL240 | 3642 | 3535 |
| CHEMBL228 | 1303 | 1146 |

The tracked [manifest copy](exp_log/reports/chembl_deco_skew_pruned_manifest.json) records all clients
and splits; the actual dataset lives under the repository's ignored `data/` directory.

**Ready-to-run configuration:** `configs/chembl_deco_skew_pruned.yaml`, 100 rounds, private
calibration in both modes, strict scorer for full monitoring, and the same 20-round checkpoint
cadence / 25 dev prompts / 96 generation tokens used for the prior skew pair. The local run changes
only `fed.mode` and its output prefix. This config was prepared, not submitted.

```bash
# Build a new directory; the builder will not overwrite an existing dataset.
python scripts/prune_chembl.py --source_dir data/chembl_deco_skew \
  --out_dir data/chembl_deco_skew_pruned

python train_fed.py --config configs/chembl_deco_skew_pruned.yaml
python train_fed.py --config configs/chembl_deco_skew_pruned.yaml \
  --set fed.mode=local out_dir=runs/chembl_deco_skew_pruned_local

# For the new strict-scorer experiment, select on dev with the all-cell failure penalty.
python scripts/summarize_sweep.py --run runs/NEW_RUN --select pct_calib_err_penalized
# Evaluate its chosen snapshot with --scorer clogp_residual_deco_strict.
```

To reproduce the historical sensitivity scores without regeneration (repeat for LOCAL):

```bash
python scripts/rescore_eval.py \
  --run runs/anvil/exp43_skew_fed_anvil_20261004-231441_j21068552 \
  --eval runs/anvil/exp43_skew_fed_anvil_20261004-231441_j21068552/evals/eval_round_0080__20261004-231441_j21068552.json \
  --scorer clogp_residual_deco --out_dir exp_log/reports/exp43_ligand_scoring
```

**Limits.** This removes the identified disconnected-component shortcut; it does not establish
biological activity, remove all possible descriptor shortcuts (e.g. excessively long attached
chains), or convert soft skew into genuinely absent alpha regions. The old alpha supports were
5th–95th percentiles, not min–max training ranges. More favorable strict-score results do not
establish a federated advantage on the new dataset; that requires matched training and evaluation.

**Validation (completed).** Eight focused CPU regression tests passed in the `steer` environment:
fragment-pruning invariance, strict rejection, attachment failures, valid multi-attachment
decoration, dataset filtering/reference isolation, scorer registration, supplemental full-eval
integration, and checkpoint selection minimizing the new error. The existing failure-penalty,
metric-edge-case, support-split and near-tie tests also passed. The exported dataset CDF exactly
matches the reference independently rebuilt by the training pipeline. Both selected Exp43 test
files were rescored successfully, and the comparison report displays both new penalty-1 metrics.
`git diff --check` passed.

### 2026-10-05 — Matched federated/local pair launched on the pruned dataset

**Status: submitted.** This supersedes the previous entry's closing note that
`configs/chembl_deco_skew_pruned.yaml` was prepared but not submitted. Experiment log: `MOL-17`.

**What this run is for.** The previous entry's strict-contract result (FED 0.2156 vs LOCAL 0.2298
on the all-cell penalty-1 score) was a *post-hoc rescoring of Exp43's existing generations*, under
a task definition those runs were not trained for, and it is the only one of the three scoring
variants that favoured federated. It therefore establishes nothing about C1 on its own. This pair
trains, selects and evaluates entirely under the single-parent-ligand contract on the pruned data,
which is what the earlier result would need in order to mean anything.

**Design: only the direction varies.** Both arms use `fed.calibration=private`. MOL-14's confound
was that the federated arm had shared calibration (every client pinned to gain 1.053) while the
local arm necessarily had private calibration (gains 1.251–1.670), so direction-sharing and
calibration-sharing moved together; the correlation between a client's locally chosen gain and how
much federation hurt it was +0.76. Holding calibration private in both arms removes that factor.
`fed.py` rejects `local` + `shared`, so private-in-both is the only legal matched setting and a
full 2×2 remains unavailable.

**The strict contract is applied at every stage**, which is the part that distinguishes this from a
rescoring: in-training dev monitoring (`monitor.scorer=clogp_residual_deco_strict`), checkpoint
selection (`pct_calib_err_penalized`, so a checkpoint that produces unscorable molecules is
penalised rather than having those cells dropped from its mean), and the test evaluation
(`--scorer clogp_residual_deco_strict`, 150 prompts per client). The legacy score is then computed
on the *same* generations (`RESCORE=clogp_residual_deco`), so the strict-vs-legacy gap is measured
on identical text instead of being inferred across runs.

| job | cc | Anvil |
|---|---|---|
| federated, private calibration, 100 rounds | 11169098 | 21102995 |
| local, private calibration, 100 rounds | 11169100 | 21102996 |

Duplicated across both clusters the molecule workstream is allowed to use; the loser of each pair
is cancelled once its counterpart is running and past startup. Launch script
`exp_log/launch/exp45_pruned_pair.sh`.

**Code changes required, both minimal.** `sbatch/train_eval_chembl.sbatch` and
`sbatch/train_eval_chembl_anvil.sbatch` take a `SELECT` variable, forwarded to both
`summarize_sweep.py` invocations; its default equals that script's own default (`pct_calib_err`),
so no previously launched experiment changes behaviour. `scripts/setup_anvil.sh` also syncs
`chembl_deco_skew_pruned`.

**Validation before submission.** On cc and on Anvil independently: the config loads, the three
decoration scorers are registered, the dataset reads 28,732 records split 26,200 / 513 / 2,019 in
agreement with the pruning manifest, and `clogp_residual_deco_strict` reproduces the stored label
of the first 200 records to within 1e-6. RDKit differs between the hosts (cc 2025.09.6, Anvil
2026.03.6; torch 2.5.1 on both), and both reproduce the stored labels, but results should carry
the host that produced them.

**What the outcomes mean.** A federated win under matched training would be the first on this task
and the first evidence for C1 outside Newsroom; before claiming it, the explanations discarded in
MOL-14 must be re-checked against the new per-client gains, and the unscorable-sweep rate compared
across arms, since under a penalised metric an arm can win by failing less rather than by
calibrating better. No win would mean the shortcut removal does not change the conclusion, and
ChEMBL's role reduces to C2/portability with C1 tested on Newsroom — the §3.3 reading.


### 2026-10-06 — Exp45 pruned pair completed: local still leads, sharing benefit unproven

**Runs:** Anvil FED **21102995**, LOCAL **21102996**. Both logs show
`configs/chembl_deco_skew_pruned.yaml`, the same client sample counts and strict scorer;
local changes `fed.mode=local`. The configuration uses private calibration in both arms,
100 rounds and seed 0. Dev selection uses `pct_calib_err_penalized`; the selected checkpoints
are FED r100 and LOCAL r80. No additional training or algorithm changes were made for this analysis.

This supersedes the preceding submission-status entry. **The matched, cleaned-data comparison
has not established a federated advantage.** The earlier calibration-sharing confound and the
post-hoc strict-score win on Exp43 are insufficient explanations of the new result. Cleaning
was necessary for a well-defined parent-ligand task; it did not make the shared-direction
method outperform local in this run.

| Selected test metric | FED | LOCAL |
|---|---:|---:|
| Percentile error, complete scorable sweeps | 0.207377 | **0.197065** |
| In-support error, complete scorable sweeps | 0.170150 | **0.164085** |
| Out-of-support error, complete scorable sweeps | 0.244470 | **0.227660** |
| All-cell percentile error, failure penalty 1 (selection metric) | 0.210637 | **0.207778** |
| Unscorable sweep rate | **0.019556** | 0.046407 |
| Spearman | 0.825169 | **0.842031** |
| Achieved percentile range | 0.498166 | **0.535229** |
| Adjacent tie rate | 0.430609 | **0.385090** |

The all-cell gap is **0.002859** (~1.36% of FED's error), smaller than the complete-sweep
calibration gap **0.010312** (~4.97%). Local trades more failures for better calibration on
scorable sweeps. The in/out-support figures above are conditional metrics and must not be
called failure-penalized in/out-support errors. Local wins on 7/8 clients' conditional overall
errors (CHEMBL243 is effectively tied), 6/8 all-cell errors, 6/8 in-support errors and 5/8
out-of-support errors. No significance claim is made from these single-seed logs.

| Client | FED all-cell error | LOCAL all-cell error | FED conditional OOS | LOCAL conditional OOS |
|---|---:|---:|---:|---:|
| CHEMBL243 | 0.2813 | 0.3075 | 0.2821 | 0.2844 |
| CHEMBL204 | 0.2218 | 0.2061 | 0.2307 | 0.1890 |
| CHEMBL325 | 0.1994 | 0.1701 | 0.2196 | 0.1782 |
| CHEMBL2835 | 0.2213 | 0.2154 | 0.2609 | 0.2696 |
| CHEMBL4078 | 0.1944 | 0.1748 | 0.2332 | 0.2194 |
| CHEMBL2039 | 0.1575 | 0.1492 | 0.2428 | 0.2032 |
| CHEMBL240 | 0.1696 | 0.2067 | 0.1931 | 0.1995 |
| CHEMBL228 | 0.2398 | 0.2323 | 0.2934 | 0.2780 |

**Behavioral finding: the federated response is narrower, mainly on the high side.** From the
logged complete-sweep mean percentile curves, the equal-client mean response at alpha 0 is
approximately 0.287 FED / 0.285 LOCAL; at alpha 1 it is 0.785 FED / 0.820 LOCAL. FED has more
adjacent ties. These aggregate curves are conditional on different complete-sweep subsets;
they support an under-reaching/saturation diagnosis, not a causal decomposition of it.
For example, CHEMBL204's achieved means at requested `[0, .25, .5, .75, 1]` are:

- FED: `[.171, .292, .462, .602, .710]`.
- LOCAL: `[.192, .353, .533, .696, .814]`.

The effect is not confined to CHEMBL4078. CHEMBL325 has all-cell error .1994 FED / .1701 LOCAL;
CHEMBL2039 has .1575 / .1492. Conversely, FED's all-cell error is substantially better on
CHEMBL243 (.2813 / .3075) and CHEMBL240 (.1696 / .2067). Local's missing-side performance
is not uniformly better either: using the logged rounded endpoint means to remove the
opposite endpoint from OOS error gives CHEMBL243 high-side error approximately .3095 FED /
.3108 LOCAL and CHEMBL228 low-side error .3921 / .3720. These are derived conditional means,
not common-valid-prompt comparisons or bootstrap results.

**Checkpoint selection is not the sole explanation.** At the same completed rounds 20/40/60/80/100,
FED dev conditional error is .246/.222/.209/.207/.205 and LOCAL is
.216/.211/.196/.190/.197. At round 100 their training losses are .0625/.0641 and dev losses
are .4519/.4366. Both fit their training data strongly; FED did not simply receive less training.

**What remains plausible, not established:** forcing all private client models to share one
weight direction can lose useful client-specific adaptations. Token-prediction training does
not isolate a purely attribute-only direction, and private calibration fitted on local data
does not by itself impose correct response curves in locally sparse regions. Sharing the
matrix therefore provides a possible transfer mechanism, not guaranteed coverage transfer.
Near-zero update cosines are a diagnostic, not proof of destructive interference. Gains are
.606–.956 FED vs 1.191–1.544 LOCAL, but the effective product is gain × direction; comparing
gains without direction norms cannot establish the cause of the narrower output range.

**The cleaned dataset still has soft support.** On its stored global reference, CHEMBL4078's
1,073 train examples span alpha approximately [.0970, .9975], with 27 below .25 and 89 above
.75, despite its reported central-90% support [.3000, .7982]. CHEMBL243 retains 17 examples
above .75; CHEMBL228 retains 12 below .25. Pruning removes disconnected-fragment shortcuts,
not tail examples. These results therefore test sparse-region generalization, not exclusively
transfer into truly unseen recipient regions.

**Next discriminating checks:** (1) establish uncertainty using multiple matched seeds and
paired per-prompt all-cell errors; (2) separate optimization from the sharing constraint—the
trainer's `reset_shared_opt_state=True` resets direction Adam state in fedavg mode but not in
local mode, so a matched optimizer-state ablation is informative; (3) evaluate interior alpha
regions absent from a recipient but demonstrably covered by donors, with the reference fixed
before constructing the split. A coverage-aware calibration variant, if tested, is a new
hypothesis and must be evaluated prospectively against local, not assumed to explain away
this result. Do not change the scoring definition to recover a federated win.

**Evidence/limits:** analysis uses the two supplied Anvil logs and the pruned dataset reference.
Their raw Anvil generation grids/snapshots were not found in the local workspace, so this entry
does not claim a new molecule-by-molecule audit or a paired confidence interval. Locally present
cc duplicate artifacts belong to different executions and were not substituted for these jobs.
Machine-readable extracted results: [Exp45 comparison](exp_log/reports/exp45_pruned_comparison.json).

### 2026-10-06 — Exp45 per-client pattern: coverage, endpoint response and failures

Expanded the same Anvil pair into a [per-client analysis](exp_log/reports/exp45_per_client_analysis.md),
with [structured results](exp_log/reports/exp45_per_client_analysis.json) and
[response curves](exp_log/reports/exp45_per_client_curves.png). Checked extracted client metrics
against both source logs, training score lists against the stored CDFs, and reconstructed
central-90% support against logged values. No training or scoring changes.

- Local's strongest conditional overall wins are CHEMBL325, CHEMBL204, CHEMBL4078 and
  CHEMBL2039. These clients show fewer adjacent response ties and higher achieved percentiles
  at requested alpha 1. FED wins the penalty-1 metric on CHEMBL243 and CHEMBL240, chiefly
  through fewer unscorable sweeps (2.0% vs 14.7%, and 1.3% vs 11.3%, respectively).
- CHEMBL204, CHEMBL325 and CHEMBL2039 contribute **91.2% of the net macro OOS advantage**
  for local. All have broad training coverage, with actual alpha minima near zero and maxima
  near one. Thus the main OOS advantage is not evidence of extrapolation from a hard narrow
  training interval.
- For **five clients** (204, 325, 2835, 2039, 240), the only OOS test alphas are `{0,1}`.
  Their conditional OOS error is exactly `(1 - pct_range) / 2`, where `pct_range` is the mean
  achieved endpoint difference. Verified numerically for both runs. Response span and OOS
  performance on these clients are therefore the same evidence, not independent findings.
- CHEMBL243 has 1,274/1,554 training examples below .25 and only 17 above .75. At requested
  alpha 1, achieved means are .638 FED / .614 LOCAL: neither reaches the high target well.
  CHEMBL228 has 700/1,146 above .75 and only 12 below .25. At requested alpha 0, achieved
  means are .495 / .475: local's small improvement still leaves poor low-side performance.
  CHEMBL4078 is concentrated in the middle (957/1,073 in [.25,.75)), but retains 27 low-tail
  and 89 high-tail examples; local's win there is under sparse tails, not absent tails.
- No simple sample-size explanation: local wins on the smallest client (2039, 643 records)
  and also on 204 (2,876), while FED wins on the largest (240, 3,535). CHEMBL2835 improves
  in-support but worsens OOS under local; 2039 and 228 show the opposite tradeoff.

**Interpretation:** local often fits a broader client-specific response, while FED improves
reliability on two clients. The strongest one-sided recipients remain difficult for both.
These observations are consistent with restricted response under a shared direction but do
not establish its cause; the optimizer-state difference remains an ablation to test. A fixed-CDF
recipient interior-alpha holdout with donor coverage would more directly test sharing benefits.
All curves/errors except penalty-1 scores condition on each model's own complete scorable
sweeps. This is one seed and dev-selected checkpoints differ; no paired significance test or
new individual-molecule audit is claimed.

### 2026-10-06 — What coverage calibration can diagnose or repair

Reviewed `federated-steering-plan.md` §2.1 against `fedsteer/warp.py` and the coverage
loss in `fedsteer/fed.py`. The augmentation is a plausible **joint-training regularizer**,
not an established remedy for Exp45 and not a post-hoc endpoint correction.

- With frozen P_i/D, gain 1 and no offset, W_i(0)=W_0+P_i and
  W_i(1)=W_0+P_i+D for every admissible warp. Changing the warp cannot change either
  endpoint or the continuous set of coefficient values available in [0,1]. In the historical
  learned-gain runs the high endpoint is W_0+P_i+s_i D; gain is an additional unresolved
  factor. Exp45 endpoint differences exclude a pure warp-shape-only explanation at frozen
  parameters, but cannot distinguish direction orientation, effective magnitude, or adapter
  co-adaptation. Five clients' OOS metric tests only these endpoints.
- During training, calibration changes the coefficients at which supervised examples train
  P_i and D. The coverage loss directly updates only warp parameters, but subsequent NLL
  updates of P_i/D change. It can therefore improve endpoints indirectly. Conversely, a
  donor's useful coefficient need not be useful on a recipient's different adapter/prompts;
  matching warp values is not matching generated attributes.
- NR-48 in `exp_log/EXPERIMENT_LOG.md` reports suggestive evidence for this coupling on
  Newsroom: at round 20, nypost's alpha-0 mean percentile is .55 with private warps versus
  .36 with coverage borrowing. Since h(0)=0, the endpoint change reflects a different
  trained private adapter. The proposed division of attribute signal between P_i and D is
  a mechanism hypothesis, not established by those aggregate outputs. This is single-seed
  interim Newsroom evidence, not a ChEMBL result or proof of superiority to shared calibration.

**Discriminating diagnostics, not yet executed:** freeze each selected P_i/D (and historical
gain), sweep the coefficient directly, and fit a client-level monotone remapping on dev,
evaluating error and strict validity on held-out prompts. Improvement within the original
coefficient interval measures available calibration headroom. A separately labelled wider
coefficient sweep tests magnitude/offset limitations; improved reach must retain validity.
Failure to find useful responses leaves a P_i/D response-path limitation, not proof that D
alone is poor. Then compare matched gain-1, no-offset joint-training arms with/without
coverage borrowing and local-only, controlling shared-optimizer-state resets. Frozen
calibration cannot test the training-mediated benefit; naive cross-run D/P swaps disrupt
co-adaptation and do not isolate intrinsic direction quality.

### 2026-10-06 — Coverage-borrowing arms launched (with/without borrowing, local-only, optimizer control)

**Status: submitted, 8 jobs (4 arms × 2 clusters).** Experiment log: `MOL-18`. This runs the
comparison the preceding entry ends with — "matched gain-1, no-offset joint-training arms
with/without coverage borrowing and local-only, controlling shared-optimizer-state resets" — and
runs it **prospectively against local**, which that entry requires. The scoring definition is
unchanged; NR-48's Newsroom repair is a reason to test the mechanism here, not evidence about it.

All four arms differ in one line only, and follow `federated-steering-plan.md` §2.1 as NR-46 did:
gain fixed at 1 (`fed.fix_gain=true` ⇒ u = 0, s = exp(0) = 1), no offset, `kumaraswamy_mix` warp,
identity penalty off (`fed.warp_reg=0`), full participation, `data/chembl_deco_skew_pruned`,
100 rounds, rotation 0, seed 0, strict scorer at every stage, dev selection on
`pct_calib_err_penalized`.

| arm | what it isolates | cc | Anvil |
|---|---|---|---|
| COV — `calibration=coverage`, λ_max = 1 | borrowing present | 11180420 | 21142036 |
| PRIV — `calibration=private` | borrowing absent | 11180421 | 21142037 |
| LOCAL — `mode=local` | the bar the plan sets | 11180422 | 21142038 |
| NORESET — `reset_shared_opt_state=false` | the Adam-state reset | 11180423 | 21142039 |

COV − PRIV is the borrowing effect; PRIV − NORESET is the optimizer-reset effect that has been
confounded with sharing in every fed-vs-local comparison so far, because
`reset_shared_opt_state=True` acts only in fedavg mode (`fed.py:270`). Local mode cannot be made to
reset without a code change, so the control is a fedavg arm that *keeps* the state. The four arms
form one race group, so they cannot split across hosts with different RDKit versions.

**These arms are not comparable with Exp45's numbers**, which used a learned gain (.606–.956 FED,
1.191–1.544 LOCAL). Here gain is 1 in all four. They are comparable with each other.

**Pre-flight measurement that justifies running it.** Borrowing is in an active regime on this
dataset, and λ_ik lands on the designed gaps rather than spreading uniformly: CHEMBL243 (low
specialist) borrows at .84–.89 above α 0.8 and ≈.1 below 0.2; CHEMBL228 (high specialist) borrows
at .82–.93 below 0.2; the middle-only CHEMBL4078 borrows at **both** ends (.92/.83 low, .59/.76
high); the broad donors CHEMBL240 and CHEMBL204 stay ≤ .34. λ mean .354, and no grid point has zero
total evidence, so the keep-prior rule is unexercised here as on Newsroom. λ_max = 1 is NR-48's
value and is untuned for ChEMBL; NR-49's λ sweep informs any follow-up.

This addresses the preceding entry's *training-mediated* question only. The frozen-P_i/D coefficient
sweep and the fitted monotone remapping it also lists remain unexecuted, and frozen calibration
cannot test the training-mediated route in any case.

**Reading order when they finish.** COV vs PRIV restricted to each client's borrowed region
(CHEMBL243 above .75, CHEMBL228 below .25, CHEMBL4078 both ends), since that is where λ is large;
then COV vs LOCAL, which is the actual bar; then PRIV vs NORESET; then the unscorable-sweep rate per
arm, because a penalty-1 metric can be won by failing less rather than calibrating better; then the
logged warps and server table z, to confirm borrowing moved the warps at all. One seed, dev-selected
checkpoints, so no significance claim — multiple matched seeds remain the preceding entry's item (1).

**Infrastructure defect fixed in passing.** NR-50 found that Quadro RTX 6000 (Turing) nodes run
these jobs several times slower and closed by noting `sbatch/train_eval_chembl.sbatch` had not been
fixed. It had not: the cc arms were submitted able to land there, and a Turing copy running 20
minutes would have satisfied the race watcher and cancelled the Anvil copies in favour of a job that
cannot finish. `ccc0232–0236` are now excluded in the pending cc jobs and in that sbatch file;
`ccc0496–0499` are Blackwell, not Turing, and stay available. Measured on this task: Exp45's cc
duplicate on ccc0235 runs 855 s/round against ~180 s/round on an A100, so 100 rounds exceeds the
24 h limit and times out before its test evaluation.
