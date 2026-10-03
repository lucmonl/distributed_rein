# Federated steering on ChEMBL: scaffold decoration with a lipophilicity knob

Experiment plan · drafted 2 October 2026 · proposal, nothing run yet.
Companion to `federated-steering-plan.md` (method, metrics and baselines are reused unchanged unless stated).
Dataset scouting that led here: `dataset-candidates.md`. Measured statistics: `data/chembl/` + §3 below.

Status markers: ✅ measured / done · ⏳ planned · ⚠️ known risk.

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
