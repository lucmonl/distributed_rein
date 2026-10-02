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
a(y) \;=\; \text{cLogP}(y) - \text{cLogP}(\text{scaffold}(y))
\]

— how lipophilic the decorations are. **α(y) = F(a(y))**, with F the equal-weight mixture CDF over participating clients, exactly as in `federated-steering-plan.md` §2 (`alpha_mode: global`).

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
2. **An RDKit scorer** module (`fedsteer/scorers/chembl.py`): `a(y) = Crippen.MolLogP(y) − Crippen.MolLogP(MurckoScaffold(y))`, returning `None` for unparseable output so invalid generations are counted, not crashed on.
3. **Prompt format**, fixed for all methods:
   `Target: {pref_name}\nCore: {scaffold_smiles}\nLigand:` → ` {smiles}`.
4. Everything else — training loop, α reference, snapshots, eval harness, `compare_runs.py` — is reused.

**Base model.** Start with Llama-3.2-1B-Instruct for the smoke test (it is what the Newsroom pilot used), but the headline runs should use whichever of Llama-3.2-1B / Qwen3-4B clears gate G0 below. ⚠️ Neither is chemistry-pretrained; validity is the main risk and G0 exists to catch it early.

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
| **Faithfulness to the input (new)** | AlignScore | **scaffold retention**: does the generated molecule still contain the requested core (substructure match)? Deterministic |
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
| **G0** | E0 validity after plain SFT | **≥ 90%** valid SMILES, **≥ 80%** scaffold retention | Switch to **SELFIES** (invalid-by-construction is impossible), then to Qwen3-4B, then to a chemistry-pretrained base |
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
