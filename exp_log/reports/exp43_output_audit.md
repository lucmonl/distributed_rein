# Exp43: concrete output audit (2026-10-05)

Inputs: `runs/anvil/exp43_skew_fed_anvil_20261004-231441_j21068552/evals/eval_round_0080__20261004-231441_j21068552.json`
and `runs/anvil/exp43_skew_local_anvil_20261004-231949_j21068553/evals/eval_round_0100__20261004-231949_j21068553.json`.
Training records: `data/chembl_deco_skew/data.jsonl`; percentile reference: each run's `alpha_reference.json`.

Inspected all 5,960 generated cells per model (150 prompts for seven clients, 142 for CHEMBL2039, five alphas).
Reassembled each generation with the existing RDKit-based `rejoin_decorations`, then counted connected
components and unresolved dummy atoms. Diagnostic strict validity requires successful assembly, one
connected component, and no dummy atoms. The original scorer does not enforce the last two conditions.
Percentile-error recalculations below use the saved four-decimal score grids, so they are approximate.
This is an output audit, not a change to the production scorer or a retraining experiment.

## A concrete scoring loophole

`fedsteer/molecules.py:202` zips and sanitizes decorations without requiring a connected final molecule
or complete consumption of attachment labels. `clogp_residual_deco` scores the whole assembled object,
including unattached fragments. `assembled_descriptors` also calls a successfully sanitized assembly valid.
A dot between labelled decorations is expected; the defect is extra disconnected components **after assembly**.

Example: CHEMBL4078/CHEMBL4077879 at alpha 0.

| Model | Output | Recorded residual | Approx. achieved percentile |
|---|---|---:|---:|
| Federated | `[1*]C` | 0.3422 | 0.582 |
| Local | `[1*]C.[Br-]` | -2.6538 | 0.020 |

The attached ligand is identical. Removing the unattached bromide from the local assembly gives residual
0.3422 again. The apparent low-alpha improvement is entirely due to the disconnected fragment.

Example: CHEMBL228/CHEMBL562802, local model.

| Requested alpha | Output | Recorded residual | Approx. achieved percentile |
|---|---|---:|---:|
| 0.75 | `[1*]Cl` | 0.6534 | 0.694 |
| 1 | `Cl.Cl.Cl.Cl.[1*]Cl.Cl.Cl.Cl` | 3.6060 | 0.994 |

At alpha 1, the assembled object has eight components: the ligand plus seven unattached `Cl` fragments.
The ligand component still has residual 0.6534. Its apparently near-perfect endpoint is not a change to
the attached ligand. Recomputing the exact unrounded component score maps to about percentile 0.704;
the difference from 0.694 reflects the saved-score rounding and the discrete empirical CDF.

This shortcut has training-data precedents. Of CHEMBL4078's 1,135 training targets, 62 have at least one
unlabelled dot-separated fragment; CHEMBL228 has 157/1,303. CHEMBL4078 contains 12 targets with a free
`[Br-]` fragment, including one exactly `[1*]C.[Br-]`. This is consistent with learning a data shortcut,
not evidence of intentional optimization against the evaluator. The data builder preserves salt/extra
components: it requires reconstruction of the original input, which can itself be disconnected.

## Frequency and effect on the comparison

| Audit result | Federated | Local |
|---|---:|---:|
| Disconnected assembled outputs / 5,960 | 97 | 227 |
| Outputs with unresolved dummy atoms | 8 | 9 |
| Strictly valid cells / 5,960 | 5,834 (97.89%) | 5,671 (95.15%) |
| CHEMBL4078 disconnected outputs at alpha 0 / 150 | 19 | 60 |
| CHEMBL228 disconnected outputs at alpha 1 / 150 | 28 | 55 |

Apply penalty 1 to unsuccessful assemblies, disconnected assemblies, or assemblies with unresolved
attachment points; retain ordinary percentile error for the remaining cells. Average cells per client,
then average clients equally. Keep the original reference CDF and checkpoint choices fixed.

| Error (lower is better) | Federated | Local |
|---|---:|---:|
| Previously added penalty-1 score (only non-finite scores fail) | 0.200779 | 0.194735 |
| Diagnostic strict penalty-1 score | **0.215762** | 0.229889 |
| Diagnostic strict penalty-1 score, original out-of-support grid cells | **0.247981** | 0.272226 |
| CHEMBL4078 diagnostic strict penalty-1 score | **0.228198** | 0.274035 |

Thus the previous penalty is insufficient: many task-invalid objects get finite scores and evade it.
The aggregate ranking reverses under this single-connected-ligand requirement. This sensitivity result
does not prove federation is universally better. Rebuilding training data and its CDF may change both models.

## Genuine steering also exists

CHEMBL4078/CHEMBL3957322 gives a clear local success with all outputs connected and all labels consumed.

| Alpha | Federated decorations | Achieved percentile | Local decorations | Achieved percentile |
|---|---|---:|---|---:|
| 0 | `[1*]C.[2*]C.[3*]N` | 0.519 | `[1*]N.[2*]O.[3*]O` | 0.151 |
| 0.25 | `[1*]C.[2*]C.[3*]C#N` | 0.645 | `[1*]N.[2*]O.[3*]C#N` | 0.181 |
| 0.5 | `[1*]C.[2*]Cl.[3*]C#N` | 0.762 | `[1*]C.[2*]O.[3*]C#N` | 0.369 |
| 0.75 | `[1*]C.[2*]Cl.[3*]C#N` | 0.762 | `[1*]C.[2*]Cl.[3*]C#N` | 0.762 |
| 1 | `[1*]C.[2*]Cl.[3*]C#N` | 0.762 | `[1*]Cl.[2*]Br.[3*]Cl` | 0.952 |

Neither local endpoint decoration string occurs exactly among this client's training targets. But its
training targets contain labelled N, O, Cl and Br fragments in 118, 235, 192 and 42 rows respectively
(counting rows containing each fragment). This supports compositional generalization, rather than exact
full-target memorization, for this example. It does not establish which training records caused it.
Federated steering moves in the right direction initially and then repeats the same output.

There are also clear federated successes. On CHEMBL243/CHEMBL311953 the achieved percentile sweeps are:

- Federated: `[0.107, 0.107, 0.107, 0.488, 0.890]`.
- Local: `[0.107, 0.107, 0.107, 0.120, 0.120]`.

Every output in both sweeps passes the strict assembly check. Local saturates; federation changes the
attached decorations at high alpha. This is an illustrative example, not an unbiased effect-size estimate.

## What “unseen alpha” means here

The displayed support is the local 5th–95th percentile interval mapped to the global reference. It is not
training min–max. The skew procedure deliberately keeps a nonzero probability of retaining both tails.

| Client | Training examples | Examples below alpha .25 | Above alpha .75 | Actual training alpha min–max |
|---|---:|---:|---:|---|
| CHEMBL243 | 1,560 | 1,274 | 17 | .0017–.9667 |
| CHEMBL228 | 1,303 | 12 | 825 | .1140–.9997 |
| CHEMBL4078 | 1,135 | 43 | 100 | .0018–.9965 |

Consequently, much of the purported out-of-support evaluation is sparse-region generalization, not
extrapolation beyond the observed training range. Even truly unseen alpha values can change outputs:
alpha enters a continuous coefficient `s*h(alpha)` on a learned matrix, rather than selecting a separately
trained discrete category. Extrapolation can succeed, saturate, or produce malformed decorations.

## Interpretation and next steps

Both models learn a coarse direction, with repeated outputs and incomplete calibration. On the strict
out-of-support diagnostic, only about 25.0% of federated and 29.1% of local requested cells both pass the
assembly check and land within .1 percentile of the requested alpha. Local therefore has more close hits
as well as more failures; a single aggregate score conceals this tradeoff. Their strict penalized errors
(.248/.272) beat a hypothetical always-median response on these same out-of-support grid cells (.456),
but this reference is not a measured frozen-base or zero-direction baseline.

There is no basis for calling them equally incapable or for saying federation did not learn steering.
The outputs show genuine changes under alpha, successes in both directions of the comparison, and a
substantial validity/scoring problem concentrated in local's apparently strongest regions. The existing
shared-versus-private calibration confound remains; gain differences alone do not establish causality.

Priorities: define parent-ligand standardization before constructing scaffolds, decorations, scores and
CDFs; require complete label matching, no dummy atoms and a connected assembled ligand at evaluation;
then repeat the private-calibration matched comparison. Preserve soft-skew testing but distinguish it
from donor-covered, genuinely absent recipient regions. Reassess target activity separately: this audit
establishes descriptor steering and structural assembly, not biological efficacy.
