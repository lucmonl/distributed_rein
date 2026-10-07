# Exp45 per-client analysis (2026-10-06)

Anvil FED 21102995 (selected r100) versus LOCAL 21102996 (selected r80), seed 0, pruned dataset and strict single-parent-ligand scorer. All extracted per-client metrics were checked against the source logs. Dataset CDFs and central-90% support were checked against the stored reference and logged support.

**Finding:** local improves calibration most on CHEMBL325, CHEMBL204, CHEMBL4078 and CHEMBL2039. FED improves reliability especially on CHEMBL243 and CHEMBL240. Strong one-sided skew remains difficult for both.

Error pairs below are FED / LOCAL; lower is better. In/out/overall conditional errors exclude any prompt with an unscorable cell anywhere in its alpha sweep. Penalty-1 error includes every cell.

| Client | Train n | Conditional overall | In-support | Out-of-support | Penalty 1 |
|---|---:|---:|---:|---:|---:|
| CHEMBL243 | 1554 | 0.2782 / 0.2781 | 0.2626 / 0.2528 | 0.2821 / 0.2844 | 0.2813 / 0.3075 |
| CHEMBL204 | 2876 | 0.2218 / 0.2016 | 0.2158 / 0.2100 | 0.2307 / 0.1890 | 0.2218 / 0.2061 |
| CHEMBL325 | 2020 | 0.1965 / 0.1676 | 0.1811 / 0.1606 | 0.2196 / 0.1782 | 0.1994 / 0.1701 |
| CHEMBL2835 | 1493 | 0.2183 / 0.2154 | 0.1899 / 0.1793 | 0.2609 / 0.2696 | 0.2213 / 0.2154 |
| CHEMBL4078 | 1073 | 0.1896 / 0.1730 | 0.1243 / 0.1034 | 0.2332 / 0.2194 | 0.1944 / 0.1748 |
| CHEMBL2039 | 643 | 0.1551 / 0.1431 | 0.0966 / 0.1031 | 0.2428 / 0.2032 | 0.1575 / 0.1492 |
| CHEMBL240 | 3535 | 0.1671 / 0.1689 | 0.1497 / 0.1485 | 0.1931 / 0.1995 | 0.1696 / 0.2067 |
| CHEMBL228 | 1146 | 0.2325 / 0.2288 | 0.1412 / 0.1550 | 0.2934 / 0.2780 | 0.2398 / 0.2323 |

## Training coverage

Counts use the same equal-client mixture CDF and average tie handling as training. Support is the mapped local 5th–95th percentile interval, not min/max.

| Client | Central 90% support | Actual train min–max | Counts in four alpha quarters | OOS test alphas |
|---|---:|---:|---|---|
| CHEMBL243 | 0.009–0.406 | 0.0016–0.9732 | 1274, 233, 30, 17 | [0, 0.5, 0.75, 1] |
| CHEMBL204 | 0.025–0.916 | 0.0004–0.9992 | 1359, 666, 435, 416 | [0, 1] |
| CHEMBL325 | 0.170–0.988 | 0.0006–0.9995 | 300, 525, 351, 844 | [0, 1] |
| CHEMBL2835 | 0.138–0.885 | 0.0164–0.9890 | 368, 412, 409, 304 | [0, 1] |
| CHEMBL4078 | 0.300–0.798 | 0.0970–0.9975 | 27, 479, 478, 89 | [0, 0.25, 1] |
| CHEMBL2039 | 0.128–0.904 | 0.0006–0.9994 | 102, 225, 205, 111 | [0, 1] |
| CHEMBL240 | 0.165–0.960 | 0.0117–0.9993 | 417, 766, 1092, 1260 | [0, 1] |
| CHEMBL228 | 0.457–0.971 | 0.1125–0.9996 | 12, 79, 355, 700 | [0, 0.25, 1] |

## Pattern and interpretation

1. **The strongest OOS gains are mostly on broadly covered clients.** CHEMBL204, CHEMBL325 and CHEMBL2039 contribute 91.2% of the net equal-client OOS error reduction. Their central supports span .025–.916, .170–.988 and .128–.904. Each has hundreds or about a hundred examples in both extreme alpha quarters. These are weak evidence for transfer into genuinely absent regions.
2. **For five clients, OOS error is exactly an endpoint-span metric.** CHEMBL204, CHEMBL325, CHEMBL2835, CHEMBL2039 and CHEMBL240 have OOS test grid {0,1}. With achieved percentiles p0,p1, OOS error = mean[(p0 + 1 - p1)/2] = (1 - pct_range)/2. This identity was verified against all ten logged values. It means a wider response automatically improves this metric; it does not establish correct intermediate calibration or accuracy on a held-out interior interval.
3. **Local usually extends the high endpoint more on the major winning clients.** At requested alpha 1: CHEMBL204 .710→.814; CHEMBL325 .837→.905; CHEMBL4078 .744→.821; CHEMBL2039 .802→.855 (FED→LOCAL). Their adjacent tie rates also fall: 45.5→39.2%, 43.4→32.6%, 41.1→32.4%, 39.2→31.7%. These are observable changes in response, consistent with a more restricted federated steering response. They do not identify whether sharing or optimization causes the restriction. Means are conditional on different complete-sweep subsets.
4. **The most one-sided clients remain weak on the missing side.** CHEMBL243 has 82.0% of training examples below .25 and only 17 above .75. At requested alpha 1, means are .638 FED / .614 LOCAL; local does not solve high-side extrapolation. CHEMBL228 has 61.1% above .75 and only 12 below .25. At requested alpha 0, means are .495 / .475; both remain far above the requested low endpoint. Local slightly improves this client, but it is not strong extrapolation. CHEMBL4078 is concentrated in the middle (89.2% in [.25,.75)), yet has 27 low-tail and 89 high-tail training examples; it is a meaningful local advantage under sparse tails, not zero-tail training.
5. **Training-set size is not a simple explanation.** Local wins the penalty metric on the smallest client CHEMBL2039 (643 samples), whereas FED wins on the largest CHEMBL240 (3535). Local also wins on CHEMBL204 (2876). Eight clients and one seed do not support a stable statistical association or a causal size claim.
6. **Failure accounting changes two important comparisons.** CHEMBL243 is effectively tied in conditional error (.2782/.2781), but unscorable sweeps rise from 2.0% FED to 14.7% LOCAL and penalty error favors FED (.2813/.3075). CHEMBL240 has similar conditional error (.1671/.1689), but unscorable sweeps are 1.3%/11.3%, giving a substantial FED penalty advantage (.1696/.2067). Sweep failure percentages are not cell failure percentages.
7. **Two clients have mixed results.** CHEMBL2835 improves in-support under local but worsens OOS (.2609/.2696); CHEMBL2039 improves OOS under local but worsens in-support (.0966/.1031). CHEMBL228 also worsens in-support under local (.1412/.1550). An overall local win does not imply a win at every alpha.

**Design implication:** current evidence supports better local calibration/response span on several clients and better FED reliability on two. It does not demonstrate useful shared-direction transfer into the most deficient recipient regions. A discriminating follow-up is a prespecified recipient interior-alpha holdout with donor coverage, fixed reference CDF, matched optimizer-state handling and multiple seeds. Neither scalar gains nor these aggregate curves alone establish the cause.

**Limits:** selected checkpoints differ; this is one seed. Conditional metrics use different successful-prompt subsets. No raw Anvil generation grid was analyzed here and no paired confidence interval was computed. No model or scoring behavior was changed.

Artifacts: [structured analysis](exp45_per_client_analysis.json), [response curves and training coverage](exp45_per_client_curves.png), [source extraction](exp45_pruned_comparison.json).
