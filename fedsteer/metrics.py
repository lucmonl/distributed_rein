"""Attribute scorers and direction-quality metrics (plan §4), shared by the
training monitor and eval_direction.py so both report identical numbers."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional, Sequence

import numpy as np
from scipy.stats import spearmanr

from .data import ClientQuantiles
from .extractive import fragment_stats, tokenize
from .mathcot import cot_tokens
from .molecules import (clogp_residual, clogp_residual_deco, clogp_residual_self,
                        clogp_residual_strict)

# scorer(generated_text, source_record) -> attribute value on the same scale as the
# training ``score`` field
SCORERS = {
    "words": lambda text, rec: float(len(text.split())),
    "density": lambda text, rec: fragment_stats(text, rec["article"])["density"],
    # ChEMBL: decoration lipophilicity against the core given in the prompt (nan if the
    # generated molecule does not parse).  The baseline must be the *requested* core:
    # with the generation's own scaffold as baseline, a model that swapped the core
    # would move the quantity being subtracted.  `_strict` additionally requires the
    # core to survive (robustness check); `_self` is a drift diagnostic only.
    "clogp_residual": clogp_residual,
    "clogp_residual_strict": clogp_residual_strict,
    "clogp_residual_self": clogp_residual_self,
    # decoration format: the model emits only the decorations and they are zipped onto
    # the core from the prompt, so the core cannot be rewritten (entry 34)
    "clogp_residual_deco": clogp_residual_deco,
    # math CoT: solution length in tokens of a fixed ruler tokenizer (fedsteer/mathcot.py)
    "cot_tokens": cot_tokens,
}

LOWER_IS_BETTER = {"calib_mae_iqr", "pct_calib_err", "pct_err_in_support", "pct_err_out_support",
                   "unscorable_row_rate",
                   "no_effect_rate", "adjacent_tie_rate", "adjacent_decrease_rate",
                   "text_tie_rate", "near_tie_rate", "near_no_effect_rate", "endpoint_near_tie_rate"}
SUMMARY_KEYS = ["pct_err_in_support", "pct_err_out_support", "reach_rate", "concordance", "spearman",
                "endpoint_increase_rate", "adjacent_increase_rate", "adjacent_tie_rate", "adjacent_decrease_rate",
                "no_effect_rate", "pct_calib_err", "pct_range", "order_rate", "norm_range", "calib_mae_iqr",
                "text_tie_rate", "near_tie_rate", "near_no_effect_rate", "endpoint_near_tie_rate",
                "adjacent_increase_rate_nt", "concordance_nt",
                # share of prompts the scorer could not score at every alpha (ChEMBL:
                # unparseable output, or a lost core under the strict scorer)
                "unscorable_row_rate"]


def metrics_for_client(scores: np.ndarray, alphas: list[float], q: ClientQuantiles,
                       support: Optional[tuple[float, float]] = None) -> dict:
    """scores: [n_prompts, n_alphas]

    ``q`` is the reference CDF that defines alpha (the client's own CDF, or the shared
    mixture in global mode).  Percentile metrics map each output's score through it,
    the same transform used for the training labels, so they are bounded and robust
    to heavy-tailed attributes such as density.  ``support`` (lo, hi) splits the alpha
    grid into the part covered by the client's own data and the part it can only
    reach through the shared direction.
    """
    scores = np.asarray(scores, dtype=float)
    # A scorer returns nan when it cannot score the output at all (an unparseable
    # molecule, say).  Row-wise metrics -- ordering, concordance, Spearman, range --
    # need a complete alpha sweep, so drop prompts with any unscorable output and
    # report how many were dropped; validity itself is reported by the task's own
    # quality script.  Attributes that are always scorable (density) are unaffected.
    n_all = len(scores)
    keep = np.all(np.isfinite(scores), axis=1)
    if not keep.all():
        scores = scores[keep]
    unscorable = {"n_prompts": int(n_all), "unscorable_row_rate": float(1.0 - keep.mean())}
    if len(scores) == 0:
        return {**unscorable, "concordance": None, "pct_calib_err": None, "spearman": None}
    order = np.all(np.diff(scores, axis=1) > 0, axis=1).mean()
    # pairwise concordance: fraction of (alpha_i < alpha_j) pairs with score_i < score_j (ties = 1/2)
    i, j = np.triu_indices(len(alphas), 1)
    diff = scores[:, j] - scores[:, i]
    concord = ((diff > 0) + 0.5 * (diff == 0)).mean()
    pct = np.vectorize(q.cdf)(scores)
    pct_err = np.abs(pct - np.asarray(alphas)[None, :])
    # Spearman per article; an article whose outputs do not change with alpha counts as 0
    constant = np.ptp(scores, axis=1) == 0
    rhos = [0.0 if c else spearmanr(alphas, s).correlation for s, c in zip(scores, constant)]
    adj = np.diff(scores, axis=1)
    rng = (scores[:, -1] - scores[:, 0]).mean() / max(q.iqr(), 1e-9)
    targets = np.array([q.quantile(a) for a in alphas])
    mae = np.abs(scores - targets[None, :]).mean() / max(q.iqr(), 1e-9)
    out = {}
    if support is not None:
        # in-support: alphas inside the part of the axis the client's own data covers;
        # out-of-support: alphas the client can only reach through the shared direction
        lo, hi = support
        a = np.asarray(alphas)
        ins = (a >= lo) & (a <= hi)
        out["support"] = [round(float(lo), 4), round(float(hi), 4)]
        out["pct_err_in_support"] = float(pct_err[:, ins].mean()) if ins.any() else None
        out["pct_err_out_support"] = float(pct_err[:, ~ins].mean()) if (~ins).any() else None
        # reach: at out-of-support alphas, did the output leave the client's own range in
        # the requested direction?
        below, above = a < lo, a > hi
        hits = [pct[:, below] < lo] if below.any() else []
        hits += [pct[:, above] > hi] if above.any() else []
        out["reach_rate"] = float(np.concatenate([h.ravel() for h in hits]).mean()) if hits else None
    return {
        **unscorable,
        **out,
        "concordance": float(concord),
        "pct_calib_err": float(pct_err.mean()),
        "pct_range": float((pct[:, -1] - pct[:, 0]).mean()),
        "mean_pct_by_alpha": pct.mean(axis=0).round(3).tolist(),
        "order_rate": float(order),
        "no_effect_rate": float(constant.mean()),              # all alphas give the same score
        "adjacent_increase_rate": float((adj > 0).mean()),     # P(score goes up | next alpha step)
        "adjacent_tie_rate": float((adj == 0).mean()),
        "adjacent_decrease_rate": float((adj < 0).mean()),
        "endpoint_increase_rate": float((scores[:, -1] > scores[:, 0]).mean()),  # highest alpha above lowest
        "spearman": float(np.mean(rhos)),
        "norm_range": float(rng),
        "calib_mae_iqr": float(mae),
        "mean_score_by_alpha": scores.mean(axis=0).round(3).tolist(),
        "target_by_alpha": targets.round(3).tolist(),
    }


def constant_output_pct_err(alphas: list[float]) -> float:
    """Percentile calibration error of a model that always outputs the client median."""
    return float(np.mean(np.abs(np.asarray(alphas) - 0.5)))


def summarize(results: dict[str, dict], keys=SUMMARY_KEYS) -> dict:
    out = {}
    for k in keys:
        vals = [r[k] for r in results.values() if r.get(k) is not None]
        if vals:
            worst = max(vals) if k in LOWER_IS_BETTER else min(vals)
            out[k] = {"mean": float(np.mean(vals)), "worst": float(worst)}
    return out


# ---------------------------------------------------------------------------
# Near-ties: outputs that differ only trivially across alpha
# ---------------------------------------------------------------------------

_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
                         "\u00a0": " "})
_ELLIPSIS = re.compile(r"\s*(\.\.\.|\u2026)\s*$")
NEAR_TIE_THRESHOLD = 0.95


def normalize_output(text: str) -> str:
    """Unify quotes/dashes/whitespace; if the text ends in an ellipsis (a cut-off lead, as in
    nypost.com's summaries), drop the ellipsis and the possibly truncated last word."""
    t = " ".join(text.translate(_QUOTES).split())
    if _ELLIPSIS.search(t):
        t = _ELLIPSIS.sub("", t)
        t = t.rsplit(" ", 1)[0] if " " in t else ""
    return t


def near_identical(a: str, b: str, threshold: float = NEAR_TIE_THRESHOLD) -> bool:
    """True if two outputs are the same up to normalization and a small token-level edit
    (difflib ratio >= threshold on tokens; 0.95 ~ at most ~5% of tokens differ)."""
    na, nb = normalize_output(a), normalize_output(b)
    if na == nb:
        return True
    ta, tb = tokenize(na), tokenize(nb)
    if not ta or not tb:
        return False
    return SequenceMatcher(None, ta, tb, autojunk=False).ratio() >= threshold


def text_tie_metrics(outputs: Sequence[Sequence[str]], scores, alphas: Sequence[float],
                     threshold: float = NEAR_TIE_THRESHOLD) -> dict:
    """outputs[article][alpha] texts; scores[article][alpha] attribute scores.

    text_tie_rate          adjacent alphas give exactly the same text
    near_tie_rate          adjacent alphas give near-identical text
    near_no_effect_rate    every adjacent pair near-identical (alpha effectively had no effect)
    endpoint_near_tie_rate lowest and highest alpha near-identical
    adjacent_increase_rate_nt / concordance_nt: the score-based rates with near-identical
        pairs counted as ties, so trivial changes (a few extra copied characters) no longer
        count as successful steering
    """
    scores = np.asarray(scores, dtype=float)
    n, k = scores.shape
    exact = np.array([[outputs[i][j] == outputs[i][j + 1] for j in range(k - 1)] for i in range(n)])
    near = np.array([[near_identical(outputs[i][j], outputs[i][j + 1], threshold) for j in range(k - 1)]
                     for i in range(n)])
    adj = np.diff(scores, axis=1)
    inc_nt = (adj > 0) & ~near
    ii, jj = np.triu_indices(k, 1)
    conc = []
    for i in range(n):
        for a, b in zip(ii, jj):
            if near_identical(outputs[i][a], outputs[i][b], threshold) or scores[i, b] == scores[i, a]:
                conc.append(0.5)
            else:
                conc.append(float(scores[i, b] > scores[i, a]))
    return {
        "text_tie_rate": float(exact.mean()),
        "near_tie_rate": float(near.mean()),
        "near_no_effect_rate": float(near.all(axis=1).mean()),
        "endpoint_near_tie_rate": float(np.mean([near_identical(o[0], o[-1], threshold) for o in outputs])),
        "adjacent_increase_rate_nt": float(inc_nt.mean()),
        "concordance_nt": float(np.mean(conc)),
    }
