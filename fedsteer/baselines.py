"""Baselines B1 (prompting) and B4 (federated activation steering, CAA).

Both run on a client's model *without* the learned weight direction (B_d zeroed), so
each client keeps its private adapter (house style) and only the control mechanism
changes.  B3 (one-shot merged direction) needs no code here: it is the method's
evaluation with a merged direction attached (scripts/merge_local_directions.py,
eval_direction.py --shared).

B1  the target level is stated in the prompt (0-100 extractiveness scale), optionally
    with k examples from the client's *own* training data whose alpha is closest to the
    target (all a real client could provide).
B4  per client, a mean-difference vector of residual-stream activations (teacher-forced,
    averaged over summary tokens) between its top- and bottom-quartile summaries; the
    server averages the vectors uniformly; at alpha the vector is added to the output of
    one decoder layer at every position with coefficient s_i * (2 alpha - 1) * mean
    hidden norm, where the scalar s_i is fitted per client on a few dev articles.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from .data import collate
from .metrics import metrics_for_client, text_tie_metrics
from .model import generate_at_alpha


def zero_direction(model) -> None:
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("lora_B_d"):
                p.zero_()


def zero_adapter(model) -> None:
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("lora_B_p"):
                p.zero_()


# ----------------------------------------------------------------------------- B1

def level_instruction(alpha: float) -> str:
    level = int(round(100 * alpha))
    return (f"Target extractiveness: {level} on a 0-100 scale, where 0 means rewriting everything in your own "
            f"words (no phrases copied from the article) and 100 means copying whole sentences word for word "
            f"from the article.")


def truncate_words(text: str, n: int) -> str:
    w = text.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def pick_shots(pool: list[dict], alpha: float, k: int, exclude_url: Optional[str] = None) -> list[dict]:
    """k examples from the client's own training pool with alpha closest to the target."""
    cands = [r for r in pool if r.get("url") != exclude_url]
    return sorted(cands, key=lambda r: (abs(r["alpha"] - alpha), r.get("url", "")))[:k]


def molecule_level_instruction(alpha: float) -> str:
    level = int(round(100 * alpha))
    return (f"Target decoration lipophilicity: {level} on a 0-100 scale, where 0 means the groups you add to "
            f"the core are as polar as possible (they lower the molecule's logP relative to the core) and 100 "
            f"means they are as lipophilic as possible (they raise it as much as possible).")


def prompt_with_level_molecule(rec: dict, alpha: float, shots: Sequence[dict]) -> str:
    parts = [f"Design a ligand for {rec['target_name']}.",
             f"Decorate this core scaffold: {rec['scaffold']}",
             molecule_level_instruction(alpha)]
    if shots:
        parts.append("Here are example ligands for this target at about this level:")
        for s in shots:
            parts.append(f"Core: {s['scaffold']}\nLigand: {s['target']}")
        parts.append("Now decorate the requested core at the target lipophilicity.")
    parts.append("Answer with one SMILES string.")
    return "\n\n".join(parts)


def math_level_instruction(alpha: float) -> str:
    level = int(round(100 * alpha))
    return (f"Target solution length: {level} on a 0-100 scale, where 0 means the shortest possible "
            f"solution (only the essential steps, as few words as possible) and 100 means the longest, "
            f"most detailed step-by-step solution (every step written out and explained).")


def prompt_with_level_math(rec: dict, alpha: float, shots: Sequence[dict]) -> str:
    parts = [rec["problem"], math_level_instruction(alpha)]
    if shots:
        parts.append("Here are example solutions at about this length:")
        for s in shots:
            parts.append(f"Problem: {s['problem']}\nSolution: {s['target']}")
        parts.append("Now solve the problem above at the target length.")
    parts.append("Please reason step by step, and put your final answer within \\boxed{}.")
    return "\n\n".join(parts)


def prompt_with_level(rec: dict, alpha: float, shots: Sequence[dict], shot_words: int = 150,
                      task: str = "auto") -> str:
    """B1's prompt. ``task`` selects the template; ``auto`` picks the molecule template
    for records that carry a scaffold and the math template for records with a problem."""
    if task == "molecule" or (task == "auto" and "scaffold" in rec):
        return prompt_with_level_molecule(rec, alpha, shots)
    if task == "math" or (task == "auto" and "problem" in rec):
        return prompt_with_level_math(rec, alpha, shots)
    parts = ["Write a short summary of the following news article.", level_instruction(alpha)]
    if shots:
        parts.append("Here are example summaries from this outlet at about this level:")
        for s in shots:
            parts.append(f"Article:\n{truncate_words(s['article'], shot_words)}\nSummary: {s['target']}")
        parts.append("Now summarize this article at the target extractiveness.")
    parts.append(f"Article:\n{rec['article']}")
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------- B4

def _layer_module(model, layer: int):
    return model.model.layers[layer]


@contextmanager
def add_to_residual(model, layer: int, vec: Optional[torch.Tensor]):
    """Add ``vec`` to the output hidden states of decoder layer ``layer`` (all positions)."""
    if vec is None:
        yield
        return

    def hook(_m, _inp, out):
        if isinstance(out, tuple):
            return (out[0] + vec.to(out[0].dtype),) + tuple(out[1:])
        return out + vec.to(out.dtype)

    h = _layer_module(model, layer).register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@torch.no_grad()
def mean_summary_activation(model, fmt, recs: list[dict], layer: int, batch_size: int = 16, bf16: bool = True):
    """Mean over examples of the mean hidden state (output of ``layer``) over summary tokens;
    also returns the mean hidden-state norm (for scaling the steering coefficient)."""
    model.eval()
    device = next(model.parameters()).device
    feats = [dict(fmt.encode(r["prompt"], r["target"]), alpha=0.0) for r in recs]
    sums, norms, n = None, [], 0
    for i in range(0, len(feats), batch_size):
        b = {k: v.to(device) for k, v in collate(feats[i:i + batch_size], fmt.tok.pad_token_id).items()}
        b.pop("alpha")
        labels = b.pop("labels")
        with model.steer_control.use_alpha(0.0), torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            hs = model(**b, output_hidden_states=True).hidden_states[layer + 1].float()
        mask = (labels != -100).unsqueeze(-1).float()
        per_ex = (hs * mask).sum(1) / mask.sum(1).clamp_min(1)
        norms.append(hs[mask.squeeze(-1).bool()].norm(dim=-1).mean().item())
        sums = per_ex.sum(0) if sums is None else sums + per_ex.sum(0)
        n += per_ex.shape[0]
    return sums / n, float(np.mean(norms))


def caa_vector(model, fmt, own_train: list[dict], layer: int, n_per_side: int = 200, seed: int = 0):
    """Client-side CAA vector: top-quartile minus bottom-quartile mean activation of the
    client's *own* training summaries (quartiles of its own alpha distribution)."""
    rng = np.random.default_rng(seed)
    a = np.array([r["alpha"] for r in own_train])
    lo, hi = np.quantile(a, 0.25), np.quantile(a, 0.75)
    bottom = [r for r in own_train if r["alpha"] <= lo]
    top = [r for r in own_train if r["alpha"] >= hi]
    pick = lambda xs: [xs[i] for i in rng.choice(len(xs), min(n_per_side, len(xs)), replace=False)]
    m_top, n1 = mean_summary_activation(model, fmt, pick(top), layer)
    m_bot, n2 = mean_summary_activation(model, fmt, pick(bottom), layer)
    return m_top - m_bot, (n1 + n2) / 2


# ------------------------------------------------------------------------ shared

def score_grid_custom(model, fmt, recs, alphas, score, prompt_fn: Optional[Callable] = None,
                      steer_fn: Optional[Callable] = None, layer: int = 0, max_new_tokens: int = 128,
                      batch_size: int = 16):
    """Like fedsteer.evaluate.score_grid, but alpha acts through the prompt (``prompt_fn(rec,
    alpha)``) and/or an activation addition (``steer_fn(alpha) -> vector``); the weight
    direction is not used (generation runs at alpha = 0 with B_d zeroed)."""
    grid = np.zeros((len(recs), len(alphas)))
    texts = [[None] * len(alphas) for _ in recs]
    for j, a in enumerate(alphas):
        prompts = [prompt_fn(r, a) if prompt_fn else r["prompt"] for r in recs]
        vec = steer_fn(a) if steer_fn else None
        with add_to_residual(model, layer, vec):
            outs = generate_at_alpha(model, fmt, prompts, 0.0, max_new_tokens=max_new_tokens, batch_size=batch_size)
        for i, (o, r) in enumerate(zip(outs, recs)):
            texts[i][j] = o[0]
            grid[i, j] = score(o[0], r)
    return grid, texts


def result_from_grid(grid, texts, recs, alphas, ref_q, support=None) -> dict:
    res = metrics_for_client(grid, alphas, ref_q, support=support)
    res.update(text_tie_metrics(texts, grid, alphas))
    res["grid"] = grid.round(4).tolist()
    res["outputs"] = texts
    res["record_ids"] = [r.get("url", str(i)) for i, r in enumerate(recs)]
    return res
