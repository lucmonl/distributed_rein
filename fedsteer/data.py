"""Client datasets.

Input format: one JSONL file with records

    {"client": str, "split": "train" | "test" | "drift" | ..., "prompt": str,
     "target": str, "score": float, ...}

``score`` is the attribute value a(y) of the target (e.g. extractive density).
The control coordinate is the client-local percentile of the score among that
client's *train* examples:  alpha_i(y) = F_i(a(y)).
"""

from __future__ import annotations

import bisect
import json
import random
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import torch


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class ClientQuantiles:
    """Empirical CDF of the attribute on a client's training targets."""
    sorted_scores: list[float]

    @classmethod
    def fit(cls, scores: Sequence[float]) -> "ClientQuantiles":
        return cls(sorted(float(s) for s in scores))

    def cdf(self, score: float, rng: Optional[random.Random] = None) -> float:
        """Mid-rank percentile in (0, 1).  Ties get the average rank, or a random
        rank within the tie block if ``rng`` is given (for discrete scores)."""
        n = len(self.sorted_scores)
        lo = bisect.bisect_left(self.sorted_scores, score)
        hi = bisect.bisect_right(self.sorted_scores, score)
        if hi > lo and rng is not None:
            rank = lo + rng.random() * (hi - lo)
        else:
            rank = (lo + hi) / 2 if hi > lo else lo
        return min(max(rank / n, 0.5 / n), 1 - 0.5 / n)

    def quantile(self, alpha: float) -> float:
        """Calibration target F_i^{-1}(alpha), linear interpolation."""
        s = self.sorted_scores
        pos = alpha * (len(s) - 1)
        i = int(pos)
        j = min(i + 1, len(s) - 1)
        return s[i] + (pos - i) * (s[j] - s[i])

    def iqr(self) -> float:
        return self.quantile(0.75) - self.quantile(0.25)


def build_clients(records: list[dict], clients: Sequence[str], tie_break: str = "average",
                  max_train: Optional[int] = None, seed: int = 0
                  ) -> tuple[dict[str, list[dict]], dict[str, ClientQuantiles]]:
    """Return per-client train examples (with ``alpha`` filled) and client CDFs.

    Quantiles are fitted on the *full* train split of each client, so subsampling
    (``max_train``, used for the data-budget study) keeps alpha on the same scale.
    """
    rng = random.Random(seed)
    by_client: dict[str, list[dict]] = {c: [] for c in clients}
    for r in records:
        if r.get("split", "train") == "train" and r["client"] in by_client:
            by_client[r["client"]].append(r)
    quantiles, out = {}, {}
    for c, rows in by_client.items():
        if not rows:
            raise ValueError(f"client {c} has no train examples")
        q = ClientQuantiles.fit([r["score"] for r in rows])
        tb_rng = random.Random(f"{seed}-{c}") if tie_break == "random" else None
        rows = [dict(r, alpha=q.cdf(r["score"], tb_rng)) for r in rows]
        rng.shuffle(rows)
        out[c] = rows[:max_train] if max_train else rows
        quantiles[c] = q
    return out, quantiles


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

class ChatFormatter:
    """Renders (prompt, target) with the model's chat template and masks the loss
    to the assistant turn (target text plus the template's end-of-turn tokens)."""

    def __init__(self, tokenizer, system_prompt: Optional[str] = None,
                 max_prompt_tokens: int = 1024, max_target_tokens: int = 256):
        self.tok = tokenizer
        self.system_prompt = system_prompt
        self.max_prompt_tokens = max_prompt_tokens
        self.max_target_tokens = max_target_tokens

    def _messages(self, prompt: str) -> list[dict]:
        msgs = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        return msgs + [{"role": "user", "content": prompt}]

    def prompt_text(self, prompt: str) -> str:
        return self.tok.apply_chat_template(self._messages(prompt), tokenize=False, add_generation_prompt=True)

    def prompt_ids(self, prompt: str) -> list[int]:
        ids = self.tok(self.prompt_text(prompt), add_special_tokens=False)["input_ids"]
        return ids[-self.max_prompt_tokens:]  # keep the end (instruction + generation header)

    def encode(self, prompt: str, target: str) -> dict:
        msgs = self._messages(prompt)
        p_text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = self.tok.apply_chat_template(msgs + [{"role": "assistant", "content": target}],
                                            tokenize=False, add_generation_prompt=False)
        if not full.startswith(p_text):
            raise ValueError("chat template renders the prompt differently with an assistant turn")
        p_ids = self.tok(p_text, add_special_tokens=False)["input_ids"][-self.max_prompt_tokens:]
        t_ids = self.tok(full[len(p_text):], add_special_tokens=False)["input_ids"][: self.max_target_tokens]
        return {"input_ids": p_ids + t_ids, "labels": [-100] * len(p_ids) + t_ids}


def collate(features: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad for training."""
    n = max(len(f["input_ids"]) for f in features)
    ids = torch.full((len(features), n), pad_id, dtype=torch.long)
    labels = torch.full((len(features), n), -100, dtype=torch.long)
    mask = torch.zeros((len(features), n), dtype=torch.long)
    for i, f in enumerate(features):
        k = len(f["input_ids"])
        ids[i, :k] = torch.tensor(f["input_ids"])
        labels[i, :k] = torch.tensor(f["labels"])
        mask[i, :k] = 1
    alpha = torch.tensor([float(f["alpha"]) for f in features])
    return {"input_ids": ids, "attention_mask": mask, "labels": labels, "alpha": alpha}


class ClientStream:
    """Endless shuffled minibatch stream for one client.  Tokenized lazily and
    cached; the RNG state is part of the client state so resuming is exact."""

    def __init__(self, examples: list[dict], formatter: ChatFormatter, batch_size: int, seed: int):
        self.examples = examples
        self.fmt = formatter
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self._order: list[int] = []
        self._cache: dict[int, dict] = {}

    def _feature(self, i: int) -> dict:
        if i not in self._cache:
            ex = self.examples[i]
            self._cache[i] = dict(self.fmt.encode(ex["prompt"], ex["target"]), alpha=ex["alpha"])
        return self._cache[i]

    def next_batch(self) -> dict[str, torch.Tensor]:
        feats = []
        while len(feats) < self.batch_size:
            if not self._order:
                self._order = list(range(len(self.examples)))
                self.rng.shuffle(self._order)
            feats.append(self._feature(self._order.pop()))
        return collate(feats, self.fmt.tok.pad_token_id)

    def state(self) -> dict:
        return {"rng": self.rng.getstate(), "order": list(self._order)}

    def load(self, state: dict) -> None:
        st = state["rng"]
        self.rng.setstate((st[0], tuple(st[1]), st[2]))
        self._order = list(state["order"])

    def __iter__(self) -> Iterator[dict]:
        while True:
            yield self.next_batch()
