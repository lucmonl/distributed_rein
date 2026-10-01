"""Summary quality metrics (reimplemented; no extra packages needed).

* AlignScore (Zha et al., ACL 2023): factual consistency of a summary with its article.
  Uses the official AlignScore-large checkpoint (RoBERTa-large + 3-way alignment head)
  with the paper's inference protocol: the article is split into ~350-word chunks, the
  summary into sentences; each sentence's score is its best alignment probability over
  chunks, and the summary's score is the mean over sentences.  Must be read at matched
  extractiveness (Ladhak et al., ACL 2022): copying is trivially consistent.
* BERTScore F1 (Zhang et al., ICLR 2020) against the reference summary: roberta-large,
  layer 17, greedy cosine matching, no idf weighting, no baseline rescaling.
* Surface: length (tokens), repeated-trigram rate within the summary, empty outputs.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Sequence

import torch
from torch import nn

from .extractive import tokenize

ROBERTA = "FacebookAI/roberta-large"
_SENT = re.compile(r"(?<=[.!?])[\"'”’)]*\s+(?=[\"'“‘(]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    sents = [s.strip() for s in _SENT.split(" ".join(text.split())) if s.strip()]
    return sents or ([text.strip()] if text.strip() else [])


def chunk_words(text: str, chunk_size: int = 350) -> list[str]:
    """Consecutive sentences grouped into chunks of about ``chunk_size`` words."""
    chunks, cur, n = [], [], 0
    for s in split_sentences(text):
        w = len(s.split())
        if cur and n + w > chunk_size:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text]


def surface(summary: str) -> dict:
    toks = tokenize(summary)
    tri = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
    rep = 1 - len(set(tri)) / len(tri) if tri else 0.0
    return {"length": len(toks), "rep3": rep, "empty": float(len(toks) == 0)}


class AlignScorer:
    def __init__(self, device="cuda", ckpt: str | None = None, batch_size: int = 32, chunk_size: int = 350):
        from transformers import AutoTokenizer, RobertaConfig, RobertaModel
        if ckpt is None:
            hits = glob.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                                          "hub/models--yzha--AlignScore/snapshots/*/AlignScore-large.ckpt"))
            if not hits:
                raise FileNotFoundError("AlignScore-large.ckpt not in HF cache; "
                                        "hf_hub_download('yzha/AlignScore', 'AlignScore-large.ckpt') first")
            ckpt = hits[0]
        self.tok = AutoTokenizer.from_pretrained(ROBERTA)
        self.model = RobertaModel(RobertaConfig.from_pretrained(ROBERTA))   # includes the pooler
        self.tri = nn.Linear(self.model.config.hidden_size, 3)
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
        base = {k[len("base_model."):]: v for k, v in sd.items() if k.startswith("base_model.")}
        missing, unexpected = self.model.load_state_dict(base, strict=False)
        missing = [m for m in missing if "position_ids" not in m]
        if missing:
            raise RuntimeError(f"AlignScore checkpoint is missing {missing[:5]}")
        self.tri.load_state_dict({"weight": sd["tri_layer.weight"], "bias": sd["tri_layer.bias"]})
        self.device = device
        self.model.to(device).eval()
        self.tri.to(device).eval()
        self.batch_size = batch_size
        self.chunk_size = chunk_size

    @torch.no_grad()
    def _probs(self, premises: list[str], hyps: list[str]) -> torch.Tensor:
        out = []
        for i in range(0, len(premises), self.batch_size):
            enc = self.tok(premises[i:i + self.batch_size], hyps[i:i + self.batch_size], truncation="only_first",
                           max_length=512, padding=True, return_tensors="pt").to(self.device)
            with torch.autocast(self.device if isinstance(self.device, str) else self.device.type,
                                dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")):
                pooled = self.model(**enc).pooler_output
                logits = self.tri(pooled)
            out.append(torch.softmax(logits.float(), dim=-1)[:, 0].cpu())   # class 0 = ALIGNED
        return torch.cat(out) if out else torch.zeros(0)

    def score(self, articles: Sequence[str], summaries: Sequence[str]) -> list[float]:
        pairs, index = [], []   # index[k] = (summary id, sentence id)
        for sid, (art, summ) in enumerate(zip(articles, summaries)):
            chunks = chunk_words(art, self.chunk_size)
            for j, sent in enumerate(split_sentences(summ)):
                for ch in chunks:
                    pairs.append((ch, sent))
                    index.append((sid, j))
        probs = self._probs([p for p, _ in pairs], [h for _, h in pairs]).tolist()
        best: dict[tuple[int, int], float] = {}
        for (sid, j), p in zip(index, probs):
            best[(sid, j)] = max(best.get((sid, j), 0.0), p)
        per_summary: dict[int, list[float]] = {}
        for (sid, _), p in best.items():
            per_summary.setdefault(sid, []).append(p)
        return [sum(per_summary[i]) / len(per_summary[i]) if i in per_summary else 0.0
                for i in range(len(summaries))]


class BERTScorer:
    def __init__(self, device="cuda", layer: int = 17, batch_size: int = 64):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(ROBERTA)
        self.model = AutoModel.from_pretrained(ROBERTA).to(device).eval()
        self.model.encoder.layer = self.model.encoder.layer[:layer]   # layer-17 representations
        self.device, self.batch_size = device, batch_size

    @torch.no_grad()
    def _embed(self, texts: list[str]):
        embs = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[i:i + self.batch_size], truncation=True, max_length=510, padding=True,
                           return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state.float()
            h = torch.nn.functional.normalize(h, dim=-1)
            for k in range(h.shape[0]):
                m = enc["attention_mask"][k].bool().clone()
                m[0] = False                                   # drop <s>
                m[int(enc["attention_mask"][k].sum()) - 1] = False   # drop </s>
                embs.append(h[k][m])
        return embs

    def score(self, cands: Sequence[str], refs: Sequence[str]) -> list[dict]:
        ce, re_ = self._embed(list(cands)), self._embed(list(refs))
        out = []
        for c, r in zip(ce, re_):
            if len(c) == 0 or len(r) == 0:
                out.append({"P": 0.0, "R": 0.0, "F1": 0.0})
                continue
            sim = c @ r.T
            p, rr = sim.max(dim=1).values.mean().item(), sim.max(dim=0).values.mean().item()
            out.append({"P": p, "R": rr, "F1": 2 * p * rr / (p + rr) if p + rr > 0 else 0.0})
        return out
