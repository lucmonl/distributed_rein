"""Extractive fragment statistics (Grusky, Naaman & Artzi, NAACL 2018).

Reimplements ``newsroom.analyze.Fragments``: greedily match, for each summary
position, the longest shared token span with the article; then

    coverage    = sum |f| / |S|         fraction of summary tokens copied
    density     = sum |f|^2 / |S|       average length of the copied fragment each token belongs to
    compression = |A| / |S|

Newsroom tokenizes with spaCy and lowercases.  spaCy is not available here, so a
regex tokenizer is used; `scripts/newsroom_stats.py` checks agreement with the
dataset's precomputed values.
"""

from __future__ import annotations

import re
from collections import defaultdict
from urllib.parse import urlparse

_TOKEN = re.compile(r"\w+(?:[''’]\w+)*|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def fragments(summary: list[str], article: list[str]) -> list[int]:
    """Lengths of the greedy extractive fragments (same semantics as the reference
    implementation, but jumps directly to candidate positions)."""
    positions = defaultdict(list)
    for j, tok in enumerate(article):
        positions[tok].append(j)
    out = []
    i, n_s, n_a = 0, len(summary), len(article)
    while i < n_s:
        best, j_next = 0, 0
        for j in positions.get(summary[i], ()):
            if j < j_next:          # reference scan skips positions inside the last match
                continue
            k = 0
            while i + k < n_s and j + k < n_a and summary[i + k] == article[j + k]:
                k += 1
            best = max(best, k)
            j_next = j + k
        if best:
            out.append(best)
        i += max(best, 1)
    return out


def fragment_stats(summary: str, article: str) -> dict[str, float]:
    s, a = tokenize(summary), tokenize(article)
    if not s:
        return {"coverage": 0.0, "density": 0.0, "compression": float("inf"), "summary_tokens": 0}
    f = fragments(s, a)
    return {
        "coverage": sum(f) / len(s),
        "density": sum(x * x for x in f) / len(s),
        "compression": len(a) / len(s),
        "summary_tokens": len(s),
    }


def density(summary: str, article: str) -> float:
    return fragment_stats(summary, article)["density"]


_SECOND_LEVEL = {"co", "com", "org", "net", "ac", "gov"}


def publication(url: str) -> str:
    """Publication id from a URL: registered domain without ``www``/subdomains,
    keeping e.g. ``dailymail.co.uk``."""
    host = urlparse(url).netloc.lower().split(":")[0]
    parts = [p for p in host.split(".") if p]
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in _SECOND_LEVEL:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])
