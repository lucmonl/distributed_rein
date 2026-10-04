"""Math CoT attribute and quality helpers (math-cot-experiment-plan.md).

The attribute is the CoT length

    a(y) = number of tokens of y under a FIXED ruler tokenizer (Qwen3-4B-Instruct-2507),

independent of the backbone, so a Llama smoke run and a Qwen run are scored on the same
scale and the training labels (scripts/build_math_fed.py) use the same function.

Quality: answer accuracy against the gold \\boxed{} answer (Math-Verify), plus format
(boxed answer present) and degenerate-length checks (repetition loops, gzip ratio).
Math-Verify is imported lazily so the scorer works without it.
"""

from __future__ import annotations

import gzip
import os
import re
from collections import Counter
from functools import lru_cache
from typing import Optional

RULER = os.environ.get("FEDSTEER_RULER", "Qwen/Qwen3-4B-Instruct-2507")


@lru_cache(maxsize=1)
def ruler():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(RULER)


def cot_tokens(text: str, rec: Optional[dict] = None) -> float:
    """Scorer: length of the generated solution in ruler tokens."""
    return float(len(ruler()(text, add_special_tokens=False).input_ids))


def extract_boxed(text: str) -> Optional[str]:
    """Content of the LAST \\boxed{...} (or \\fbox{...}) in ``text``, braces matched; None if absent."""
    if not isinstance(text, str):
        return None
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


def is_correct(text: str, gold: str) -> bool:
    """Math-Verify equivalence of the generation's final answer and the gold answer."""
    from math_verify import parse, verify
    pred = extract_boxed(text)
    if pred is None:
        return False
    try:
        g = parse(f"\\boxed{{{gold}}}")
        p = parse(f"\\boxed{{{pred}}}")
        if verify(g, p):
            return True
    except Exception:
        pass
    # MCQ letters and plain strings: normalized exact match
    norm = lambda s: re.sub(r"\\text\{([^}]*)\}|[\s$()\\]", r"\1", s).strip(".").upper()
    return norm(pred) == norm(gold)


def repetition_loop(text: str, n: int = 20, times: int = 3) -> bool:
    """True if some word n-gram occurs ``times`` or more: the signature of a generation that
    reaches a length by looping rather than by reasoning.  n = 20 words: with n = 10 the check
    mostly fired on restated formulas (entry 37: base 0.110 -> 0.020, G0 0.090 -> 0.020)."""
    w = text.split()
    if len(w) < n * times:
        return False
    c = Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return max(c.values()) >= times


def gzip_ratio(text: str) -> float:
    """compressed / raw bytes; low values mean repetitive text."""
    b = text.encode()
    return len(gzip.compress(b)) / max(len(b), 1)
