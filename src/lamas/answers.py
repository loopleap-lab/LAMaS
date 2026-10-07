"""Pull the final answer out of a solution text.

These keys are what "agreement" means for the stop policy and for majority votes: two
solutions agree when their keys are equal.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Callable, List, Optional, Sequence

_BOXED = re.compile(r"\\boxed{((?:[^{}]|{[^{}]*})*)}", re.S)
_FINAL_LINE = re.compile(r"(?:final answer|answer)\s*(?:is|:)\s*(.+)", re.I)


def extract_boxed(text: str) -> Optional[str]:
    r"""The content of the last ``\boxed{...}``."""
    if not text:
        return None
    found = _BOXED.findall(text)
    return found[-1].strip() if found else None


def extract_final_line(text: str, max_len: int = 80) -> Optional[str]:
    """A short ``Final answer: ...`` / ``The answer is ...`` statement, normalised."""
    if not text:
        return None
    boxed = extract_boxed(text)
    if boxed is not None and len(boxed) <= max_len:
        return _normalise(boxed)
    hits = _FINAL_LINE.findall(text)
    if hits:
        value = hits[-1].strip().splitlines()[0].strip()
        if 0 < len(value) <= max_len:
            return _normalise(value)
    return None


def _normalise(value: str) -> str:
    value = value.strip().strip(".").strip().strip("*`$ ")
    try:
        number = float(value.replace(",", ""))
        return str(int(number)) if number.is_integer() else str(number)
    except ValueError:
        return re.sub(r"\s+", " ", value.lower())


def text_hash(text: str) -> str:
    return hashlib.md5((text or "").strip().encode("utf-8")).hexdigest()


def consensus(keys: Sequence[str]) -> float:
    """Share of the most common key: 1.0 when everything agrees."""
    if not keys:
        return 0.0
    return Counter(keys).most_common(1)[0][1] / len(keys)


def majority(solutions: List[str], key_fn: Callable[[str], Optional[str]]) -> Optional[str]:
    """The most recent solution that carries the most common answer key."""
    if not solutions:
        return None
    keys = [key_fn(s) or "h:" + text_hash(s) for s in solutions]
    best = Counter(keys).most_common(1)[0][0]
    for solution, key in zip(reversed(solutions), reversed(keys)):
        if key == best:
            return solution
    return solutions[-1]
