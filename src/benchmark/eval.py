from __future__ import annotations

import re
from typing import Optional

ANS_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
INVALID_ANSWER = "[invalid]"


def extract_answer(text: str | None) -> str:
    """Extract final numeric answer using GSM8K style `#### value` markers."""
    if not text:
        return INVALID_ANSWER
    match = ANS_RE.search(text)
    if not match:
        return INVALID_ANSWER
    return match.group(1).strip()


def compare_answers(candidate: str | None, reference: str | None) -> Optional[bool]:
    """Return True/False when both answers can be parsed; otherwise None."""
    cand = extract_answer(candidate)
    gold = extract_answer(reference)
    if INVALID_ANSWER in (cand, gold):
        return None
    return cand == gold
