from __future__ import annotations

import re


ID_RE = re.compile(r"(?:DOC|PART|INV|SIG|CASE):[A-Za-z0-9_\\-]+")


def extract_citations(text: str) -> set[str]:
    return set(ID_RE.findall(text))


def groundedness_score(citations: set[str], allowed_ids: set[str]) -> float:
    if not citations:
        return 0.0
    supported = sum(1 for c in citations if c in allowed_ids)
    return supported / len(citations)
