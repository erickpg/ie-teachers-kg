"""String normalisation helpers."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional

from rapidfuzz import process

_SUFFIX_PAT = re.compile(r",?\s+(inc|corp|co|llc|sl|sa|ltd|gmbh|s\.a\.|s\.l\.|plc)\.?:?$", re.IGNORECASE)


def _strip_suffixes(name: str) -> str:
    text = name
    while True:
        new = _SUFFIX_PAT.sub("", text)
        if new == text:
            break
        text = new
    return text


def _smart_case(text: str) -> str:
    tokens = []
    for token in text.split():
        clean = token.strip(" ,.;:/")
        if not clean:
            continue
        if len(clean) <= 4 and clean.isupper():
            tokens.append(clean)
        elif clean.isupper() and any(ch.isdigit() for ch in clean):
            tokens.append(clean)
        else:
            tokens.append(clean.title())
    return " ".join(tokens)


def normalize_name(name: str) -> str:
    """Unicode, whitespace, and suffix cleanup with friendly casing."""

    text = unicodedata.normalize("NFKC", (name or "").strip())
    text = _strip_suffixes(text)
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ,.;:/")
    return _smart_case(text)


def cluster_and_canonicalize(
    names: List[str], score_cutoff: int = 90, aliases: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Cluster noisy labels and map them to canonical representatives."""

    cleaned = [normalize_name(n) for n in names if n]
    if not cleaned:
        return {}
    unique = sorted(set(cleaned))
    mapping: Dict[str, str] = {}
    counts = Counter(cleaned)

    if aliases:
        for raw, canon in aliases.items():
            raw_norm = normalize_name(raw)
            canon_norm = normalize_name(canon)
            mapping[raw_norm] = canon_norm
            mapping[canon_norm] = canon_norm

    for name in unique:
        if name in mapping:
            continue
        matches = process.extract(name, unique, score_cutoff=score_cutoff)
        if not matches:
            mapping[name] = name
            continue
        candidates = [match_name for match_name, _score, _idx in matches]
        best = max(candidates, key=lambda item: (counts.get(item, 0), len(item)))
        for candidate in candidates:
            if candidate not in mapping:
                mapping[candidate] = best
    return mapping
