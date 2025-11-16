"""String normalisation helpers."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

from rapidfuzz import process

_SUFFIX_PAT = re.compile(r",?\s+(inc|corp|co|llc|sl|sa|ltd)\.?:?$", re.IGNORECASE)


def _strip_suffixes(name: str) -> str:
    while True:
        new = _SUFFIX_PAT.sub("", name)
        if new == name:
            break
        name = new
    return name


def normalize_name(name: str) -> str:
    """Basic unicode and whitespace cleanup plus suffix trimming."""

    text = unicodedata.normalize("NFKC", name or "")
    text = text.strip()
    text = _strip_suffixes(text)
    text = re.sub(r"\s+", " ", text)
    return text.title()


def cluster_and_canonicalize(names: List[str], score_cutoff: int = 90) -> Dict[str, str]:
    """Return mapping of each name to a canonical representative using rapidfuzz."""

    unique = sorted({normalize_name(n) for n in names if n})
    if not unique:
        return {}
    mapping: Dict[str, str] = {}
    for name in unique:
        if name in mapping:
            continue
        matches = process.extract(name, unique, score_cutoff=score_cutoff)
        canon = name
        for match_name, _score, _ in matches:
            mapping[match_name] = canon
    return mapping
