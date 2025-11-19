"""String normalisation helpers."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from rapidfuzz import process

_SUFFIX_PAT = re.compile(r",?\s+(inc|corp|co|llc|sl|sa|ltd)\.?:?$", re.IGNORECASE)
_SMALL_WORDS = {"de", "of", "la", "del", "da", "do", "das", "dos", "y", "en"}


def _strip_suffixes(name: str) -> str:
    while True:
        new = _SUFFIX_PAT.sub("", name)
        if new == name:
            break
        name = new
    return name


def normalize_name(name: Optional[str]) -> str:
    """Unicode-aware normalisation keeping key punctuation and acronyms."""

    text = unicodedata.normalize("NFKC", name or "")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.strip()
    text = _strip_suffixes(text)
    text = re.sub(r"[^0-9A-Za-z&/\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    def _collapse_acronyms(value: str) -> str:
        """Collapse spaced acronyms like 'I E' -> 'IE'."""

        def repl(match: re.Match) -> str:
            return "".join(ch for ch in match.group(0) if ch.isalpha())

        return re.sub(r"\b(?:([A-Z])\s+){1,}([A-Z])\b", repl, value)

    text = _collapse_acronyms(text)
    text = re.sub(r"\bI\s*E\s*B\s+Business School\b", "IE Business School", text, flags=re.IGNORECASE)

    tokens = []
    for token in text.split(" "):
        if not token:
            continue
        lower = token.lower()
        if token.isupper() and len(token) > 1:
            tokens.append(token)
        elif lower in _SMALL_WORDS:
            tokens.append(lower)
        else:
            tokens.append(token.capitalize())
    return " ".join(tokens)


def _valid_name(name: str) -> bool:
    if not name:
        return False
    return sum(ch.isalnum() for ch in name) >= 3


def cluster_and_canonicalize(
    names: List[str],
    score_cutoff: int = 90,
    stopwords: Optional[Iterable[str]] = None,
    alias_map: Optional[Dict[str, str]] = None,
    entity: str = "generic",
) -> Dict[str, str]:
    """Return mapping of each name to a canonical representative using rapidfuzz."""

    stop = {s.lower() for s in (stopwords or [])}
    normalized: List[str] = []
    for n in names:
        norm = normalize_name(n)
        if norm:
            normalized.append(norm)
    normalized = [n for n in normalized if _valid_name(n) and n.lower() not in stop]
    unique = sorted(set(normalized))
    mapping: Dict[str, str] = {}

    if alias_map:
        for alias, canon in alias_map.items():
            alias_norm = normalize_name(alias)
            canon_norm = normalize_name(canon)
            if alias_norm and canon_norm:
                mapping[alias_norm] = canon_norm

    if entity == "location" and unique:
        from rules import canon_location  # local import to avoid circular dependency

        buckets = defaultdict(list)
        for item in unique:
            canon = canon_location(item)
            if not canon:
                continue
            buckets[canon].append(item)
        for country, items in buckets.items():
            preferred = country if country else max(items, key=len)
            for it in items:
                mapping[it] = preferred
        return mapping

    for name in unique:
        if name in mapping or name.lower() in stop:
            continue
        matches = process.extract(name, unique, score_cutoff=score_cutoff)
        canon = name
        for match_name, _score, _ in matches:
            if match_name.lower() in stop:
                continue
            mapping.setdefault(match_name, canon)
    return mapping
