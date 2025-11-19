"""Regex driven helpers that complement the HF NER output."""

from __future__ import annotations

import re
from typing import List

from rules import plausible_org, strip_preps

# Capture title-cased spans that often denote organisations.
# Examples handled:
#   - "at Universidad Complutense de Madrid"
#   - "with Banco Santander"
#   - "Harvard University"
#   - "London School of Economics and Political Science"
_ORG_PATTERNS = [
    re.compile(
        r"(?P<org>(?:University|Universidad|Université|Università|Universidade|College|School|Institute|Instituto|Faculty|Facultad|Escuela|Polytechnic|Politécnica)[^.;,]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:at|for|from|in|with|by)\s+(?P<org>[A-Z][A-Za-z0-9&'’/().\-]+(?:\s+[A-Z][A-Za-z0-9&'’/().\-]+){1,})",
    ),
]


def org_candidates_from_text(text: str) -> List[str]:
    """Return plausible organisation names extracted with regexes."""

    if not text:
        return []
    candidates = []
    seen = set()
    for pattern in _ORG_PATTERNS:
        for match in pattern.finditer(text):
            org = match.group("org")
            org = org.strip(" \t-•,.;:()[]")
            org = strip_preps(org)
            if not org:
                continue
            if not plausible_org(org):
                continue
            key = org.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(org)
    return candidates
