"""Regex-driven extraction utilities for structured attributes."""

from __future__ import annotations

import re
from typing import Dict, Literal, Optional, Tuple

from .normalize import normalize_name

SECTION_STOP_ORGS = {
    "academic experience",
    "academic exp",
    "corporate experience",
    "academic background",
    "courses taught",
    "education",
    "publications",
    "professional experience",
}

ORG_ALIASES: Dict[str, str] = {
    "ie": "IE Business School",
    "ie university": "IE Business School",
    "instituto de empresa": "IE Business School",
    "ie business school": "IE Business School",
    "mba ie": "IE Business School",
    "u. de navarra": "Universidad de Navarra",
    "u de navarra": "Universidad de Navarra",
    "u navarra": "Universidad de Navarra",
    "universidad navarra": "Universidad de Navarra",
    "university of navarra": "Universidad de Navarra",
}

LOCATION_ALIASES: Dict[str, str] = {
    "spaing": "Spain",
    "spain": "Spain",
    "u.k.": "United Kingdom",
    "uk": "United Kingdom",
    "uae": "United Arab Emirates",
}

DEGREE_MAP: Dict[str, str] = {
    "postdoctoral": "Postdoc",
    "post-doctoral": "Postdoc",
    "postdoctoral studies": "Postdoc",
    "postdoc": "Postdoc",
    "phd": "PhD",
    "ph.d.": "PhD",
    "doctor": "PhD",
    "e.m.b.a.": "MBA",
    "m.b.a.": "MBA",
    "mba": "MBA",
    "msc": "MSc",
    "m.sc.": "MSc",
    "ms": "MSc",
    "master": "MSc",
    "ma": "MA",
    "m.a.": "MA",
    "bsc": "BSc",
    "b.sc.": "BSc",
    "ba": "BA",
    "b.a.": "BA",
    "meng": "MEng",
    "beng": "BEng",
    "certificate": "Certificate",
    "executive certificate": "Certificate",
    "diploma": "Certificate",
}

_UNI_HINTS = [
    "university",
    "college",
    "school",
    "institute",
    "politécnica",
    "escuela",
    "facultad",
]

_COMP_HINTS = [
    "studio",
    "capital",
    "partners",
    "group",
    "consulting",
    "bank",
    "digital",
    "bcg",
    "etisalat",
    "vidivixi",
    "halliburton",
    "ge",
    "permasteelisa",
    "millwood",
    "corp",
    "company",
]

_EXPECTED_RELATION_TYPES: Dict[str, Literal["university", "company"]] = {
    "studied_at": "university",
    "worked_at": "company",
    "teaches": "university",
}


def canon_org(name: Optional[str]) -> Optional[str]:
    """Normalise organisation names and expand known aliases."""

    if not name:
        return None
    norm = normalize_name(name)
    low = norm.lower()
    return ORG_ALIASES.get(low, norm)


def canon_location(name: Optional[str]) -> Optional[str]:
    """Normalise location names and expand aliases."""

    if not name:
        return None
    norm = normalize_name(name)
    low = norm.lower()
    return LOCATION_ALIASES.get(low, norm)


def classify_org(name: Optional[str]) -> Optional[str]:
    """Heuristically classify the organisation type."""

    if not name:
        return None
    low = name.lower()
    for keyword in _UNI_HINTS:
        if keyword in low:
            return "university"
    for keyword in _COMP_HINTS:
        if keyword in low:
            return "company"
    return None


def normalize_degree(text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return canonical degree level and extracted field if available."""

    if not text:
        return None, None
    clean = text.strip()
    lowered = clean.lower()
    lowered_simple = lowered.replace(".", "")
    level = None
    for key, value in DEGREE_MAP.items():
        key_norm = key.lower().replace(".", "")
        if key_norm in lowered_simple:
            level = value
            break
    field = None
    match = re.search(r"in ([A-Za-z0-9 &/\-]+)", clean, re.IGNORECASE)
    if match:
        field = match.group(1).strip()
    return level, field


def year_bin(year: Optional[int], width: int = 5) -> Optional[str]:
    """Return the inclusive year bin for the provided width."""

    if year is None or year < 1900:
        return None
    start = year - ((year - 1900) % width)
    end = start + width - 1
    return f"{start}-{end}"
