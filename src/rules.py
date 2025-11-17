"""Regex-driven extraction utilities for structured attributes."""

from __future__ import annotations

import re
from typing import Dict, Literal, Optional, Tuple

from normalize import normalize_name

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

def _alias_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


ORG_ALIAS_BY_KEY: Dict[str, str] = {
    "iebusinessschool": "IE Business School",
    "iebbusinessschool": "IE Business School",
    "ieuniversity": "IE Business School",
    "institutodeempresa": "IE Business School",
    "mbaie": "IE Business School",
    "universityofnavarra": "Universidad de Navarra",
    "universidaddenavarra": "Universidad de Navarra",
    "udenavarra": "Universidad de Navarra",
}

# Backwards-compatible alias map for clustering helpers (uses human-readable keys).
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

_COUNTRY_PATTERNS = [
    (r"\b(usa|u\.s\.a\.|u\.s\.|united states)\b", "USA"),
    (r"\b(uk|u\.k\.|united kingdom|england|scotland|wales|northern ireland)\b", "United Kingdom"),
    (r"\b(uae|united arab emirates)\b", "United Arab Emirates"),
    (r"\b(spain|españa)\b", "Spain"),
]

_EXPECTED_RELATION_TYPES: Dict[str, Literal["university", "company"]] = {
    "studied_at": "university",
    "worked_at": "company",
    "teaches": "university",
}


def canon_org(name: Optional[str]) -> str:
    """Normalise organisation names and expand known aliases."""

    if not name:
        return ""
    norm = normalize_name(name)
    key = _alias_key(norm)
    return ORG_ALIAS_BY_KEY.get(key, norm)

def canon_location(name: str) -> str:
    """Return a coarse, country-first canonical location (e.g., 'USA', 'Spain').
    If no country pattern is found, return a cleaned version of the input.
    """
    if not name:
        return ""
    n = normalize_name(name)
    low = n.lower()
    n = LOCATION_ALIASES.get(low, n)
    low = n.lower()
    for pat, canon in _COUNTRY_PATTERNS:
        if re.search(pat, low):
            return canon
    return n  # fallback: leave as cleaned full string (city/state etc.)

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
