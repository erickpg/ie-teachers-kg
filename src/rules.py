"""Regex-driven extraction utilities for structured attributes."""

from __future__ import annotations

import re
from typing import Dict, Literal, Optional, Tuple

from normalize import normalize_name

# Hard exceptions always win
ORG_HARD_POSITIVE = {
    "ie business school": "university",
    "ie university": "university",
    "universidad de navarra": "university",
    "universidad complutense de madrid": "university",
}
ORG_HARD_NEGATIVE = {
    "bank of spain": "company",
    "banco de españa": "company",
}

# Section names we use as priors
SECTION_PRIOR = {
    "academic_background": "university",
    "academic_experience": "university",
    "corporate_experience": "company",
}

# Lexical cues (soft signals)
UNIV_CUES = [
    "university",
    "universidad",
    "université",
    "università",
    "universidade",
    "college",
    "school of",
    "faculty",
    "facultad",
    "escuela",
    "polytechnic",
    "politécnica",
    "institute of technology",
    "institut of technology",
]
COMP_CUES = [
    "inc",
    "llc",
    "ltd",
    "ltda",
    "s.a.",
    "s.l.",
    "gmbh",
    "ag",
    "bv",
    "spa",
    "pty",
    "partners",
    "group",
    "consulting",
    "capital",
    "bank",
    "studio",
    "lab",
    "labs",
    "holding",
    "fund",
]
GOV_CUES = [
    "ministry",
    "council",
    "agency",
    "commission",
    "secretariat",
    "court",
    "bank of",
    "central bank",
    "city of",
    "state of",
    "department of",
]

SECTION_STOP_ORGS = {
    "academic experience",
    "academic exp",
    "corporate experience",
    "academic background",
    "courses taught",
    "education",
    "publications",
    "professional experience",
    # Sustainability/SDG headings that are frequently mistaken for org names
    "quality education",
    "industry innovation and infrastructure",
    "innovation and infraestructure",
    "sustainable cities and communities",
    "work and economic growth",
    "reduced inequalities",
    "responsible consumption and production",
    "peace justice and strong institutions",
    "partnerships for the goals",
    "climate action",
    "life below water",
    "life on land",
    "no poverty",
    "zero hunger",
    "good health and well-being",
    "gender equality",
    "clean water and sanitation",
    "affordable and clean energy",
    "decent work and economic growth",
}

HONORS_THESIS_TOKENS = (
    "summa cum laude",
    "magna cum laude",
    "cum laude",
    "thesis",
    "dissertation",
    "with a thesis",
    "with thesis",
    "honors",
    "honours",
)

PREP_STRIP_RE = re.compile(r"^(?:by|at|in|of|from|with)\s+", re.I)


def strip_preps(s: str) -> str:
    return PREP_STRIP_RE.sub("", s or "").strip()


def looks_like_honor(s: str) -> bool:
    low = (s or "").lower()
    return any(tok in low for tok in HONORS_THESIS_TOKENS)


def plausible_org(s: str) -> bool:
    if not s:
        return False
    t = strip_preps(normalize_name(s))
    low = t.lower()
    if low in SECTION_STOP_ORGS:
        return False
    if looks_like_honor(t):
        return False
    if len(t) > 100:
        return False
    toks = [w for w in re.split(r"\W+", t) if w]
    if len(toks) < 2:
        return False
    letters = sum(ch.isalpha() for ch in t)
    if letters / max(1, len(t)) < 0.6:
        return False
    return True

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
    """Backwards compatible alias for the new soft label helper."""

    if not name:
        return None
    label = soft_name_label(name)
    return None if label == "unknown" else label


def soft_name_label(n: str) -> Literal["university", "company", "government", "unknown"]:
    """Heuristic classifier from the NAME only (soft)."""

    low = (n or "").lower()
    if low in ORG_HARD_POSITIVE:
        return "university"
    if low in ORG_HARD_NEGATIVE:
        return ORG_HARD_NEGATIVE[low]
    if any(c in low for c in UNIV_CUES):
        return "university"
    if any(c in low for c in GOV_CUES):
        return "government"
    if any(c in low for c in COMP_CUES):
        return "company"
    return "unknown"


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
