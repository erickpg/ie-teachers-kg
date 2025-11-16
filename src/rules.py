"""Regex and rule helpers for structured information extraction."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from normalize import normalize_name

ORG_ALIASES: Dict[str, str] = {
    # IE variants collapse into a single canonical label
    "ie": "IE Business School",
    "ie university": "IE Business School",
    "ie business school": "IE Business School",
    "instituto de empresa": "IE Business School",
    "ie (instituto de empresa)": "IE Business School",
    "mba ie": "IE Business School",
    "ie business": "IE Business School",
    "ie school": "IE Business School",
    "ie business school (mba)": "IE Business School",
    # Universidad de Navarra variants
    "u. de navarra": "Universidad de Navarra",
    "u de navarra": "Universidad de Navarra",
    "u navarra": "Universidad de Navarra",
    "universidad navarra": "Universidad de Navarra",
    "university of navarra": "Universidad de Navarra",
    "universidad de navarra": "Universidad de Navarra",
}

LOCATION_ALIASES: Dict[str, str] = {
    "spaing": "Spain",
    "spain": "Spain",
    "u.k.": "United Kingdom",
    "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "uae": "United Arab Emirates",
    "u.a.e": "United Arab Emirates",
}

UNIV_HINTS = [
    "university",
    "universidad",
    "college",
    "school of",
    "business school",
    "institute",
    "institut",
    "polytechnic",
    "politécnica",
    "escuela",
    "facultad",
]

COMP_HINTS = [
    "inc",
    "ltd",
    "llc",
    "gmbh",
    "s.a.",
    "s.a",
    "s.l.",
    "studio",
    "capital",
    "partners",
    "group",
    "consulting",
    "bank",
    "digital",
    "bcg",
    "boston consulting group",
    "etisalat",
    "vidivixi",
    "hub",
    "point",
    "holding",
    "becquerel",
    "a&m",
    "bearingpoint",
]

DEGREE_MAP = {
    "phd": "PhD",
    "ph.d.": "PhD",
    "doctor": "PhD",
    "msc": "MSc",
    "ms": "MSc",
    "m.sc.": "MSc",
    "master": "MSc",
    "master's": "MSc",
    "ma": "MA",
    "m.a.": "MA",
    "mba": "MBA",
    "bsc": "BSc",
    "b.sc.": "BSc",
    "ba": "BA",
    "b.a.": "BA",
    "meng": "MEng",
    "m.eng.": "MEng",
    "beng": "BEng",
    "b.eng.": "BEng",
    "bachelor": "BSc",
}

_DEGREE_PAT = re.compile(r"\b(PhD|MBA|MSc|MA|MS|MEng|BSc|BA|BEng|Doctor|Master|Bachelor)\b", re.IGNORECASE)
_FIELD_PAT = re.compile(r"\bin\s+([A-Z][A-Za-z &/\-]+)")
_YEAR_PAT = re.compile(r"(19|20)\d{2}")
_COURSE_PAT = re.compile(r"course(?:s)?\s+(?:such as|including)\s+([^.;]+)", re.IGNORECASE)
_COURSE_SPLIT = re.compile(r",| and ")


def extract_degrees_and_years(text: str) -> List[Dict[str, Optional[str]]]:
    """Return degree, field, and year hints from free text snippets."""

    text = text or ""
    results: List[Dict[str, Optional[str]]] = []
    for match in _DEGREE_PAT.finditer(text):
        degree = match.group(0)
        window = text[match.end() : match.end() + 80]
        field_match = _FIELD_PAT.search(window)
        year_match = _YEAR_PAT.search(window)
        level, field_norm = normalize_degree(degree)
        results.append(
            {
                "degree": degree,
                "degree_canon": level,
                "field": field_norm or (field_match.group(1) if field_match else None),
                "year": int(year_match.group(0)) if year_match else None,
            }
        )
    return results


def extract_courses(text: str) -> List[str]:
    """Extract course names from descriptive text snippets."""

    text = text or ""
    courses: List[str] = []
    for match in _COURSE_PAT.finditer(text):
        chunk = match.group(1)
        for piece in _COURSE_SPLIT.split(chunk):
            clean = piece.strip(" .;")
            if clean:
                courses.append(clean)
    return courses


def normalize_degree(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (degree_level, field) parsed from the free-text degree chunk."""

    if not text:
        return None, None
    cleaned = text.strip()
    low = cleaned.lower()
    level = next((val for key, val in DEGREE_MAP.items() if key in low), None)
    field = None
    match = re.search(r"\bin\s+([A-Z][A-Za-z&\-\s]{2,})", cleaned)
    if match:
        field = match.group(1).strip()
    return level, field


def classify_org(name: str) -> Optional[str]:
    """Heuristically classify an organisation as a university or company."""

    if not name:
        return None
    canonical = canon_org(name)
    low = canonical.lower()
    academic_aliases = {normalize_name(val).lower() for val in ORG_ALIASES.values()}
    if low in academic_aliases:
        return "university"
    if any(hint in low for hint in UNIV_HINTS):
        return "university"
    if any(hint in low for hint in COMP_HINTS):
        return "company"
    return None


def year_bin(year: Optional[int], width: int = 5) -> Optional[str]:
    """Bin a year into inclusive `width`-wide buckets."""

    if year is None:
        return None
    if not isinstance(year, int) or year < 1900:
        return None
    start = year - (year % width)
    end = start + width - 1
    return f"{start}-{end}"


def canon_org_with_flag(name: str) -> Tuple[str, bool]:
    """Return canonical org name and whether it matched a known alias."""

    norm = normalize_name(name)
    low = norm.lower()
    if low in ORG_ALIASES:
        return normalize_name(ORG_ALIASES[low]), True
    return norm, False


def canon_org(name: str) -> str:
    """Apply unicode cleanup and alias collapsing to organisation names."""

    canonical, _ = canon_org_with_flag(name)
    return canonical


def canon_location_with_flag(name: str) -> Tuple[str, bool]:
    """Return canonical location plus whether it matched a known alias."""

    norm = normalize_name(name)
    low = norm.lower()
    if low in LOCATION_ALIASES:
        return normalize_name(LOCATION_ALIASES[low]), True
    return norm, False


def canon_location(name: str) -> str:
    """Apply alias collapsing for noisy location labels."""

    canonical, _ = canon_location_with_flag(name)
    return canonical
