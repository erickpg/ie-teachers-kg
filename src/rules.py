"""Regex-driven extraction utilities for structured attributes."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

_DEGREE_PAT = re.compile(
    r"\b(PhD|MBA|MSc|MA|MS|MEng|BSc|BA|BEng|Doctor|Master|Bachelor)\b",
    re.IGNORECASE,
)
_FIELD_PAT = re.compile(r"in ([A-Z][A-Za-z &/]+)")
_YEAR_PAT = re.compile(r"(19|20)\d{2}")
_COURSE_PAT = re.compile(r"course(?:s)?\s+such as\s+([^.;]+)", re.IGNORECASE)
_COURSE_SPLIT = re.compile(r",| and ")

_UNI_KEYWORDS = ["university", "college", "school", "institute"]
_COMP_KEYWORDS = ["consulting", "bank", "corp", "company", "group", "llc", "sa", "sl"]


def extract_degrees_and_years(text: str) -> List[Dict[str, Optional[str]]]:
    """Return list of detected degree, field and year information."""

    results: List[Dict[str, Optional[str]]] = []
    for match in _DEGREE_PAT.finditer(text):
        degree = match.group(1)
        span = text[match.end() : match.end() + 80]
        field_match = _FIELD_PAT.search(span)
        year_match = _YEAR_PAT.search(span)
        results.append(
            {
                "degree": degree,
                "field": field_match.group(1) if field_match else None,
                "year": int(year_match.group(0)) if year_match else None,
            }
        )
    return results


def extract_courses(text: str) -> List[str]:
    """Extract course names using lightweight patterns."""

    courses: List[str] = []
    for match in _COURSE_PAT.finditer(text):
        chunk = match.group(1)
        for piece in _COURSE_SPLIT.split(chunk):
            clean = piece.strip(" .")
            if clean:
                courses.append(clean)
    return courses


def classify_org(name: str) -> Optional[str]:
    """Heuristically classify an organisation as university/company."""

    low = (name or "").lower()
    if any(key in low for key in _UNI_KEYWORDS):
        return "university"
    if any(key in low for key in _COMP_KEYWORDS):
        return "company"
    return None


def year_bin(year: int, width: int = 5) -> Optional[str]:
    """Return the inclusive 5-year bin label for the provided year."""

    if year is None or year < 1900:
        return None
    start = year - ((year - 1900) % width)
    end = start + width - 1
    return f"{start}-{end}"
