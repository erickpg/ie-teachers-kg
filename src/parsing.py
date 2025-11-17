"""Utilities for cleaning and segmenting professor bios."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional

from bs4 import BeautifulSoup


_SECTION_MAP = {
    "corporate experience": "corporate_experience",
    "academic experience": "academic_experience",
    "academic background": "academic_background",
}


def clean_html(html: str) -> str:
    """Return plain text cleaned from the incoming HTML snippet."""

    soup = BeautifulSoup(html or "", "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(" \n", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sections(html: str) -> Dict[str, str]:
    """Split the biography by <h4> headings into predefined sections."""

    soup = BeautifulSoup(html or "", "lxml")
    sections: Dict[str, List[str]] = defaultdict(list)

    current_key = "intro"
    for element in soup.recursiveChildGenerator():
        if getattr(element, "name", None) == "h4":
            heading = element.get_text(" ", strip=True).lower()
            current_key = _SECTION_MAP.get(heading, "intro")
            continue
        if getattr(element, "string", None):
            text = element.string.strip()
            if not text:
                continue
            sections[current_key].append(text)

    if not sections:
        sections["intro"].append(clean_html(html))

    return {k: " ".join(v).strip() for k, v in sections.items() if v}


def sentences(text: str) -> List[str]:
    """Split a block of text into light-weight sentences."""

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


_BULLET_SPLIT = re.compile(r"(?:\n+|•+|\u2022+|\*+|\s*[-–]\s+)")


def _strip_tags(text: str) -> str:
    """Remove remaining HTML tags from a snippet."""

    return re.sub(r"<[^>]+>", " ", text or "")


def iter_bullets(section_text: str) -> List[str]:
    """Return a list of bullet/line items for the provided section text."""

    if not section_text:
        return []

    soup = BeautifulSoup(section_text, "lxml")
    bullets = [
        re.sub(r"\s+", " ", li.get_text(" ", strip=True))
        for li in soup.find_all("li")
    ]
    if bullets:
        return [b.strip(" -•") for b in bullets if b.strip(" -•")]

    text = soup.get_text("\n", strip=True) or section_text
    text = _strip_tags(text)
    items = [re.sub(r"\s+", " ", part).strip(" -•") for part in _BULLET_SPLIT.split(text)]
    return [item for item in items if item]


_YEAR_RANGE = re.compile(
    r"(?P<start>(?:19|20)\d{2})\s*(?:[–-]\s*(?P<end>(?:19|20)\d{2}|present|ongoing))?",
    re.IGNORECASE,
)


def _extract_years(line: str) -> Dict[str, Optional[int]]:
    match = list(_YEAR_RANGE.finditer(line or ""))
    if not match:
        return {"start_year": None, "end_year": None, "end_year_text": None}
    last = match[-1]
    start = int(last.group("start"))
    end_raw = last.group("end")
    end_year = None
    end_year_text: Optional[str] = None
    if end_raw:
        if end_raw.isdigit():
            end_year = int(end_raw)
        else:
            end_year_text = end_raw.title()
    return {"start_year": start, "end_year": end_year, "end_year_text": end_year_text}


def _split_parts(line: str) -> List[str]:
    parts = [p.strip(" -") for p in re.split(r"[,;]", line) if p.strip(" -")]
    return parts


def parse_corporate_bullet(line: str) -> Optional[Dict[str, Optional[str]]]:
    """Parse a corporate experience bullet into structured fields."""

    text = (line or "").strip()
    if not text:
        return None
    years = _extract_years(text)
    parts = _split_parts(_YEAR_RANGE.sub("", text))
    if len(parts) < 2:
        return None
    role = parts[0]
    org = parts[1]
    location = parts[2] if len(parts) > 2 else None
    return {
        "role": role or None,
        "org": org or None,
        "location": location or None,
        **years,
    }


def parse_academic_background_bullet(line: str) -> Optional[Dict[str, Optional[str]]]:
    """Parse an academic background bullet."""

    text = (line or "").strip()
    if not text:
        return None
    years = _extract_years(text)
    parts = _split_parts(_YEAR_RANGE.sub("", text))
    if not parts:
        return None
    degree_text = parts[0]
    university_raw = parts[1] if len(parts) > 1 else None
    universities: List[str] = []
    if university_raw:
        universities = [p.strip() for p in re.split(r"[\//&]", university_raw) if p.strip()]
    location = parts[2] if len(parts) > 2 else None
    location_list: List[str] = []
    if location:
        location_list = [p.strip() for p in re.split(r"[\//&]", location) if p.strip()]
    return {
        "degree_text": degree_text or None,
        "university": universities or None,
        "location": location or None,
        "location_list": location_list or None,
        "year": years["start_year"] or years["end_year"],
    }


_COURSE_PATTERNS = [
    re.compile(r"Adjunct (?:Level [^ ]+ )?Professor of (?P<course>[^,]+)", re.IGNORECASE),
    re.compile(r"Professor of (?P<course>[^,]+)", re.IGNORECASE),
    re.compile(r"taught (?P<course>.+?) at ", re.IGNORECASE),
]


def _extract_course(text: str) -> Optional[str]:
    for pattern in _COURSE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group("course").strip()
    return None


def parse_academic_experience_bullet(line: str) -> Optional[Dict[str, Optional[str]]]:
    """Parse an academic experience bullet with role/course/center data."""

    text = (line or "").strip()
    if not text:
        return None
    years = _extract_years(text)
    parts = _split_parts(_YEAR_RANGE.sub("", text))
    if len(parts) < 2:
        return None
    role = parts[0]
    center = parts[1]
    location = parts[2] if len(parts) > 2 else None
    course = _extract_course(text)
    return {
        "role": role or None,
        "course": course or None,
        "center": center or None,
        "location": location or None,
        **years,
    }
