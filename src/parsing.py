"""Utilities for cleaning and segmenting professor bios."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

from fusion import Candidate, Relation
from normalize import normalize_name
from rules import (
    canon_location_with_flag,
    canon_org_with_flag,
    classify_org,
    normalize_degree,
    year_bin,
)

_SECTION_MAP = {
    "corporate experience": "corporate_experience",
    "professional experience": "corporate_experience",
    "work experience": "corporate_experience",
    "academic experience": "academic_experience",
    "academic background": "academic_background",
    "education": "academic_background",
}

_SECTION_RELATION: Dict[str, Relation] = {
    "corporate_experience": "worked_at",
    "academic_background": "studied_at",
    "academic_experience": "worked_at",
}

ROLE_COMPANY_LINE = re.compile(
    r"^[•\-\*]?\s*(?P<role>[^,;|]+?)\s*,\s*(?P<org>[^,;|]+?)\s*,\s*(?P<location>[^,;|]+?)\s*,\s*(?P<years>[^,;|]+?)\s*$",
    flags=re.IGNORECASE,
)
DEGREE_LINE = re.compile(
    r"^[•\-\*]?\s*(?P<degree>[^,;|]+?)\s*,\s*(?P<university>[^,;|]+?)\s*,\s*(?P<location>[^,;|]+?)\s*,\s*(?P<year>\d{4})\s*$",
    flags=re.IGNORECASE,
)
YEAR_RANGE = re.compile(
    r"(?P<start>(19|20)\d{2})\s*(?:–|-|to|—)\s*(?P<end>(present|now|nowadays|(19|20)\d{2}))",
    flags=re.IGNORECASE,
)


def _heading_key(text: str) -> Optional[str]:
    clean = re.sub(r"[\s:\-–—]+", " ", text or "").strip().lower()
    for key, value in _SECTION_MAP.items():
        if key in clean:
            return value
    return None


def _normalise_heading(text: str) -> str:
    return _heading_key(text) or "intro"


def clean_html(html: str) -> str:
    """Return plain text cleaned from the incoming HTML snippet."""

    soup = BeautifulSoup(html or "", "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(" \n", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sections(html: str) -> Dict[str, str]:
    """Split the biography by headings into predefined sections."""

    soup = BeautifulSoup(html or "", "lxml")
    sections: Dict[str, List[str]] = defaultdict(list)
    current_key = "intro"

    for element in soup.recursiveChildGenerator():
        name = getattr(element, "name", None)
        if name == "h4":
            heading = element.get_text(" ", strip=True)
            current_key = _normalise_heading(heading)
            continue
        if name == "strong" and getattr(getattr(element, "parent", None), "name", "").lower() != "h4":
            heading = element.get_text(" ", strip=True)
            candidate = _heading_key(heading)
            if candidate:
                current_key = candidate
                continue
        if name == "li":
            text = element.get_text(" ", strip=True)
            if text:
                sections[current_key].append(f"• {text}")
            continue
        string = getattr(element, "string", None)
        if string:
            parent = getattr(element, "parent", None)
            if getattr(parent, "name", "") == "h4":
                continue
            text = string.strip()
            if text:
                candidate = _heading_key(text)
                if candidate and len(text) <= 80:
                    current_key = candidate
                    continue
                sections[current_key].append(text)

    if not sections:
        sections["intro"].append(clean_html(html))

    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


def sentences(text: str) -> List[str]:
    """Split a block of text into light-weight sentences."""

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def section_relation(section: str) -> Relation:
    """Return the relation that best matches the section name."""

    return _SECTION_RELATION.get(section, "unknown")


def parse_corporate_bullet(line: str) -> Optional[Dict[str, Optional[str]]]:
    """Parse a corporate experience bullet line if the pattern matches."""

    if not line:
        return None
    match = ROLE_COMPANY_LINE.match(line.strip())
    if not match:
        return None
    years_raw = match.group("years").strip()
    start = end = None
    end_year_text: Optional[str] = None
    range_match = YEAR_RANGE.search(years_raw)
    if range_match:
        start = int(range_match.group("start"))
        end_txt = range_match.group("end").lower()
        if end_txt in {"present", "now", "nowadays"}:
            end = None
        else:
            end = int(end_txt)
    else:
        end_year_text = years_raw
    return {
        "role": match.group("role").strip(),
        "org": match.group("org").strip(),
        "location": match.group("location").strip(),
        "start_year": start,
        "end_year": end,
        "end_year_text": end_year_text,
    }


def parse_degree_bullet(line: str) -> Optional[Dict[str, Optional[str]]]:
    """Parse an academic background bullet line if the pattern matches."""

    if not line:
        return None
    match = DEGREE_LINE.match(line.strip())
    if not match:
        return None
    return {
        "degree_text": match.group("degree").strip(),
        "university": match.group("university").strip(),
        "location": match.group("location").strip(),
        "year": int(match.group("year")),
    }


def iter_bullets(section_text: str) -> List[str]:
    """Return cleaned bullet/line items for a section block."""

    if not section_text:
        return []
    chunks: List[str] = []
    for raw in re.split(r"[\n\r]+", section_text):
        stripped = raw.strip()
        if not stripped:
            continue
        # Split additional inline bullets
        pieces = re.split(r"(?<=\))\s+•\s+|\s+•\s+|\s+-\s+", stripped)
        for piece in pieces:
            clean_piece = piece.strip()
            if clean_piece:
                chunks.append(clean_piece)
    return chunks


def build_pattern_candidates(section: str, lines: List[str]) -> List[Candidate]:
    """Create pattern-based candidates from parsed lines."""

    candidates: List[Candidate] = []
    if not lines:
        return candidates
    for idx, raw in enumerate(lines):
        snippet = raw.strip()
        if not snippet:
            continue
        if section == "corporate_experience":
            parsed = parse_corporate_bullet(snippet)
            if not parsed or not parsed.get("org"):
                continue
            org_raw = parsed["org"]
            org_norm = normalize_name(org_raw)
            org_canon, alias_hit = canon_org_with_flag(org_raw)
            loc_raw = parsed.get("location")
            loc_norm = normalize_name(loc_raw) if loc_raw else None
            loc_canon = None
            if loc_raw:
                loc_canon, _ = canon_location_with_flag(loc_raw)
            candidate = Candidate(
                relation="worked_at",
                source="pattern",
                section=section,
                line_idx=idx,
                text_span=snippet,
                org_raw=org_raw,
                org_norm=org_norm,
                org_canon=org_canon,
                org_type_guess=classify_org(org_canon) or "unknown",
                location_raw=loc_raw,
                location_norm=loc_norm,
                location_canon=loc_canon,
                role=parsed.get("role"),
                start_year=parsed.get("start_year"),
                end_year=parsed.get("end_year"),
                end_year_text=parsed.get("end_year_text"),
            )
            candidate.meta = {"sources": ["pattern"], "alias_hit": alias_hit, "parser": "corporate_bullet"}
            candidates.append(candidate)
        elif section == "academic_background":
            parsed = parse_degree_bullet(snippet)
            if not parsed or not parsed.get("university"):
                continue
            org_raw = parsed["university"]
            org_norm = normalize_name(org_raw)
            org_canon, alias_hit = canon_org_with_flag(org_raw)
            loc_raw = parsed.get("location")
            loc_norm = normalize_name(loc_raw) if loc_raw else None
            loc_canon = None
            if loc_raw:
                loc_canon, _ = canon_location_with_flag(loc_raw)
            level, field = normalize_degree(parsed.get("degree_text"))
            year_val = parsed.get("year")
            candidate = Candidate(
                relation="studied_at",
                source="pattern",
                section=section,
                line_idx=idx,
                text_span=snippet,
                org_raw=org_raw,
                org_norm=org_norm,
                org_canon=org_canon,
                org_type_guess=classify_org(org_canon) or "unknown",
                location_raw=loc_raw,
                location_norm=loc_norm,
                location_canon=loc_canon,
                degree_text=parsed.get("degree_text"),
                degree_level=level,
                field=field,
                year=year_val,
                year_bin=year_bin(year_val),
            )
            candidate.meta = {"sources": ["pattern"], "alias_hit": alias_hit, "parser": "degree_bullet"}
            candidates.append(candidate)
    return candidates


def pattern_candidates_from_section(section: str, section_text: str) -> List[Candidate]:
    """Convenience helper that splits text and builds pattern candidates."""

    lines = iter_bullets(section_text)
    if not lines and section_text:
        lines = [section_text]
    return build_pattern_candidates(section, lines)
