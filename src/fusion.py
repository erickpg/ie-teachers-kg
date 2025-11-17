"""Line-level fusion between parser output and NER evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Literal, Optional

from rapidfuzz import fuzz

from rules import (
    SECTION_STOP_ORGS,
    canon_location,
    canon_org,
    classify_org,
    normalize_degree,
    year_bin,
)

Relation = Literal["worked_at", "studied_at", "teaches", "unknown"]
OrgType = Literal["university", "company", "unknown"]

_SECTION_RELATIONS: Dict[str, Relation] = {
    "corporate_experience": "worked_at",
    "academic_background": "studied_at",
    "academic_experience": "teaches",
}

_EXPECTED_TYPES: Dict[Relation, OrgType] = {
    "worked_at": "company",
    "studied_at": "university",
    "teaches": "university",
    "unknown": "unknown",
}


@dataclass
class LineCandidate:
    """Single fused candidate extracted from one line."""

    relation: Relation
    section: str
    text_span: str
    role: Optional[str] = None
    course: Optional[str] = None
    org_raw: Optional[str] = None
    org_canon: Optional[str] = None
    org_type: OrgType = "unknown"
    location_raw: Optional[str] = None
    location_canon: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    year: Optional[int] = None
    year_bin: Optional[str] = None
    degree_text: Optional[str] = None
    degree_level: Optional[str] = None
    field: Optional[str] = None
    sources: List[str] = dc_field(default_factory=list)
    meta: Dict[str, Any] = dc_field(default_factory=dict)


def _relation_for_section(section: str) -> Relation:
    return _SECTION_RELATIONS.get(section, "unknown")


def _expected_type(relation: Relation) -> OrgType:
    return _EXPECTED_TYPES.get(relation, "unknown")


def _alnum_length(text: Optional[str]) -> int:
    if not text:
        return 0
    return sum(ch.isalnum() for ch in text)


def _select_ner_org(ner: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    if not ner:
        return None
    orgs = ner.get("orgs") or []
    if not orgs:
        return None
    return orgs[0]["text"].strip()


def _select_ner_location(ner: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    locs = ner.get("locs") or []
    if not locs:
        return None
    return locs[0]["text"].strip()


def _extract_year_from_dates(ner: Dict[str, List[Dict[str, Any]]]) -> Optional[int]:
    for ent in ner.get("dates") or []:
        match = re.search(r"(19|20)\d{2}", ent["text"])
        if match:
            return int(match.group(0))
    return None


def _build_base_orgs(
    relation: Relation, parsed: Optional[Dict[str, Any]], ner: Dict[str, Any]
) -> List[Optional[str]]:
    parsed_orgs: List[Optional[str]] = []
    if parsed:
        if relation == "studied_at":
            unis = parsed.get("university")
            if isinstance(unis, list):
                parsed_orgs = unis
            elif unis:
                parsed_orgs = [unis]
        elif relation == "worked_at":
            if parsed.get("org"):
                parsed_orgs = [parsed.get("org")]
        elif relation == "teaches":
            if parsed.get("center"):
                parsed_orgs = [parsed.get("center")]

    parsed_orgs = [org for org in parsed_orgs if org]
    if parsed_orgs:
        return parsed_orgs

    ner_org = _select_ner_org(ner)
    if ner_org:
        return [ner_org]
    return [None]


def _score_candidate(
    relation: Relation,
    section: str,
    org_raw: Optional[str],
    org_from_parser: bool,
    ner_org: Optional[str],
    org_type: OrgType,
) -> float:
    score = 0.0
    if org_from_parser:
        score += 0.6
    if org_from_parser and ner_org and org_raw:
        if fuzz.WRatio(org_raw, ner_org) >= 90:
            score += 0.2
    if _relation_for_section(section) == relation:
        score += 0.1
    expected = _expected_type(relation)
    if expected != "unknown" and org_type == expected:
        score += 0.1
    if (org_raw or "").lower() in SECTION_STOP_ORGS or _alnum_length(org_raw) < 3:
        score -= 0.2
    return round(score, 3)


def fuse_line(
    section: str,
    line: str,
    parsed: Optional[Dict[str, Any]],
    ner: Optional[Dict[str, List[Dict[str, Any]]]],
) -> List[LineCandidate]:
    """Fuse parser and NER output for a single line."""

    relation = _relation_for_section(section)
    ner = ner or {"orgs": [], "locs": [], "dates": []}
    sources: List[str] = []
    if parsed:
        sources.append("parser")
    if any(ner.values()):
        sources.append("ner")

    base_orgs = _build_base_orgs(relation, parsed, ner)
    location = parsed.get("location") if parsed else None
    location_list = parsed.get("location_list") if parsed else None
    if not location:
        location = _select_ner_location(ner)
    start_year = parsed.get("start_year") if parsed else None
    end_year = parsed.get("end_year") if parsed else None
    end_year_text = parsed.get("end_year_text") if parsed else None
    year = parsed.get("year") if parsed else None
    if year is None and start_year is None:
        year = _extract_year_from_dates(ner)
    degree_text = parsed.get("degree_text") if parsed else None
    degree_level = None
    field = None
    if relation == "studied_at":
        degree_level, field = normalize_degree(degree_text)
        if year is None:
            year = start_year
    year_value = year or start_year
    year_bin_value = year_bin(year_value) if year_value else None
    candidate_list: List[LineCandidate] = []

    ner_org_text = _select_ner_org(ner)

    for idx, org_raw in enumerate(base_orgs):
        resolved_location = location
        if relation == "studied_at" and isinstance(location_list, list):
            if idx < len(location_list):
                resolved_location = location_list[idx]
        org_canon = canon_org(org_raw)
        location_canon = canon_location(resolved_location)
        org_type = classify_org(org_canon) or "unknown"
        candidate = LineCandidate(
            relation=relation,
            section=section,
            text_span=line,
            role=parsed.get("role") if parsed else None,
            course=parsed.get("course") if parsed else None,
            org_raw=org_raw,
            org_canon=org_canon,
            org_type=org_type,
            location_raw=resolved_location,
            location_canon=location_canon,
            start_year=start_year,
            end_year=end_year,
            year=year,
            year_bin=year_bin_value,
            degree_text=degree_text,
            degree_level=degree_level,
            field=field,
            sources=list(dict.fromkeys(sources)),
            meta={},
        )
        if end_year_text:
            candidate.meta["end_year_text"] = end_year_text
        candidate.meta["sources"] = candidate.sources or ["ner"]

        org_from_parser = org_raw is not None and (parsed is not None)
        if org_from_parser and ner_org_text and org_raw:
            similarity = fuzz.WRatio(org_raw, ner_org_text)
            if similarity < 90:
                candidate.meta["alt_org"] = ner_org_text
        candidate.meta["score"] = _score_candidate(
            relation,
            section,
            org_raw,
            org_from_parser,
            ner_org_text,
            org_type,
        )
        candidate_list.append(candidate)

    return candidate_list
