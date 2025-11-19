"""Line-level fusion between parser output and NER evidence."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Literal, Optional

from rapidfuzz import fuzz

from rules import (
    SECTION_STOP_ORGS,
    SECTION_PRIOR,
    canon_location,
    canon_org,
    classify_org,
    plausible_org,
    soft_name_label,
    strip_preps,
    normalize_degree,
    ORG_HARD_NEGATIVE,
    ORG_HARD_POSITIVE,
    UNIV_CUES,
    COMP_CUES,
    GOV_CUES,
    year_bin,
)
from pattern_extractor import org_candidates_from_text

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

_ORG_KEYWORDS = tuple(
    sorted({kw.lower() for kw in (*UNIV_CUES, *COMP_CUES, *GOV_CUES) if kw})
)
_ORG_KEYWORD_RE = (
    re.compile(r"\b(" + "|".join(re.escape(k) for k in _ORG_KEYWORDS) + r")\b", re.I)
    if _ORG_KEYWORDS
    else None
)
_WINDOW_STOP_CHARS = set(",;:.()[]{}|/\n\r\t•") | {"-", "–", "—"}
_CONNECTOR_TOKENS = ("at", "with", "by", "from", "for")
_ROLE_STOPWORDS = {
    "senior",
    "advisor",
    "adviser",
    "associate",
    "assistant",
    "director",
    "managing",
    "member",
    "chair",
    "professor",
    "adjunct",
    "visiting",
    "deputy",
    "chief",
    "head",
    "partner",
    "consultant",
    "manager",
    "executive",
    "president",
    "vice",
    "chairman",
    "chairwoman",
    "board",
    "founder",
    "cofounder",
    "co-founder",
    "cfo",
    "ceo",
    "cio",
    "cto",
    "cmo",
    "svp",
    "vp",
    "lecturer",
}
_CONNECTOR_CUTS = [f" {tok} " for tok in _CONNECTOR_TOKENS]
_ACRONYM_CANDIDATE_RE = re.compile(r"\b[A-Z][A-Z0-9&]{2,}\b")


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


def _has_connector_before(text: str, start: int) -> bool:
    window = text[max(0, start - 30) : start].lower().rstrip()
    return any(
        window.endswith(f" {tok}") or window.endswith(tok)
        for tok in _CONNECTOR_TOKENS
    )


def _clean_candidate_snippet(snippet: str, cue_offset: int) -> str:
    """Trim leading role words and connectors from an extracted snippet."""

    if not snippet:
        return ""
    before = snippet[:cue_offset]
    lower_before = before.lower()
    cut_index = -1
    for connector in _CONNECTOR_CUTS:
        idx = lower_before.rfind(connector)
        if idx != -1:
            cut_index = idx + len(connector)
            break
    if cut_index != -1:
        snippet = snippet[cut_index:]
    snippet = snippet.strip(" ,;:()[]{}-–—•")
    snippet = re.sub(r"^(?:and|&|the)\s+", "", snippet, flags=re.IGNORECASE)
    tokens = snippet.split()
    while tokens and tokens[0].lower().strip(".,") in _ROLE_STOPWORDS:
        tokens.pop(0)
    snippet = " ".join(tokens)
    snippet = strip_preps(snippet)
    snippet = re.sub(r"\s+", " ", snippet).strip(" ,;:()[]{}-–—•")
    lower_snip = snippet.lower()
    for stop in _ROLE_STOPWORDS:
        marker = f" and {stop}"
        idx = lower_snip.find(marker)
        if idx != -1:
            snippet = snippet[:idx]
            snippet = snippet.strip(" ,;:()[]{}-–—•")
            break
    snippet = re.sub(r"\s+(?:in|at)\s+[A-Z][A-Za-z\- ]{0,40}$", "", snippet)
    return snippet


def _extract_window(text: str, start: int, end: int) -> tuple[str, int]:
    left = start
    while left > 0 and text[left - 1] not in _WINDOW_STOP_CHARS:
        left -= 1
    right = end
    text_len = len(text)
    while right < text_len and text[right] not in _WINDOW_STOP_CHARS:
        right += 1
    snippet = text[left:right]
    snippet = snippet.strip(" ,;:()[]{}-–—•")
    return snippet, left


def _heuristic_org_candidates(line: str) -> List[str]:
    """Lightweight rule-based extractor backing off the HF NER results."""

    if not line:
        return []
    matches: List[re.Match[str]] = []
    if _ORG_KEYWORD_RE:
        matches.extend(list(_ORG_KEYWORD_RE.finditer(line)))
    matches.extend(
        match
        for match in _ACRONYM_CANDIDATE_RE.finditer(line)
        if _has_connector_before(line, match.start())
    )
    if not matches:
        return []
    matches.sort(key=lambda m: m.start())
    candidates: List[str] = []
    for match in matches:
        snippet, left_idx = _extract_window(line, match.start(), match.end())
        cue_offset = match.start() - left_idx
        candidate = _clean_candidate_snippet(snippet, cue_offset)
        needle = match.group().strip()
        if needle and needle.lower() not in (candidate or "").lower():
            candidate = needle
        if not candidate:
            continue
        if not plausible_org(candidate):
            continue
        candidates.append(candidate)
    return list(dict.fromkeys(candidates))


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
    relation: Relation,
    parsed: Optional[Dict[str, Any]],
    ner: Dict[str, Any],
    line: str,
) -> tuple[List[Optional[str]], bool]:
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
        return parsed_orgs, False

    heuristic_orgs = _heuristic_org_candidates(line)
    if heuristic_orgs:
        return heuristic_orgs, True

    pattern_orgs = org_candidates_from_text(line)
    if pattern_orgs:
        return pattern_orgs

    ner_org = _select_ner_org(ner)
    if ner_org:
        return [ner_org], False
    return [None], False


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

    base_orgs, used_heuristics = _build_base_orgs(relation, parsed, ner, line)
    if used_heuristics:
        sources.append("heuristic")
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
        org_canon = canon_org(strip_preps(org_raw))
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


def collect_org_evidence(records) -> Dict[str, Dict[str, Any]]:
    """Aggregate soft evidence for each canonical organisation."""

    evd = defaultdict(
        lambda: {
            "mentions": 0,
            "by_relation": Counter(),
            "by_section_prior": Counter(),
            "by_name_label": Counter(),
        }
    )

    def _add(org_raw: str, relation: str, section: str):
        if not org_raw:
            return
        if not plausible_org(org_raw):
            return
        org = canon_org(strip_preps(org_raw))
        lbl = soft_name_label(org)
        prior = SECTION_PRIOR.get(section, None)

        entry = evd[org]
        entry["mentions"] += 1
        entry["by_relation"][relation] += 1
        if prior:
            entry["by_section_prior"][prior] += 1
        entry["by_name_label"][lbl] += 1

    for rec in records:
        for study in rec.get("studies", []):
            _add(
                study.get("university_canon") or study.get("university"),
                "studied_at",
                study.get("source_section", ""),
            )
        for work in rec.get("work", []):
            _add(
                work.get("company_canon")
                or work.get("company")
                or work.get("university_canon")
                or work.get("university"),
                "worked_at",
                work.get("source_section", ""),
            )
        for course in rec.get("courses", []):
            _add(
                course.get("center_canon")
                or course.get("center")
                or course.get("university")
                or course.get("company"),
                "teaches",
                course.get("source_section", ""),
            )

    return evd


def resolve_final_org_types(org_evd: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Resolve the final organisation type via weighted voting."""

    final: Dict[str, str] = {}
    for org, evidence in org_evd.items():
        low = org.lower()
        if low in ORG_HARD_POSITIVE:
            final[org] = "university"
            continue
        if low in ORG_HARD_NEGATIVE:
            final[org] = ORG_HARD_NEGATIVE[low]
            continue

        score = Counter()
        score["university"] += 3 * evidence["by_relation"].get("studied_at", 0)
        score["university"] += 2 * evidence["by_relation"].get("teaches", 0)
        score["company"] += 2 * evidence["by_relation"].get("worked_at", 0)

        for lbl, cnt in evidence["by_name_label"].items():
            if lbl in ("university", "company", "government"):
                score[lbl] += 2 * cnt

        for prior, cnt in evidence["by_section_prior"].items():
            if prior in ("university", "company"):
                score[prior] += cnt

        if score:
            winner = score.most_common(1)[0][0]
        else:
            winner = "unknown"

        name_label = soft_name_label(org)
        if winner == "university" and name_label in ("company", "government"):
            winner = name_label

        final[org] = winner

    return final
