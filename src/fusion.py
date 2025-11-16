"""Light-weight alignment and scoring between pattern and NER candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from rapidfuzz import fuzz

Relation = Literal["studied_at", "worked_at", "teaches", "unknown"]
OrgType = Literal["university", "company", "unknown"]
Source = Literal["pattern", "ner_en", "ner_xlm"]

FUSION_WEIGHTS: Dict[str, float] = {
    "bias_pattern": 0.55,
    "bias_ner_en": 0.50,
    "bias_ner_xlm": 0.50,
    "agreement_bonus": 0.20,
    "section_prior_match": 0.10,
    "alias_canon_bonus": 0.05,
    "org_type_bonus": 0.05,
    "penalty_conflict_type": -0.10,
    "penalty_garbage": -0.15,
}

_SECTION_PRIORS: Dict[str, Relation] = {
    "corporate_experience": "worked_at",
    "academic_background": "studied_at",
}


@dataclass
class Candidate:
    """Evidence fragment extracted from either a pattern or NER source."""

    relation: Relation
    source: Source
    section: str
    line_idx: int
    text_span: str
    org_raw: Optional[str] = None
    org_norm: Optional[str] = None
    org_canon: Optional[str] = None
    org_type_guess: OrgType = "unknown"
    location_raw: Optional[str] = None
    location_norm: Optional[str] = None
    location_canon: Optional[str] = None
    role: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    end_year_text: Optional[str] = None
    degree_text: Optional[str] = None
    degree_level: Optional[str] = None
    field: Optional[str] = None
    year: Optional[int] = None
    year_bin: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scored:
    """Candidate enriched with a scalar score for downstream selection."""

    key: str
    score: float
    reasons: List[str]
    merged: Candidate


def _org_string(cand: Candidate) -> str:
    return cand.org_canon or cand.org_norm or cand.org_raw or ""


def _merge_values(primary: Optional[Any], secondary: Optional[Any]) -> Optional[Any]:
    if primary not in (None, ""):
        return primary
    return secondary


def _merge_candidate(base: Candidate, incoming: Candidate) -> Candidate:
    merged = Candidate(
        relation=base.relation or incoming.relation,
        source=base.source,
        section=base.section or incoming.section,
        line_idx=min(base.line_idx, incoming.line_idx),
        text_span=max(base.text_span, incoming.text_span, key=len),
        org_raw=_merge_values(base.org_raw, incoming.org_raw),
        org_norm=_merge_values(base.org_norm, incoming.org_norm),
        org_canon=_merge_values(base.org_canon, incoming.org_canon),
        org_type_guess=base.org_type_guess if base.org_type_guess != "unknown" else incoming.org_type_guess,
        location_raw=_merge_values(base.location_raw, incoming.location_raw),
        location_norm=_merge_values(base.location_norm, incoming.location_norm),
        location_canon=_merge_values(base.location_canon, incoming.location_canon),
        role=_merge_values(base.role, incoming.role),
        start_year=_merge_values(base.start_year, incoming.start_year),
        end_year=_merge_values(base.end_year, incoming.end_year),
        end_year_text=_merge_values(base.end_year_text, incoming.end_year_text),
        degree_text=_merge_values(base.degree_text, incoming.degree_text),
        degree_level=_merge_values(base.degree_level, incoming.degree_level),
        field=_merge_values(base.field, incoming.field),
        year=_merge_values(base.year, incoming.year),
        year_bin=_merge_values(base.year_bin, incoming.year_bin),
        meta={**incoming.meta, **base.meta},
    )
    sources = list(dict.fromkeys(incoming.meta.get("sources", []) + base.meta.get("sources", [])))
    merged.meta["sources"] = sources or [base.source]
    if incoming.meta.get("alias_hit") or base.meta.get("alias_hit"):
        merged.meta["alias_hit"] = True
    if incoming.meta.get("agreement") or base.meta.get("agreement"):
        merged.meta["agreement"] = True
    return merged


def align_candidates(pattern_cands: List[Candidate], ner_cands: List[Candidate]) -> List[Candidate]:
    """Align candidates when their organisations overlap in nearby lines."""

    aligned: List[Candidate] = []
    used: set[int] = set()
    for pattern in pattern_cands:
        merged = Candidate(**{**pattern.__dict__, "meta": dict(pattern.meta)})
        merged.meta.setdefault("sources", [pattern.source])
        for idx, ner in enumerate(ner_cands):
            if idx in used or pattern.section != ner.section:
                continue
            if abs(pattern.line_idx - ner.line_idx) > 1:
                continue
            org_a = _org_string(pattern)
            org_b = _org_string(ner)
            if not org_a or not org_b:
                continue
            similarity = fuzz.WRatio(org_a, org_b)
            if similarity >= 90:
                merged = _merge_candidate(merged, ner)
                merged.meta["agreement"] = True
                used.add(idx)
        aligned.append(merged)
    for idx, candidate in enumerate(ner_cands):
        if idx in used:
            continue
        carry = Candidate(**{**candidate.__dict__, "meta": dict(candidate.meta)})
        carry.meta.setdefault("sources", [candidate.source])
        aligned.append(carry)
    return aligned


def make_key(cand: Candidate) -> str:
    """Build a stable key per relation for deduplication."""

    if cand.relation == "worked_at":
        return f"worked_at|{_org_string(cand)}"
    if cand.relation == "studied_at":
        degree_key = cand.degree_level or cand.degree_text or "unknown"
        year_key = cand.year or cand.year_bin or "any"
        return f"studied_at|{_org_string(cand)}|{degree_key}|{year_key}"
    if cand.relation == "teaches":
        return f"teaches|{cand.role or cand.text_span}"
    return f"unknown|{cand.section}|{cand.line_idx}"


def _expected_type(relation: Relation) -> Optional[OrgType]:
    if relation == "worked_at":
        return "company"
    if relation == "studied_at":
        return "university"
    return None


def score_candidate(candidate: Candidate, weights: Optional[Dict[str, float]] = None) -> Scored:
    """Score a candidate using the linear model and track textual reasons."""

    weights = weights or FUSION_WEIGHTS
    score = 0.0
    reasons: List[str] = []
    sources = candidate.meta.get("sources") or [candidate.source]
    for src in dict.fromkeys(sources):
        weight_key = f"bias_{src}"
        if weight_key in weights:
            score += weights[weight_key]
            reasons.append(f"{weight_key}+{weights[weight_key]:.2f}")
    if candidate.meta.get("agreement") and "agreement_bonus" in weights:
        score += weights["agreement_bonus"]
        reasons.append(f"agreement+{weights['agreement_bonus']:.2f}")
    prior = _SECTION_PRIORS.get(candidate.section)
    if prior and prior == candidate.relation and "section_prior_match" in weights:
        score += weights["section_prior_match"]
        reasons.append(f"section_prior+{weights['section_prior_match']:.2f}")
    if candidate.meta.get("alias_hit") and "alias_canon_bonus" in weights:
        score += weights["alias_canon_bonus"]
        reasons.append(f"alias_bonus+{weights['alias_canon_bonus']:.2f}")
    expected = _expected_type(candidate.relation)
    if expected and candidate.org_type_guess != "unknown":
        if candidate.org_type_guess == expected and "org_type_bonus" in weights:
            score += weights["org_type_bonus"]
            reasons.append(f"org_type+{weights['org_type_bonus']:.2f}")
        elif candidate.org_type_guess != expected and "penalty_conflict_type" in weights:
            score += weights["penalty_conflict_type"]
            reasons.append(f"type_penalty{weights['penalty_conflict_type']:+.2f}")
    org_label = _org_string(candidate)
    if (not org_label or len(org_label) <= 2) and "penalty_garbage" in weights:
        score += weights["penalty_garbage"]
        reasons.append(f"garbage{weights['penalty_garbage']:+.2f}")
    return Scored(key=make_key(candidate), score=score, reasons=reasons, merged=candidate)


def select_entities(scored: List[Scored], threshold: float = 0.65) -> List[Scored]:
    """Keep the best-scoring candidate per key above the threshold."""

    best: Dict[str, Scored] = {}
    for item in scored:
        existing = best.get(item.key)
        if existing is None or item.score > existing.score:
            best[item.key] = item
    selected = [entry for entry in best.values() if entry.score >= threshold]
    return sorted(selected, key=lambda s: s.score, reverse=True)


def fusion_pipeline(
    sections: Dict[str, str],
    build_pattern,
    build_ner,
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.65,
) -> Dict[str, List[Scored]]:
    """Helper to run pattern + NER candidates through alignment and scoring."""

    weights = weights or FUSION_WEIGHTS
    fused: Dict[str, List[Scored]] = {"worked_at": [], "studied_at": [], "teaches": [], "unknown": []}
    for section, text in sections.items():
        pattern_cands = build_pattern(section, text)
        ner_cands = build_ner(section, text)
        aligned = align_candidates(pattern_cands, ner_cands)
        scored = [score_candidate(c, weights) for c in aligned]
        accepted = select_entities(scored, threshold)
        for item in accepted:
            fused.setdefault(item.merged.relation, []).append(item)
    return fused
