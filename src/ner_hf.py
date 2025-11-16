"""Hugging Face NER helpers used by the notebook pipeline."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from transformers import pipeline

from fusion import Candidate
from normalize import normalize_name
from rules import (
    canon_location_with_flag,
    canon_org_with_flag,
    classify_org,
    normalize_degree,
    year_bin,
)


def load_pipelines() -> Dict[str, Any]:
    """Load both HF NER pipelines with deterministic settings."""

    common_args = {
        "aggregation_strategy": "simple",
        "grouped_entities": True,
    }
    models = {
        "bert": "dslim/bert-base-NER",
        "xlm": "Davlan/xlm-roberta-base-ner-hrl",
    }
    return {name: pipeline("ner", model=ckpt, **common_args) for name, ckpt in models.items()}


def _standardize(entity: Dict[str, Any]) -> Dict[str, Any]:
    text = (entity.get("word") or entity.get("entity_group") or "").replace("##", "").strip()
    return {
        "text": text,
        "label": entity.get("entity_group") or entity.get("entity"),
        "score": float(entity.get("score", 0.0)),
        "start": int(entity.get("start", -1)),
        "end": int(entity.get("end", -1)),
    }


def run_ner(text: str, pipe: Any) -> List[Dict[str, Any]]:
    """Run the provided NER pipeline and standardize the output."""

    if not text.strip():
        return []
    raw = pipe(text)
    return [_standardize(ent) for ent in raw]


def _overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return not (a["end"] <= b["start"] or b["end"] <= a["start"])


def merge_entities(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge two entity lists keeping the longest/highest-scoring overlaps."""

    merged: List[Dict[str, Any]] = []
    candidates = sorted(a + b, key=lambda e: (e["start"], -e["end"]))

    for ent in candidates:
        replaced = False
        for idx, existing in enumerate(merged):
            if _overlap(ent, existing):
                len_ent = ent["end"] - ent["start"]
                len_ex = existing["end"] - existing["start"]
                if len_ent > len_ex or (len_ent == len_ex and ent["score"] > existing["score"]):
                    merged[idx] = ent
                replaced = True
                break
        if not replaced:
            merged.append(ent)
    return merged


_SECTION_REL = {
    "corporate_experience": "worked_at",
    "academic_background": "studied_at",
    "academic_experience": "worked_at",
}

_LOC_LABELS = {"LOC", "GPE", "LOCATION"}
_YEAR_RE = re.compile(r"(19|20)\d{2}")
_PIPE_TO_SOURCE = {"bert": "ner_en", "xlm": "ner_xlm"}


def _closest_location(org: Dict[str, Any], locations: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not locations:
        return None
    org_center = (org.get("start", 0) + org.get("end", 0)) / 2
    best = None
    best_dist = float("inf")
    for loc in locations:
        loc_center = (loc.get("start", 0) + loc.get("end", 0)) / 2
        dist = abs(org_center - loc_center)
        if dist < best_dist:
            best = loc
            best_dist = dist
    return best


def _year_from_text(text: str) -> int | None:
    match = _YEAR_RE.search(text)
    if match:
        return int(match.group(0))
    return None


def build_ner_candidates(
    section: str, line: str, line_idx: int, pipes: Dict[str, Any]
) -> List[Candidate]:
    """Run both NER models on a line and convert ORG hits into candidates."""

    candidates: List[Candidate] = []
    if not line or not line.strip():
        return candidates
    for pipe_name, pipe in pipes.items():
        ents = run_ner(line, pipe)
        orgs = [ent for ent in ents if "ORG" in (ent.get("label") or "").upper()]
        locs = [ent for ent in ents if (ent.get("label") or "").upper() in _LOC_LABELS]
        source = _PIPE_TO_SOURCE.get(pipe_name, "ner_en")
        degree_level, field = normalize_degree(line)
        year_val = _year_from_text(line)
        for org in orgs:
            org_raw = org.get("text")
            if not org_raw:
                continue
            org_norm = normalize_name(org_raw)
            org_canon, alias_hit = canon_org_with_flag(org_raw)
            loc_hit = _closest_location(org, locs)
            loc_raw = loc_hit.get("text") if loc_hit else None
            loc_norm = normalize_name(loc_raw) if loc_raw else None
            loc_canon = None
            if loc_raw:
                loc_canon, _ = canon_location_with_flag(loc_raw)
            candidate = Candidate(
                relation=_SECTION_REL.get(section, "unknown"),
                source=source,  # type: ignore[arg-type]
                section=section,
                line_idx=line_idx,
                text_span=line,
                org_raw=org_raw,
                org_norm=org_norm,
                org_canon=org_canon,
                org_type_guess=classify_org(org_canon) or "unknown",
                location_raw=loc_raw,
                location_norm=loc_norm,
                location_canon=loc_canon,
                degree_text=line if degree_level else None,
                degree_level=degree_level,
                field=field,
                year=year_val,
                year_bin=year_bin(year_val),
            )
            candidate.meta = {
                "sources": [source],
                "alias_hit": alias_hit,
                "ner_score": org.get("score"),
                "ner_label": org.get("label"),
            }
            candidates.append(candidate)
    return candidates
