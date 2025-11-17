"""Hugging Face NER helpers used by the notebook pipeline."""

from __future__ import annotations

from typing import Any, Dict, List

from transformers import pipeline

from .rules import SECTION_STOP_ORGS


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


def ner_on_line(line: str, pipes: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Run NER on a single line returning grouped entities."""

    results = {"orgs": [], "locs": [], "dates": []}
    if not line.strip():
        return results

    orgs: List[Dict[str, Any]] = []
    locs: List[Dict[str, Any]] = []
    dates: List[Dict[str, Any]] = []

    for name, pipe in (pipes or {}).items():
        for ent in run_ner(line, pipe):
            label = (ent.get("label") or "").upper()
            enriched = {**ent, "source": name}
            if label in {"ORG", "ORGANIZATION"}:
                if enriched["text"].lower() in SECTION_STOP_ORGS:
                    continue
                orgs = merge_entities(orgs, [enriched])
            elif label in {"LOC", "GPE"}:
                locs = merge_entities(locs, [enriched])
            elif label == "DATE":
                dates = merge_entities(dates, [enriched])

    results["orgs"] = orgs
    results["locs"] = locs
    results["dates"] = dates
    return results
