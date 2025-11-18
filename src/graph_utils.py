"""NetworkX helpers to assemble and export the knowledge graph."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import networkx as nx


_NODE_TYPES = {
    "Professor": "professor",
    "University": "university",
    "Company": "company",
    "Course": "course",
    "Degree": "degree",
    "Location": "location",
}


def new_graph() -> nx.Graph:
    """Create a graph with helpful defaults."""

    G = nx.Graph()
    G.graph["name"] = "IE Teachers KG"
    return G


def _node_id(node_type: str, name: str) -> tuple[str, str]:
    return (node_type, name)


def _resolve_node(node_type: str, value: Union[str, tuple[str, str]]) -> tuple[str, str]:
    if isinstance(value, tuple):
        return value
    return _node_id(node_type, value)


_ALLOWED_ATTR_TYPES = (str, int, float, bool)


def _sanitize_value(value: Any) -> Any:
    """Return a value that is safe to serialize in graph exports."""

    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, _ALLOWED_ATTR_TYPES):
        return value
    if isinstance(value, (list, tuple, set)):
        parts = [str(v) for v in value if v is not None]
        return " | ".join(parts) if parts else None
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _sanitize_attrs(attrs: Dict[str, object] | None) -> Dict[str, object]:
    clean: Dict[str, object] = {}
    for key, value in (attrs or {}).items():
        safe_value = _sanitize_value(value)
        if safe_value is not None:
            clean[str(key)] = safe_value
    return clean


def _add_node(G: nx.Graph, node_type: str, name: str, **attrs) -> tuple[str, str]:
    node_key = _node_id(node_type, name)
    data = G.nodes.get(node_key, {})
    data.setdefault("type", node_type)
    data.update(_sanitize_attrs(attrs))
    G.add_node(node_key, **data)
    return node_key


def add_professor(G: nx.Graph, prof_id: str, **attrs) -> tuple[str, str]:
    return _add_node(G, "Professor", prof_id, **attrs)


def add_university(
    G: nx.Graph, name: str, location: str | None = None, **attrs
) -> tuple[str, str]:
    if location:
        attrs.setdefault("location", location)
    return _add_node(G, "University", name, **attrs)


def add_company(G: nx.Graph, name: str, location: str | None = None, **attrs) -> tuple[str, str]:
    if location:
        attrs.setdefault("location", location)
    return _add_node(G, "Company", name, **attrs)


def add_course(G: nx.Graph, name: str, **attrs) -> tuple[str, str]:
    return _add_node(G, "Course", name, **attrs)


def add_degree(G: nx.Graph, level: str, field: str | None = None, **attrs) -> tuple[str, str]:
    if field:
        attrs.setdefault("field", field)
    label = f"{level}-{field}" if field else level
    return _add_node(G, "Degree", label, **attrs)


def add_location(G: nx.Graph, name: str, **attrs) -> tuple[str, str]:
    return _add_node(G, "Location", name, **attrs)


def _edge(G: nx.Graph, a: tuple[str, str], b: tuple[str, str], **attrs) -> None:
    G.add_edge(a, b, **_sanitize_attrs(attrs))


def link_studied_at(
    G: nx.Graph,
    prof_id: Union[str, tuple[str, str]],
    univ_name: Union[str, tuple[str, str]],
    **attrs,
) -> None:
    prof_node = _resolve_node("Professor", prof_id)
    univ_node = _resolve_node("University", univ_name)
    _edge(G, prof_node, univ_node, relation="studied_at", **attrs)


def link_worked_at(
    G: nx.Graph,
    prof_id: Union[str, tuple[str, str]],
    comp_name: Union[str, tuple[str, str]],
    **attrs,
) -> None:
    prof_node = _resolve_node("Professor", prof_id)
    comp_node = _resolve_node("Company", comp_name)
    _edge(G, prof_node, comp_node, relation="worked_at", **attrs)


def link_teaches(
    G: nx.Graph,
    prof_id: Union[str, tuple[str, str]],
    course_name: Union[str, tuple[str, str]],
    **attrs,
) -> None:
    prof_node = _resolve_node("Professor", prof_id)
    course_node = _resolve_node("Course", course_name)
    _edge(G, prof_node, course_node, relation="teaches", **attrs)


def link_located_in(
    G: nx.Graph,
    org_name: Union[str, tuple[str, str]],
    location: Union[str, tuple[str, str]],
    org_type: str,
) -> None:
    org_node = org_name
    if isinstance(org_name, str):
        node_type = "University" if org_type == "university" else "Company"
        org_node = _resolve_node(node_type, org_name)
    loc_node = add_location(G, location) if isinstance(location, str) else location
    _edge(G, org_node, loc_node, relation="located_in", org_type=org_type)


def _to_stringed_graph(G: nx.Graph) -> nx.Graph:
    """Return a copy of the graph with string ids and sanitized attributes."""

    H = nx.Graph()
    for node, attrs in G.nodes(data=True):
        H.add_node(str(node), **_sanitize_attrs(attrs))
    for u, v, attrs in G.edges(data=True):
        H.add_edge(str(u), str(v), **_sanitize_attrs(attrs))
    return H


def save_graph(G: nx.Graph, out_dir: str | os.PathLike[str]) -> Dict[str, str]:
    """Persist graph artifacts (CSV + GEXF/GraphML) and return their paths."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    nodes_path = out_path / "nodes.csv"
    edges_path = out_path / "edges.csv"
    gexf_path = out_path / "graph.gexf"
    graphml_path = out_path / "graph.graphml"

    H = _to_stringed_graph(G)

    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "type"])
        for node, data in H.nodes(data=True):
            writer.writerow([node, data.get("type", "unknown")])

    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["source", "target", "relation", "attributes"]
        writer.writerow(header)
        for u, v, data in H.edges(data=True):
            writer.writerow([u, v, data.get("relation", ""), _sanitize_attrs(data)])

    paths: Dict[str, str] = {
        "nodes": str(nodes_path),
        "edges": str(edges_path),
    }

    try:
        nx.write_gexf(H, gexf_path)
        paths["gexf"] = str(gexf_path)
    except Exception as exc:  # pragma: no cover - defensive
        paths["gexf_error"] = repr(exc)

    try:
        nx.write_graphml(H, graphml_path)
        paths["graphml"] = str(graphml_path)
    except Exception as exc:  # pragma: no cover - defensive
        paths["graphml_error"] = repr(exc)

    return paths


def top_k_by_degree(G: nx.Graph, node_type: str, k: int = 10) -> List[Tuple[str, int]]:
    filtered = [(n, d) for n, d in G.degree() if G.nodes[n].get("type") == node_type]
    return sorted(filtered, key=lambda x: x[1], reverse=True)[:k]
