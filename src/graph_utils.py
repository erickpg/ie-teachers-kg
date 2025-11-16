"""NetworkX helpers to assemble and export the knowledge graph."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Tuple

import networkx as nx

_NODE_TYPES = {
    "Professor": "professor",
    "University": "university",
    "Company": "company",
    "Course": "course",
    "Degree": "degree",
    "Location": "location",
}

_ALLOWED_SCALARS = (str, int, float, bool)


def new_graph() -> nx.Graph:
    """Create a graph with helpful defaults."""

    graph = nx.Graph()
    graph.graph["name"] = "IE Teachers KG"
    return graph


def _node_key(node_type: str, name: str) -> Tuple[str, str]:
    return (node_type, name)


def _sanitize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, _ALLOWED_SCALARS):
        return value
    if isinstance(value, (list, tuple, set)):
        return " | ".join(str(item) for item in value)
    return str(value)


def _sanitize_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in attrs.items():
        sanitized = _sanitize_value(value)
        if sanitized is not None:
            cleaned[key] = sanitized
    return cleaned


def _add_node(G: nx.Graph, node_type: str, name: str, **attrs) -> None:
    node_id = _node_key(node_type, name)
    existing = dict(G.nodes.get(node_id, {}))
    existing.setdefault("type", node_type)
    existing.update(_sanitize_attrs(attrs))
    G.add_node(node_id, **existing)


def add_professor(G: nx.Graph, prof_id: str, **attrs) -> None:
    _add_node(G, "Professor", prof_id, **attrs)


def add_university(G: nx.Graph, name: str, location: str | None = None, **attrs) -> None:
    if location:
        attrs.setdefault("location", location)
    _add_node(G, "University", name, **attrs)


def add_company(G: nx.Graph, name: str, location: str | None = None, **attrs) -> None:
    if location:
        attrs.setdefault("location", location)
    _add_node(G, "Company", name, **attrs)


def add_course(G: nx.Graph, name: str, **attrs) -> None:
    _add_node(G, "Course", name, **attrs)


def add_degree(G: nx.Graph, level: str, field: str | None = None, **attrs) -> None:
    attrs.setdefault("level", level)
    if field:
        attrs.setdefault("field", field)
    label = f"{level} — {field}" if field else level
    _add_node(G, "Degree", label, **attrs)


def add_location(G: nx.Graph, name: str, **attrs) -> None:
    _add_node(G, "Location", name, **attrs)


def _add_edge(G: nx.Graph, source: Tuple[str, str], target: Tuple[str, str], **attrs) -> None:
    G.add_edge(source, target, **_sanitize_attrs(attrs))


def link_studied_at(G: nx.Graph, prof_id: str, univ_name: str, **attrs) -> None:
    source = _node_key("Professor", prof_id)
    target = _node_key("University", univ_name)
    _add_edge(G, source, target, relation="studied_at", **attrs)


def link_worked_at(G: nx.Graph, prof_id: str, comp_name: str, **attrs) -> None:
    source = _node_key("Professor", prof_id)
    target = _node_key("Company", comp_name)
    _add_edge(G, source, target, relation="worked_at", **attrs)


def link_teaches(G: nx.Graph, prof_id: str, course_name: str, **attrs) -> None:
    source = _node_key("Professor", prof_id)
    target = _node_key("Course", course_name)
    _add_edge(G, source, target, relation="teaches", **attrs)


def link_located_in(G: nx.Graph, org_name: str, location: str, org_type: str) -> None:
    add_location(G, location)
    org_node = _node_key(org_type.title(), org_name)
    loc_node = _node_key("Location", location)
    _add_edge(G, org_node, loc_node, relation="located_in", org_type=org_type)


def _node_label(node: Tuple[str, str]) -> str:
    if isinstance(node, tuple) and len(node) == 2:
        return node[1]
    return str(node)


def save_graph(G: nx.Graph, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    nodes_path = os.path.join(out_dir, "nodes.csv")
    edges_path = os.path.join(out_dir, "edges.csv")
    gexf_path = os.path.join(out_dir, "graph.gexf")

    with open(nodes_path, "w", newline="", encoding="utf-8") as f_nodes:
        writer = csv.writer(f_nodes)
        writer.writerow(["id", "label", "type"])
        for node, data in G.nodes(data=True):
            writer.writerow([str(node), _node_label(node), data.get("type", "unknown")])

    with open(edges_path, "w", newline="", encoding="utf-8") as f_edges:
        writer = csv.writer(f_edges)
        writer.writerow(["source", "target", "relation", "attributes"])
        for source, target, data in G.edges(data=True):
            attr_json = json.dumps(data, ensure_ascii=False)
            writer.writerow([str(source), str(target), data.get("relation", ""), attr_json])

    nx.write_gexf(G, gexf_path)


def top_k_by_degree(G: nx.Graph, node_type: str, k: int = 10) -> List[Tuple[str, int]]:
    filtered = []
    for node, degree in G.degree():
        data = G.nodes[node]
        if data.get("type") == node_type:
            filtered.append((_node_label(node), degree))
    return sorted(filtered, key=lambda item: item[1], reverse=True)[:k]
