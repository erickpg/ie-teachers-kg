"""NetworkX helpers to assemble and export the knowledge graph."""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

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


def _add_node(G: nx.Graph, node_id: str, node_type: str, **attrs) -> None:
    data = G.nodes.get(node_id, {})
    data.setdefault("type", node_type)
    data.update(attrs)
    G.add_node(node_id, **data)


def add_professor(G: nx.Graph, prof_id: str, **attrs) -> None:
    _add_node(G, prof_id, "Professor", **attrs)


def add_university(G: nx.Graph, name: str, location: str | None = None, **attrs) -> None:
    if location:
        attrs.setdefault("location", location)
    _add_node(G, name, "University", **attrs)


def add_company(G: nx.Graph, name: str, location: str | None = None, **attrs) -> None:
    if location:
        attrs.setdefault("location", location)
    _add_node(G, name, "Company", **attrs)


def add_course(G: nx.Graph, name: str, **attrs) -> None:
    _add_node(G, name, "Course", **attrs)


def add_degree(G: nx.Graph, level: str, field: str | None = None, **attrs) -> None:
    if field:
        attrs.setdefault("field", field)
    _add_node(G, f"{level}-{field}" if field else level, "Degree", **attrs)


def add_location(G: nx.Graph, name: str, **attrs) -> None:
    _add_node(G, name, "Location", **attrs)


def link_studied_at(G: nx.Graph, prof_id: str, univ_name: str, **attrs) -> None:
    G.add_edge(prof_id, univ_name, relation="studied_at", **attrs)


def link_worked_at(G: nx.Graph, prof_id: str, comp_name: str, **attrs) -> None:
    G.add_edge(prof_id, comp_name, relation="worked_at", **attrs)


def link_teaches(G: nx.Graph, prof_id: str, course_name: str, **attrs) -> None:
    G.add_edge(prof_id, course_name, relation="teaches", **attrs)


def link_located_in(G: nx.Graph, org_name: str, location: str, org_type: str) -> None:
    add_location(G, location)
    G.add_edge(org_name, location, relation="located_in", org_type=org_type)


def save_graph(G: nx.Graph, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    nodes_path = os.path.join(out_dir, "nodes.csv")
    edges_path = os.path.join(out_dir, "edges.csv")
    gexf_path = os.path.join(out_dir, "graph.gexf")

    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "type"])
        for node, data in G.nodes(data=True):
            writer.writerow([node, data.get("type", "unknown")])

    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["source", "target", "relation", "attributes"]
        writer.writerow(header)
        for u, v, data in G.edges(data=True):
            writer.writerow([u, v, data.get("relation", ""), data])

    nx.write_gexf(G, gexf_path)


def top_k_by_degree(G: nx.Graph, node_type: str, k: int = 10) -> List[Tuple[str, int]]:
    filtered = [(n, d) for n, d in G.degree() if G.nodes[n].get("type") == node_type]
    return sorted(filtered, key=lambda x: x[1], reverse=True)[:k]
