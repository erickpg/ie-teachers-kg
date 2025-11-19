"""NetworkX helpers to assemble and export the knowledge graph."""

from __future__ import annotations

import csv
import json
import os
from string import Template
from typing import Dict, List, Tuple, Union

import networkx as nx


_NODE_TYPES = {
    "Professor": "professor",
    "University": "university",
    "Company": "company",
    "Course": "course",
    "Degree": "degree",
    "Location": "location",
}

_NODE_COLORS = {
    "Professor": "#2563eb",
    "University": "#16a34a",
    "Company": "#ea580c",
    "Course": "#a855f7",
    "Degree": "#eab308",
    "Location": "#14b8a6",
    "unknown": "#6b7280",
}

_EDGE_STYLES = {
    "studied_at": {
        "color": "#0ea5e9",
        "dasharray": "6,4",
        "width": 2.0,
        "distance": 140,
        "label": "Studied at",
    },
    "worked_at": {
        "color": "#f97316",
        "dasharray": "0",
        "width": 2.4,
        "distance": 120,
        "label": "Worked at",
    },
    "teaches": {
        "color": "#c026d3",
        "dasharray": "2,2",
        "width": 1.8,
        "distance": 110,
        "label": "Teaches",
    },
    "located_in": {
        "color": "#0d9488",
        "dasharray": "1,6",
        "width": 1.5,
        "distance": 160,
        "label": "Located in",
    },
}

_EDGE_STYLE_DEFAULT = {
    "color": "#94a3b8",
    "dasharray": "0",
    "width": 1.5,
    "distance": 120,
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


def _sanitize_attrs(attrs: Dict[str, object]) -> Dict[str, object]:
    clean: Dict[str, object] = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            clean[key] = ", ".join(str(v) for v in value if v is not None)
        elif isinstance(value, dict):
            clean[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            clean[key] = value
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


def _node_dom_id(node: tuple[str, str] | str) -> str:
    if isinstance(node, tuple) and len(node) == 2:
        return f"{node[0]}::{node[1]}"
    return str(node)


def _node_label(node: tuple[str, str] | str) -> str:
    if isinstance(node, tuple) and len(node) == 2:
        return node[1]
    return str(node)


def _relation_label(relation: str) -> str:
    return relation.replace("_", " ").title()


def _style_for_relation(relation: str) -> Dict[str, object]:
    style = dict(_EDGE_STYLE_DEFAULT)
    custom = _EDGE_STYLES.get(relation)
    if custom:
        style.update(custom)
    style.setdefault("label", _relation_label(relation))
    return style


def _scale_sizes(centrality: Dict[tuple[str, str], float]) -> Dict[tuple[str, str], float]:
    if not centrality:
        return {}
    values = list(centrality.values())
    min_c = min(values)
    max_c = max(values)
    min_size = 8.0
    max_size = 28.0
    if max_c == min_c:
        return {node: (min_size + max_size) / 2.0 for node in centrality}
    span = max_c - min_c
    return {
        node: min_size + ((value - min_c) / span) * (max_size - min_size)
        for node, value in centrality.items()
    }


def _format_tooltip(title: str, node_type: str, attrs: Dict[str, object]) -> str:
    details = []
    for key, value in attrs.items():
        if key == "type":
            continue
        details.append(f"{key}: {value}")
    if details:
        return f"{title} ({node_type})\n" + "\n".join(details)
    return f"{title} ({node_type})"


def _format_edge_tooltip(label: str, attrs: Dict[str, object]) -> str:
    info = []
    for key, value in attrs.items():
        if key == "relation":
            continue
        info.append(f"{key}: {value}")
    if info:
        return f"{label}\n" + "\n".join(info)
    return label


def _graph_to_vis_payload(G: nx.Graph) -> Dict[str, object]:
    centrality = nx.degree_centrality(G) if G.number_of_nodes() else {}
    sizes = _scale_sizes(centrality)
    nodes_payload: List[Dict[str, object]] = []
    node_types_present: set[str] = set()
    for node, data in G.nodes(data=True):
        node_type = data.get("type", "unknown")
        node_types_present.add(node_type)
        attrs = _sanitize_attrs(data)
        label = _node_label(node)
        tooltip = _format_tooltip(label, node_type, attrs)
        nodes_payload.append(
            {
                "id": _node_dom_id(node),
                "label": label,
                "type": node_type,
                "color": _NODE_COLORS.get(node_type, _NODE_COLORS["unknown"]),
                "size": sizes.get(node, 10.0),
                "tooltip": tooltip,
            }
        )

    edges_payload: List[Dict[str, object]] = []
    relations_present: set[str] = set()
    for source, target, data in G.edges(data=True):
        relation = data.get("relation", "related_to")
        relations_present.add(relation)
        attrs = _sanitize_attrs(data)
        style = _style_for_relation(relation)
        tooltip = _format_edge_tooltip(style.get("label", relation), attrs)
        edges_payload.append(
            {
                "source": _node_dom_id(source),
                "target": _node_dom_id(target),
                "relation": relation,
                "label": style.get("label", relation),
                "color": style["color"],
                "dasharray": style["dasharray"],
                "width": style["width"],
                "distance": style["distance"],
                "tooltip": tooltip,
            }
        )

    node_legend = [
        {
            "type": node_type,
            "label": node_type,
            "color": _NODE_COLORS.get(node_type, _NODE_COLORS["unknown"]),
        }
        for node_type in sorted(node_types_present)
    ]

    edge_legend = []
    for relation in sorted(relations_present):
        style = _style_for_relation(relation)
        edge_legend.append(
            {
                "relation": relation,
                "label": style.get("label", relation),
                "color": style["color"],
                "dasharray": style["dasharray"],
                "width": style["width"],
            }
        )

    return {
        "nodes": nodes_payload,
        "edges": edges_payload,
        "node_legend": node_legend,
        "edge_legend": edge_legend,
    }


def _legend_markup(items: List[Dict[str, object]], kind: str) -> str:
    if kind == "node":
        return "\n".join(
            '<li><span class="swatch" style="background:{color}"></span>{label}</li>'.format(
                color=item["color"], label=item["label"]
            )
            for item in items
        )
    return "\n".join(
        '<li><span class="edge" style="border-bottom:{width}px {style} {color}"></span>{label}</li>'.format(
            width=item["width"],
            style="solid" if item["dasharray"] in {"0", "0,0"} else "dashed",
            color=item["color"],
            label=item["label"],
        )
        for item in items
    )


def _write_graph_html(G: nx.Graph, html_path: str) -> None:
    payload = _graph_to_vis_payload(G)
    nodes_json = json.dumps(payload["nodes"], ensure_ascii=False)
    edges_json = json.dumps(payload["edges"], ensure_ascii=False)
    legend_nodes = _legend_markup(payload["node_legend"], "node")
    legend_edges = _legend_markup(payload["edge_legend"], "edge")
    template = Template(
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>IE Teachers Knowledge Graph</title>
    <style>
        :root {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: #0f172a;
            background-color: #f8fafc;
        }
        body {
            margin: 0;
            padding: 1.5rem;
        }
        h1 {
            margin-top: 0;
        }
        #legend {
            display: flex;
            gap: 2rem;
            flex-wrap: wrap;
            margin-bottom: 1.5rem;
        }
        .legend-block {
            background: #ffffff;
            border-radius: 0.75rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
            padding: 1rem 1.5rem;
        }
        .legend-block ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .legend-block li {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 0.35rem 0;
            font-size: 0.95rem;
        }
        .swatch {
            width: 1.25rem;
            height: 1.25rem;
            border-radius: 50%;
            border: 2px solid #e2e8f0;
        }
        .edge {
            width: 2.5rem;
            border-bottom: 2px solid #94a3b8;
        }
        #graph {
            width: 100%;
            height: 720px;
            background: #ffffff;
            border-radius: 1rem;
            box-shadow: 0 15px 40px rgba(15, 23, 42, 0.12);
        }
        svg {
            width: 100%;
            height: 100%;
            border-radius: 1rem;
        }
        text {
            font-size: 10px;
            pointer-events: none;
        }
    </style>
</head>
<body>
    <h1>IE Teachers Knowledge Graph</h1>
    <p>Nodes are sized by degree centrality, coloured by entity type, and edges use distinct colours and dash styles per relation.</p>
    <section id="legend">
        <div class="legend-block">
            <h3>Entities</h3>
            <ul>
                $legend_nodes
            </ul>
        </div>
        <div class="legend-block">
            <h3>Relations</h3>
            <ul>
                $legend_edges
            </ul>
        </div>
    </section>
    <div id="graph"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js" integrity="sha512-s97F9bBlt3F4iAfDdc0AGJi/7luWGINuD/7++UZ5EONosFVJeFt3uTJS3BM4tiT6ak1jaRrI2BGG99nmHn7+Vw==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <script>
        const nodes = $nodes_json;
        const links = $edges_json;
        const width = document.getElementById('graph').clientWidth;
        const height = document.getElementById('graph').clientHeight;

        const svg = d3.select('#graph')
            .append('svg')
            .attr('viewBox', '0 0 ' + width + ' ' + height);

        const link = svg.append('g')
            .attr('stroke-linecap', 'round')
            .selectAll('line')
            .data(links)
            .join('line')
            .attr('stroke', d => d.color)
            .attr('stroke-width', d => d.width)
            .attr('stroke-dasharray', d => d.dasharray);

        link.append('title').text(d => d.tooltip);

        const node = svg.append('g')
            .attr('stroke', '#0f172a')
            .attr('stroke-width', 0.75)
            .selectAll('circle')
            .data(nodes)
            .join('circle')
            .attr('r', d => d.size)
            .attr('fill', d => d.color)
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));

        node.append('title').text(d => d.tooltip);

        const labels = svg.append('g')
            .selectAll('text')
            .data(nodes)
            .join('text')
            .attr('text-anchor', 'middle')
            .attr('dy', d => d.size + 8)
            .text(d => d.label);

        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(d => d.distance).strength(0.15))
            .force('charge', d3.forceManyBody().strength(-220))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(d => d.size + 12));

        simulation.on('tick', () => {
            svg.selectAll('line')
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            labels
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
    </script>
</body>
</html>
"""
    )

    html = template.substitute(
        legend_nodes=legend_nodes,
        legend_edges=legend_edges,
        nodes_json=nodes_json,
        edges_json=edges_json,
    )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


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


def save_graph(G: nx.Graph, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    nodes_path = os.path.join(out_dir, "nodes.csv")
    edges_path = os.path.join(out_dir, "edges.csv")
    gexf_path = os.path.join(out_dir, "graph.gexf")
    html_path = os.path.join(out_dir, "graph.html")

    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "type"])
        for node, data in G.nodes(data=True):
            writer.writerow([str(node), data.get("type", "unknown")])

    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["source", "target", "relation", "attributes"]
        writer.writerow(header)
        for u, v, data in G.edges(data=True):
            writer.writerow([str(u), str(v), data.get("relation", ""), _sanitize_attrs(data)])

    nx.write_gexf(G, gexf_path)
    _write_graph_html(G, html_path)


def top_k_by_degree(G: nx.Graph, node_type: str, k: int = 10) -> List[Tuple[str, int]]:
    filtered = [(n, d) for n, d in G.degree() if G.nodes[n].get("type") == node_type]
    return sorted(filtered, key=lambda x: x[1], reverse=True)[:k]
