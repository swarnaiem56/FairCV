"""NetworkX-based hiring graph visualizer.

Loads hiring_graph.graphml and gnn_bias_results.csv, then colors university
and company nodes by residual:
  red   = negative residual (penalized)
  green = positive residual (boosted)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go


DEFAULT_GRAPH_PATH = Path("hiring_graph.graphml")
DEFAULT_RESULTS_PATH = Path("gnn_bias_results.csv")
DEFAULT_OUTPUT_PATH = Path("graph_bias_visualization.html")

BASE_NODE_COLORS = {
    "candidate": "#60a5fa",
    "skill": "#a78bfa",
    "location": "#94a3b8",
    "unknown": "#cbd5e1",
}


def load_inputs(graph_path: Path, results_path: Path):
    graph = nx.read_graphml(graph_path)
    results = pd.read_csv(results_path)
    return graph, results


def residual_to_color(residual, max_abs):
    if residual is None or pd.isna(residual):
        return "#e5e7eb"
    if max_abs <= 1e-9:
        return "#e5e7eb"

    ratio = max(-1.0, min(1.0, float(residual) / max_abs))
    neutral = (229, 231, 235)
    green = (22, 163, 74)
    red = (220, 38, 38)

    if ratio >= 0:
        mix = ratio
        rgb = tuple(neutral[i] + (green[i] - neutral[i]) * mix for i in range(3))
    else:
        mix = -ratio
        rgb = tuple(neutral[i] + (red[i] - neutral[i]) * mix for i in range(3))

    return "#%02x%02x%02x" % tuple(max(0, min(255, int(value))) for value in rgb)


def node_type(node_data):
    return str(node_data.get("type", "unknown"))


def residual_maps(graph: nx.Graph, results: pd.DataFrame):
    residual_by_candidate = results.set_index("candidate_id")["residual"].to_dict()
    residual_by_university = results.groupby("university_name")["residual"].mean().to_dict()

    candidate_to_company = {}
    for company, candidate, edge_data in graph.edges(data=True):
        if edge_data.get("relation") == "hired":
            candidate_to_company[candidate] = company

    company_frame = results.copy()
    company_frame["hired_company"] = company_frame["candidate_id"].map(candidate_to_company)
    residual_by_company = company_frame.groupby("hired_company")["residual"].mean().to_dict()

    company_counts = company_frame.groupby("hired_company")["candidate_id"].count().to_dict()
    university_counts = results.groupby("university_name")["candidate_id"].count().to_dict()

    return residual_by_candidate, residual_by_university, residual_by_company, company_counts, university_counts


def layout_positions(graph: nx.Graph):
    groups = {"company": [], "candidate": [], "university": [], "skill": [], "location": [], "unknown": []}
    for node, data in graph.nodes(data=True):
        groups.setdefault(node_type(data), []).append(node)

    y_levels = {
        "company": 2.8,
        "candidate": 0.0,
        "university": -2.8,
        "skill": -4.6,
        "location": -3.8,
        "unknown": 0.0,
    }

    positions = {}
    for group_name, nodes in groups.items():
        if not nodes:
            continue
        nodes = sorted(nodes)
        span = max(6.0, len(nodes) * 0.9)
        step = span / max(len(nodes), 1)
        start = -span / 2.0
        for idx, node in enumerate(nodes):
            positions[node] = (start + idx * step, y_levels.get(group_name, 0.0))
    return positions


def build_figure(graph: nx.Graph, results: pd.DataFrame, title: str):
    residual_by_candidate, residual_by_university, residual_by_company, company_counts, university_counts = residual_maps(
        graph, results
    )
    max_abs = float(results["residual"].abs().max()) if not results.empty else 0.0
    positions = layout_positions(graph)

    edge_x = []
    edge_y = []
    for u, v in graph.edges():
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.6, color="#9ca3af"),
        hoverinfo="none",
        name="edges",
    )

    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_text = []

    for node, data in graph.nodes(data=True):
        x, y = positions[node]
        ntype = node_type(data)
        node_x.append(x)
        node_y.append(y)

        if ntype == "company":
            residual = residual_by_company.get(node)
            node_color.append(residual_to_color(residual, max_abs))
            node_size.append(20 + 2.0 * company_counts.get(node, 0))
            node_text.append(f"company={node}<br>avg_residual={residual:+.3f}" if residual is not None else f"company={node}<br>avg_residual=nan")
        elif ntype == "university":
            university_name = data.get("name", node)
            residual = residual_by_university.get(university_name)
            node_color.append(residual_to_color(residual, max_abs))
            node_size.append(16 + 1.8 * university_counts.get(university_name, 0))
            node_text.append(
                f"university={university_name}<br>avg_residual={residual:+.3f}"
                if residual is not None
                else f"university={university_name}<br>avg_residual=nan"
            )
        elif ntype == "candidate":
            residual = residual_by_candidate.get(node)
            node_color.append(BASE_NODE_COLORS["candidate"])
            node_size.append(10)
            node_text.append(f"candidate={node}<br>residual={residual:+.3f}" if residual is not None else f"candidate={node}")
        else:
            node_color.append(BASE_NODE_COLORS.get(ntype, BASE_NODE_COLORS["unknown"]))
            node_size.append(9)
            node_text.append(f"node={node}<br>type={ntype}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(color=node_color, size=node_size, line=dict(width=0.8, color="#111827")),
        name="nodes",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        title_x=0.5,
        showlegend=False,
        hovermode="closest",
        margin=dict(b=10, l=10, r=10, t=55),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f8fafc",
    )
    return fig


def run(graph_path: Path, results_path: Path, output_path: Path):
    graph, results = load_inputs(graph_path, results_path)
    fig = build_figure(graph, results, title="Hiring Graph Residual Map")
    fig.write_html(output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    run(args.graph, args.results, args.output)


if __name__ == "__main__":
    main()
