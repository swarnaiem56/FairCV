import argparse
import ast
import math
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go


DEFAULT_GRAPH_PATH = Path("hiring_graph.graphml")
DEFAULT_BIAS_PATH = Path("graph_bias_stats.csv")


BASE_NODE_COLORS = {
    "company": "#f59e0b",
    "candidate": "#60a5fa",
    "university": "#34d399",
    "skill": "#a78bfa",
    "location": "#94a3b8",
}

BIAS_COLOR = "#ef4444"  # red = biased nodes


def load_inputs(graph_path: Path, bias_path: Path):
    graph = nx.read_graphml(graph_path)
    bias_df = pd.read_csv(bias_path)
    return graph, bias_df


def get_biased_companies(bias_df: pd.DataFrame, threshold: float):
    filtered = bias_df[
        (bias_df["dominant_share"] >= threshold)
        & (bias_df["dominant_value"].astype(str).str.lower() != "unknown")
    ]
    return set(filtered["company"].unique())


def _safe_parse_counts(raw):
    if pd.isna(raw):
        return {}
    if isinstance(raw, dict):
        return raw
    text = str(raw)
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return {}


def compute_bias_scores(graph: nx.Graph, biased_companies):
    node_bias_score = {node: 0 for node in graph.nodes()}

    for company in biased_companies:
        if company not in graph:
            continue

        node_bias_score[company] += 3

        # Company -> Candidate edges with relation='hired'
        for _, candidate, edge_data in graph.out_edges(company, data=True):
            if edge_data.get("relation") != "hired":
                continue

            node_bias_score[candidate] += 1

            # Candidate -> University edges with relation='attended'
            for _, university, c_edge_data in graph.out_edges(candidate, data=True):
                if c_edge_data.get("relation") == "attended":
                    node_bias_score[university] += 1

    return node_bias_score


def make_node_color(node_type: str, bias_score: int):
    if bias_score > 0:
        return BIAS_COLOR
    return BASE_NODE_COLORS.get(node_type, "#cbd5e1")


def _fast_layout(graph: nx.Graph):
    """Simple hierarchical layout: companies on top, candidates/universities below."""
    import math
    pos = {}
    companies = sorted([n for n, d in graph.nodes(data=True) if d.get('type') == 'company'])
    
    # Position companies horizontally at top
    num_companies = len(companies)
    for i, comp in enumerate(companies):
        x = (i - num_companies / 2) * 2
        pos[comp] = (x, 5)
    
    # Position other nodes around their companies
    node_idx = 0
    for node, data in graph.nodes(data=True):
        if node in pos:
            continue
        node_idx += 1
        angle = (node_idx * 137.5) * math.pi / 180  # Golden angle
        radius = 1 + (node_idx % 5) * 0.5
        x = math.cos(angle) * radius
        y = math.sin(angle) * radius - 3
        pos[node] = (x, y)
    
    return pos


def build_plotly_figure(graph: nx.Graph, node_bias_score, title: str):
    # Create subgraph with high-bias nodes + their neighbors for faster layout
    high_bias_nodes = {n for n, score in node_bias_score.items() if score > 0}
    companies = {n for n, d in graph.nodes(data=True) if d.get('type') == 'company'}
    
    # Include companies, high-bias nodes, and their immediate neighbors
    viz_nodes = companies | high_bias_nodes
    for node in list(high_bias_nodes):
        viz_nodes.update(graph.predecessors(node))
        viz_nodes.update(graph.successors(node))
    
    subgraph = graph.subgraph(viz_nodes).copy()
    
    # Use fast custom layout
    pos = _fast_layout(subgraph)

    edge_x = []
    edge_y = []
    for u, v in subgraph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#9ca3af"),
        hoverinfo="none",
        mode="lines",
        name="edges",
    )

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    for node, data in subgraph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        node_type = data.get("type", "unknown")
        bias_score = node_bias_score.get(node, 0)

        color = make_node_color(node_type, bias_score)
        node_color.append(color)

        # Slightly enlarge high-bias nodes
        node_size.append(10 + min(bias_score, 6) * 1.5)

        node_text.append(
            f"node={node}<br>"
            f"type={node_type}<br>"
            f"bias_score={bias_score}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            color=node_color,
            size=node_size,
            line_width=0.7,
            line_color="#111827",
        ),
        name="nodes",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        title_x=0.5,
        showlegend=False,
        hovermode="closest",
        margin=dict(b=10, l=10, r=10, t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f8fafc",
    )
    return fig


def make_company_bias_table(bias_df: pd.DataFrame, threshold: float):
    rows = []
    for company, group in bias_df.groupby("company"):
        gender_row = group[group["attribute_type"] == "gender"]
        tier_row = group[group["attribute_type"] == "university_tier"]

        if gender_row.empty or tier_row.empty:
            continue

        g = gender_row.iloc[0]
        t = tier_row.iloc[0]

        is_biased = (g["dominant_share"] >= threshold and str(g["dominant_value"]).lower() != "unknown") or (
            t["dominant_share"] >= threshold and str(t["dominant_value"]).lower() != "unknown"
        )

        rows.append(
            {
                "company": company,
                "gender_dominant": g["dominant_value"],
                "gender_share": round(float(g["dominant_share"]), 4),
                "tier_dominant": t["dominant_value"],
                "tier_share": round(float(t["dominant_share"]), 4),
                "bias_flag": "biased" if is_biased else "balanced",
                "gender_counts": _safe_parse_counts(g.get("all_counts")),
                "tier_counts": _safe_parse_counts(t.get("all_counts")),
            }
        )

    return pd.DataFrame(rows).sort_values(["bias_flag", "company"], ascending=[True, True])


def run_dashboard(graph_path: Path, bias_path: Path):
    import streamlit as st

    st.set_page_config(page_title="FairGraph-CV Bias Visualizer", layout="wide")
    st.title("FairGraph-CV: Color-Coded Bias Graph")
    st.caption("Red nodes indicate bias-linked nodes based on company-level dominant-share threshold.")

    with st.sidebar:
        st.header("Controls")
        threshold = st.slider("Bias threshold", min_value=0.50, max_value=0.95, value=0.70, step=0.01)

    graph, bias_df = load_inputs(graph_path, bias_path)
    biased_companies = get_biased_companies(bias_df, threshold)
    node_bias_score = compute_bias_scores(graph, biased_companies)

    title = f"Hiring Network (threshold={threshold:.2f}, biased_companies={len(biased_companies)})"
    fig = build_plotly_figure(graph, node_bias_score, title=title)

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Biased Companies")
        if biased_companies:
            st.write(sorted(biased_companies))
        else:
            st.write("No biased companies at current threshold.")

        st.subheader("Bias Stats")
        summary_df = make_company_bias_table(bias_df, threshold)
        st.dataframe(summary_df, use_container_width=True)


def run_cli(graph_path: Path, bias_path: Path, threshold: float, out_html: Path):
    graph, bias_df = load_inputs(graph_path, bias_path)
    biased_companies = get_biased_companies(bias_df, threshold)
    node_bias_score = compute_bias_scores(graph, biased_companies)

    fig = build_plotly_figure(
        graph,
        node_bias_score,
        title=f"Hiring Network Bias View (threshold={threshold:.2f})",
    )
    fig.write_html(str(out_html), include_plotlyjs="cdn")

    summary_df = make_company_bias_table(bias_df, threshold)
    print("Biased companies:", sorted(biased_companies))
    print("Summary rows:", len(summary_df))
    print(f"Saved visualization HTML to: {out_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Color-coded graph visualization with bias highlighting.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH, help="Path to hiring_graph.graphml")
    parser.add_argument("--bias", type=Path, default=DEFAULT_BIAS_PATH, help="Path to graph_bias_stats.csv")
    parser.add_argument("--threshold", type=float, default=0.70, help="Dominant-share threshold for bias flag")
    parser.add_argument("--html", type=Path, default=Path("graph_bias_visualization.html"), help="Output HTML path for CLI mode")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard mode")
    args = parser.parse_args()

    if args.dashboard:
        # Run with: streamlit run graph_visualizer.py -- --dashboard
        run_dashboard(args.graph, args.bias)
    else:
        run_cli(args.graph, args.bias, args.threshold, args.html)
