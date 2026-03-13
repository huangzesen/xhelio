"""DAG → Plotly visualization for pipeline replay UI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from data_ops.dag import PipelineDAG

# Tool type → color mapping (matches frontend TOOL_COLORS)
_TOOL_COLORS = {
    "run_code": "#6366f1",          # indigo
    "render_plotly_json": "#f59e0b", # amber
    "generate_mpl_script": "#f59e0b",
    "generate_jsx_component": "#f59e0b",
    "load_file": "#10b981",          # emerald
    "manage_data": "#8b5cf6",        # violet
    "describe_data": "#64748b",      # slate
    "preview_data": "#64748b",
}
_DEFAULT_COLOR = "#94a3b8"  # gray


def dag_to_plotly(
    dag: "PipelineDAG",
    highlight_op_id: str | None = None,
) -> dict:
    """Convert a PipelineDAG to a Plotly figure dict."""
    order = dag.topological_order()
    if not order:
        return {"data": [], "layout": _make_layout()}

    graph = dag._graph
    depths = _compute_depths(graph, order)
    positions = _layout_by_depth(order, depths)

    highlighted = set()
    if highlight_op_id and highlight_op_id in graph:
        highlighted = nx.ancestors(graph, highlight_op_id) | {highlight_op_id}

    edge_x, edge_y = [], []
    for u, v in graph.edges():
        if u in positions and v in positions:
            x0, y0 = positions[u]
            x1, y1 = positions[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    traces = []
    if edge_x:
        traces.append({
            "type": "scatter",
            "x": edge_x, "y": edge_y,
            "mode": "lines",
            "line": {"width": 1.5, "color": "#cbd5e1"},
            "hoverinfo": "none",
            "showlegend": False,
        })

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for op_id in order:
        if op_id not in positions:
            continue
        x, y = positions[op_id]
        attrs = dag.node(op_id)
        tool = attrs.get("tool", "unknown")
        status = attrs.get("status", "unknown")

        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{op_id}<br>{tool}<br>status: {status}")

        color = _TOOL_COLORS.get(tool, _DEFAULT_COLOR)
        if highlighted and op_id not in highlighted:
            color = "#e2e8f0"
        if status != "success":
            color = "#ef4444"
        node_color.append(color)
        node_size.append(16 if op_id == highlight_op_id else 12)

    traces.append({
        "type": "scatter",
        "x": node_x, "y": node_y,
        "mode": "markers+text",
        "text": [op_id for op_id in order if op_id in positions],
        "textposition": "top center",
        "textfont": {"size": 9, "color": "#475569"},
        "marker": {"size": node_size, "color": node_color, "line": {"width": 1, "color": "#fff"}},
        "hovertext": node_text,
        "hoverinfo": "text",
        "showlegend": False,
    })

    return {"data": traces, "layout": _make_layout()}


def _compute_depths(graph: nx.DiGraph, order: list[str]) -> dict[str, int]:
    depths: dict[str, int] = {}
    for node in order:
        preds = list(graph.predecessors(node))
        if not preds:
            depths[node] = 0
        else:
            depths[node] = max(depths.get(p, 0) for p in preds) + 1
    return depths


def _layout_by_depth(order: list[str], depths: dict[str, int]) -> dict[str, tuple[float, float]]:
    layers: dict[int, list[str]] = {}
    for node in order:
        d = depths.get(node, 0)
        layers.setdefault(d, []).append(node)

    positions = {}
    for depth, nodes in layers.items():
        for i, node in enumerate(nodes):
            y_offset = -(i - (len(nodes) - 1) / 2)
            positions[node] = (depth, y_offset)
    return positions


def _make_layout() -> dict:
    return {
        "showlegend": False,
        "hovermode": "closest",
        "xaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
        "yaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
        "margin": {"l": 20, "r": 20, "t": 20, "b": 20},
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
    }
