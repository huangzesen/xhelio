#!/usr/bin/env python
"""Plot a static flowchart of the data pipeline from a session's operations log.

Usage:
    python scripts/plot_pipeline.py                        # latest session
    python scripts/plot_pipeline.py SESSION_ID             # specific session
    python scripts/plot_pipeline.py --list                 # list sessions with operations
    python scripts/plot_pipeline.py SESSION_ID -o out.png  # save to file

Renders the minimal DAG (from OperationsLog.get_pipeline) as a Plotly figure
using a layered layout. Nodes are colored by tool type.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go

from data_ops.operations_log import OperationsLog


# ── Colors & display names ──────────────────────────────────────────────
TOOL_STYLE = {
    "fetch_data":        {"color": "#4C78A8", "symbol": "diamond", "label": "fetch"},
    "custom_operation":  {"color": "#F58518", "symbol": "square",  "label": "compute"},
    "store_dataframe":   {"color": "#72B7B2", "symbol": "square",  "label": "create"},
    "render_plotly_json": {"color": "#E45756", "symbol": "circle",  "label": "plot"},
    "manage_plot":       {"color": "#BAB0AC", "symbol": "circle",  "label": "export"},
}
# Dimmed variants for orphan nodes (not connected to the final plot)
TOOL_STYLE_ORPHAN = {
    "fetch_data":        {"color": "#B8C9DD", "symbol": "diamond", "label": "fetch"},
    "custom_operation":  {"color": "#F5C68A", "symbol": "square",  "label": "compute"},
    "store_dataframe":   {"color": "#B5DAD6", "symbol": "square",  "label": "create"},
    "render_plotly_json": {"color": "#F0ABAA", "symbol": "circle",  "label": "plot"},
    "manage_plot":       {"color": "#D5D0CD", "symbol": "circle",  "label": "export"},
}
DEFAULT_STYLE = {"color": "#888888", "symbol": "circle", "label": "?"}
DEFAULT_STYLE_ORPHAN = {"color": "#CCCCCC", "symbol": "circle", "label": "?"}

# Text wrapping width (characters) for node labels
LABEL_WRAP_WIDTH = 18

# Layout constants
X_SPACING = 1.6   # horizontal spacing between nodes in the same layer
Y_SPACING = 1.8   # vertical spacing between layers


def _wrap_text(text: str, width: int = LABEL_WRAP_WIDTH) -> str:
    """Wrap text into multiple lines using <br> for Plotly HTML text.

    Splits on comma boundaries first, then wraps remaining long segments.
    """
    if len(text) <= width:
        return text

    # Split on ", " boundaries first
    parts = text.split(", ")
    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}, {part}" if current else part
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            # If the single part is still too long, hard-wrap it
            while len(part) > width:
                lines.append(part[:width])
                part = part[width:]
            current = part
    if current:
        lines.append(current)
    return "<br>".join(lines)


def _is_connected(rec: dict) -> bool:
    """Check if a pipeline record contributes to any end-state product."""
    return bool(rec.get("contributes_to"))


def _load_operations(session_id: str | None, sessions_dir: Path) -> tuple[list[dict], str]:
    """Load operations.json for a session. Returns (records, session_id)."""
    if session_id:
        ops_file = sessions_dir / session_id / "operations.json"
        if not ops_file.exists():
            print(f"Error: {ops_file} not found", file=sys.stderr)
            sys.exit(1)
        return json.loads(ops_file.read_text()), session_id

    # Find latest session with operations
    for d in sorted(sessions_dir.iterdir(), reverse=True):
        ops_file = d / "operations.json"
        if ops_file.exists():
            try:
                ops = json.loads(ops_file.read_text())
                if ops:
                    print(f"Using latest session: {d.name} ({len(ops)} operations)")
                    return ops, d.name
            except (json.JSONDecodeError, OSError):
                continue

    print("No sessions with operations found", file=sys.stderr)
    sys.exit(1)


def _list_sessions(sessions_dir: Path):
    """Print sessions that have operations."""
    for d in sorted(sessions_dir.iterdir(), reverse=True):
        ops_file = d / "operations.json"
        if ops_file.exists():
            try:
                ops = json.loads(ops_file.read_text())
                if ops:
                    tools = {}
                    for r in ops:
                        t = r["tool"]
                        tools[t] = tools.get(t, 0) + 1
                    summary = ", ".join(f"{v}x {k}" for k, v in tools.items())
                    print(f"  {d.name}  ({len(ops)} ops: {summary})")
            except (json.JSONDecodeError, OSError):
                continue


def _assign_layers(pipeline: list[dict]) -> dict[str, int]:
    """Assign each operation to a layer (y-level) via topological depth.

    Layer 0 = operations with no inputs in the pipeline (sources).
    Each subsequent layer is max(layer of input producers) + 1.
    """
    id_rec = {r["id"]: r for r in pipeline}
    label_producer: dict[str, str] = {}
    for rec in pipeline:
        for label in rec["outputs"]:
            label_producer[label] = rec["id"]

    cache: dict[str, int] = {}

    def depth(op_id: str) -> int:
        if op_id in cache:
            return cache[op_id]
        rec = id_rec[op_id]
        parent_depths = []
        for inp in rec["inputs"]:
            src = label_producer.get(inp)
            if src and src in id_rec:
                parent_depths.append(depth(src))
        layer = (max(parent_depths) + 1) if parent_depths else 0
        cache[op_id] = layer
        return layer

    for rec in pipeline:
        depth(rec["id"])
    return cache


def _product_group_key(rec: dict) -> tuple[str, ...]:
    """Return a canonical grouping key from contributes_to (sorted product IDs).

    Nodes contributing to the same set of renders belong to the same product
    group and should be placed near each other.
    """
    ct = rec.get("contributes_to", [])
    return tuple(sorted(ct)) if ct else ("__orphan__",)


def _compute_positions(
    pipeline: list[dict],
    layers: dict[str, int],
) -> dict[str, tuple[float, float]]:
    """Compute (x, y) positions for nodes using product-grouped layered layout.

    Nodes are first grouped by which render(s) they contribute to.  Within
    each layer, groups are laid out side-by-side with extra spacing between
    groups, and nodes within a group are ordered by barycenter to minimise
    edge crossings.
    """
    id_rec = {r["id"]: r for r in pipeline}
    label_producer: dict[str, str] = {}
    for rec in pipeline:
        for label in rec["outputs"]:
            label_producer[label] = rec["id"]

    # Group by layer
    layer_groups: dict[int, list[str]] = {}
    for op_id, layer in layers.items():
        layer_groups.setdefault(layer, []).append(op_id)

    max_layer = max(layers.values()) if layers else 0
    positions: dict[str, tuple[float, float]] = {}

    # Extra gap inserted between product groups within a layer
    GROUP_GAP = X_SPACING * 0.8

    def _place_layer(op_ids: list[str], layer: int) -> None:
        """Place *op_ids* for a given layer, grouping by product."""
        # Partition nodes into product groups
        groups: dict[tuple[str, ...], list[str]] = {}
        for op_id in op_ids:
            key = _product_group_key(id_rec[op_id])
            groups.setdefault(key, []).append(op_id)

        # Compute a barycenter for each node (from already-placed parents)
        barycenters: dict[str, float] = {}
        for op_id in op_ids:
            rec = id_rec[op_id]
            parent_xs = []
            for inp in rec["inputs"]:
                src = label_producer.get(inp)
                if src and src in positions:
                    parent_xs.append(positions[src][0])
            barycenters[op_id] = (
                sum(parent_xs) / len(parent_xs) if parent_xs else 0
            )

        # Sort nodes within each group by barycenter
        for key in groups:
            groups[key].sort(key=lambda oid: barycenters[oid])

        # Order groups by average barycenter so the overall left-to-right
        # placement follows the upstream layout.
        def group_center(key: tuple[str, ...]) -> float:
            members = groups[key]
            return sum(barycenters[oid] for oid in members) / len(members)

        ordered_keys = sorted(groups.keys(), key=group_center)

        # Lay out groups side by side, inserting GROUP_GAP between them.
        all_ordered: list[str] = []
        # Build a list of (op_id, is_group_start) so we know where gaps go.
        segments: list[list[str]] = [groups[k] for k in ordered_keys]

        total_n = sum(len(s) for s in segments)
        # Total width: (total_n - 1) * X_SPACING  + (num_gaps) * GROUP_GAP
        num_gaps = max(0, len(segments) - 1)
        total_width = (total_n - 1) * X_SPACING + num_gaps * GROUP_GAP

        # Start from leftmost position (centred around 0)
        cursor = -total_width / 2
        for seg_idx, seg in enumerate(segments):
            for i, op_id in enumerate(seg):
                positions[op_id] = (cursor, -layer * Y_SPACING)
                if i < len(seg) - 1:
                    cursor += X_SPACING
            if seg_idx < len(segments) - 1:
                cursor += X_SPACING + GROUP_GAP

    # Place each layer
    for layer in range(0, max_layer + 1):
        op_ids = layer_groups.get(layer, [])
        if not op_ids:
            continue
        _place_layer(op_ids, layer)

    return positions


def _bezier_path(
    sx: float, sy: float, dx: float, dy: float, n_points: int = 20,
) -> tuple[list[float | None], list[float | None]]:
    """Generate a vertical bezier curve from (sx,sy) to (dx,dy)."""
    xs, ys = [], []
    # Control points: vertical drop with slight curve
    cy1 = sy + (dy - sy) * 0.4
    cy2 = sy + (dy - sy) * 0.6
    for i in range(n_points + 1):
        t = i / n_points
        it = 1 - t
        # Cubic bezier
        x = it**3 * sx + 3 * it**2 * t * sx + 3 * it * t**2 * dx + t**3 * dx
        y = it**3 * sy + 3 * it**2 * t * cy1 + 3 * it * t**2 * cy2 + t**3 * dy
        xs.append(x)
        ys.append(y)
    xs.append(None)  # break between edges
    ys.append(None)
    return xs, ys


def _collapse_families(pipeline: list[dict]) -> list[dict]:
    """Collapse multi-state product families into a single representative record.

    Returns a new pipeline list where each multi-state family is represented
    by a single merged record (using the first member's ID as the node ID).
    The merged record has ``state_count`` > 1 and ``_family_members`` listing
    all member op IDs for hover text.
    """
    # Identify families with > 1 state
    collapsed: dict[str, list[dict]] = {}  # family_id → members
    for rec in pipeline:
        fam = rec.get("product_family")
        if fam and rec.get("state_count", 1) > 1:
            collapsed.setdefault(fam, []).append(rec)

    if not collapsed:
        return pipeline

    # Set of op IDs that are non-representative family members (to skip)
    skip_ids: set[str] = set()
    # family_id → merged record
    merged: dict[str, dict] = {}
    for fam_id, members in collapsed.items():
        # Use the first member as the representative
        rep = dict(members[0])
        rep["_family_members"] = [m["id"] for m in members]
        # Merge contributes_to from all members
        all_contrib: set[str] = set()
        for m in members:
            all_contrib.update(m.get("contributes_to", []))
        rep["contributes_to"] = sorted(all_contrib)
        merged[fam_id] = rep
        for m in members[1:]:
            skip_ids.add(m["id"])

    result = []
    for rec in pipeline:
        if rec["id"] in skip_ids:
            continue
        if rec["id"] in merged:
            result.append(merged[rec["id"]])
        else:
            result.append(rec)
    return result


def build_figure(pipeline: list[dict], session_id: str) -> go.Figure:
    """Build a Plotly figure showing the pipeline DAG.

    Multi-state product families are collapsed into a single node.
    """
    if not pipeline:
        fig = go.Figure()
        fig.add_annotation(text="Empty pipeline", showarrow=False, font_size=20)
        return fig

    # Collapse multi-state families before layout
    pipeline = _collapse_families(pipeline)

    layers = _assign_layers(pipeline)
    max_layer = max(layers.values()) if layers else 0
    positions = _compute_positions(pipeline, layers)

    id_rec = {r["id"]: r for r in pipeline}
    label_producer: dict[str, str] = {}
    for rec in pipeline:
        for label in rec["outputs"]:
            label_producer[label] = rec["id"]

    n_connected = sum(1 for r in pipeline if _is_connected(r))
    n_orphans = len(pipeline) - n_connected
    if n_orphans:
        print(f"  {n_connected} connected to plot, {n_orphans} orphaned")

    # ── Draw edges ──────────────────────────────────────────────────────
    # Separate connected vs orphan edges for different colors
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    orphan_edge_x: list[float | None] = []
    orphan_edge_y: list[float | None] = []
    annotations = []

    for rec in pipeline:
        dst_id = rec["id"]
        dx, dy = positions[dst_id]
        is_connected = _is_connected(rec)
        for inp in rec["inputs"]:
            src_id = label_producer.get(inp)
            if src_id and src_id in positions:
                sx, sy = positions[src_id]
                bx, by = _bezier_path(sx, sy, dx, dy)

                if is_connected and _is_connected(id_rec.get(src_id, {})):
                    edge_x.extend(bx)
                    edge_y.extend(by)
                    arrow_color = "#AAAAAA"
                    label_color = "#888888"
                else:
                    orphan_edge_x.extend(bx)
                    orphan_edge_y.extend(by)
                    arrow_color = "#DDDDDD"
                    label_color = "#BBBBBB"

                # Arrowhead at destination
                annotations.append(dict(
                    ax=bx[-3], ay=by[-3],
                    x=dx, y=dy,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=2, arrowsize=1, arrowwidth=1.5,
                    arrowcolor=arrow_color,
                ))

                # Edge label at midpoint
                mid_idx = len(bx) // 2
                mx, my = bx[mid_idx], by[mid_idx]
                short_label = inp if len(inp) <= 28 else inp[:25] + "..."
                annotations.append(dict(
                    x=mx, y=my,
                    text=f"<i>{short_label}</i>",
                    showarrow=False,
                    font=dict(size=8, color=label_color),
                    bgcolor="rgba(255,255,255,0.85)",
                    borderpad=1,
                ))

    edge_traces = []
    if edge_x:
        edge_traces.append(go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(color="#BBBBBB", width=1.5),
            hoverinfo="none",
            showlegend=False,
        ))
    if orphan_edge_x:
        edge_traces.append(go.Scatter(
            x=orphan_edge_x, y=orphan_edge_y,
            mode="lines",
            line=dict(color="#E8E8E8", width=1, dash="dot"),
            hoverinfo="none",
            showlegend=False,
        ))

    # ── Draw nodes ──────────────────────────────────────────────────────
    # Group by (tool, connected) for separate traces
    trace_key = lambda tool, connected: f"{tool}|{'c' if connected else 'o'}"
    node_groups: dict[str, dict] = {}

    for rec in pipeline:
        tool = rec["tool"]
        op_id = rec["id"]
        is_connected = _is_connected(rec)
        style = (TOOL_STYLE if is_connected else TOOL_STYLE_ORPHAN).get(
            tool, DEFAULT_STYLE if is_connected else DEFAULT_STYLE_ORPHAN)
        x, y = positions[op_id]

        # Build node label — collapsed families get special label
        state_count = rec.get("state_count", 1)
        family_members = rec.get("_family_members")
        if family_members and state_count > 1:
            node_text = f"plot<br><sub>({state_count} states)</sub>"
        else:
            outputs_str = ", ".join(rec["outputs"]) if rec["outputs"] else ""
            node_text = style["label"]
            if outputs_str:
                wrapped = _wrap_text(outputs_str)
                node_text += f"<br><sub>{wrapped}</sub>"

        # Hover text with full details
        hover_parts = [f"<b>{op_id}</b>: {rec['tool']}"]
        if family_members and state_count > 1:
            hover_parts.append(f"<b>{state_count} states:</b> {', '.join(family_members)}")
        if not is_connected:
            hover_parts.append("<i>(not connected to plot)</i>")
        if rec["outputs"]:
            hover_parts.append(f"out: {', '.join(rec['outputs'])}")
        if rec["inputs"]:
            hover_parts.append(f"in: {', '.join(rec['inputs'])}")
        args_preview = {k: v for k, v in rec["args"].items()
                        if k not in ("figure_json", "figure", "code") and v}
        if args_preview:
            hover_parts.append(f"args: {args_preview}")
        if rec["args"].get("code"):
            code = rec["args"]["code"]
            code_short = code[:100] + "..." if len(code) > 100 else code
            hover_parts.append(f"code: {code_short}")
        hover = "<br>".join(hover_parts)

        key = trace_key(tool, is_connected)
        if key not in node_groups:
            node_groups[key] = {
                "tool": tool, "connected": is_connected, "style": style,
                "x": [], "y": [], "text": [], "hover": [],
            }
        node_groups[key]["x"].append(x)
        node_groups[key]["y"].append(y)
        node_groups[key]["text"].append(node_text)
        node_groups[key]["hover"].append(hover)

    node_traces = []
    seen_legend: set[str] = set()
    for data in node_groups.values():
        style = data["style"]
        is_connected = data["connected"]
        legend_name = style["label"] if is_connected else f"{style['label']} (orphan)"
        show_legend = legend_name not in seen_legend
        seen_legend.add(legend_name)

        node_traces.append(go.Scatter(
            x=data["x"], y=data["y"],
            mode="markers+text",
            marker=dict(
                size=40,
                color=style["color"],
                symbol=style["symbol"],
                line=dict(width=2, color="white"),
                opacity=0.9 if is_connected else 0.5,
            ),
            text=data["text"],
            textposition="bottom center",
            textfont=dict(size=10, color="#333333" if is_connected else "#AAAAAA"),
            hovertext=data["hover"],
            hoverinfo="text",
            name=legend_name,
            showlegend=show_legend,
        ))

    # ── Layout ──────────────────────────────────────────────────────────
    fig = go.Figure(data=edge_traces + node_traces)

    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    x_range = max(all_x) - min(all_x) if all_x else 0
    y_range = max(all_y) - min(all_y) if all_y else 0
    x_margin = max(1.5, x_range * 0.2)
    y_margin = max(1.2, y_range * 0.15)

    fig.update_layout(
        title=dict(text=f"Data Pipeline: {session_id}", font_size=14),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[min(all_x) - x_margin, max(all_x) + x_margin],
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[min(all_y) - y_margin, max(all_y) + y_margin],
            scaleanchor="x",
        ),
        plot_bgcolor="white",
        annotations=annotations,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=80, b=20),
        height=max(450, (max_layer + 1) * 180 + 100),
        width=max(600, int(x_range * 120) + 200),
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot pipeline flowchart from a session")
    parser.add_argument("session_id", nargs="?", help="Session ID (default: latest)")
    parser.add_argument("--list", action="store_true", help="List sessions with operations")
    parser.add_argument("-o", "--output", help="Save to file (png/pdf/html) instead of showing")
    parser.add_argument("--sessions-dir", default=None,
                        type=Path, help="Sessions directory (default: {data_dir}/sessions)")
    args = parser.parse_args()

    if args.sessions_dir is None:
        from config import get_data_dir
        args.sessions_dir = get_data_dir() / "sessions"

    if args.list:
        print("Sessions with operations:")
        _list_sessions(args.sessions_dir)
        return

    records, session_id = _load_operations(args.session_id, args.sessions_dir)

    # Build pipeline
    log = OperationsLog()
    log.load_from_records(records)
    all_labels = set()
    for r in records:
        if r["status"] == "success":
            all_labels.update(r["outputs"])
    pipeline = log.get_pipeline(all_labels)
    print(f"Pipeline: {len(pipeline)} steps (from {len(records)} total operations)")

    # Build and show/save figure
    fig = build_figure(pipeline, session_id)

    if args.output:
        out = Path(args.output)
        if out.suffix == ".html":
            fig.write_html(str(out))
        else:
            fig.write_image(str(out), scale=2)
        print(f"Saved to {out}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
