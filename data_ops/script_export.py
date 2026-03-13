"""Pipeline script export — generate standalone Python replay scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_ops.dag import PipelineDAG


def export_script(dag: "PipelineDAG", target_op_id: str) -> str:
    """Generate a Python script that replays the subgraph for target_op_id."""
    sub = dag.subgraph(target_op_id)
    order = sub.topological_order()

    lines = [
        '#!/usr/bin/env python',
        '"""Auto-generated pipeline replay script."""',
        '',
        'from pathlib import Path',
        'from data_ops.store import DataStore',
        'from agent.tool_context import ReplayContext',
        'from agent.tool_handlers import TOOL_REGISTRY',
        '',
        'store = DataStore(data_dir=Path("./replay_data"))',
        'ctx = ReplayContext(store=store)',
        '',
    ]

    for i, op_id in enumerate(order):
        node = sub.node(op_id)
        if node["status"] != "success":
            lines.append(
                f'# Step {i + 1}: {node["tool"]} ({op_id}) — SKIPPED (original failed)'
            )
            lines.append('')
            continue

        lines.append(f'# Step {i + 1}: {node["tool"]} ({op_id})')
        lines.append(f'TOOL_REGISTRY[{node["tool"]!r}](ctx, {node["args"]!r})')
        lines.append('')

    return '\n'.join(lines)
