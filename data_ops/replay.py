"""Pipeline replay engine — re-executes a PipelineDAG subgraph."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.tool_context import ToolContext
    from data_ops.dag import PipelineDAG


@dataclass
class StepResult:
    op_id: str
    tool: str
    status: str  # "success", "skipped", "error"
    duration_ms: float
    error: str | None = None
    outputs: dict | None = None


@dataclass
class ReplayResult:
    """Matches frontend ReplayResult: {steps_completed, steps_total, errors, figure, figure_url}."""
    steps_completed: int
    steps_total: int
    errors: list[dict] = field(default_factory=list)
    figure: dict | None = None
    figure_url: str | None = None
    _steps: list[StepResult] = field(default_factory=list, repr=False)


class ReplayEngine:
    """Re-executes a PipelineDAG subgraph using tool handlers."""

    def __init__(self, dag: "PipelineDAG", ctx: "ToolContext"):
        self.dag = dag
        self.ctx = ctx

    def replay(self, target_op_id: str) -> ReplayResult:
        """Replay all ancestors of target_op_id, then target itself."""
        from agent.tool_handlers import TOOL_REGISTRY

        # 1. Extract minimal subgraph
        sub = self.dag.subgraph(target_op_id)

        # 2. Get topological execution order
        order = sub.topological_order()

        # 3. Execute each step
        steps: list[StepResult] = []
        figures: list[dict] = []

        for op_id in order:
            node = sub.node(op_id)

            if node["status"] != "success":
                steps.append(StepResult(
                    op_id=op_id, tool=node["tool"], status="skipped",
                    duration_ms=0, error="Original execution failed",
                ))
                continue

            handler = TOOL_REGISTRY.get(node["tool"])
            if handler is None:
                steps.append(StepResult(
                    op_id=op_id, tool=node["tool"], status="skipped",
                    duration_ms=0, error=f"No handler for {node['tool']}",
                ))
                continue

            t0 = time.monotonic()
            try:
                from agent.tool_caller import ToolCaller
                _replay_caller = ToolCaller(agent_id="replay", agent_type="replay")
                result = handler(self.ctx, node["args"], _replay_caller)
                elapsed = (time.monotonic() - t0) * 1000
                status = result.get("status", "success")
                steps.append(StepResult(
                    op_id=op_id, tool=node["tool"], status=status,
                    duration_ms=elapsed, outputs=node.get("outputs"),
                ))
                if "figure" in result:
                    figures.append(result["figure"])
            except Exception as e:
                elapsed = (time.monotonic() - t0) * 1000
                steps.append(StepResult(
                    op_id=op_id, tool=node["tool"], status="error",
                    duration_ms=elapsed, error=str(e),
                ))

        # 4. Build result — only actual execution errors, not skips
        errors = [
            {"op_id": s.op_id, "tool": s.tool, "error": s.error}
            for s in steps if s.status == "error"
        ]

        return ReplayResult(
            steps_completed=sum(1 for s in steps if s.status == "success"),
            steps_total=len(steps),
            errors=errors,
            figure=figures[-1] if figures else None,
            _steps=steps,
        )
