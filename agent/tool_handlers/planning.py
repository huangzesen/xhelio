"""Planning tool handler — extracted from agent/core.py."""
from __future__ import annotations

from typing import TYPE_CHECKING

from agent.event_bus import PLAN_UPDATE
from agent.session_persistence import save_plan as _persist_plan
from agent.tool_caller import OrchestratorState

if TYPE_CHECKING:
    from agent.tool_caller import ToolCaller
    from agent.tool_context import ToolContext

def _ensure_orch(ctx) -> OrchestratorState:
    """Get or create orchestrator state — safe for mutation."""
    if "orchestrator" not in ctx.agent_state:
        ctx.agent_state["orchestrator"] = OrchestratorState()
    return ctx.agent_state["orchestrator"]


def handle_plan(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    """Handler for the unified plan tool.

    Actions:
      create  — save a new plan (replaces any existing plan)
      update  — tick a step's status or add a note
      check   — return the current plan
      drop    — discard the plan
    """
    orch = _ensure_orch(ctx)
    action = tool_args.get("action", "check")

    if action == "create":
        tasks = tool_args.get("tasks")
        if not tasks or not isinstance(tasks, list):
            return {"status": "error", "message": "create requires a non-empty 'tasks' array"}

        orch.current_plan = {
            "summary": tool_args.get("summary", ""),
            "reasoning": tool_args.get("reasoning", ""),
            "tasks": [
                {
                    "description": t.get("description", ""),
                    "instruction": t.get("instruction", ""),
                    "mission": t.get("mission"),
                    "status": "pending",
                }
                for t in tasks
            ],
        }
        n = len(orch.current_plan["tasks"])

        _persist_plan(ctx)
        if ctx.event_bus is not None:
            ctx.event_bus.emit(
                PLAN_UPDATE, level="info",
                msg=f"[Plan] Created plan with {n} steps",
                data={"action": "create", "plan": orch.current_plan},
            )
        return {"status": "success", "task_count": n}

    if action == "update":

        if orch.current_plan is None:
            return {"status": "error", "message": "No active plan"}
        try:
            step = int(tool_args.get("step"))
        except (TypeError, ValueError):
            return {"status": "error", "message": "step must be an integer"}
        tasks = orch.current_plan["tasks"]
        if not (0 <= step < len(tasks)):
            return {"status": "error", "message": f"Invalid step index (0..{len(tasks) - 1})"}
        new_status = tool_args.get("status")
        if new_status:
            tasks[step]["status"] = new_status
        note = tool_args.get("note")
        if note:
            tasks[step]["note"] = note

        _persist_plan(ctx)
        if ctx.event_bus is not None:
            ctx.event_bus.emit(
                PLAN_UPDATE, level="info",
                msg=f"[Plan] Step {step} → {tasks[step]['status']}",
                data={"action": "update", "plan": orch.current_plan},
            )
        return {"status": "success", "step": step, "new_status": tasks[step]["status"]}

    if action == "check":

        if orch.current_plan is None:
            return {"status": "error", "message": "No active plan"}
        return {"status": "success", "plan": orch.current_plan}

    if action == "drop":

        had_plan = orch.current_plan is not None
        orch.current_plan = None
        _persist_plan(ctx)
        if had_plan:
            if ctx.event_bus is not None:
                ctx.event_bus.emit(
                    PLAN_UPDATE, level="info",
                    msg="[Plan] Dropped",
                    data={"action": "drop", "plan": None},
                )
        return {"status": "success", "dropped": had_plan}

    return {"status": "error", "message": f"Unknown action: {action}"}
