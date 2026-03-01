"""Planning tool handlers."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent


def handle_request_planning(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    request = tool_args["request"]
    reasoning = tool_args.get("reasoning", "")
    time_start = tool_args.get("time_start", "")
    time_end = tool_args.get("time_end", "")
    structured_time_range = (
        f"{time_start} to {time_end}" if time_start and time_end else ""
    )
    from agent.event_bus import DEBUG

    orch._event_bus.emit(
        DEBUG, level="debug", msg=f"[Planner] Planning requested: {reasoning}"
    )
    summary = orch._handle_planning_request(
        request, structured_time_range=structured_time_range
    )
    return {"status": "success", "result": summary, "planning_used": True}
