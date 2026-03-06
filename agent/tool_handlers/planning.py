"""Planning tool handlers."""

from __future__ import annotations
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent


def handle_request_planning(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    """Run the planner to research datasets and produce a structured plan.

    The planner researches (search_datasets, list_parameters, web_search)
    and returns a plan dict. The orchestrator LLM then executes the plan
    via its existing delegation tools (delegate_to_envoy, delegate_to_viz, etc.).

    Runs asynchronously - the planner will send a message to the orchestrator
    when complete.

    Returns a tool result indicating planning has started.
    """
    request = tool_args["request"]
    reasoning = tool_args.get("reasoning", "")
    time_start = tool_args.get("time_start", "")
    time_end = tool_args.get("time_end", "")

    from agent.event_bus import DEBUG, PLAN_CREATED

    orch._event_bus.emit(
        DEBUG, level="debug", msg=f"[Planner] Planning requested: {reasoning}"
    )

    planner = orch._get_or_create_planner_agent()

    planning_msg = request
    if time_start and time_end:
        planning_msg += (
            f"\n\nSuggested time range: {time_start} to {time_end}. "
            "Use this as your starting point, but adjust if data "
            "availability checks show it is inappropriate."
        )

    planning_msg += (
        "\n\n[Tip: Call events(action='check') to see what happened "
        "earlier in this session.]"
    )

    def run_planning():
        """Run planning in background thread."""
        try:
            planner.start_planning(planning_msg)
        finally:
            planner.reset()

    # Start planning in background - doesn't block
    thread = threading.Thread(target=run_planning, daemon=True)
    thread.start()

    return {
        "status": "success",
        "message": "Planning started. Planner will respond when complete.",
        "planning_used": True,
    }
