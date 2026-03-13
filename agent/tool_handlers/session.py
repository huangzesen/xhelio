"""Session tool handlers."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.tool_caller import ToolCaller
    from agent.tool_context import ToolContext


def handle_ask_clarification(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    return {
        "status": "clarification_needed",
        "question": tool_args.get("question", ""),
        "options": tool_args.get("options", []),
        "context": tool_args.get("context", ""),
    }



def handle_events(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    action = tool_args.get("action", "")

    if action == "check":
        return _handle_check_events(ctx, tool_args)
    elif action == "details":
        return _handle_event_details(ctx, tool_args)
    else:
        return {"status": "error", "message": f"Unknown action: {action}"}


def _handle_check_events(ctx: "ToolContext", tool_args: dict) -> dict:
    _orch_state = ctx.agent_state.get("orchestrator")
    feed = _orch_state.event_feed if _orch_state else None
    if feed is None:
        return {"status": "error", "message": "Event feed not available"}
    from agent.truncation import get_item_limit

    max_events = min(tool_args.get("max_events", 50), get_item_limit("items.events"))
    event_types = tool_args.get("event_types")
    return feed.check(max_events=max_events, event_types=event_types)


def _handle_event_details(ctx: "ToolContext", tool_args: dict) -> dict:
    _orch_state = ctx.agent_state.get("orchestrator")
    feed = _orch_state.event_feed if _orch_state else None
    if feed is None:
        return {"status": "error", "message": "Event feed not available"}
    return feed.get_details(tool_args.get("event_ids", []))


def handle_events_admin(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    action = tool_args.get("action", "")

    if action == "check":
        return _handle_check_events(ctx, tool_args)
    elif action == "details":
        return _handle_event_details(ctx, tool_args)
    elif action == "peek":
        return _handle_peek_events(ctx, tool_args)
    elif action == "peek_details":
        return _handle_peek_details(ctx, tool_args)
    else:
        return {"status": "error", "message": f"Unknown action: {action}"}


def _handle_peek_events(ctx: "ToolContext", tool_args: dict) -> dict:
    _orch_state = ctx.agent_state.get("orchestrator")
    feed = _orch_state.event_feed if _orch_state else None
    if feed is None:
        return {"status": "error", "message": "Event feed not available"}
    from agent.truncation import get_item_limit

    max_events = min(tool_args.get("max_events", 50), get_item_limit("items.events"))
    event_types = tool_args.get("event_types")
    agent_filter = tool_args.get("agent")
    since_seconds = tool_args.get("since_seconds")
    return feed.peek(
        max_events=max_events,
        event_types=event_types,
        agent_filter=agent_filter,
        since_seconds=since_seconds,
    )


def _handle_peek_details(ctx: "ToolContext", tool_args: dict) -> dict:
    _orch_state = ctx.agent_state.get("orchestrator")
    feed = _orch_state.event_feed if _orch_state else None
    if feed is None:
        return {"status": "error", "message": "Event feed not available"}
    return feed.peek_details(tool_args.get("event_ids", []))


def handle_manage_workers(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    action = tool_args.get("action", "list")
    if ctx.work_tracker is None:
        return {"status": "error", "message": "Work tracker not available"}
    if action == "cancel":
        work_id = tool_args.get("work_id")
        if work_id:
            cancelled = ctx.work_tracker.cancel(work_id)
            return {"status": "success", "cancelled": cancelled}
        else:
            count = ctx.work_tracker.cancel_all()
            return {"status": "success", "cancelled_count": count}
    # Default: list
    active = ctx.work_tracker.list_active()
    return {"status": "success", "active_work": active, "count": len(active)}
