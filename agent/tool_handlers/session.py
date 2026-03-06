"""Session tool handlers."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent


def handle_ask_clarification(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    return {
        "status": "clarification_needed",
        "question": tool_args.get("question", ""),
        "options": tool_args.get("options", []),
        "context": tool_args.get("context", ""),
    }


def handle_get_session_assets(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    return orch._handle_get_session_assets()


def handle_restore_plot(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    return orch._handle_restore_plot()


def handle_events(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    action = tool_args.get("action", "")

    if action == "check":
        return _handle_check_events(orch, tool_args)
    elif action == "details":
        return _handle_event_details(orch, tool_args)
    else:
        return {"status": "error", "message": f"Unknown action: {action}"}


def _handle_check_events(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    feed = getattr(orch._tls, "active_sub_agent_feed", None) or orch._event_feed
    from agent.truncation import get_item_limit

    max_events = min(tool_args.get("max_events", 50), get_item_limit("items.events"))
    event_types = tool_args.get("event_types")
    return feed.check(max_events=max_events, event_types=event_types)


def _handle_event_details(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    feed = getattr(orch._tls, "active_sub_agent_feed", None) or orch._event_feed
    return feed.get_details(tool_args.get("event_ids", []))


def handle_events_admin(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    action = tool_args.get("action", "")

    if action == "check":
        return _handle_check_events(orch, tool_args)
    elif action == "details":
        return _handle_event_details(orch, tool_args)
    elif action == "peek":
        return _handle_peek_events(orch, tool_args)
    elif action == "peek_details":
        return _handle_peek_details(orch, tool_args)
    else:
        return {"status": "error", "message": f"Unknown action: {action}"}


def _handle_peek_events(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    feed = getattr(orch._tls, "active_sub_agent_feed", None) or orch._event_feed
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


def _handle_peek_details(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    feed = getattr(orch._tls, "active_sub_agent_feed", None) or orch._event_feed
    return feed.peek_details(tool_args.get("event_ids", []))


def handle_list_active_work(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    return orch._handle_list_active_work(tool_args)


def handle_cancel_work(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    return orch._handle_cancel_work(tool_args)
