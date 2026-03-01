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


def handle_check_events(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    feed = getattr(orch._tls, "active_sub_agent_feed", None) or orch._event_feed
    from agent.truncation import get_item_limit

    max_events = min(
        tool_args.get("max_events", 50), get_item_limit("items.check_events")
    )
    event_types = tool_args.get("event_types")
    compact = tool_args.get("compact", False)
    return feed.check(max_events=max_events, event_types=event_types, compact=compact)


def handle_get_event_details(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    feed = getattr(orch._tls, "active_sub_agent_feed", None) or orch._event_feed
    compact = tool_args.get("compact", False)
    return feed.get_details(tool_args.get("event_ids", []), compact=compact)


def handle_list_active_work(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    return orch._handle_list_active_work(tool_args)


def handle_cancel_work(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    return orch._handle_cancel_work(tool_args)
