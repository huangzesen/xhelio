"""Permission tool handler — synchronous user approval gate."""

from __future__ import annotations
from typing import TYPE_CHECKING
from uuid import uuid4

from agent.tool_handlers.decorator import tool_handler

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent


@tool_handler("ask_user_permission")
def handle_ask_user_permission(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    """Block until user approves or denies the requested action."""
    action = tool_args.get("action", "unknown")
    description = tool_args.get("description", "")
    command = tool_args.get("command", "")

    if not description or not command:
        return {
            "status": "error",
            "message": "Both 'description' and 'command' are required.",
        }

    request_id = f"perm-{uuid4().hex[:12]}"
    result = orch.request_permission(
        request_id=request_id,
        action=action,
        description=description,
        command=command,
    )

    return {
        "status": "approved" if result["approved"] else "denied",
        "approved": result["approved"],
        "reason": result["reason"],
    }
