"""Handler for the envoy_query tool — unified envoy capability discovery.

Currently stubbed — returns envoys registered in the kind registry.
Will be rebuilt when MCP-backed envoys are re-added.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.tool_caller import ToolCaller
    from agent.tool_context import ToolContext


def handle_envoy_query(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    """Handle envoy_query tool calls."""
    envoy_id = tool_args.get("envoy")
    search = tool_args.get("search")

    # List all registered envoys
    from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY

    if not envoy_id and not search:
        kinds = ENVOY_KIND_REGISTRY._mission_kinds
        envoys = [
            {"id": mid, "kind": kind}
            for mid, kind in sorted(kinds.items())
        ]
        return {"status": "success", "envoys": envoys, "count": len(envoys)}

    if envoy_id:
        if envoy_id not in ENVOY_KIND_REGISTRY._mission_kinds:
            return {"status": "error", "message": f"Envoy '{envoy_id}' not found."}
        kind = ENVOY_KIND_REGISTRY.get_kind(envoy_id)
        return {"status": "success", "id": envoy_id, "kind": kind}

    return {"status": "success", "envoys": [], "count": 0}
