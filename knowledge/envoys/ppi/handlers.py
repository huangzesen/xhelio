"""Tool handlers for PPI envoy kind."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent


def handle_fetch_data_ppi(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    """Fetch data from PDS PPI archive — delegates to the shared fetch handler."""
    from agent.tool_handlers.data_ops import handle_fetch_data
    return handle_fetch_data(orch, tool_args)
