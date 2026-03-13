from __future__ import annotations
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from agent.tool_context import ToolContext
    from agent.tool_caller import ToolCaller

# Handler signature: handler(ctx, args, caller) → dict
# ctx is the SessionContext. caller is the ToolCaller (agent identity).
ToolHandler = Callable[["ToolContext", dict, "ToolCaller"], dict]

# Populated incrementally as handlers are extracted from core.py
TOOL_REGISTRY: dict[str, ToolHandler] = {}

# ── Session, visualization, planning ──
from agent.tool_handlers.session import (
    handle_events,
    handle_events_admin,
)
from agent.tool_handlers.visualization import (
    handle_render_plotly_json,
    handle_manage_plot,
    handle_generate_mpl_script,
    handle_manage_mpl_output,
    handle_generate_jsx_component,
    handle_manage_jsx_output,
)
from agent.tool_handlers.figure import handle_manage_figure
from agent.tool_handlers.files import handle_manage_files
# ── Memory, document, discovery ──
from agent.tool_handlers.memory import handle_review_memory
from agent.tool_handlers.document import (
    handle_read_document,
    handle_function_docs,
)
from agent.tool_handlers.envoy_query import handle_envoy_query

# ── Delegation ── (handlers moved to OrchestratorAgent._local_tools)

# ── File I/O ──
from agent.tool_handlers.permission import handle_ask_user_permission
from agent.tool_handlers.sandbox_packages import handle_manage_sandbox_packages
from agent.tool_handlers.sandbox import handle_run_code
from agent.tool_handlers.manage_envoy import handle_manage_envoy
# planning handler moved to OrchestratorAgent._local_tools

# ── Data ops ──
from agent.tool_handlers.data_ops import (
    handle_assets,
    handle_manage_data,
)

# ── Pipeline ──
from agent.tool_handlers.pipeline import (
    handle_pipeline,
)

TOOL_REGISTRY.update(
    {
        # Session (ask_clarification, manage_workers → OrchestratorAgent._local_tools)
        "xhelio__manage_sandbox_packages": handle_manage_sandbox_packages,
        "xhelio__events": handle_events,
        "xhelio__events_admin": handle_events_admin,
        # Visualization
        "xhelio__render_plotly_json": handle_render_plotly_json,
        "xhelio__manage_plot": handle_manage_plot,
        "xhelio__generate_mpl_script": handle_generate_mpl_script,
        "xhelio__manage_mpl_output": handle_manage_mpl_output,
        "xhelio__generate_jsx_component": handle_generate_jsx_component,
        "xhelio__manage_jsx_output": handle_manage_jsx_output,
        # Memory
        "xhelio__review_memory": handle_review_memory,
        # Document
        "xhelio__read_document": handle_read_document,
        "xhelio__function_docs": handle_function_docs,
        # Discovery
        "xhelio__envoy_query": handle_envoy_query,
        # File management
        "xhelio__manage_files": handle_manage_files,
        # Data ops
        "xhelio__assets": handle_assets,
        "xhelio__run_code": handle_run_code,
        "xhelio__manage_data": handle_manage_data,
        # Pipeline
        "xhelio__pipeline": handle_pipeline,
        # Envoy management
        "xhelio__manage_envoy": handle_manage_envoy,
        # Figure management
        "xhelio__manage_figure": handle_manage_figure,
    }
)

# Merge decorator-registered handlers (modules must be imported above so
# decorators execute at import time before this line runs).
from .decorator import get_all_handlers
TOOL_REGISTRY.update(get_all_handlers())


# =============================================================================
# Registry protocol adapter
# =============================================================================


class _ToolHandlerRegistryAdapter:
    name = "tools.handlers"
    description = "Tool name to Python handler function dispatch"

    def get(self, key: str):
        return TOOL_REGISTRY.get(key)

    def list_all(self) -> dict:
        return dict(TOOL_REGISTRY)


TOOL_HANDLER_REGISTRY = _ToolHandlerRegistryAdapter()
from agent.registry_protocol import register_registry  # noqa: E402
register_registry(TOOL_HANDLER_REGISTRY)


# =============================================================================
# MCP tool wrapper factory & envoy registration helpers
# =============================================================================


def make_mcp_handler(kind_name: str, mcp_tool_name: str):
    """Create a TOOL_REGISTRY-compatible handler for an MCP tool.

    The returned handler dispatches to ctx.mcp_client.call_tool().
    If no MCP client is available (e.g., during replay without MCP),
    returns an error dict.
    """
    def handler(ctx, tool_args: dict, caller=None) -> dict:
        client = ctx.mcp_client
        if client is None:
            return {"status": "error", "error": "MCP client not available"}
        return client.call_tool(kind_name, mcp_tool_name, tool_args)
    handler.__name__ = f"mcp_{kind_name}_{mcp_tool_name}"
    handler.__doc__ = f"MCP handler for {kind_name}:{mcp_tool_name}"
    return handler


def register_envoy_tools(kind_name: str, tools: list[dict]) -> list[str]:
    """Register envoy tools in TOOL_REGISTRY with namespace prefix.

    Args:
        kind_name: Envoy kind name (e.g., "cdaweb").
        tools: List of tool dicts with "name" and optional "pipeline_relevant".

    Returns:
        List of registered namespaced tool names.
    """
    from data_ops.dag import register_pipeline_tool

    registered = []
    for tool in tools:
        name = tool.get("name")
        if not name:
            continue
        namespaced = f"{kind_name}:{name}"
        TOOL_REGISTRY[namespaced] = make_mcp_handler(kind_name, name)
        if tool.get("pipeline_relevant", False):
            register_pipeline_tool(namespaced)
        registered.append(namespaced)
    return registered
