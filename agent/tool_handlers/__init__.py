from __future__ import annotations
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent

ToolHandler = Callable[["OrchestratorAgent", dict], dict]

# Populated incrementally as handlers are extracted from core.py
TOOL_REGISTRY: dict[str, ToolHandler] = {}

# ── Session, visualization, planning ──
from agent.tool_handlers.session import (
    handle_ask_clarification,
    handle_manage_session_assets,
    handle_events,
    handle_events_admin,
    handle_manage_workers,
)
from agent.tool_handlers.visualization import (
    handle_render_plotly_json,
    handle_manage_plot,
    handle_generate_mpl_script,
    handle_manage_mpl_output,
    handle_generate_jsx_component,
    handle_manage_jsx_output,
)
# ── Memory, document, discovery ──
from agent.tool_handlers.memory import handle_review_memory
from agent.tool_handlers.document import (
    handle_read_document,
    handle_search_function_docs,
    handle_get_function_docs,
)
from agent.tool_handlers.discovery import (
    handle_search_datasets,
    handle_list_parameters,
    handle_browse_datasets,
    handle_list_missions,
    handle_get_dataset_docs,
    handle_search_full_catalog,
    handle_web_search,
)
from agent.tool_handlers.envoy_query import handle_envoy_query

# ── Delegation ──
from agent.tool_handlers.delegation import (
    handle_delegate_to_envoy,
    handle_delegate_to_viz,
    handle_delegate_to_data_ops,
    handle_delegate_to_data_io,
    handle_delegate_to_insight,
    handle_delegate_to_planner,
)

# ── File I/O ──
from agent.tool_handlers.file_io import handle_load_file
from agent.tool_handlers.permission import handle_ask_user_permission
from agent.tool_handlers.package_install import handle_install_package
from agent.tool_handlers.sandbox_packages import handle_manage_sandbox_packages
from agent.tool_handlers.sandbox import handle_run_code
from agent.tool_handlers.manage_envoy import handle_manage_envoy

# ── Data ops ──
from agent.tool_handlers.data_ops import (
    handle_fetch_data,
    handle_list_fetched_data,
    handle_list_assets,
    handle_describe_data,
    handle_preview_data,
    handle_manage_data,
)

# ── Pipeline ──
from agent.tool_handlers.pipeline import (
    handle_pipeline,
    handle_modify_pipeline_node,
    handle_execute_pipeline,
    handle_save_pipeline,
    handle_run_pipeline,
    handle_search_pipelines,
)

TOOL_REGISTRY.update(
    {
        # Session
        "ask_clarification": handle_ask_clarification,
        "install_package": handle_install_package,
        "manage_sandbox_packages": handle_manage_sandbox_packages,
        "manage_session_assets": handle_manage_session_assets,
        "events": handle_events,
        "events_admin": handle_events_admin,
        "manage_workers": handle_manage_workers,
        # Visualization
        "render_plotly_json": handle_render_plotly_json,
        "manage_plot": handle_manage_plot,
        "generate_mpl_script": handle_generate_mpl_script,
        "manage_mpl_output": handle_manage_mpl_output,
        "generate_jsx_component": handle_generate_jsx_component,
        "manage_jsx_output": handle_manage_jsx_output,
        # Planning
        "delegate_to_planner": handle_delegate_to_planner,
        "plan_check": lambda orch, args: orch._handle_plan_check(args),
        # Memory
        "review_memory": handle_review_memory,
        # Document
        "read_document": handle_read_document,
        "search_function_docs": handle_search_function_docs,
        "get_function_docs": handle_get_function_docs,
        # Discovery
        "search_datasets": handle_search_datasets,
        "list_parameters": handle_list_parameters,
        "browse_datasets": handle_browse_datasets,
        "list_missions": handle_list_missions,
        "get_dataset_docs": handle_get_dataset_docs,
        "search_full_catalog": handle_search_full_catalog,
        "web_search": handle_web_search,
        "envoy_query": handle_envoy_query,
        # Delegation
        "delegate_to_envoy": handle_delegate_to_envoy,
        "delegate_to_viz": handle_delegate_to_viz,
        "delegate_to_data_ops": handle_delegate_to_data_ops,
        "delegate_to_data_io": handle_delegate_to_data_io,
        "delegate_to_insight": handle_delegate_to_insight,
        # File I/O
        "load_file": handle_load_file,
        # Data ops
        "fetch_data": handle_fetch_data,
        "list_fetched_data": handle_list_fetched_data,
        "list_assets": handle_list_assets,
        "run_code": handle_run_code,
        "describe_data": handle_describe_data,
        "preview_data": handle_preview_data,
        "manage_data": handle_manage_data,
        # Pipeline
        "pipeline": handle_pipeline,
        # Envoy management
        "manage_envoy": handle_manage_envoy,
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
