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
    handle_get_session_assets,
    handle_restore_plot,
    handle_events,
    handle_events_admin,
    handle_list_active_work,
    handle_cancel_work,
)
from agent.tool_handlers.visualization import (
    handle_render_plotly_json,
    handle_manage_plot,
    handle_generate_mpl_script,
    handle_manage_mpl_output,
    handle_generate_jsx_component,
    handle_manage_jsx_output,
)
from agent.tool_handlers.planning import handle_request_planning

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

# ── Delegation ──
from agent.tool_handlers.delegation import (
    handle_delegate_to_envoy,
    handle_delegate_to_viz,
    handle_delegate_to_data_ops,
    handle_delegate_to_data_io,
    handle_delegate_to_insight,
)

# ── File I/O ──
from agent.tool_handlers.file_io import handle_load_file

# ── Data ops ──
from agent.tool_handlers.data_ops import (
    handle_fetch_data,
    handle_list_fetched_data,
    handle_custom_operation,
    handle_store_dataframe,
    handle_describe_data,
    handle_preview_data,
    handle_save_data,
    handle_merge_datasets,
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
        "get_session_assets": handle_get_session_assets,
        "restore_plot": handle_restore_plot,
        "events": handle_events,
        "events_admin": handle_events_admin,
        "list_active_work": handle_list_active_work,
        "cancel_work": handle_cancel_work,
        # Visualization
        "render_plotly_json": handle_render_plotly_json,
        "manage_plot": handle_manage_plot,
        "generate_mpl_script": handle_generate_mpl_script,
        "manage_mpl_output": handle_manage_mpl_output,
        "generate_jsx_component": handle_generate_jsx_component,
        "manage_jsx_output": handle_manage_jsx_output,
        # Planning
        "request_planning": handle_request_planning,
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
        "custom_operation": handle_custom_operation,
        "store_dataframe": handle_store_dataframe,
        "describe_data": handle_describe_data,
        "preview_data": handle_preview_data,
        "save_data": handle_save_data,
        "merge_datasets": handle_merge_datasets,
        # Pipeline
        "pipeline": handle_pipeline,
    }
)
