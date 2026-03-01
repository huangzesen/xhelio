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
    handle_check_events,
    handle_get_event_details,
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
from agent.tool_handlers.memory import handle_recall_memories, handle_review_memory
from agent.tool_handlers.document import (
    handle_read_document,
    handle_search_function_docs,
    handle_get_function_docs,
)
from agent.tool_handlers.discovery import (
    handle_browse_tools,
    handle_load_tools,
    handle_search_datasets,
    handle_list_parameters,
    handle_get_data_availability,
    handle_browse_datasets,
    handle_list_missions,
    handle_get_dataset_docs,
    handle_search_full_catalog,
    handle_google_search,
)

# ── Delegation ──
from agent.tool_handlers.delegation import (
    handle_delegate_to_mission,
    handle_delegate_to_viz_plotly,
    handle_delegate_to_viz_mpl,
    handle_delegate_to_viz_jsx,
    handle_delegate_to_data_ops,
    handle_delegate_to_data_extraction,
    handle_delegate_to_insight,
)

# ── Data ops ──
from agent.tool_handlers.data_ops import (
    handle_fetch_data,
    handle_list_fetched_data,
    handle_custom_operation,
    handle_store_dataframe,
    handle_describe_data,
    handle_preview_data,
    handle_save_data,
)

# ── Pipeline ──
from agent.tool_handlers.pipeline import (
    handle_get_pipeline_info,
    handle_modify_pipeline_node,
    handle_execute_pipeline,
    handle_save_pipeline,
    handle_run_pipeline,
    handle_search_pipelines,
)

TOOL_REGISTRY.update({
    # Session
    "ask_clarification": handle_ask_clarification,
    "get_session_assets": handle_get_session_assets,
    "restore_plot": handle_restore_plot,
    "check_events": handle_check_events,
    "get_event_details": handle_get_event_details,
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
    # Memory
    "recall_memories": handle_recall_memories,
    "review_memory": handle_review_memory,
    # Document
    "read_document": handle_read_document,
    "search_function_docs": handle_search_function_docs,
    "get_function_docs": handle_get_function_docs,
    # Discovery
    "browse_tools": handle_browse_tools,
    "load_tools": handle_load_tools,
    "search_datasets": handle_search_datasets,
    "list_parameters": handle_list_parameters,
    "get_data_availability": handle_get_data_availability,
    "browse_datasets": handle_browse_datasets,
    "list_missions": handle_list_missions,
    "get_dataset_docs": handle_get_dataset_docs,
    "search_full_catalog": handle_search_full_catalog,
    "google_search": handle_google_search,
    # Delegation
    "delegate_to_mission": handle_delegate_to_mission,
    "delegate_to_viz_plotly": handle_delegate_to_viz_plotly,
    "delegate_to_viz_mpl": handle_delegate_to_viz_mpl,
    "delegate_to_viz_jsx": handle_delegate_to_viz_jsx,
    "delegate_to_data_ops": handle_delegate_to_data_ops,
    "delegate_to_data_extraction": handle_delegate_to_data_extraction,
    "delegate_to_insight": handle_delegate_to_insight,
    # Data ops
    "fetch_data": handle_fetch_data,
    "list_fetched_data": handle_list_fetched_data,
    "custom_operation": handle_custom_operation,
    "store_dataframe": handle_store_dataframe,
    "describe_data": handle_describe_data,
    "preview_data": handle_preview_data,
    "save_data": handle_save_data,
    # Pipeline
    "get_pipeline_info": handle_get_pipeline_info,
    "modify_pipeline_node": handle_modify_pipeline_node,
    "execute_pipeline": handle_execute_pipeline,
    "save_pipeline": handle_save_pipeline,
    "run_pipeline": handle_run_pipeline,
    "search_pipelines": handle_search_pipelines,
})
