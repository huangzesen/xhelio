"""
Data I/O agent.

Loads local files and turns unstructured text into structured DataFrames. Handles:
- File imports (CSV, JSON, Parquet, Excel via load_file)
- Event catalogs, ICME lists, flare lists from search results
- Tables extracted from documents (PDF, images)
- Any text-to-DataFrame conversion

Uses load_file for tabular imports, store_dataframe for text-to-DataFrame,
and read_document for document reading.
"""

import threading

from .llm import LLMAdapter
from .sub_agent import SubAgent
from .tools import get_function_schemas
from .event_bus import EventBus
from .agent_registry import DATA_IO_TOOLS
from knowledge.prompt_builder import build_data_io_prompt


class DataIOAgent(SubAgent):
    """A SubAgent specialized for file I/O and converting unstructured text to DataFrames."""

    _has_deferred_reviews = True

    _PARALLEL_SAFE_TOOLS = {
        "list_fetched_data", "get_session_assets",
        "review_memory", "events",
    }

    def __init__(
        self,
        adapter: LLMAdapter,
        model_name: str,
        tool_executor,
        *,
        event_bus: EventBus | None = None,
        cancel_event: threading.Event | None = None,
    ):
        tool_schemas = get_function_schemas(names=DATA_IO_TOOLS)

        super().__init__(
            agent_id="DataIOAgent",
            adapter=adapter,
            model_name=model_name,
            tool_executor=tool_executor,
            system_prompt=build_data_io_prompt(),
            tool_schemas=tool_schemas,
            event_bus=event_bus,
            cancel_event=cancel_event,
        )
