"""
Data extraction agent.

Turns unstructured text into structured DataFrames. Handles:
- Event catalogs, ICME lists, flare lists from search results
- Tables extracted from documents (PDF, images)
- Any text-to-DataFrame conversion

Uses store_dataframe to create DataFrames and read_document to read
documents.
"""

from .llm import LLMAdapter
from .sub_agent import SubAgent
from .tools import get_function_schemas
from .event_bus import EventBus
from .agent_registry import EXTRACTION_TOOLS
from knowledge.prompt_builder import build_data_extraction_prompt


class DataExtractionAgent(SubAgent):
    """A SubAgent specialized for converting unstructured text to DataFrames."""

    _has_deferred_reviews = True

    def __init__(
        self,
        adapter: LLMAdapter,
        model_name: str,
        tool_executor,
        *,
        event_bus: EventBus | None = None,
    ):
        tool_schemas = get_function_schemas(names=EXTRACTION_TOOLS)

        super().__init__(
            agent_id="DataExtractionAgent",
            adapter=adapter,
            model_name=model_name,
            tool_executor=tool_executor,
            system_prompt=build_data_extraction_prompt(),
            tool_schemas=tool_schemas,
            event_bus=event_bus,
        )
