"""
Data extraction sub-agent.

Turns unstructured text into structured DataFrames. Handles:
- Event catalogs, ICME lists, flare lists from search results
- Tables extracted from documents (PDF, images)
- Any text-to-DataFrame conversion

Uses store_dataframe to create DataFrames and read_document to read
documents.
"""

from .llm import LLMAdapter
from .base_agent import BaseSubAgent
from knowledge.prompt_builder import build_data_extraction_prompt

# DataExtraction agent gets extraction + document + conversation tools
EXTRACTION_CATEGORIES = ["data_extraction", "document", "conversation"]
EXTRACTION_EXTRA_TOOLS = ["list_fetched_data", "review_memory"]


class DataExtractionAgent(BaseSubAgent):
    """An LLM session specialized for converting unstructured text to DataFrames."""

    def __init__(
        self,
        adapter: LLMAdapter,
        model_name: str,
        tool_executor,
        verbose: bool = False,
        cancel_event=None,
        event_bus=None,
    ):
        super().__init__(
            adapter=adapter,
            model_name=model_name,
            tool_executor=tool_executor,
            verbose=verbose,
            agent_name="DataExtraction Agent",
            system_prompt=build_data_extraction_prompt(),
            tool_categories=EXTRACTION_CATEGORIES,
            extra_tool_names=EXTRACTION_EXTRA_TOOLS,
            cancel_event=cancel_event,
            event_bus=event_bus,
        )
