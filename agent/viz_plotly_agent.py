"""
Visualization agent for Plotly-based interactive plots.

Owns all visualization through tools:
- render_plotly_json — create/update plots via Plotly figure JSON with data_label placeholders
- manage_plot — export, reset, zoom, get state
- list_fetched_data — discover available data in memory
- describe_data — inspect data statistics and structure
- preview_data — view actual data values
"""

from __future__ import annotations

import re
import threading
from typing import TYPE_CHECKING

from .llm import LLMAdapter
from .sub_agent import SubAgent
from .tools import get_function_schemas
from .event_bus import EventBus
from .agent_registry import VIZ_PLOTLY_TOOLS
from knowledge.prompt_builder import build_viz_plotly_prompt

if TYPE_CHECKING:
    from .memory import MemoryStore


def _extract_labels_from_instruction(instruction: str) -> list[str]:
    """Extract data labels from a task instruction that has store contents appended.

    The orchestrator appends lines like "  - AC_H0_MFI.Magnitude (37800 pts)"
    to the instruction. This extracts the label portion.
    """
    labels = []
    for match in re.finditer(r"^\s+-\s+(\S+)\s+\(", instruction, re.MULTILINE):
        labels.append(match.group(1))
    return labels


class VizPlotlyAgent(SubAgent):
    """A SubAgent specialized for Plotly visualization.

    Persistent — stays alive across delegations. The LLM inspects data
    via describe_data/preview_data/list_fetched_data before plotting,
    all within a single persistent session.
    """

    _has_deferred_reviews = True

    _PARALLEL_SAFE_TOOLS = {
        "list_fetched_data", "describe_data", "preview_data",
        "review_memory", "events",
    }

    def __init__(
        self,
        adapter: LLMAdapter,
        model_name: str,
        tool_executor,
        *,
        gui_mode: bool = False,
        event_bus: EventBus | None = None,
        memory_store: MemoryStore | None = None,
        memory_scope: str = "",
        cancel_event: threading.Event | None = None,
    ):
        self.gui_mode = gui_mode

        # Build tool schemas
        tool_schemas = get_function_schemas(names=VIZ_PLOTLY_TOOLS)

        super().__init__(
            agent_id="VizAgent[Plotly]",
            adapter=adapter,
            model_name=model_name,
            tool_executor=tool_executor,
            system_prompt=build_viz_plotly_prompt(gui_mode=gui_mode),
            tool_schemas=tool_schemas,
            event_bus=event_bus,
            memory_store=memory_store,
            memory_scope=memory_scope or "visualization",
            cancel_event=cancel_event,
        )
