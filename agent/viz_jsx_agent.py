"""
JSX/Recharts visualization agent.

Owns JSX-based visualization through:
- generate_jsx_component — create interactive Recharts dashboards via JSX code
- manage_jsx_output — list, view source, recompile, delete JSX outputs
- list_fetched_data — discover available data in memory
- describe_data — inspect data statistics and structure
- preview_data — view actual data values
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .llm import LLMAdapter
from .sub_agent import SubAgent
from .tools import get_function_schemas
from .event_bus import EventBus
from .agent_registry import VIZ_JSX_TOOLS
from knowledge.prompt_builder import build_viz_jsx_prompt

if TYPE_CHECKING:
    from .memory import MemoryStore


class VizJsxAgent(SubAgent):
    """A SubAgent specialized for JSX/Recharts visualization.

    Persistent — stays alive across delegations. The LLM inspects data
    and generates JSX code in a single persistent session.

    The LLM generates JSX/TSX code that is compiled server-side via esbuild
    and rendered in a sandboxed iframe on the frontend.
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
        session_dir: Path | None = None,
    ):
        self.gui_mode = gui_mode
        self.session_dir = session_dir

        # Build tool schemas
        tool_schemas = get_function_schemas(names=VIZ_JSX_TOOLS)

        super().__init__(
            agent_id="VizAgent[JSX]",
            adapter=adapter,
            model_name=model_name,
            tool_executor=tool_executor,
            system_prompt=build_viz_jsx_prompt(gui_mode=gui_mode),
            tool_schemas=tool_schemas,
            event_bus=event_bus,
            memory_store=memory_store,
            memory_scope=memory_scope or "visualization",
        )
