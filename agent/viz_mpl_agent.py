"""
Matplotlib visualization agent.

Owns matplotlib-based visualization through:
- generate_mpl_script — create plots via matplotlib Python scripts in a subprocess sandbox
- manage_mpl_output — list, view scripts, rerun, delete MPL outputs
- list_fetched_data — discover available data in memory
- describe_data — inspect data statistics and structure
- preview_data — view actual data values
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from .sub_agent import SubAgent
from .tools import get_function_schemas
from .event_bus import EventBus
from .agent_registry import VIZ_MPL_TOOLS
from knowledge.prompt_builder import build_viz_mpl_prompt

if TYPE_CHECKING:
    from .memory import MemoryStore


class VizMplAgent(SubAgent):
    """A SubAgent specialized for matplotlib visualization.

    Persistent — stays alive across delegations. The LLM inspects data
    and generates matplotlib scripts in a single persistent session.

    Unlike VizPlotlyAgent, this agent generates standalone Python scripts
    that execute in a subprocess sandbox. The session_dir is needed to locate
    the data directory for script execution.
    """

    _has_deferred_reviews = True

    _PARALLEL_SAFE_TOOLS = {
        "list_fetched_data", "describe_data", "preview_data",
        "review_memory", "events",
    }

    # Mpl scripts are slow to generate — use shorter timeout with fewer
    # retries so we reset quickly instead of waiting 4+ minutes.
    _llm_retry_timeout = 60.0    # 60s per attempt (default 120s)
    _llm_max_retries = 2         # 2 retries max (default 4)
    _llm_reset_threshold = 2     # reset after 2 consecutive failures

    def __init__(
        self,
        service,
        tool_executor,
        *,
        gui_mode: bool = False,
        event_bus: EventBus | None = None,
        memory_store: MemoryStore | None = None,
        memory_scope: str = "",
        session_dir: Path | None = None,
        cancel_event: threading.Event | None = None,
    ):
        self.gui_mode = gui_mode
        self.session_dir = session_dir

        # Build tool schemas
        tool_schemas = get_function_schemas(names=VIZ_MPL_TOOLS)

        super().__init__(
            agent_id="VizAgent[Mpl]",
            service=service,
            agent_type="viz_mpl",
            tool_executor=tool_executor,
            system_prompt=build_viz_mpl_prompt(gui_mode=gui_mode),
            tool_schemas=tool_schemas,
            event_bus=event_bus,
            memory_store=memory_store,
            memory_scope=memory_scope or "visualization",
            cancel_event=cancel_event,
        )
