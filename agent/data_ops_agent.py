"""
Data operations agent for transformations and analysis.

The LLM explores data structure, researches function APIs via
search_function_docs/get_function_docs, and writes computation code via
run_code — all within a single persistent session.

The orchestrator delegates computation requests here, keeping fetching
in mission agents and visualization in the visualization agent.
"""

from __future__ import annotations

import threading
from typing import Callable, TYPE_CHECKING

from .sub_agent import SubAgent
from .tools import get_function_schemas
from .event_bus import EventBus
from .agent_registry import DATAOPS_TOOLS
from knowledge.prompt_builder import build_data_ops_prompt

if TYPE_CHECKING:
    from .memory import MemoryStore


class DataOpsAgent(SubAgent):
    """A SubAgent specialized for data transformations.

    Persistent — stays alive across delegations. The LLM researches data
    and functions, then writes computation code, all in a single session.
    """

    _has_deferred_reviews = True

    _PARALLEL_SAFE_TOOLS = {
        "describe_data", "preview_data", "list_fetched_data",
        "search_function_docs", "get_function_docs",
        "manage_session_assets", "review_memory", "events",
    }

    def __init__(
        self,
        service,
        tool_executor,
        *,
        agent_id: str = "DataOpsAgent",
        event_bus: EventBus | None = None,
        memory_store: MemoryStore | None = None,
        memory_scope: str = "",
        active_missions_fn: Callable[[], set[str]] | None = None,
        cancel_event: threading.Event | None = None,
    ):
        self._active_missions_fn = active_missions_fn
        # Build tool schemas
        tool_schemas = get_function_schemas(names=DATAOPS_TOOLS)

        super().__init__(
            agent_id=agent_id,
            service=service,
            agent_type="data_ops",
            tool_executor=tool_executor,
            system_prompt=build_data_ops_prompt(),
            tool_schemas=tool_schemas,
            event_bus=event_bus,
            memory_store=memory_store,
            memory_scope=memory_scope or "data_ops",
            cancel_event=cancel_event,
        )

    def _get_active_missions(self) -> set[str] | None:
        """Return active mission IDs for scope filtering."""
        if self._active_missions_fn:
            return self._active_missions_fn()
        return None
