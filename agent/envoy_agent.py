"""
Mission-specific agent for executing tasks within a single mission's context.

Each EnvoyAgent gets a focused system prompt (via build_envoy_prompt)
and its own persistent LLM chat session, so it has deep knowledge of one
mission's data products without context pollution from other missions.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from .sub_agent import SubAgent
from .event_bus import EventBus
from .tools import get_function_schemas
from .agent_registry import ENVOY_TOOL_REGISTRY
from knowledge.prompt_builder import build_envoy_prompt

if TYPE_CHECKING:
    from .memory import MemoryStore


class EnvoyAgent(SubAgent):
    """A SubAgent specialized for a single mission.

    One persistent agent per mission stays alive across delegations (LLM
    session reused via Interactions API).  When the primary is busy,
    ephemeral overflow instances are created with a custom ``agent_id``
    (e.g. ``EnvoyAgent[PSP]#0``) and cleaned up after delegation.
    """

    _has_deferred_reviews = True

    # All envoy tools are read-only discovery or data-fetch operations —
    # safe to run concurrently. SPICE tools are added dynamically via
    # register_spice_tools() in agent_registry.py.
    _PARALLEL_SAFE_TOOLS = {
        "browse_datasets", "search_datasets", "search_full_catalog",
        "list_parameters", "get_dataset_docs",
        "list_fetched_data", "list_missions", "events", "fetch_data",
        "review_memory", "get_session_assets",
    }

    def __init__(
        self,
        mission_id: str,
        service,
        tool_executor,
        *,
        agent_id: str | None = None,
        event_bus: EventBus | None = None,
        memory_store: MemoryStore | None = None,
        memory_scope: str = "",
        cancel_event: threading.Event | None = None,
        sandbox_config: dict | None = None,
    ):
        self.mission_id = mission_id
        self.sandbox_config = sandbox_config

        # Resolve tool set from per-mission registry (group-based)
        tool_list = ENVOY_TOOL_REGISTRY.get_tools(mission_id)

        # Build FunctionSchema list from the chosen tool set
        tool_schemas = get_function_schemas(names=tool_list)

        super().__init__(
            agent_id=agent_id or f"EnvoyAgent[{mission_id}]",
            service=service,
            agent_type="envoy",
            tool_executor=tool_executor,
            system_prompt=build_envoy_prompt(mission_id),
            tool_schemas=tool_schemas,
            event_bus=event_bus,
            memory_store=memory_store,
            memory_scope=memory_scope or f"envoy:{mission_id}",
            cancel_event=cancel_event,
        )

    def _on_tool_result_hook(
        self, tool_name: str, tool_args: dict, result: dict
    ) -> str | None:
        """Intercept clarification_needed results and return formatted question."""
        if not isinstance(result, dict):
            return None
        if result.get("status") == "clarification_needed":
            question = result["question"]
            if result.get("context"):
                question = f"{result['context']}\n\n{question}"
            if result.get("options"):
                question += "\n\nOptions:\n" + "\n".join(
                    f"  {i + 1}. {opt}" for i, opt in enumerate(result["options"])
                )
            return question
        return None
