"""
Data operations agent with two-phase compute.

Phase 1 (Think): Explore data structure and research function APIs via
an ephemeral chat with function_docs + data inspection tools.

Phase 2 (Execute): Write and run computation code via custom_operation,
enriched with the think phase's research findings.

The orchestrator delegates computation requests here, keeping fetching
in mission agents and visualization in the visualization agent.
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from .llm import LLMAdapter
from .sub_agent import SubAgent
from .tools import get_function_schemas
from .tool_loop import run_tool_loop, extract_text_from_response
from .event_bus import EventBus, DEBUG, PROGRESS
from .loop_guard import LoopGuard
from .turn_limits import get_limit
from .model_fallback import get_active_model
from .agent_registry import DATAOPS_TOOLS, DATAOPS_THINK_TOOLS
from .llm_utils import send_with_timeout
from .logging import get_logger
from knowledge.prompt_builder import build_data_ops_prompt, build_data_ops_think_prompt

if TYPE_CHECKING:
    from .memory import MemoryStore


class DataOpsAgent(SubAgent):
    """A SubAgent specialized for data transformations with think→execute pattern.

    Persistent — stays alive across delegations. The think phase runs
    as an ephemeral LLM chat to research data/functions before execution.
    """

    _has_deferred_reviews = True

    def __init__(
        self,
        adapter: LLMAdapter,
        model_name: str,
        tool_executor,
        *,
        agent_id: str = "DataOpsAgent",
        event_bus: EventBus | None = None,
        memory_store: MemoryStore | None = None,
        memory_scope: str = "",
        active_missions_fn: Callable[[], set[str]] | None = None,
    ):
        self._active_missions_fn = active_missions_fn
        # Build tool schemas
        tool_schemas = get_function_schemas(names=DATAOPS_TOOLS)

        # Think-phase tool schemas
        self._think_tool_schemas = get_function_schemas(names=DATAOPS_THINK_TOOLS)

        super().__init__(
            agent_id=agent_id,
            adapter=adapter,
            model_name=model_name,
            tool_executor=tool_executor,
            system_prompt=build_data_ops_prompt(),
            tool_schemas=tool_schemas,
            event_bus=event_bus,
            memory_store=memory_store,
            memory_scope=memory_scope or "data_ops",
        )

    def _get_active_missions(self) -> set[str] | None:
        """Return active mission IDs for scope filtering."""
        if self._active_missions_fn:
            return self._active_missions_fn()
        return None

    def _handle_request(self, msg):
        """Override to inject think→execute pattern."""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Prepend incremental memory context if available
        memory_prefix = self._build_memory_prefix()
        if memory_prefix:
            content = f"{memory_prefix}\n\n{content}"

        # Phase 1: Think — research data and functions
        self._event_bus.emit(
            DEBUG, agent=self.agent_id,
            msg="[DataOps] Think phase: researching data & functions...",
        )

        think_prompt = build_data_ops_think_prompt()
        chat = self.adapter.create_chat(
            model=get_active_model(self.model_name),
            system_prompt=think_prompt,
            tools=self._think_tool_schemas,
            thinking="high",
        )

        response = send_with_timeout(
            chat=chat,
            message=content,
            timeout_pool=self._timeout_pool,
            cancel_event=self._cancel_event,
            retry_timeout=180,
            agent_name=f"{self.agent_id}/Think",
            logger=get_logger(),
        )
        self._track_usage(response)

        response = run_tool_loop(
            chat=chat,
            response=response,
            tool_executor=self.tool_executor,
            agent_name=f"{self.agent_id}/Think",
            max_total_calls=get_limit("think.max_total_calls"),
            max_iterations=get_limit("think.max_iterations"),
            track_usage=self._track_usage,
            cancel_event=self._cancel_event,
            send_fn=lambda msg: send_with_timeout(
                chat=chat, message=msg,
                timeout_pool=self._timeout_pool,
                cancel_event=self._cancel_event,
                retry_timeout=180,
                agent_name=f"{self.agent_id}/Think",
                logger=get_logger(),
            ),
            adapter=self.adapter,
        )

        think_text = extract_text_from_response(response)
        self._event_bus.emit(
            PROGRESS, agent=self.agent_id,
            msg="[DataOps] Think phase complete",
        )

        # Phase 2: Execute with enriched context
        # Fresh loop guard for the execute phase
        self._guard = LoopGuard(
            max_total_calls=get_limit("sub_agent.max_total_calls"),
            max_iterations=get_limit("sub_agent.max_iterations"),
        )
        if think_text:
            enriched = (
                f"{content}\n\n"
                f"## Research Findings\n{think_text}\n\n"
                f"Now write the code using custom_operation."
            )
        else:
            enriched = content

        llm_response = self._llm_send(enriched)
        result = self._process_response(llm_response, msg)
        self._deliver_result(msg, result)
        self._run_deferred_reviews(msg)
