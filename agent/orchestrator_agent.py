"""OrchestratorAgent — top-level orchestrator, BaseAgent subclass.

The orchestrator routes user requests to specialized sub-agents and
handles data operations directly. It is the only agent that:

- Uses streaming responses (streaming=True)
- Has delegation tools (delegate_to_envoy, delegate_to_viz, etc.)
- Uses orchestrator-level loop guard limits
- Manages cancellation holdback for fire-and-forget delegations
- Has a priority inbox (user messages before subagent results)

The actual run_loop and message processing are inherited from BaseAgent.
Orchestrator-specific behavior is expressed through:
- Construction (tool schemas, system prompt, parallel safe tools)
- _pre_request / _post_request hooks
- _get_guard_limits override
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING
from uuid import uuid4

from .logging import get_logger

logger = get_logger()

if TYPE_CHECKING:
    from .session_context import SessionContext

from .base_agent import (
    BaseAgent, Message, _make_message,
    MSG_CANCEL, MSG_USER_INPUT, MSG_REQUEST, MSG_DELEGATION_RESULT,
)
from .event_bus import (
    EventBus, USER_MESSAGE, AGENT_RESPONSE, ROUND_START, ROUND_END,
)
from .llm import LLMService
from .turn_limits import get_limit


class OrchestratorAgent(BaseAgent):
    """Top-level orchestrator agent.

    Attributes:
        agent_type: ``"orchestrator"`` — identity for grouping and config.
    """

    agent_type = "orchestrator"

    # Delegation tools can run concurrently — they block until
    # the sub-agent finishes but don't share mutable state.
    _PARALLEL_SAFE_TOOLS: set[str] = {
        "delegate_to_envoy",
        "delegate_to_viz",
        "delegate_to_data_ops",
        "delegate_to_data_io",
    }

    def __init__(
        self,
        session_ctx: SessionContext,
        service: LLMService,
        *,
        cancel_event: threading.Event | None = None,
        event_bus: EventBus | None = None,
        system_prompt: str = "",
        tool_schemas: list | None = None,
    ):
        agent_id = f"orchestrator:{uuid4().hex[:6]}"

        # Use provided system prompt or build from prompt_builder
        if not system_prompt:
            try:
                from knowledge.prompt_builder import build_orchestrator_system_prompt
                system_prompt = build_orchestrator_system_prompt()
            except (ImportError, Exception):
                system_prompt = (
                    "You are the orchestrator agent. You route user requests to "
                    "specialized sub-agents and handle data operations directly."
                )

        super().__init__(
            agent_id=agent_id,
            service=service,
            tool_schemas=tool_schemas or [],
            system_prompt=system_prompt,
            session_ctx=session_ctx,
            event_bus=event_bus,
            cancel_event=cancel_event,
            streaming=True,
        )

        # -- Orchestrator-private tools --
        # These are plumbing tools only the orchestrator should use.
        # They are NOT in TOOL_REGISTRY — invisible to sub-agents.
        from .tool_handlers.delegation import (
            handle_delegate_to_envoy,
            handle_delegate_to_viz,
            handle_delegate_to_data_ops,
            handle_delegate_to_data_io,
        )
        from .tool_handlers.session import (
            handle_ask_clarification,
            handle_manage_workers,
        )
        from .tool_handlers.planning import handle_plan

        _PRIVATE_TOOLS = {
            "delegate_to_envoy": handle_delegate_to_envoy,
            "delegate_to_viz": handle_delegate_to_viz,
            "delegate_to_data_ops": handle_delegate_to_data_ops,
            "delegate_to_data_io": handle_delegate_to_data_io,
            "ask_clarification": handle_ask_clarification,
            "manage_workers": handle_manage_workers,
            "plan": handle_plan,
        }
        self._local_tools.update(_PRIVATE_TOOLS)

        # Inject schemas for private tools (not in tool_registry.json)
        from .tools import get_function_schemas_for_agent
        private_schemas = get_function_schemas_for_agent(
            names=list(_PRIVATE_TOOLS.keys()), agent_ctx="ctx:orchestrator"
        )
        self._tool_schemas.extend(private_schemas)

        # Register orchestrator state section
        from .tool_caller import OrchestratorState
        if "orchestrator" not in self.session_ctx.agent_state:
            self.session_ctx.agent_state["orchestrator"] = OrchestratorState()

        # Cancellation holdback — buffer subagent results during cancel
        self._held_results: list[dict] = []
        self._held_results_lock = threading.Lock()
        self._was_cancelled = False

        # Cycle tracking (orchestrator concept, not generic agent)
        self._cycle_number = 0
        self._turn_number = 0

    # ------------------------------------------------------------------
    # Public API — cycle number
    # ------------------------------------------------------------------

    @property
    def cycle_number(self) -> int:
        """Current orchestrator cycle count (read-only public API)."""
        return self._cycle_number

    @cycle_number.setter
    def cycle_number(self, value: int) -> None:
        self._cycle_number = value

    # ------------------------------------------------------------------
    # Guard limits — orchestrator gets more generous limits
    # ------------------------------------------------------------------

    def _get_guard_limits(self) -> tuple[int, int, int]:
        """Orchestrator uses orchestrator-specific limits."""
        return (
            get_limit("orchestrator.max_total_calls"),
            get_limit("orchestrator.dup_free_passes"),
            get_limit("orchestrator.dup_hard_block"),
        )

    # ------------------------------------------------------------------
    # Message routing — drain-user-first
    # ------------------------------------------------------------------

    def _handle_message(self, msg: Message) -> None:
        """Route messages with user-first priority.

        When multiple messages are queued, user messages are processed
        before subagent results. Multiple queued user messages are
        merged into a single request.
        """
        if msg.type == MSG_CANCEL:
            self._handle_cancel(msg)
        elif msg.type == MSG_USER_INPUT:
            self._cycle_number += 1
            # Drain additional user messages and merge
            merged = msg.content if isinstance(msg.content, str) else str(msg.content)
            while not self.inbox.empty():
                try:
                    next_msg = self.inbox.get_nowait()
                except queue.Empty:
                    break
                if next_msg.type == MSG_USER_INPUT:
                    next_content = (
                        next_msg.content
                        if isinstance(next_msg.content, str)
                        else str(next_msg.content)
                    )
                    merged += "\n\n---\n\n" + next_content
                else:
                    # Put non-user messages back
                    self.inbox.put(next_msg)
                    break
            # Emit lifecycle events for persistence and SSE
            self._event_bus.emit(
                USER_MESSAGE,
                agent=self.agent_id,
                level="info",
                msg=merged,
                data={"text": merged},
            )
            self._event_bus.emit(
                ROUND_START,
                agent=self.agent_id,
                level="info",
                msg="Round started",
            )
            # Create merged request and process
            merged_msg = _make_message(
                type=MSG_REQUEST,
                sender=msg.sender,
                content=merged,
                reply_to=msg.reply_to,
                reply_event=msg._reply_event,
            )
            self._handle_request(merged_msg)
        elif msg.type == MSG_DELEGATION_RESULT:
            # Subagent results go back to LLM for synthesis
            self._handle_request(msg)
        else:
            super()._handle_message(msg)

    # ------------------------------------------------------------------
    # Pre/post request hooks
    # ------------------------------------------------------------------

    def _pre_request(self, msg: Message) -> str:
        """Inject session context before sending to LLM.

        Prepends:
        - Data store context (available datasets, labels)
        - Plot state (current figure info)
        - Held results from cancelled fire-and-forget delegations
        """
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Drain held fire-and-forget results
        with self._held_results_lock:
            if self._held_results:
                held = self._held_results.copy()
                self._held_results.clear()
                results_text = "\n".join(
                    f"- {r.get('agent', 'unknown')}: {r.get('summary', 'completed')}"
                    for r in held
                )
                content = (
                    f"[Background tasks completed while you were idle]\n"
                    f"{results_text}\n\n{content}"
                )

        # Cancelled context
        if self._was_cancelled:
            content = (
                "[Previous operation was cancelled by the user. "
                "Do not repeat the cancelled work.]\n\n" + content
            )
            self._was_cancelled = False

        return content

    def _post_request(self, msg: Message, result: dict) -> None:
        """Post-request: emit agent response + round end, increment turn counter."""
        text = result.get("text", "")
        generated = result.get("generated", False)
        if text:
            self._event_bus.emit(
                AGENT_RESPONSE,
                agent=self.agent_id,
                level="info",
                msg=text,
                data={"text": text, "generated": generated},
            )
        # Emit round_end only for user-initiated rounds (MSG_REQUEST),
        # not for delegation-result synthesis turns (MSG_DELEGATION_RESULT).
        # ROUND_START is emitted once per user message in _handle_message,
        # so ROUND_END must match 1:1 to keep event log pairs consistent.
        if msg.type == MSG_REQUEST:
            from agent.token_tracking import get_token_usage
            usage = get_token_usage(self)
            self._event_bus.emit(
                ROUND_END,
                agent=self.agent_id,
                level="info",
                msg="Round complete",
                data={
                    "token_usage": usage,
                    "round_token_usage": usage,
                },
            )
            # Auto-save session to disk after each user round so that
            # metadata, chat history, and figure state survive crashes.
            from agent.session_persistence import save_session
            try:
                save_session(self)
            except Exception:
                pass  # Never break the agent loop for a save failure

            # Trigger async memory extraction (daemon thread, non-blocking)
            if self.session_ctx.memory_hooks is not None:
                try:
                    self.session_ctx.memory_hooks.maybe_extract()
                except Exception as e:
                    logger.warning(f"[orchestrator] memory_hooks.maybe_extract failed: {e}", exc_info=True)

            # Trigger async eureka discovery (daemon thread, non-blocking)
            if self.session_ctx.eureka_hooks is not None:
                try:
                    self.session_ctx.eureka_hooks.maybe_extract_eurekas()
                except Exception as e:
                    logger.warning(f"[orchestrator] eureka_hooks.maybe_extract_eurekas failed: {e}", exc_info=True)

        self._turn_number += 1

    # ------------------------------------------------------------------
    # Cancellation holdback
    # ------------------------------------------------------------------

    def _handle_cancel(self, msg: Message) -> None:
        """Cancel with holdback — mark cancelled, don't discard held results."""
        super()._handle_cancel(msg)
        self._was_cancelled = True

    def hold_result(self, result: dict) -> None:
        """Buffer a subagent result during cancellation (fire-and-forget)."""
        with self._held_results_lock:
            self._held_results.append(result)

    # ------------------------------------------------------------------
    # SSE bridge
    # ------------------------------------------------------------------

    def subscribe_sse(self, callback) -> None:
        """Wire the SSE bridge callback into the event bus.

        Creates an SSEEventListener that translates SessionEvents into
        typed dicts and forwards them through *callback* to the async
        SSE stream consumed by the frontend.
        """
        from .event_bus import SSEEventListener

        self._sse_listener = SSEEventListener(callback)
        self._event_bus.subscribe(self._sse_listener)

    def unsubscribe_sse(self) -> None:
        """Disconnect the SSE listener from the event bus."""
        listener = getattr(self, "_sse_listener", None)
        if listener is not None:
            self._event_bus.unsubscribe(listener)
            self._sse_listener = None
