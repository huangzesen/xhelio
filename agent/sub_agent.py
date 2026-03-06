"""
Sub-agent base class for all agents.

Every agent is an autonomous agent with an inbox (queue.Queue). All tools
execute inline (sync) on the agent thread — the LLM always receives actual
results, never "started" acks.

Key concepts:
    - **Sync by default**: every tool call executes inline and returns actual
      results before the LLM's next turn. No fire-and-forget, no polling,
      no background threads for individual tools.
    - **Native parallel tool calling**: when the LLM emits multiple tool calls
      in a single response and all are in ``_PARALLEL_SAFE_TOOLS``, they run
      concurrently via ThreadPoolExecutor. Otherwise they run sequentially.
    - **2-state lifecycle**: SLEEPING (waiting for inbox) and ACTIVE (processing).
    - **Persistent LLM session**: each sub-agent keeps its chat session
      (and Interactions API ``interaction_id``) across messages, so sub-agents
      remember prior context.
"""

from __future__ import annotations

import enum
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .memory import MemoryStore

from .event_bus import (
    EventBus,
    SessionEvent,
    get_event_bus,
    set_event_bus,
    LLM_CALL,
    TOOL_STARTED,
    TOOL_RESULT,
    SUB_AGENT_TOOL,
    SUB_AGENT_ERROR,
    CONTEXT_COMPACTION,
    DEBUG,
    AGENT_STATE_CHANGE,
    TEXT_DELTA,
    MEMORY_EXTRACTION_DONE,
)
from .tool_timing import ToolTimer, stamp_tool_result

# ---------------------------------------------------------------------------
# Context compaction prompt
# ---------------------------------------------------------------------------

_COMPACTION_PROMPT = (
    "Summarize the following conversation history concisely for an AI agent "
    "that needs to continue the session. Preserve:\n"
    "- ALL errors, failures, and their details verbatim\n"
    "- Key decisions and user preferences\n"
    "- Data that was fetched/computed and current state\n"
    "- Tool calls that produced important results\n\n"
    "Drop thinking blocks and routine acknowledgments. "
    "Output ONLY the summary, no commentary.\n\n"
    "Conversation history:\n"
)


# ---------------------------------------------------------------------------
# AgentState — formal lifecycle state model
# ---------------------------------------------------------------------------


class AgentState(enum.Enum):
    """Lifecycle state of a sub-agent (or the orchestrator).

    SLEEPING ──(inbox message)──────────────► ACTIVE
    ACTIVE   ──(all done)──────────────────► SLEEPING

    | State      | Thread            | In-flight work | Can accept input |
    |------------|-------------------|---------------|-----------------|
    | ACTIVE     | Running           | Yes           | No              |
    | SLEEPING   | Blocked on inbox  | None          | Yes             |

    All tools execute inline (sync) — including parallel tool execution, which
    blocks on a ThreadPoolExecutor barrier but stays ACTIVE throughout.
    """

    ACTIVE = "active"
    SLEEPING = "sleeping"


from .llm import LLMAdapter, ChatSession, LLMResponse, FunctionSchema, ToolCall
from .logging import get_logger

from .loop_guard import LoopGuard
from .model_fallback import get_active_model
from .turn_limits import get_limit
from .truncation import trunc, trunc_items
from .llm_utils import send_with_timeout, track_llm_usage, _is_stale_interaction_error
from .token_counter import count_tokens, count_tool_tokens
import config


logger = get_logger()


# ---------------------------------------------------------------------------
# Message — the universal inbox item
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A message delivered to a sub-agent's inbox.

    Attributes:
        id:        Unique message ID.
        type:      One of "request", "tool_result", "tool_error", "cancel",
                   "user_input", "delegation_result".
        sender:    Agent ID, "user", or "tool_runner".
        content:   Payload — str for requests, dict for tool results.
        reply_to:  Links back to original message/task_id.
        timestamp: ``time.monotonic()`` when created.
        _reply_event: Internal Event for callers waiting on a result.
        _reply_value: Internal slot for the agent's response.
    """

    id: str
    type: str
    sender: str
    content: Any
    reply_to: str | None = None
    timestamp: float = field(default_factory=time.monotonic)
    # Synchronous reply channel (set by send callers)
    _reply_event: threading.Event | None = field(default=None, repr=False)
    _reply_value: Any = field(default=None, repr=False)


def _make_message(
    type: str,
    sender: str,
    content: Any,
    *,
    reply_to: str | None = None,
    reply_event: threading.Event | None = None,
) -> Message:
    return Message(
        id=f"msg_{uuid4().hex[:12]}",
        type=type,
        sender=sender,
        content=content,
        reply_to=reply_to,
        _reply_event=reply_event,
    )


# ---------------------------------------------------------------------------
# SubAgent — autonomous agent with inbox and persistent LLM session
# ---------------------------------------------------------------------------


class SubAgent:
    """Base class for all agent-model sub-agents.

    Subclasses override:
        - ``_build_system_prompt()`` to customise the system prompt
        - ``_get_tool_schemas()`` to provide tool schemas
        - ``_on_tool_result_hook()`` for interception (e.g. clarification)
        - ``_post_response_hook()`` for side effects after LLM responds
        - ``_PARALLEL_SAFE_TOOLS`` (class var) to list tools safe for concurrent execution

    The sub-agent owns a single thread that reads from its inbox and processes
    messages sequentially. When the LLM emits multiple tool calls in a single
    response, they run in parallel via ThreadPoolExecutor if all are in
    ``_PARALLEL_SAFE_TOOLS``; otherwise they run sequentially.
    """

    # Subclasses set to True to enable deferred memory reviews after
    # the main result has been delivered back to the orchestrator.
    _has_deferred_reviews: bool = False

    # Tools safe for concurrent execution. Subclasses override with their
    # read-only / side-effect-free tools. If empty, all multi-tool responses
    # fall back to sequential execution.
    _PARALLEL_SAFE_TOOLS: set[str] = set()

    # Inbox polling interval (seconds). Shorter = faster test turnaround.
    _inbox_timeout: float = 1.0

    # LLM timeout/retry overrides. None = use global defaults from llm_utils.
    # Subclasses can override for agent-specific behaviour.
    _llm_retry_timeout: float | None = None   # seconds per attempt before retry
    _llm_max_retries: int | None = None        # max retry attempts
    _llm_reset_threshold: int | None = None    # consecutive failures before session reset

    def __init__(
        self,
        agent_id: str,
        adapter: LLMAdapter,
        model_name: str,
        tool_executor: Callable,
        *,
        system_prompt: str = "",
        tool_schemas: list[FunctionSchema] | None = None,
        event_bus: EventBus | None = None,
        cancel_event: threading.Event | None = None,
        memory_store: MemoryStore | None = None,
        memory_scope: str = "",
    ):
        self.agent_id = agent_id
        self.adapter = adapter
        self.model_name = model_name
        self.tool_executor = tool_executor
        self.system_prompt = system_prompt

        self._tool_schemas = list(tool_schemas) if tool_schemas else []
        self._event_bus = event_bus if event_bus is not None else get_event_bus()
        self._cancel_event = cancel_event

        # Inbox — the single coordination primitive
        self.inbox: queue.Queue[Message] = queue.Queue()

        # Persistent LLM session state
        self._interaction_id: str | None = None
        self._chat: ChatSession | None = None

        # Persistent loop guard — survives across tool-result-triggered
        # _process_response() calls. Reset on each new _handle_request().
        self._guard: LoopGuard | None = None

        # Lifecycle
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._idle = threading.Event()
        self._idle.set()  # starts idle
        self._state = AgentState.SLEEPING

        # Token tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_thinking_tokens = 0
        self._total_cached_tokens = 0
        self._api_calls = 0
        self._last_tool_context = "send_message"

        # Token decomposition (system/tools/history breakdown)
        self._system_prompt_tokens = 0
        self._tools_tokens = 0
        self._token_decomp_dirty = True
        self._latest_input_tokens = 0

        # Memory store for core memory injection (baked into system prompt)
        self._memory_store: MemoryStore | None = memory_store
        self._memory_scope: str = memory_scope

        # Timeout pool for LLM calls
        self._timeout_pool = ThreadPoolExecutor(max_workers=1)

        # Subscribe to event bus for routing tool results to inbox
        self._event_bus.subscribe(self._on_event)
        # Listen for memory mutations to refresh core memory
        self._event_bus.subscribe(self._on_memory_event)


    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the sub-agent's main loop thread."""
        if self._thread and self._thread.is_alive():
            return
        self._shutdown.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name=f"agent-{self.agent_id}",
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal shutdown and wait for the agent thread to exit."""
        self._shutdown.set()
        self._event_bus.unsubscribe(self._on_event)
        if self._thread:
            self._thread.join(timeout=timeout)
        self._timeout_pool.shutdown(wait=False)

    @property
    def is_idle(self) -> bool:
        return self._idle.is_set()

    @property
    def state(self) -> AgentState:
        """Current lifecycle state."""
        return self._state

    def _set_state(self, new_state: AgentState, reason: str = "") -> None:
        """Transition to a new state, keeping ``_idle`` in sync and emitting an event."""
        old = self._state
        if old == new_state:
            return
        self._state = new_state
        # Keep legacy _idle Event in sync
        if new_state == AgentState.SLEEPING:
            self._idle.set()
        else:
            self._idle.clear()
        suffix = f" ({reason})" if reason else ""
        self._event_bus.emit(
            AGENT_STATE_CHANGE,
            agent=self.agent_id,
            level="debug",
            msg=f"[{self.agent_id}] {old.value} → {new_state.value}{suffix}",
            data={
                "agent_id": self.agent_id,
                "old": old.value,
                "new": new_state.value,
                "reason": reason,
            },
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Wait for messages, process them. Sub-agent persists between messages."""
        set_event_bus(self._event_bus)
        while not self._shutdown.is_set():
            try:
                msg = self.inbox.get(timeout=self._inbox_timeout)
            except queue.Empty:
                continue
            self._set_state(AgentState.ACTIVE, reason=f"received {msg.type}")
            try:
                self._handle_message(msg)
            except Exception as e:
                logger.error(
                    f"[{self.agent_id}] Unhandled error in message handler: {e}",
                    exc_info=True,
                )
                # If the message has a reply channel, unblock the caller
                if msg._reply_event:
                    msg._reply_value = {
                        "text": f"Internal error: {e}",
                        "failed": True,
                        "errors": [str(e)],
                    }
                    msg._reply_event.set()
            finally:
                self._set_state(AgentState.SLEEPING, reason="all tools done")

    def _handle_message(self, msg: Message) -> None:
        """Route message by type."""
        if msg.type == "cancel":
            self._handle_cancel(msg)
        elif msg.type in ("request", "user_input"):
            self._handle_request(msg)
        else:
            logger.warning(f"[{self.agent_id}] Unknown message type: {msg.type}")

    # ------------------------------------------------------------------
    # Request handling — the core LLM interaction
    # ------------------------------------------------------------------

    def _handle_request(self, msg: Message) -> None:
        """Send request to LLM, process response with tool calls."""
        # Fresh loop guard for each new request — persists across
        # tool-result-triggered _process_response() calls within this request.
        self._guard = LoopGuard(
            max_total_calls=get_limit("sub_agent.max_total_calls"),
            dup_free_passes=get_limit("sub_agent.dup_free_passes"),
            dup_hard_block=get_limit("sub_agent.dup_hard_block"),
        )
        content = (
            msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        )
        # Prepend current time
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        content = f"[Current time: {current_time}]\n\n{content}"
        response = self._llm_send(content)
        result = self._process_response(response)
        self._deliver_result(msg, result)
        self._run_deferred_reviews(msg)

    def _handle_cancel(self, msg: Message) -> None:
        """Cancel active tools. Sub-agent stays alive."""
        if self._cancel_event:
            self._cancel_event.set()

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _check_and_compact(self) -> None:
        """Check context usage and compact messages if nearing the limit."""
        if self._chat is None:
            return
        ctx_window = self._chat.context_window()
        if ctx_window <= 0:
            return
        ctx_tokens = self._chat.estimate_context_tokens()
        if ctx_tokens <= 0 or ctx_tokens < ctx_window * 0.8:
            return

        from .model_fallback import get_active_model

        def summarizer(text: str) -> str:
            response = self.adapter.generate(
                model=get_active_model(self.model_name),
                contents=_COMPACTION_PROMPT + text,
                temperature=0.1,
                max_output_tokens=2048,
            )
            return response.text.strip() if response and response.text else ""

        before = ctx_tokens
        if self._chat.compact(summarizer=summarizer):
            after = self._chat.estimate_context_tokens()
            self._event_bus.emit(
                CONTEXT_COMPACTION,
                agent=self.agent_id,
                level="info",
                data={
                    "before_tokens": before,
                    "after_tokens": after,
                    "context_window": ctx_window,
                },
            )

    def _llm_send(self, message: Any) -> LLMResponse:
        """Send a message to the LLM, reusing the persistent chat session."""
        if self._chat is None:
            self._chat = self.adapter.create_chat(
                model=get_active_model(self.model_name),
                system_prompt=self._build_core_memory(),
                tools=self._tool_schemas or None,
                thinking="high",
                interaction_id=self._interaction_id,
            )

        self._check_and_compact()

        retry_timeout = self._llm_retry_timeout
        if retry_timeout is None:
            retry_timeout = config.LLM_RETRY_TIMEOUT if hasattr(config, "LLM_RETRY_TIMEOUT") else 180

        try:
            response = send_with_timeout(
                chat=self._chat,
                message=message,
                timeout_pool=self._timeout_pool,
                cancel_event=self._cancel_event,
                retry_timeout=retry_timeout,
                agent_name=self.agent_id,
                logger=logger,
                on_reset=self._on_reset,
                max_retries=self._llm_max_retries,
                reset_threshold=self._llm_reset_threshold,
            )
        except Exception as exc:
            # Handle stale Interactions API session
            if self._interaction_id and _is_stale_interaction_error(exc):
                self._event_bus.emit(
                    DEBUG,
                    agent=self.agent_id,
                    level="warning",
                    msg=f"[{self.agent_id}] Stale interaction — starting fresh session",
                )
                self._interaction_id = None
                self._chat = self.adapter.create_chat(
                    model=get_active_model(self.model_name),
                    system_prompt=self._build_system_prompt(),
                    tools=self._tool_schemas or None,
                    thinking="high",
                )
                response = send_with_timeout(
                    chat=self._chat,
                    message=message,
                    timeout_pool=self._timeout_pool,
                    cancel_event=self._cancel_event,
                    retry_timeout=retry_timeout,
                    agent_name=self.agent_id,
                    logger=logger,
                    max_retries=self._llm_max_retries,
                    reset_threshold=self._llm_reset_threshold,
                )
            else:
                raise

        self._track_usage(response)
        # Preserve interaction ID for session reuse
        if hasattr(self._chat, "interaction_id") and self._chat.interaction_id:
            self._interaction_id = self._chat.interaction_id
        return response

    def _on_reset(self, chat, failed_message):
        """Rollback reset: new chat, drop failed turn, inject context.

        Same strategy as OrchestratorAgent._on_reset — drop the failed
        assistant turn, tell the model what it tried and what the results were.
        """
        # Prefer client-side history (Interactions API) over metadata stubs
        if hasattr(chat, "get_client_history"):
            history = chat.get_client_history()
        else:
            history = chat.get_history()

        from .core import OrchestratorAgent
        tool_summary = OrchestratorAgent._summarize_tool_calls(history)
        result_summary = OrchestratorAgent._summarize_tool_results(failed_message)

        while history and history[-1].get("role") == "assistant":
            history.pop()

        # Drop orphaned tool_result-only user messages (stale tool_call_ids)
        while history and history[-1].get("role") == "user":
            content = history[-1].get("content")
            if isinstance(content, list) and all(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in content
            ):
                history.pop()
            else:
                break

        self._event_bus.emit(
            LLM_CALL,
            agent=self.agent_id,
            level="warning",
            msg=f"[{self.agent_id}] Session rollback — new chat ({len(history)} msgs kept)",
        )

        self._chat = self.adapter.create_chat(
            model=get_active_model(self.model_name),
            system_prompt=self._build_system_prompt(),
            tools=self._tool_schemas or None,
            thinking="high",
            history=history,
        )

        rollback_msg = (
            "Your previous response was lost due to a server error. "
            "Here is what happened:\n\n"
            f"You called these tools:\n{tool_summary}\n\n"
            f"The tools executed successfully and returned:\n{result_summary}\n\n"
            "Data already fetched is still available in memory. "
            "Please continue based on these results."
        )
        return self._chat, rollback_msg

    def _build_system_prompt(self) -> str:
        """Return the system prompt. Override in subclasses."""
        return self.system_prompt

    # ------------------------------------------------------------------
    # Memory injection — diff-based incremental injection
    # ------------------------------------------------------------------

    def _get_active_missions(self) -> set[str] | None:
        """Return active mission IDs for scope filtering. Override in DataOpsAgent."""
        return None

    def _build_core_memory(self) -> str:
        """Core memory = system prompt + long-term memory. Set once at chat creation.

        Memory is baked into the system prompt rather than prepended to each
        user message. This improves prefix cache stability and simplifies
        context management.
        """
        parts = [self._build_system_prompt()]
        if self._memory_store and self._memory_scope:
            section = self._memory_store.format_for_injection(
                scope=self._memory_scope,
                active_missions=self._get_active_missions(),
                include_review_instruction=True,
            )
            if section:
                parts.append(section)
        return "\n\n".join(parts)

    def _on_memory_mutated(self) -> None:
        """Refresh core memory when long-term memory changes mid-session."""
        if self._chat is not None:
            self._chat.update_system_prompt(self._build_core_memory())

    def _on_memory_event(self, event: SessionEvent) -> None:
        """Event bus listener for memory mutation events."""
        if event.type == MEMORY_EXTRACTION_DONE:
            self._on_memory_mutated()

    # ------------------------------------------------------------------
    # Response processing — native parallel tool dispatch
    # ------------------------------------------------------------------

    def _process_response(self, response: LLMResponse) -> dict:
        """Handle tool calls and collect text output.

        When the LLM emits multiple tool calls in a single response:
          - If ALL are in ``_PARALLEL_SAFE_TOOLS``: run concurrently via ThreadPoolExecutor
          - Otherwise: run sequentially on the agent thread

        All tools return actual results to the LLM — no "started" acks, no
        polling loops. The LLM always gets real results before its next turn.

        Returns a result dict: {"text": ..., "failed": ..., "errors": [...]}.
        """
        # Use persistent guard if set (from _handle_request), else create one
        guard = self._guard or LoopGuard(
            max_total_calls=get_limit("sub_agent.max_total_calls"),
            dup_free_passes=get_limit("sub_agent.dup_free_passes"),
            dup_hard_block=get_limit("sub_agent.dup_hard_block"),
        )
        collected_text_parts: list[str] = []
        collected_errors: list[str] = []

        while True:
            # Collect any text output
            if response.text:
                collected_text_parts.append(response.text)
                if response.tool_calls:  # intermediate text — emit it
                    self._event_bus.emit(
                        TEXT_DELTA,
                        agent=self.agent_id,
                        level="info",
                        msg=f"[{self.agent_id}] {response.text}",
                        data={"text": response.text + "\n\n", "commentary": True},
                    )

            if not response.tool_calls:
                break

            # Check cancel
            if self._cancel_event and self._cancel_event.is_set():
                return {
                    "text": "Interrupted by user.",
                    "failed": True,
                    "errors": ["Interrupted"],
                }

            # Total call limit check
            stop_reason = guard.check_limit(len(response.tool_calls))
            if stop_reason:
                self._event_bus.emit(
                    DEBUG,
                    agent=self.agent_id,
                    level="debug",
                    msg=f"[{self.agent_id}] Stopping: {stop_reason}",
                )
                break

            # Decide parallel vs sequential execution
            all_parallel_safe = (
                len(response.tool_calls) > 1
                and self._PARALLEL_SAFE_TOOLS
                and all(tc.name in self._PARALLEL_SAFE_TOOLS for tc in response.tool_calls)
            )

            if all_parallel_safe:
                result = self._execute_tools_parallel(
                    response.tool_calls, guard, collected_errors,
                )
            else:
                result = self._execute_tools_sequential(
                    response.tool_calls, guard, collected_errors,
                )

            if result.intercepted:
                # Commit tool results to session history without an API call.
                # The assistant's tool_use blocks must be paired with
                # tool_results to keep history valid for followup messages.
                if result.tool_results and self._chat:
                    self._chat.commit_tool_results(result.tool_results)
                return {
                    "text": result.intercept_text,
                    "failed": False,
                    "errors": [],
                }

            guard.record_calls(len(response.tool_calls))

            # Detect repeated unrecoverable errors — same error twice in a row
            if (
                len(collected_errors) >= 2
                and collected_errors[-1] == collected_errors[-2]
            ):
                logger.warning(
                    "[%s] Same error repeated, breaking early: %s",
                    self.agent_id,
                    collected_errors[-1],
                )
                break  # failed=True will be set by collected_errors check below

            # Feed tool results back to LLM
            response = self._llm_send(result.tool_results)

        final_text = "\n".join(collected_text_parts)
        # Always pass back the result info - let the orchestrator decide success/failure
        # based on actual outcomes (output_files, figure created, etc.) not heuristics
        return {"text": final_text, "failed": False, "errors": collected_errors}

    # ------------------------------------------------------------------
    # Tool execution — shared result container
    # ------------------------------------------------------------------

    @dataclass
    class _ToolExecResult:
        """Container for tool execution results."""
        tool_results: list  # adapter-formatted tool result messages
        intercepted: bool = False
        intercept_text: str = ""

    def _execute_single_tool(
        self,
        tc: ToolCall,
        guard: LoopGuard,
        collected_errors: list[str],
    ) -> tuple[dict | None, bool, str]:
        """Execute a single tool call. Returns (result_msg, intercepted, intercept_text).

        Handles commentary, duplicate detection, event emission, timing, and hooks.
        Returns None for result_msg if the tool was blocked by the loop guard.
        """
        tc_id = getattr(tc, "id", None)
        args = dict(tc.args) if tc.args else {}
        commentary = args.pop("commentary", None)
        if commentary:
            self._event_bus.emit(
                TEXT_DELTA,
                agent=self.agent_id,
                level="info",
                msg=f"[{self.agent_id}] {commentary}",
                data={"text": commentary + "\n\n", "commentary": True},
            )

        # Strip backward-compat _sync flag
        args.pop("_sync", None)

        # Duplicate call tracking
        verdict = guard.record_tool_call(tc.name, args)
        if verdict.blocked:
            result = {
                "status": "blocked",
                "_duplicate_warning": verdict.warning,
                "message": f"Execution skipped — duplicate call #{verdict.count}",
            }
            msg = self.adapter.make_tool_result_message(
                tc.name, result, tool_call_id=tc_id,
            )
            return msg, False, ""

        # Emit TOOL_STARTED for console/event log visibility
        task_id = f"tool_{uuid4().hex[:8]}"
        self._event_bus.emit(
            TOOL_STARTED,
            agent=self.agent_id,
            level="debug",
            msg=f"[{self.agent_id}] Tool: {tc.name}",
            data={
                "task_id": task_id,
                "tool_name": tc.name,
                "tool_args": args,
            },
        )

        timer = ToolTimer()
        try:
            # Inject reviewer identity for review_memory
            if tc.name == "review_memory":
                args = {**args, "_reviewer_agent_id": self.agent_id}
            with timer:
                result = self.tool_executor(tc.name, args, tc_id)
            if isinstance(result, dict):
                stamp_tool_result(result, timer.elapsed_ms)

            # Inject dup warning if applicable
            if verdict.warning and isinstance(result, dict):
                result["_duplicate_warning"] = verdict.warning

            result_msg = self.adapter.make_tool_result_message(
                tc.name, result, tool_call_id=tc_id,
            )

            # Emit tool result event for console/event log
            status = (
                "error"
                if (isinstance(result, dict) and result.get("status") == "error")
                else "success"
            )
            # Track dict-level errors so collected_errors reflects tool failures
            if status == "error":
                err_msg = (
                    result.get("message", "unknown error")
                    if isinstance(result, dict)
                    else str(result)
                )
                collected_errors.append(f"{tc.name}: {err_msg}")
            self._event_bus.emit(
                SUB_AGENT_TOOL,
                agent=self.agent_id,
                target=f"tool:{tc.name}",
                msg=f"[{self.agent_id}] {tc.name} → {status}",
                data={
                    "task_id": task_id,
                    "tool_name": tc.name,
                    "tool_args": args,
                    "tool_result": result,
                    "status": status,
                    "elapsed_ms": timer.elapsed_ms,
                },
            )
            # Emit for frontend display (SSE)
            self._event_bus.emit(
                TOOL_RESULT,
                agent=self.agent_id,
                target=self.agent_id,
                msg=f"[{self.agent_id}] {tc.name} → {status}",
                data={
                    "tool_name": tc.name,
                    "status": status,
                    "elapsed_ms": timer.elapsed_ms,
                },
            )

            # Run interception hook
            intercept = self._on_tool_result_hook(tc.name, args, result)
            if intercept is not None:
                return result_msg, True, intercept

            return result_msg, False, ""

        except Exception as e:
            err_result = {"status": "error", "message": str(e)}
            stamp_tool_result(err_result, timer.elapsed_ms)
            result_msg = self.adapter.make_tool_result_message(
                tc.name, err_result, tool_call_id=tc_id,
            )
            collected_errors.append(f"{tc.name}: {e}")
            self._event_bus.emit(
                SUB_AGENT_ERROR,
                agent=self.agent_id,
                level="error",
                msg=f"[{self.agent_id}] {tc.name} FAILED: {e}",
                data={
                    "task_id": task_id,
                    "tool_name": tc.name,
                    "tool_args": args,
                    "error": str(e),
                },
            )
            return result_msg, False, ""

    # ------------------------------------------------------------------
    # Sequential tool execution
    # ------------------------------------------------------------------

    def _execute_tools_sequential(
        self,
        tool_calls: list[ToolCall],
        guard: LoopGuard,
        collected_errors: list[str],
    ) -> _ToolExecResult:
        """Run tool calls one at a time on the agent thread."""
        tool_results = []
        for tc in tool_calls:
            result_msg, intercepted, intercept_text = self._execute_single_tool(
                tc, guard, collected_errors,
            )
            if result_msg is not None:
                tool_results.append(result_msg)
            if intercepted:
                return SubAgent._ToolExecResult(
                    tool_results=tool_results,
                    intercepted=True,
                    intercept_text=intercept_text,
                )
        return SubAgent._ToolExecResult(tool_results=tool_results)

    # ------------------------------------------------------------------
    # Parallel tool execution via ThreadPoolExecutor
    # ------------------------------------------------------------------

    def _execute_tools_parallel(
        self,
        tool_calls: list[ToolCall],
        guard: LoopGuard,
        collected_errors: list[str],
    ) -> _ToolExecResult:
        """Run multiple tool calls concurrently via ThreadPoolExecutor.

        Pre-checks duplicate detection sequentially (guard is not thread-safe),
        then submits non-blocked tools to the pool. Results are collected in
        original order. Hooks fire after all tools complete.
        """
        from concurrent.futures import as_completed

        timeout = get_limit("agent.parallel_tool_timeout")

        # Phase 1: Pre-check duplicates and prepare (sequential — guard not thread-safe)
        to_execute: list[tuple[int, ToolCall, dict]] = []  # (index, tc, args)
        tool_results: list[tuple[int, dict | None]] = []   # (index, result_msg)

        for i, tc in enumerate(tool_calls):
            tc_id = getattr(tc, "id", None)
            args = dict(tc.args) if tc.args else {}
            commentary = args.pop("commentary", None)
            args.pop("_sync", None)

            if commentary:
                self._event_bus.emit(
                    TEXT_DELTA,
                    agent=self.agent_id,
                    level="info",
                    msg=f"[{self.agent_id}] {commentary}",
                    data={"text": commentary + "\n\n", "commentary": True},
                )

            verdict = guard.record_tool_call(tc.name, args)
            if verdict.blocked:
                result = {
                    "status": "blocked",
                    "_duplicate_warning": verdict.warning,
                    "message": f"Execution skipped — duplicate call #{verdict.count}",
                }
                tool_results.append((i, self.adapter.make_tool_result_message(
                    tc.name, result, tool_call_id=tc_id,
                )))
            else:
                to_execute.append((i, tc, args))

        if not to_execute:
            # All blocked — return results sorted by index
            tool_results.sort(key=lambda x: x[0])
            return SubAgent._ToolExecResult(
                tool_results=[r for _, r in tool_results],
            )

        # Phase 2: Execute in parallel
        tool_names = [tc.name for _, tc, _ in to_execute]
        shown, _ = trunc_items(tool_names, "items.tool_names_log")
        ellipsis = "..." if len(tool_names) > len(shown) else ""
        self._event_bus.emit(
            DEBUG,
            agent=self.agent_id,
            level="debug",
            msg=f"[{self.agent_id}] Parallel execution of {len(to_execute)} tool(s): {', '.join(shown)}{ellipsis}",
        )

        results_map: dict[int, Any] = {}    # index → raw tool result
        errors_map: dict[int, str] = {}     # index → error message
        task_ids: dict[int, str] = {}       # index → task_id for events
        args_map: dict[int, dict] = {}      # index → processed args
        tc_map: dict[int, ToolCall] = {}    # index → ToolCall

        def _run_one(index: int, tc: ToolCall, args: dict):
            tc_id = getattr(tc, "id", None)
            task_id = f"par_{uuid4().hex[:8]}"
            task_ids[index] = task_id
            args_map[index] = args
            tc_map[index] = tc

            self._event_bus.emit(
                TOOL_STARTED,
                agent=self.agent_id,
                level="debug",
                msg=f"[{self.agent_id}] Parallel tool: {tc.name}",
                data={
                    "task_id": task_id,
                    "tool_name": tc.name,
                    "tool_args": args,
                },
            )

            if tc.name == "review_memory":
                args = {**args, "_reviewer_agent_id": self.agent_id}

            timer = ToolTimer()
            with timer:
                result = self.tool_executor(tc.name, args, tc_id)
            if isinstance(result, dict):
                stamp_tool_result(result, timer.elapsed_ms)

            status = (
                "error"
                if (isinstance(result, dict) and result.get("status") == "error")
                else "success"
            )
            self._event_bus.emit(
                SUB_AGENT_TOOL,
                agent=self.agent_id,
                target=f"tool:{tc.name}",
                msg=f"[{self.agent_id}] {tc.name} → {status}",
                data={
                    "task_id": task_id,
                    "tool_name": tc.name,
                    "tool_args": args,
                    "tool_result": result,
                    "status": status,
                    "elapsed_ms": timer.elapsed_ms,
                },
            )
            self._event_bus.emit(
                TOOL_RESULT,
                agent=self.agent_id,
                target=self.agent_id,
                msg=f"[{self.agent_id}] {tc.name} → {status}",
                data={
                    "tool_name": tc.name,
                    "status": status,
                    "elapsed_ms": timer.elapsed_ms,
                },
            )
            return index, result

        pool = ThreadPoolExecutor(max_workers=len(to_execute))
        try:
            futures = {
                pool.submit(_run_one, i, tc, args): i
                for i, tc, args in to_execute
            }

            try:
                done = set()
                for future in as_completed(futures, timeout=float(timeout)):
                    if self._cancel_event and self._cancel_event.is_set():
                        break
                    done.add(future)
                    try:
                        idx, result = future.result()
                        results_map[idx] = result
                    except Exception as e:
                        idx = futures[future]
                        errors_map[idx] = str(e)
                # Mark un-completed as timed out
                for future, idx in futures.items():
                    if future not in done:
                        errors_map[idx] = "Timed out"
            except TimeoutError:
                for future, idx in futures.items():
                    if idx not in results_map and idx not in errors_map:
                        errors_map[idx] = "Timed out"
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        # Phase 3: Build result messages and run hooks (sequential)
        for i, tc, args in to_execute:
            tc_id = getattr(tc, "id", None)
            if i in results_map:
                result = results_map[i]
                tool_results.append((i, self.adapter.make_tool_result_message(
                    tc.name, result, tool_call_id=tc_id,
                )))
                # Track dict-level errors
                if isinstance(result, dict) and result.get("status") == "error":
                    err_msg = result.get("message", "unknown error")
                    collected_errors.append(f"{tc.name}: {err_msg}")
                # Run interception hook
                intercept = self._on_tool_result_hook(tc.name, args, result)
                if intercept is not None:
                    tool_results.sort(key=lambda x: x[0])
                    return SubAgent._ToolExecResult(
                        tool_results=[r for _, r in tool_results],
                        intercepted=True,
                        intercept_text=intercept,
                    )
            elif i in errors_map:
                err_msg = errors_map[i]
                err_result = {"status": "error", "message": err_msg}
                tool_results.append((i, self.adapter.make_tool_result_message(
                    tc.name, err_result, tool_call_id=tc_id,
                )))
                collected_errors.append(f"{tc.name}: {err_msg}")
                self._event_bus.emit(
                    SUB_AGENT_ERROR,
                    agent=self.agent_id,
                    level="error",
                    msg=f"[{self.agent_id}] {tc.name} FAILED: {err_msg}",
                    data={
                        "tool_name": tc.name,
                        "tool_args": args,
                        "error": err_msg,
                    },
                )
                # Run hook on error result too
                intercept = self._on_tool_result_hook(tc.name, args, err_result)
                if intercept is not None:
                    tool_results.sort(key=lambda x: x[0])
                    return SubAgent._ToolExecResult(
                        tool_results=[r for _, r in tool_results],
                        intercepted=True,
                        intercept_text=intercept,
                    )

        # Sort by original index to preserve order
        tool_results.sort(key=lambda x: x[0])
        return SubAgent._ToolExecResult(
            tool_results=[r for _, r in tool_results],
        )

    # ------------------------------------------------------------------
    # Event bus listener — routes tool results to inbox
    # ------------------------------------------------------------------

    def _on_event(self, event: SessionEvent) -> None:
        """Event bus listener. No-op for sub-agents in sync-only mode.

        All tool results are returned inline — no async delivery via event bus.
        Kept as a subscription point for potential future use (e.g. external
        signals routed via event bus).
        """
        pass

    # ------------------------------------------------------------------
    # Public API — send message and wait for result
    # ------------------------------------------------------------------

    def send(
        self,
        content: str | dict,
        sender: str = "user",
        wait: bool = True,
        timeout: float = 300.0,
    ) -> dict | None:
        """Send a message to the sub-agent.

        Args:
            content: Message content
            sender: Message sender
            wait: If True, block until result. If False, fire-and-forget (returns None).
            timeout: Max time to wait for result (only used if wait=True)

        Returns:
            If wait=True: result dict {"text": ..., "failed": ..., "errors": [...]}
            If wait=False: None
        """
        reply_event = threading.Event() if wait else None
        msg = _make_message("request", sender, content, reply_event=reply_event)
        self.inbox.put(msg)

        if not wait:
            return None

        if not reply_event.wait(timeout=timeout):
            return {
                "text": f"Timeout after {timeout}s waiting for {self.agent_id}",
                "failed": True,
                "errors": ["timeout"],
            }
        return msg._reply_value if msg._reply_value is not None else {"text": "", "failed": True, "errors": ["no reply"]}

    def _deliver_result(self, msg: Message, result: dict) -> None:
        """Deliver result to a waiting caller (if any) or send to orchestrator."""
        if msg._reply_event:
            msg._reply_value = result
            msg._reply_event.set()
        else:
            # Fire-and-forget: send result to orchestrator so user gets notified
            text = result.get("text", "")
            if text:
                self.send_to_orchestrator(text)

    def send_to_orchestrator(self, content: str, priority: int = 1):
        """Send message to orchestrator's inbox.

        Args:
            content: Message text explaining what was done and recommendations.
            priority: Message priority (0=user, 1=subagent).
        """
        # Get orchestrator's inbox - set by orchestrator when creating subagent
        if hasattr(self, '_orchestrator_inbox') and self._orchestrator_inbox:
            msg = _make_message(
                "subagent_result",
                sender=self.agent_id,
                content=content,
            )
            # Put directly in orchestrator's inbox
            self._orchestrator_inbox.put((priority, msg.timestamp, msg))

    # ------------------------------------------------------------------
    # Deferred memory reviews
    # ------------------------------------------------------------------

    def _run_deferred_reviews(self, msg: Message) -> None:
        """Run deferred memory reviews after delivering the main result.

        Only executes if ``_has_deferred_reviews`` is True and memory is
        present in core memory (baked into system prompt at chat creation).

        The sub-agent's persistent LLM session continues after delivery, so this
        runs on the same thread — the orchestrator has already moved on.
        """
        if not self._has_deferred_reviews:
            return

        # Memory is now baked into core memory — check if store has content
        if not self._memory_store or not self._memory_scope:
            return
        has_memory = bool(
            self._memory_store.format_for_injection(
                scope=self._memory_scope,
                active_missions=self._get_active_missions(),
                include_review_instruction=False,
            )
        )
        if not has_memory:
            return

        self._event_bus.emit(
            DEBUG,
            agent=self.agent_id,
            level="debug",
            msg=f"[{self.agent_id}] Starting deferred memory review...",
        )

        try:
            review_prompt = (
                "Your main task is complete and the result has been delivered. "
                "Now review the memories that were injected into your request. "
                "Call review_memory(memory_id, stars, rating, criticism, suggestion, comment) "
                "for 1-3 memories you found relevant (or irrelevant). Only review memories you "
                "have NOT previously reviewed.\n\n"
                "When done reviewing, respond with just 'Done'."
            )

            # Reuse the persistent chat session — review context is fine
            # to keep in the sub-agent's history (provides continuity).
            # Ensure session exists (it should after _handle_request).
            if self._chat is None:
                return

            response = send_with_timeout(
                chat=self._chat,
                message=review_prompt,
                timeout_pool=self._timeout_pool,
                cancel_event=self._cancel_event,
                retry_timeout=30,
                agent_name=f"{self.agent_id}/Review",
                logger=logger,
            )
            self._track_usage(response)
            if hasattr(self._chat, "interaction_id") and self._chat.interaction_id:
                self._interaction_id = self._chat.interaction_id

            # Process up to 2 rounds of review_memory tool calls
            for _ in range(2):
                if not response.tool_calls:
                    break
                tool_results = []
                for tc in response.tool_calls:
                    if tc.name == "review_memory":
                        args = dict(tc.args) if tc.args else {}
                        args["_reviewer_agent_id"] = self.agent_id
                        result = self.tool_executor(
                            tc.name, args, getattr(tc, "id", None)
                        )
                        tool_results.append(
                            self.adapter.make_tool_result_message(
                                tc.name,
                                result,
                                tool_call_id=getattr(tc, "id", None),
                            )
                        )
                if not tool_results:
                    break
                response = send_with_timeout(
                    chat=self._chat,
                    message=tool_results,
                    timeout_pool=self._timeout_pool,
                    cancel_event=self._cancel_event,
                    retry_timeout=30,
                    agent_name=f"{self.agent_id}/Review",
                    logger=logger,
                )
                self._track_usage(response)
                if hasattr(self._chat, "interaction_id") and self._chat.interaction_id:
                    self._interaction_id = self._chat.interaction_id

            self._event_bus.emit(
                DEBUG,
                agent=self.agent_id,
                level="debug",
                msg=f"[{self.agent_id}] Deferred memory review complete",
            )
        except Exception as e:
            # Reviews are best-effort — never fail the delegation
            logger.debug(f"[{self.agent_id}] Deferred review failed: {e}")

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def _update_token_decomposition(self) -> None:
        """Recompute cached system prompt and tools token counts."""
        self._system_prompt_tokens = count_tokens(self._build_system_prompt())
        self._tools_tokens = count_tool_tokens(self._tool_schemas)
        self._token_decomp_dirty = False

    def _track_usage(self, response: LLMResponse) -> None:
        if self._token_decomp_dirty:
            self._update_token_decomposition()
        token_state = {
            "input": self._total_input_tokens,
            "output": self._total_output_tokens,
            "thinking": self._total_thinking_tokens,
            "cached": self._total_cached_tokens,
            "api_calls": self._api_calls,
        }
        track_llm_usage(
            response=response,
            token_state=token_state,
            agent_name=self.agent_id,
            last_tool_context=self._last_tool_context,
            system_tokens=self._system_prompt_tokens,
            tools_tokens=self._tools_tokens,
        )
        self._total_input_tokens = token_state["input"]
        self._total_output_tokens = token_state["output"]
        self._total_thinking_tokens = token_state["thinking"]
        self._total_cached_tokens = token_state["cached"]
        self._api_calls = token_state["api_calls"]
        self._latest_input_tokens = response.usage.input_tokens

    def get_token_usage(self) -> dict:
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "thinking_tokens": self._total_thinking_tokens,
            "cached_tokens": self._total_cached_tokens,
            "total_tokens": (
                self._total_input_tokens
                + self._total_output_tokens
                + self._total_thinking_tokens
            ),
            "api_calls": self._api_calls,
            "ctx_system_tokens": self._system_prompt_tokens,
            "ctx_tools_tokens": self._tools_tokens,
            "ctx_history_tokens": max(
                0,
                self._latest_input_tokens
                - self._system_prompt_tokens
                - self._tools_tokens,
            ),
            "ctx_total_tokens": self._latest_input_tokens,
        }

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _on_tool_result_hook(
        self, tool_name: str, tool_args: dict, result: dict
    ) -> str | None:
        """Hook called after each tool execution.

        If this returns a non-None string, the current request processing
        returns immediately with that string as the result text.
        """
        return None

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return sub-agent status for monitoring."""
        return {
            "agent_id": self.agent_id,
            "state": self._state.value,
            "idle": self.is_idle,
            "queue_depth": self.inbox.qsize(),
            "tokens": self.get_token_usage(),
        }

