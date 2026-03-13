"""
BaseAgent — unified agent lifecycle.

All agent types (orchestrator, envoy, viz, data_ops, etc.) inherit from this
single base class. The core loop is final — subclasses customize behavior
through construction parameters, _pre_request/_post_request hooks, and
_local_tools overrides.

Key concepts:
    - **Sync by default**: every tool call executes inline and returns actual
      results before the LLM's next turn.
    - **Native parallel tool calling**: when the LLM emits multiple tool calls
      in a single response and all are in ``_PARALLEL_SAFE_TOOLS``, they run
      concurrently via ThreadPoolExecutor. Otherwise they run sequentially.
    - **2-state lifecycle**: SLEEPING (waiting for inbox) and ACTIVE (processing).
    - **Persistent LLM session**: each agent keeps its chat session
      (and Interactions API ``interaction_id``) across messages.
"""

from __future__ import annotations

import enum
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .memory import MemoryStore
    from .session_context import SessionContext

from concurrent.futures import ThreadPoolExecutor

from .event_bus import (
    EventBus,
    SessionEvent,
    get_event_bus,
    set_event_bus,
    AGENT_STATE_CHANGE,
    CONTEXT_COMPACTION,
    DEBUG,
    LLM_CALL,
    MEMORY_EXTRACTION_DONE,
    MEMORY_INJECTED,
    SUB_AGENT_ERROR,
    TEXT_DELTA,
    TOOL_CALL,
    TOOL_RESULT,
)
from .llm import LLMService, LLMResponse, ChatSession, FunctionSchema, ToolCall
from .llm_session_mixin import COMPACTION_PROMPT as _COMPACTION_PROMPT
from .llm_utils import (
    send_with_timeout,
    send_with_timeout_stream,
    track_llm_usage,
    _is_stale_interaction_error,
)
from .logging import get_logger
from .loop_guard import LoopGuard
from .tool_caller import ToolCaller
from .token_counter import count_tokens, count_tool_tokens
from .tool_timing import ToolTimer, stamp_tool_result
from .truncation import trunc_items
from .turn_limits import get_limit
from pathlib import Path as _Path
import config as _config

logger = get_logger()


# ---------------------------------------------------------------------------
# AgentState — formal lifecycle state model
# ---------------------------------------------------------------------------


class AgentState(enum.Enum):
    """Lifecycle state of an agent.

    SLEEPING ──(inbox message)──────────────► ACTIVE
    ACTIVE   ──(all done)──────────────────► SLEEPING
    """

    ACTIVE = "active"
    SLEEPING = "sleeping"


# ---------------------------------------------------------------------------
# Message type constants
# ---------------------------------------------------------------------------

MSG_REQUEST = "request"
MSG_TOOL_RESULT = "tool_result"
MSG_TOOL_ERROR = "tool_error"
MSG_CANCEL = "cancel"
MSG_USER_INPUT = "user_input"
MSG_DELEGATION_RESULT = "delegation_result"


# ---------------------------------------------------------------------------
# Message — the universal inbox item
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A message delivered to an agent's inbox.

    Attributes:
        id:        Unique message ID.
        type:      One of MSG_REQUEST, MSG_TOOL_RESULT, MSG_TOOL_ERROR,
                   MSG_CANCEL, MSG_USER_INPUT, MSG_DELEGATION_RESULT.
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
# BaseAgent — unified agent with inbox and persistent LLM session
# ---------------------------------------------------------------------------


class BaseAgent:
    """Unified base class for all agents.

    Subclasses set:
        - ``agent_type`` class attribute — identity string
        - ``config_key`` property — for config resolution (defaults to agent_type)
        - ``_PARALLEL_SAFE_TOOLS`` — set of tool names safe for concurrent execution
        - ``_local_tools`` — instance-level tool handler overrides

    Subclasses may override:
        - ``_pre_request(msg)`` — transform message before LLM send
        - ``_post_request(msg, result)`` — side effects after LLM responds
        - ``_handle_message(msg)`` — message routing (must call super for processing)
        - ``_get_guard_limits()`` — per-agent loop guard limits
    """

    # Identity — must be set by subclasses
    agent_type: str = ""

    # Tools safe for concurrent execution. Subclasses override with their
    # read-only / side-effect-free tools.
    _PARALLEL_SAFE_TOOLS: set[str] = set()

    # Whether this agent runs deferred memory reviews after delivering results
    _has_deferred_reviews: bool = False

    # Inbox polling interval (seconds)
    _inbox_timeout: float = 1.0

    # LLM timeout/retry overrides. None = use global defaults.
    _llm_retry_timeout: float | None = None
    _llm_max_retries: int | None = None
    _llm_reset_threshold: int | None = None

    def __init__(
        self,
        agent_id: str,
        service: LLMService,
        system_prompt: str,
        *,
        tool_schemas: list[FunctionSchema] | None = None,
        session_ctx: SessionContext | None = None,
        event_bus: EventBus | None = None,
        cancel_event: threading.Event | None = None,
        streaming: bool = False,
        memory_store: MemoryStore | None = None,
        memory_scope: str = "",
    ):
        self.agent_id = agent_id
        self.service = service
        self.system_prompt = system_prompt
        self.session_ctx = session_ctx

        self._tool_schemas = list(tool_schemas) if tool_schemas else []

        # Auto-resolve tools from agent registry if none provided.
        # Keyed by config_key (e.g. "data_io" → "ctx:data_io" → tool list).
        # This ensures agents get their tools without explicit wiring —
        # just add entries to tool_registry.json under the agent's config_key.
        if not self._tool_schemas:
            try:
                from .agent_registry import AGENT_CALL_REGISTRY
                from .tools import get_function_schemas_for_agent
                ctx_key = f"ctx:{self.config_key}"
                tool_names = AGENT_CALL_REGISTRY.get(ctx_key)
                if tool_names:
                    self._tool_schemas = get_function_schemas_for_agent(
                        names=list(tool_names), agent_ctx=ctx_key
                    )
            except Exception:
                pass  # Fall through to no tools — matches current behavior

        self._event_bus = event_bus if event_bus is not None else get_event_bus()
        self._cancel_event = cancel_event
        self._streaming = streaming

        # Per-instance tool handler overrides (used by EurekaAgent, MemoryAgent)
        self._local_tools: dict[str, Callable] = {}

        # Intrinsic tools — available to all agents by default
        self._local_tools["vision"] = self._handle_vision
        self._local_tools["web_search"] = self._handle_web_search

        # Intrinsic schemas — always present regardless of tool_registry.json
        self._intrinsic_schemas = [
            FunctionSchema(
                name="vision",
                description="Analyze an image file using vision.",
                parameters={
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to the image file (PNG, JPG, WebP).",
                        },
                        "question": {
                            "type": "string",
                            "description": "What to analyze or look for in the image.",
                        },
                    },
                    "required": ["image_path", "question"],
                },
            ),
            FunctionSchema(
                name="web_search",
                description="Search the web for real-world context and information.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]

        # Append intrinsic schemas (always present, not filtered by tool_registry.json)
        self._tool_schemas.extend(self._intrinsic_schemas)

        # Inbox — the single coordination primitive
        self.inbox: queue.Queue[Message] = queue.Queue()

        # Lifecycle
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._idle = threading.Event()
        self._idle.set()  # starts idle
        self._state = AgentState.SLEEPING

        # Memory store for core memory injection
        self._memory_store: MemoryStore | None = memory_store
        self._memory_scope: str = memory_scope

        # Resolve provider/model from config
        self._provider, resolved_model, self._base_url = _config.resolve_agent_model(
            self.config_key
        )
        self.model_name = resolved_model

        # Persistent LLM session state
        self._interaction_id: str | None = None
        self._chat: ChatSession | None = None

        # Persistent loop guard
        self._guard: LoopGuard | None = None

        # Token tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_thinking_tokens = 0
        self._total_cached_tokens = 0
        self._api_calls = 0
        self._last_tool_context = "send_message"
        self._system_prompt_tokens = 0
        self._tools_tokens = 0
        self._token_decomp_dirty = True
        self._latest_input_tokens = 0

        # Streaming state
        self._text_already_streamed = False
        self._intermediate_text_streamed = False
        self._message_seq = 0  # increments per LLM response within a round

        # Timeout pool for LLM calls
        self._timeout_pool = ThreadPoolExecutor(max_workers=1)

        # Orchestrator inbox reference (for fire-and-forget sends)
        self._orchestrator_inbox: queue.Queue | None = None

        # Subscribe to memory mutation events for core memory refresh
        if self._memory_store:
            self._event_bus.subscribe(self._on_memory_event)

    # ------------------------------------------------------------------
    # Intrinsic tool handlers
    # ------------------------------------------------------------------

    _MIME_BY_EXT: dict[str, str] = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }

    def _handle_vision(self, ctx, args, caller) -> dict:
        """Analyze an image file using the model's vision capability."""
        image_path = args.get("image_path")
        question = args.get("question", "Describe what you see in this image.")

        if not image_path:
            return {"status": "error", "message": "Missing required parameter: image_path"}

        path = _Path(image_path)
        if not path.is_file():
            return {"status": "error", "message": f"Image file not found: {image_path}"}

        image_bytes = path.read_bytes()
        mime = self._MIME_BY_EXT.get(path.suffix.lower(), "image/png")

        response = self.service.generate_vision(question, image_bytes, mime_type=mime)
        if not response.text:
            return {
                "status": "error",
                "message": "Vision analysis returned no response — vision provider may not be configured.",
            }
        return {"status": "ok", "analysis": response.text}

    def _handle_web_search(self, ctx, args, caller) -> dict:
        """Search the web for information."""
        query = args.get("query")
        if not query:
            return {"status": "error", "message": "Missing required parameter: query"}
        resp = self.service.web_search(query)
        if not resp.text:
            return {
                "status": "error",
                "message": "Web search returned no results. The web search provider may not be configured.",
            }
        return {"status": "ok", "results": resp.text}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config_key(self) -> str:
        """Config key for model resolution. Override in subclasses."""
        return self.agent_type

    @property
    def is_idle(self) -> bool:
        return self._idle.is_set()

    @property
    def state(self) -> AgentState:
        return self._state

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the agent's main loop thread."""
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
        if self._thread:
            self._thread.join(timeout=timeout)
        self._timeout_pool.shutdown(wait=False)

    def _set_state(self, new_state: AgentState, reason: str = "") -> None:
        """Transition to a new state, keeping _idle in sync."""
        old = self._state
        if old == new_state:
            return
        self._state = new_state
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
    # Main loop (final — do not override)
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Wait for messages, process them. Agent persists between messages."""
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
                err_desc = str(e) or repr(e)
                logger.error(
                    f"[{self.agent_id}] Unhandled error in message handler: {err_desc}",
                    exc_info=True,
                )
                self._event_bus.emit(
                    DEBUG,
                    agent=self.agent_id,
                    level="error",
                    msg=f"[{self.agent_id}] Unhandled error: {err_desc}",
                )
                if msg._reply_event:
                    msg._reply_value = {
                        "text": f"Internal error: {err_desc}",
                        "failed": True,
                        "errors": [err_desc],
                    }
                    msg._reply_event.set()
            finally:
                self._set_state(AgentState.SLEEPING, reason="all done")

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _handle_message(self, msg: Message) -> None:
        """Route message by type. Subclasses may override for routing."""
        if msg.type == MSG_CANCEL:
            self._handle_cancel(msg)
        elif msg.type in (MSG_REQUEST, MSG_USER_INPUT):
            self._handle_request(msg)
        else:
            logger.warning(f"[{self.agent_id}] Unknown message type: {msg.type}")

    def _handle_request(self, msg: Message) -> None:
        """Send request to LLM, process response with tool calls."""
        from datetime import datetime, timezone

        max_calls, dup_free, dup_hard = self._get_guard_limits()
        self._guard = LoopGuard(
            max_total_calls=max_calls,
            dup_free_passes=dup_free,
            dup_hard_block=dup_hard,
        )
        content = self._pre_request(msg)
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        content = f"[Current time: {current_time}]\n\n{content}"
        response = self._llm_send(content)
        result = self._process_response(response)
        self._post_request(msg, result)
        self._deliver_result(msg, result)
        if self._has_deferred_reviews:
            self._run_deferred_reviews(msg)

    def _handle_cancel(self, msg: Message) -> None:
        """Cancel active tools. Agent stays alive."""
        if self._cancel_event:
            self._cancel_event.set()

    def _get_guard_limits(self) -> tuple[int, int, int]:
        """Return (max_total_calls, dup_free_passes, dup_hard_block)."""
        return (
            get_limit("sub_agent.max_total_calls"),
            get_limit("sub_agent.dup_free_passes"),
            get_limit("sub_agent.dup_hard_block"),
        )

    # ------------------------------------------------------------------
    # Response processing — native parallel tool dispatch
    # ------------------------------------------------------------------

    @dataclass
    class _ToolExecResult:
        """Container for tool execution results."""
        tool_results: list
        intercepted: bool = False
        intercept_text: str = ""

    def _process_response(self, response: LLMResponse) -> dict:
        """Handle tool calls and collect text output.

        Returns a result dict: {"text": ..., "failed": ..., "errors": [...]}.
        """
        guard = self._guard or LoopGuard(
            max_total_calls=get_limit("sub_agent.max_total_calls"),
            dup_free_passes=get_limit("sub_agent.dup_free_passes"),
            dup_hard_block=get_limit("sub_agent.dup_hard_block"),
        )
        collected_text_parts: list[str] = []
        collected_errors: list[str] = []

        while True:
            if response.text:
                collected_text_parts.append(response.text)
                if response.tool_calls:
                    if not (self._streaming and self._intermediate_text_streamed):
                        self._event_bus.emit(
                            TEXT_DELTA,
                            agent=self.agent_id,
                            level="info",
                            msg=f"[{self.agent_id}] {response.text}",
                            data={"text": response.text + "\n\n", "commentary": True},
                        )
                    self._intermediate_text_streamed = False

            if not response.tool_calls:
                break

            if self._cancel_event and self._cancel_event.is_set():
                return {
                    "text": "Interrupted by user.",
                    "failed": True,
                    "errors": ["Interrupted"],
                }

            stop_reason = guard.check_limit(len(response.tool_calls))
            if stop_reason:
                self._event_bus.emit(
                    DEBUG,
                    agent=self.agent_id,
                    level="debug",
                    msg=f"[{self.agent_id}] Stopping: {stop_reason}",
                )
                break

            all_parallel_safe = (
                len(response.tool_calls) > 1
                and self._PARALLEL_SAFE_TOOLS
                and all(
                    tc.name in self._PARALLEL_SAFE_TOOLS
                    for tc in response.tool_calls
                )
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
                if result.tool_results and self._chat:
                    self._chat.commit_tool_results(result.tool_results)
                return {
                    "text": result.intercept_text,
                    "failed": False,
                    "errors": [],
                }

            guard.record_calls(len(response.tool_calls))

            if (
                len(collected_errors) >= 2
                and collected_errors[-1] == collected_errors[-2]
            ):
                logger.warning(
                    "[%s] Same error repeated, breaking early: %s",
                    self.agent_id,
                    collected_errors[-1],
                )
                break

            response = self._llm_send(result.tool_results)

        final_text = "\n".join(collected_text_parts)
        has_errors = bool(collected_errors)
        no_useful_output = not final_text.strip()
        return {
            "text": final_text,
            "failed": has_errors and no_useful_output,
            "errors": collected_errors,
        }

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_single_tool(
        self,
        tc: ToolCall,
        guard: LoopGuard,
        collected_errors: list[str],
    ) -> tuple[Any, bool, str]:
        """Execute a single tool call.

        Returns (result_msg, intercepted, intercept_text).
        """
        from .tool_handlers import TOOL_REGISTRY

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

        args.pop("_sync", None)

        verdict = guard.record_tool_call(tc.name, args)
        if verdict.blocked:
            result = {
                "status": "blocked",
                "_duplicate_warning": verdict.warning,
                "message": f"Execution skipped — duplicate call #{verdict.count}",
            }
            msg = self.service.make_tool_result(
                tc.name, result, tool_call_id=tc_id,
                provider=self._provider,
            )
            return msg, False, ""

        # Emit tool_call event for Activity panel + Console
        self._event_bus.emit(
            TOOL_CALL,
            agent=self.agent_id,
            level="info",
            msg=f"[{self.agent_id}] Calling {tc.name}",
            data={"tool_name": tc.name, "tool_args": args},
        )

        timer = ToolTimer()
        try:
            # Resolve handler: local tools take priority
            handler = self._local_tools.get(tc.name) or TOOL_REGISTRY.get(tc.name)
            if handler is None:
                raise KeyError(f"Unknown tool: {tc.name}")

            # Inject reviewer identity for review_memory
            if tc.name == "xhelio__review_memory":
                args = {**args, "_reviewer_agent_id": self.agent_id}

            caller = ToolCaller(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                tool_call_id=tc_id,
            )
            with timer:
                result = handler(self.session_ctx, args, caller)

            if isinstance(result, dict):
                stamp_tool_result(result, timer.elapsed_ms)

            # Emit tool_result event for Activity panel + Console
            status = result.get("status", "success") if isinstance(result, dict) else "success"
            self._event_bus.emit(
                TOOL_RESULT,
                agent=self.agent_id,
                level="info" if status != "error" else "warning",
                msg=f"[{self.agent_id}] {tc.name} -> {status}",
                data={
                    "tool_name": tc.name,
                    "tool_result": result,
                    "status": status,
                },
            )

            if verdict.warning and isinstance(result, dict):
                result["_duplicate_warning"] = verdict.warning

            # Check for intercept sentinel
            if isinstance(result, dict) and result.get("intercept"):
                intercept_text = result.get("text", "")
                result_msg = self.service.make_tool_result(
                    tc.name, result, tool_call_id=tc_id,
                    provider=self._provider,
                )
                return result_msg, True, intercept_text

            result_msg = self.service.make_tool_result(
                tc.name, result, tool_call_id=tc_id,
                provider=self._provider,
            )

            if isinstance(result, dict) and result.get("status") == "error":
                err_msg = result.get("message", "unknown error")
                collected_errors.append(f"{tc.name}: {err_msg}")

            # Run interception hook
            intercept = self._on_tool_result_hook(tc.name, args, result)
            if intercept is not None:
                return result_msg, True, intercept

            return result_msg, False, ""

        except Exception as e:
            err_result = {"status": "error", "message": str(e)}
            stamp_tool_result(err_result, timer.elapsed_ms)
            result_msg = self.service.make_tool_result(
                tc.name, err_result, tool_call_id=tc_id,
                provider=self._provider,
            )
            collected_errors.append(f"{tc.name}: {e}")
            self._event_bus.emit(
                SUB_AGENT_ERROR,
                agent=self.agent_id,
                level="error",
                msg=f"[{self.agent_id}] {tc.name} FAILED: {e}",
                data={
                    "tool_name": tc.name,
                    "tool_args": args,
                    "error": str(e),
                },
            )
            return result_msg, False, ""

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
                return BaseAgent._ToolExecResult(
                    tool_results=tool_results,
                    intercepted=True,
                    intercept_text=intercept_text,
                )
        return BaseAgent._ToolExecResult(tool_results=tool_results)

    def _execute_tools_parallel(
        self,
        tool_calls: list[ToolCall],
        guard: LoopGuard,
        collected_errors: list[str],
    ) -> _ToolExecResult:
        """Run multiple tool calls concurrently via ThreadPoolExecutor."""
        from concurrent.futures import as_completed
        from .tool_handlers import TOOL_REGISTRY

        timeout = get_limit("agent.parallel_tool_timeout")

        # Phase 1: Pre-check duplicates (sequential — guard not thread-safe)
        to_execute: list[tuple[int, ToolCall, dict]] = []
        tool_results: list[tuple[int, Any]] = []

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
                tool_results.append((i, self.service.make_tool_result(
                    tc.name, result, tool_call_id=tc_id,
                    provider=self._provider,
                )))
            else:
                to_execute.append((i, tc, args))

        if not to_execute:
            tool_results.sort(key=lambda x: x[0])
            return BaseAgent._ToolExecResult(
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

        results_map: dict[int, Any] = {}
        errors_map: dict[int, str] = {}

        def _run_one(index: int, tc: ToolCall, args: dict):
            if tc.name == "xhelio__review_memory":
                args = {**args, "_reviewer_agent_id": self.agent_id}

            handler = self._local_tools.get(tc.name) or TOOL_REGISTRY.get(tc.name)
            if handler is None:
                raise KeyError(f"Unknown tool: {tc.name}")

            tc_id = getattr(tc, 'id', None)
            caller = ToolCaller(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                tool_call_id=tc_id,
            )
            timer = ToolTimer()
            with timer:
                result = handler(self.session_ctx, args, caller)
            if isinstance(result, dict):
                stamp_tool_result(result, timer.elapsed_ms)
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
                for future, idx in futures.items():
                    if future not in done:
                        errors_map[idx] = "Timed out"
            except TimeoutError:
                for future, idx in futures.items():
                    if idx not in results_map and idx not in errors_map:
                        errors_map[idx] = "Timed out"
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        # Phase 3: Build result messages (sequential)
        for i, tc, args in to_execute:
            tc_id = getattr(tc, "id", None)
            if i in results_map:
                result = results_map[i]
                tool_results.append((i, self.service.make_tool_result(
                    tc.name, result, tool_call_id=tc_id,
                    provider=self._provider,
                )))
                if isinstance(result, dict) and result.get("status") == "error":
                    err_msg = result.get("message", "unknown error")
                    collected_errors.append(f"{tc.name}: {err_msg}")
                # Check intercept
                if isinstance(result, dict) and result.get("intercept"):
                    tool_results.sort(key=lambda x: x[0])
                    return BaseAgent._ToolExecResult(
                        tool_results=[r for _, r in tool_results],
                        intercepted=True,
                        intercept_text=result.get("text", ""),
                    )
            elif i in errors_map:
                err_msg = errors_map[i]
                err_result = {"status": "error", "message": err_msg}
                tool_results.append((i, self.service.make_tool_result(
                    tc.name, err_result, tool_call_id=tc_id,
                    provider=self._provider,
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

        tool_results.sort(key=lambda x: x[0])
        return BaseAgent._ToolExecResult(
            tool_results=[r for _, r in tool_results],
        )

    # ------------------------------------------------------------------
    # LLM communication
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build the system prompt. Override for memory injection etc."""
        return self.system_prompt

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

    def _get_active_missions(self) -> list[str] | None:
        """Return active mission IDs for scope filtering. Override in subclasses."""
        return None

    def _on_memory_mutated(self) -> None:
        """Refresh core memory when long-term memory changes mid-session."""
        if self._chat is not None:
            self._chat.update_system_prompt(self._build_core_memory())
            # Report how many memories were injected
            count = 0
            if self._memory_store:
                scope = self._memory_scope or "generic"
                enabled = self._memory_store.get_enabled()
                count = sum(1 for m in enabled if scope in m.scopes and m.type != "review")
            self._event_bus.emit(
                MEMORY_INJECTED,
                agent=self.agent_id,
                level="info",
                msg=f"[{self.agent_id}] Memory refreshed: {count} memories",
                data={"memory_count": count, "scope": self._memory_scope or "generic"},
            )

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

        def summarizer(text: str) -> str:
            response = self.service.get_adapter(
                self._provider, self._base_url
            ).generate(
                model=self.model_name,
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
        """Send a message to the LLM, reusing the persistent chat session.

        When streaming=True, emits TEXT_DELTA events as tokens arrive.
        When streaming=False, blocks until the full response is ready.
        """
        if self._chat is None:
            self._chat = self.service.get_adapter(
                self._provider, self._base_url
            ).create_chat(
                model=self.model_name,
                system_prompt=self._build_core_memory(),
                tools=self._tool_schemas or None,
                thinking="high",
                interaction_id=self._interaction_id,
            )

        self._check_and_compact()

        retry_timeout = self._llm_retry_timeout
        if retry_timeout is None:
            retry_timeout = (
                _config.LLM_RETRY_TIMEOUT
                if hasattr(_config, "LLM_RETRY_TIMEOUT")
                else 180
            )

        try:
            if self._streaming:
                response = self._llm_send_streaming(message, retry_timeout)
            else:
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
                self._chat = self.service.get_adapter(
                    self._provider, self._base_url
                ).create_chat(
                    model=self.model_name,
                    system_prompt=self._build_system_prompt(),
                    tools=self._tool_schemas or None,
                    thinking="high",
                )
                if self._streaming:
                    response = self._llm_send_streaming(message, retry_timeout)
                else:
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

    def _llm_send_streaming(
        self, message: Any, retry_timeout: float
    ) -> LLMResponse:
        """Streaming LLM send — emits TEXT_DELTA events as tokens arrive."""
        self._message_seq += 1
        seq = self._message_seq
        first_chunk = True

        def _on_chunk(text_delta: str) -> None:
            nonlocal first_chunk
            data: dict = {"text": text_delta, "streaming": True}
            if first_chunk:
                data["message_seq"] = seq
                first_chunk = False
            self._event_bus.emit(
                TEXT_DELTA,
                agent=self.agent_id,
                level="info",
                msg=text_delta,
                data=data,
            )

        response = send_with_timeout_stream(
            chat=self._chat,
            message=message,
            timeout_pool=self._timeout_pool,
            cancel_event=self._cancel_event,
            retry_timeout=retry_timeout,
            agent_name=self.agent_id,
            logger=logger,
            on_chunk=_on_chunk,
            on_reset=self._on_reset,
            max_retries=self._llm_max_retries,
            reset_threshold=self._llm_reset_threshold,
        )

        # Mark that text was already streamed
        if response.text:
            if response.tool_calls:
                self._intermediate_text_streamed = True
            else:
                self._text_already_streamed = True

        return response

    def _on_reset(self, chat, failed_message):
        """Rollback reset: new chat, drop failed turn, inject context."""
        from agent.llm_session_mixin import (
            summarize_tool_calls,
            summarize_tool_results,
        )
        from agent.llm.interface import ToolResultBlock, ToolCallBlock

        iface = chat.interface

        # Summarize tool calls from last assistant turn
        parts = []
        last_asst = iface.last_assistant_entry()
        if last_asst:
            for block in last_asst.content:
                if isinstance(block, ToolCallBlock):
                    args_str = ", ".join(
                        f"{k}={repr(v)[:80]}" for k, v in block.args.items()
                    )
                    parts.append(f"- {block.name}({args_str})")
        tool_summary = "\n".join(parts) if parts else "(no tool calls found)"
        result_summary = summarize_tool_results(failed_message)

        # Drop failed turn
        iface.drop_trailing(lambda e: e.role == "assistant")
        iface.drop_trailing(
            lambda e: e.role == "user"
            and all(isinstance(b, ToolResultBlock) for b in e.content)
        )

        self._event_bus.emit(
            LLM_CALL,
            agent=self.agent_id,
            level="warning",
            msg=f"[{self.agent_id}] Session rollback — new chat ({len(iface.entries)} entries kept)",
        )

        self._chat = self.service.create_session(
            system_prompt=self._build_system_prompt(),
            tools=self._tool_schemas or None,
            model=self.model_name,
            thinking="high",
            tracked=False,
            provider=self._provider,
            interface=iface,
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

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def _update_token_decomposition(self) -> None:
        """Recompute cached system prompt and tools token counts."""
        self._system_prompt_tokens = count_tokens(self._build_system_prompt())
        self._tools_tokens = count_tool_tokens(self._tool_schemas)
        self._token_decomp_dirty = False

    def _track_usage(self, response: LLMResponse) -> None:
        """Accumulate token usage from an LLMResponse."""
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
        if response.usage:
            self._latest_input_tokens = response.usage.input_tokens

    def get_token_usage(self) -> dict:
        """Return token usage summary."""
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
    # Public API for session persistence
    # ------------------------------------------------------------------

    def get_chat_state(self) -> dict:
        """Serialize current chat session for persistence.

        Returns an empty dict if no chat session exists.
        """
        if self._chat is None:
            return {}
        try:
            return {
                "messages": self._chat.interface.to_dict(),
            }
        except Exception:
            return {}

    def restore_chat(self, state: dict) -> None:
        """Restore or create a chat session from saved state.

        If state contains messages, attempts to resume. On failure
        (or empty state), creates a fresh session.
        """
        messages = state.get("messages")
        if messages:
            try:
                self._chat = self.service.resume_session(state)
                return
            except Exception:
                pass
        # Fresh session
        self._chat = self.service.create_session(
            system_prompt=self.system_prompt,
            tools=self._tool_schemas,
            model=self.model_name,
            thinking="high",
            tracked=False,
        )

    def restore_token_state(self, state: dict) -> None:
        """Restore cumulative token counters from a saved session."""
        self._total_input_tokens = state.get("input_tokens", 0)
        self._total_output_tokens = state.get("output_tokens", 0)
        self._total_thinking_tokens = state.get("thinking_tokens", 0)
        self._total_cached_tokens = state.get("cached_tokens", 0)
        self._api_calls = state.get("api_calls", 0)

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return agent status for monitoring."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "state": self._state.value,
            "idle": self.is_idle,
            "queue_depth": self.inbox.qsize(),
            "tokens": self.get_token_usage(),
        }

    # ------------------------------------------------------------------
    # Hooks (overridable by subclasses)
    # ------------------------------------------------------------------

    def _pre_request(self, msg: Message) -> str:
        """Transform message content before sending to LLM.

        Returns the content string to send.
        """
        return msg.content if isinstance(msg.content, str) else json.dumps(msg.content)

    def _post_request(self, msg: Message, result: dict) -> None:
        """Called after _process_response, before _deliver_result.

        Override in subclasses for post-processing.
        """
        pass

    def _on_tool_result_hook(
        self, tool_name: str, tool_args: dict, result: dict
    ) -> str | None:
        """Hook called after each tool execution.

        If this returns a non-None string, the current request processing
        returns immediately with that string as the result text.
        """
        return None

    def _run_deferred_reviews(self, msg: Message) -> None:
        """Run deferred memory reviews after delivering the main result.

        No-op by default. Override in subclasses that set
        ``_has_deferred_reviews = True``.
        """
        pass

    def _on_memory_event(self, event: SessionEvent) -> None:
        """Event bus listener for memory mutation events."""
        if event.type == MEMORY_EXTRACTION_DONE:
            self._on_memory_mutated()

    # ------------------------------------------------------------------
    # Result delivery
    # ------------------------------------------------------------------

    def _deliver_result(self, msg: Message, result: dict) -> None:
        """Deliver result to a waiting caller, or notify orchestrator."""
        if msg._reply_event:
            msg._reply_value = result
            msg._reply_event.set()
        else:
            text = result.get("text", "")
            if text:
                self.send_to_orchestrator(text)

    def send_to_orchestrator(self, content: str, priority: int = 1) -> None:
        """Send message to orchestrator's inbox (fire-and-forget)."""
        if self._orchestrator_inbox:
            msg = _make_message(
                MSG_DELEGATION_RESULT,
                self.agent_id,
                content,
            )
            self._orchestrator_inbox.put((priority, msg.timestamp, msg))

    # ------------------------------------------------------------------
    # Public send API
    # ------------------------------------------------------------------

    def send(
        self,
        content: str | dict,
        sender: str = "user",
        wait: bool = True,
        timeout: float = 300.0,
    ) -> dict | None:
        """Send a message to the agent.

        Args:
            content: Message content
            sender: Message sender
            wait: If True, block until result. If False, fire-and-forget.
            timeout: Max time to wait for result (only used if wait=True)

        Returns:
            If wait=True: result dict {"text": ..., "failed": ..., "errors": [...]}
            If wait=False: None
        """
        reply_event = threading.Event() if wait else None
        msg = _make_message(MSG_REQUEST, sender, content, reply_event=reply_event)
        self.inbox.put(msg)

        if not wait:
            return None

        if not reply_event.wait(timeout=timeout):
            return {
                "text": f"Timeout after {timeout}s waiting for {self.agent_id}",
                "failed": True,
                "errors": ["timeout"],
            }
        if msg._reply_value is None:
            return {"text": "", "failed": True, "errors": ["no reply"]}
        return msg._reply_value
