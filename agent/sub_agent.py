"""
Sub-agent base class for all agents.

Every agent is an autonomous actor with an inbox (queue.Queue). Tools block
inline by default — the LLM always receives actual results, never "started"
acks. Agents stay responsive and persistent between requests.

Key concepts:
    - **Blocking by default**: all tool calls (including formerly-async ones
      like fetch_data) execute and return actual results before the LLM's
      next turn. Multiple non-sync tools dispatch in parallel via an implicit
      batch_sync barrier. No fire-and-forget, no polling loops.
    - **batch_sync meta-tool**: the LLM calls ``batch_sync(calls=[...])`` when
      it needs results from multiple tools before deciding. All inner tools
      run in parallel; the call blocks until all complete (or timeout).
    - **Event bus as coordination backbone**: tool results are emitted as
      SUB_AGENT_TOOL events for observability. The inbox still accepts
      messages that queue while tools execute.
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
from typing import Any, Callable, Optional, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .memory import MemoryStore

from .event_bus import (
    EventBus, SessionEvent, get_event_bus, set_event_bus,
    TOOL_STARTED, TOOL_RESULT, TOOL_ERROR,
    SUB_AGENT_TOOL, SUB_AGENT_ERROR, DEBUG,
    AGENT_STATE_CHANGE, TEXT_DELTA,
)
from .tool_timing import ToolTimer, stamp_tool_result


# ---------------------------------------------------------------------------
# AgentState — formal lifecycle state model
# ---------------------------------------------------------------------------

class AgentState(enum.Enum):
    """Lifecycle state of a sub-agent (or the orchestrator).

    SLEEPING ──(inbox message)──────────────► ACTIVE
    ACTIVE   ──(batch_sync wait)────────────► PENDING
    ACTIVE   ──(all done)──────────────────► SLEEPING
    PENDING  ──(batch results arrive)───────► ACTIVE

    | State      | Thread            | In-flight work | Can accept input |
    |------------|-------------------|---------------|-----------------|
    | ACTIVE     | Running           | Maybe         | No              |
    | PENDING    | Blocked on batch  | Yes           | No              |
    | SLEEPING   | Blocked on inbox  | None          | Yes             |

    Note: IDLE is vestigial — kept for backward compatibility with external
    state monitors, but no longer reachable in normal flow. All tools now
    block inline (no fire-and-forget async dispatch).
    """
    ACTIVE = "active"
    PENDING = "pending"
    IDLE = "idle"      # vestigial — see docstring
    SLEEPING = "sleeping"
from .llm import LLMAdapter, ChatSession, LLMResponse, FunctionSchema, ToolCall
from .logging import get_logger
from .loop_guard import LoopGuard, DupVerdict
from .agent_registry import SYNC_TOOLS
from .model_fallback import get_active_model
from .turn_limits import get_limit
from .truncation import trunc, trunc_items
from .llm_utils import send_with_timeout, track_llm_usage, _is_stale_interaction_error
from .token_counter import count_tokens, count_tool_tokens
import config


logger = get_logger()


# ---------------------------------------------------------------------------
# _BatchCollector — collects parallel tool results with barrier wait
# ---------------------------------------------------------------------------

class _BatchCollector:
    """Collects results from parallel async tool dispatches with barrier wait.

    Handles the race where a tool completes before ``add_task()`` is called
    by buffering early-arriving results in ``_early``.
    """

    def __init__(self, timeout: float, cancel_event: threading.Event | None):
        self._lock = threading.Lock()
        self._done = threading.Condition(self._lock)
        self._pending: set[str] = set()
        self._results: dict[str, dict] = {}
        self._errors: dict[str, str] = {}
        self._early: dict[str, tuple] = {}  # results arriving before add_task
        self._timeout = timeout
        self._cancel_event = cancel_event
        self._all_registered = threading.Event()

    def add_task(self, task_id: str) -> None:
        """Register a task_id. Handles early-arriving results."""
        with self._done:
            if task_id in self._early:
                result, error = self._early.pop(task_id)
                if error:
                    self._errors[task_id] = error
                else:
                    self._results[task_id] = result
                self._done.notify_all()
            else:
                self._pending.add(task_id)

    def finish_registration(self) -> None:
        """Signal that all tasks have been registered."""
        self._all_registered.set()

    def deliver(self, task_id: str, result: dict | None, error: str | None) -> bool:
        """Deliver a result. Returns True if task_id belongs to this batch."""
        with self._done:
            if task_id in self._pending:
                self._pending.discard(task_id)
                if error:
                    self._errors[task_id] = error
                else:
                    self._results[task_id] = result
                self._done.notify_all()
                return True
            if not self._all_registered.is_set():
                # Result arrived before add_task — buffer it
                self._early[task_id] = (result, error)
                return True
            return False

    def wait_all(self) -> tuple[dict, dict, set]:
        """Block until all complete, timeout, or cancel.

        Returns (results, errors, timed_out_task_ids).
        """
        deadline = time.monotonic() + self._timeout
        with self._done:
            while self._pending:
                if self._cancel_event and self._cancel_event.is_set():
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._done.wait(timeout=min(remaining, 0.5))
        with self._lock:
            return dict(self._results), dict(self._errors), set(self._pending)


# ---------------------------------------------------------------------------
# BATCH_SYNC_SCHEMA — meta-tool for scatter-gather barrier
# ---------------------------------------------------------------------------

BATCH_SYNC_SCHEMA = FunctionSchema(
    name="batch_sync",
    description=(
        "Execute multiple tools in parallel and wait for ALL results before "
        "returning. Use when you need results from several independent tools "
        "before deciding what to do next."
    ),
    parameters={
        "type": "object",
        "properties": {
            "calls": {
                "type": "array",
                "description": "List of tool calls to execute in parallel.",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {
                            "type": "string",
                            "description": "Tool name.",
                        },
                        "args": {
                            "type": "object",
                            "description": "Tool arguments.",
                        },
                    },
                    "required": ["tool"],
                },
            },
            "timeout": {
                "type": "number",
                "description": (
                    "Max seconds to wait. Partial results returned on timeout. "
                    "Default: 120."
                ),
            },
            "commentary": {
                "type": "string",
                "description": (
                    "Brief active-voice sentence describing what you are doing and why. "
                    "One sentence preferred, two max. Shown to the user in the chat."
                ),
            },
        },
        "required": ["calls", "commentary"],
    },
)


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
    # Synchronous reply channel (set by send_and_wait callers)
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
    """Base class for all actor-model sub-agents.

    Subclasses override:
        - ``_build_system_prompt()`` to customise the system prompt
        - ``_get_tool_schemas()`` to provide tool schemas
        - ``_on_tool_result_hook()`` for interception (e.g. clarification)
        - ``_post_response_hook()`` for side effects after LLM responds

    The sub-agent owns a single thread that reads from its inbox and processes
    messages sequentially. Tool calls are dispatched to background threads
    (async). The ``batch_sync`` meta-tool provides a barrier: multiple tools
    run in parallel and the call blocks until all complete.
    """

    # Subclasses set to True to enable deferred memory reviews after
    # the main result has been delivered back to the orchestrator.
    _has_deferred_reviews: bool = False

    # Inbox polling interval (seconds). Shorter = faster test turnaround.
    _inbox_timeout: float = 1.0

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
        self._tool_schemas.append(BATCH_SYNC_SCHEMA)
        self._event_bus = event_bus if event_bus is not None else get_event_bus()
        self._cancel_event = cancel_event

        # Inbox — the single coordination primitive
        self.inbox: queue.Queue[Message] = queue.Queue()

        # Persistent LLM session state
        self._interaction_id: str | None = None
        self._chat: ChatSession | None = None

        # In-flight async tools: task_id → {name, args, thread}
        self._active_tools: dict[str, dict] = {}
        self._active_tools_lock = threading.Lock()

        # Active batch_sync collector (set during _execute_batch_sync)
        self._active_batch: _BatchCollector | None = None

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

        # Memory injection — diff-based incremental injection across delegations
        self._memory_store: MemoryStore | None = memory_store
        self._memory_scope: str = memory_scope
        self._memory_epoch: int = -1  # -1 = never injected
        self._memory_snapshot: dict[str, str] = {}  # {memory_id: formatted_line}
        self._memory_reviews_pending: bool = False

        # Timeout pool for LLM calls
        self._timeout_pool = ThreadPoolExecutor(max_workers=1)

        # Subscribe to event bus for routing tool results to inbox
        self._event_bus.subscribe(self._on_event)

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
    def active_tool_count(self) -> int:
        with self._active_tools_lock:
            return len(self._active_tools)

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
            AGENT_STATE_CHANGE, agent=self.agent_id, level="debug",
            msg=f"[{self.agent_id}] {old.value} → {new_state.value}{suffix}",
            data={"agent_id": self.agent_id, "old": old.value, "new": new_state.value,
                  "reason": reason},
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
                logger.error(f"[{self.agent_id}] Unhandled error in message handler: {e}", exc_info=True)
                # If the message has a reply channel, unblock the caller
                if msg._reply_event:
                    msg._reply_value = {
                        "text": f"Internal error: {e}",
                        "failed": True,
                        "errors": [str(e)],
                    }
                    msg._reply_event.set()
            finally:
                # Determine next state based on in-flight async tools
                if self.active_tool_count > 0:
                    with self._active_tools_lock:
                        tool_names = [t["name"] for t in self._active_tools.values()]
                    shown, _ = trunc_items(tool_names, "items.tool_names_log")
                    ellipsis = "..." if len(tool_names) > len(shown) else ""
                    self._set_state(AgentState.IDLE,
                                    reason=f"waiting on {len(tool_names)} async tool(s): {', '.join(shown)}{ellipsis}")
                else:
                    self._set_state(AgentState.SLEEPING, reason="all tools done")

    def _handle_message(self, msg: Message) -> None:
        """Route message by type."""
        if msg.type == "cancel":
            self._handle_cancel(msg)
        elif msg.type in ("tool_result", "tool_error"):
            self._handle_tool_result(msg)
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
        content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        # Prepend incremental memory context if available
        memory_prefix = self._build_memory_prefix()
        if memory_prefix:
            content = f"{memory_prefix}\n\n{content}"
        response = self._llm_send(content)
        result = self._process_response(response, msg)
        self._deliver_result(msg, result)
        self._run_deferred_reviews(msg)

    def _handle_tool_result(self, msg: Message) -> None:
        """Background tool finished. Remove from active_tools, feed to LLM."""
        task_id = msg.reply_to
        with self._active_tools_lock:
            self._active_tools.pop(task_id, None)

        # Format the result for the LLM
        if msg.type == "tool_error":
            result_text = (
                f"Background task {task_id} FAILED:\n"
                f"{json.dumps(msg.content, default=str)}"
            )
        else:
            result_text = (
                f"Background task {task_id} completed:\n"
                f"{json.dumps(msg.content, default=str)}"
            )

        response = self._llm_send(result_text)
        # Process any follow-up tool calls from the LLM
        self._process_response(response, msg)

    def _handle_cancel(self, msg: Message) -> None:
        """Cancel active tools. Sub-agent stays alive."""
        if self._cancel_event:
            self._cancel_event.set()
        with self._active_tools_lock:
            cancelled = list(self._active_tools.keys())
            self._active_tools.clear()
        if cancelled:
            self._event_bus.emit(
                DEBUG, agent=self.agent_id, level="info",
                msg=f"[{self.agent_id}] Cancelled {len(cancelled)} active tools",
            )

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _llm_send(self, message: Any) -> LLMResponse:
        """Send a message to the LLM, reusing the persistent chat session."""
        if self._chat is None:
            self._chat = self.adapter.create_chat(
                model=get_active_model(self.model_name),
                system_prompt=self._build_system_prompt(),
                tools=self._tool_schemas or None,
                thinking="high",
                interaction_id=self._interaction_id,
            )

        try:
            response = send_with_timeout(
                chat=self._chat,
                message=message,
                timeout_pool=self._timeout_pool,
                cancel_event=self._cancel_event,
                retry_timeout=config.LLM_RETRY_TIMEOUT if hasattr(config, 'LLM_RETRY_TIMEOUT') else 180,
                agent_name=self.agent_id,
                logger=logger,
            )
        except Exception as exc:
            # Handle stale Interactions API session
            if self._interaction_id and _is_stale_interaction_error(exc):
                self._event_bus.emit(
                    DEBUG, agent=self.agent_id, level="warning",
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
                    retry_timeout=180,
                    agent_name=self.agent_id,
                    logger=logger,
                )
            else:
                raise

        self._track_usage(response)
        # Preserve interaction ID for session reuse
        if hasattr(self._chat, 'interaction_id') and self._chat.interaction_id:
            self._interaction_id = self._chat.interaction_id
        return response

    def _build_system_prompt(self) -> str:
        """Return the system prompt. Override in subclasses."""
        return self.system_prompt

    # ------------------------------------------------------------------
    # Memory injection — diff-based incremental injection
    # ------------------------------------------------------------------

    def _get_active_missions(self) -> set[str] | None:
        """Return active mission IDs for scope filtering. Override in DataOpsAgent."""
        return None

    def _build_memory_prefix(self) -> str | None:
        """Build a memory prefix for the current delegation request.

        First delegation: full ``[CONTEXT FROM LONG-TERM MEMORY]`` block.
        Subsequent delegations with changes: ``[MEMORY UPDATE]`` diff block.
        No changes: returns ``None`` (skip).
        """
        if not self._memory_store or not self._memory_scope:
            return None

        store = self._memory_store
        current_epoch = store.mutation_epoch

        # Fast path: no mutations since last injection
        if current_epoch == self._memory_epoch:
            return None

        active_missions = self._get_active_missions()
        epoch, new_snapshot = store.get_scoped_memory_ids(
            self._memory_scope, active_missions=active_missions,
        )

        if self._memory_epoch == -1:
            # First injection — full block
            if not new_snapshot:
                self._memory_epoch = epoch
                return None

            # Bucket by type prefix
            buckets: dict[str, list[str]] = {}
            for mid, line in new_snapshot.items():
                # line format: "[Category] [id] content"
                bracket_end = line.index("]") + 1
                category = line[1:bracket_end - 1]
                buckets.setdefault(category, []).append(f"- {line}")

            parts = ["[CONTEXT FROM LONG-TERM MEMORY]", "## Operational Knowledge"]
            category_headers = {
                "Preferences": "### Preferences",
                "Past Sessions": "### Past Sessions",
                "Lessons Learned": "### Lessons Learned",
                "Operational Reflections": "### Operational Reflections",
            }
            for category, header in category_headers.items():
                if category in buckets:
                    parts.append("")
                    parts.append(header)
                    parts.extend(buckets[category])
            # Any remaining categories not in the standard set
            for category, lines in buckets.items():
                if category not in category_headers:
                    parts.append("")
                    parts.append(f"### {category}")
                    parts.extend(lines)

            parts.append("")
            parts.append(
                "IMPORTANT: After completing your main task, you MUST call review_memory(memory_id, stars, comment) "
                "for at least 1 (up to 4) of the memories listed above that you have NOT previously reviewed. "
                "If you have already reviewed a memory (shown as \"Your previous review\" above), "
                "only re-review it if your opinion has substantially changed based on this session's experience. "
                "Do not re-submit the same or similar review.\n"
                "The comment MUST use this exact four-line format:\n"
                "(1) Rating: why this star count\n"
                "(2) Criticism: what's wrong or could be better\n"
                "(3) Suggestion: how to improve the memory\n"
                "(4) Comment: any extra observation"
            )
            parts.append("[END MEMORY CONTEXT]")

            self._memory_epoch = epoch
            self._memory_snapshot = dict(new_snapshot)
            self._memory_reviews_pending = True
            return "\n".join(parts)
        else:
            # Incremental diff
            old_ids = set(self._memory_snapshot.keys())
            new_ids = set(new_snapshot.keys())
            added = new_ids - old_ids
            removed = old_ids - new_ids

            if not added and not removed:
                # Content might have changed but same IDs — update snapshot
                self._memory_epoch = epoch
                self._memory_snapshot = dict(new_snapshot)
                return None

            parts = ["[MEMORY UPDATE]"]
            if added:
                parts.append("Added:")
                for mid in added:
                    parts.append(f"- {new_snapshot[mid]}")
            if removed:
                parts.append(f"Removed: {', '.join(removed)}")
            parts.append("[END MEMORY UPDATE]")

            self._memory_epoch = epoch
            self._memory_snapshot = dict(new_snapshot)
            self._memory_reviews_pending = True
            return "\n".join(parts)

    # ------------------------------------------------------------------
    # Response processing — batch_sync + async tool dispatch
    # ------------------------------------------------------------------

    def _process_response(self, response: LLMResponse, origin_msg: Message) -> dict:
        """Handle tool calls (batch_sync, sync, or blocking) and collect text output.

        Tool calls are partitioned into three groups:
          - batch_sync: explicit scatter-gather barrier (blocking, results inline)
          - sync: fast read-only tools from SYNC_TOOLS (blocking, results inline)
          - blocking: everything else (dispatched in parallel via implicit
            batch_sync barrier, blocks until all complete, results inline)

        All tools return actual results to the LLM — no "started" acks, no
        polling loops. This follows the "Default Serial" model: the LLM always
        gets real results before its next turn.

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
                        TEXT_DELTA, agent=self.agent_id, level="info",
                        msg=f"[{self.agent_id}] {response.text}",
                        data={"text": response.text + "\n\n", "commentary": True},
                    )

            if not response.tool_calls:
                break

            # Check cancel
            if self._cancel_event and self._cancel_event.is_set():
                return {"text": "Interrupted by user.", "failed": True, "errors": ["Interrupted"]}

            # Total call limit check
            stop_reason = guard.check_limit(len(response.tool_calls))
            if stop_reason:
                self._event_bus.emit(
                    DEBUG, agent=self.agent_id, level="debug",
                    msg=f"[{self.agent_id}] Stopping: {stop_reason}",
                )
                break

            # Partition tool calls into three groups
            batch_calls = [tc for tc in response.tool_calls if tc.name == "batch_sync"]
            sync_calls = [tc for tc in response.tool_calls
                          if tc.name != "batch_sync" and tc.name in SYNC_TOOLS]
            blocking_calls = [tc for tc in response.tool_calls
                              if tc.name != "batch_sync" and tc.name not in SYNC_TOOLS]

            tool_results = []

            # Process batch_sync calls (blocking — results returned inline)
            for tc in batch_calls:
                tc_id = getattr(tc, "id", None)
                batch_args = dict(tc.args) if tc.args else {}
                commentary = batch_args.pop("commentary", None)
                if commentary:
                    self._event_bus.emit(
                        TEXT_DELTA, agent=self.agent_id, level="info",
                        msg=f"[{self.agent_id}] {commentary}",
                        data={"text": commentary + "\n\n", "commentary": True},
                    )
                batch_result = self._execute_batch_sync(batch_args)

                if batch_result.get("status") == "intercepted":
                    return {
                        "text": batch_result["intercept_text"],
                        "failed": False,
                        "errors": [],
                    }

                # Record inner tool calls for loop guard
                inner_count = len(batch_args.get("calls", []))
                if inner_count:
                    guard.record_calls(inner_count)

                # Collect errors from batch results
                if batch_result.get("status") == "error":
                    collected_errors.append(f"batch_sync: {batch_result.get('message', '')}")
                else:
                    for entry in batch_result.get("results", []):
                        if entry.get("status") == "error":
                            err_msg = entry.get("result", {}).get("message", "")
                            collected_errors.append(f"{entry['tool']}: {err_msg}")

                tool_results.append(
                    self.adapter.make_tool_result_message(
                        "batch_sync", batch_result, tool_call_id=tc_id,
                    )
                )

            # Process sync calls (blocking — results returned inline)
            for tc in sync_calls:
                tc_id = getattr(tc, "id", None)
                args = dict(tc.args) if tc.args else {}
                commentary = args.pop("commentary", None)
                if commentary:
                    self._event_bus.emit(
                        TEXT_DELTA, agent=self.agent_id, level="info",
                        msg=f"[{self.agent_id}] {commentary}",
                        data={"text": commentary + "\n\n", "commentary": True},
                    )

                # Duplicate call tracking
                verdict = guard.record_tool_call(tc.name, args)
                if verdict.blocked:
                    result = {
                        "status": "blocked",
                        "_duplicate_warning": verdict.warning,
                        "message": f"Execution skipped — duplicate call #{verdict.count}",
                    }
                    tool_results.append(
                        self.adapter.make_tool_result_message(
                            tc.name, result, tool_call_id=tc_id,
                        )
                    )
                    continue

                # Emit TOOL_STARTED for console/event log visibility
                sync_task_id = f"sync_{uuid4().hex[:8]}"
                self._event_bus.emit(
                    TOOL_STARTED, agent=self.agent_id, level="debug",
                    msg=f"[{self.agent_id}] Sync tool: {tc.name}",
                    data={
                        "task_id": sync_task_id,
                        "tool_name": tc.name,
                        "tool_args": args,
                    },
                )

                timer = ToolTimer()
                try:
                    # Inject reviewer identity (same as _run_tool_background)
                    if tc.name == "review_memory":
                        args = {**args, "_reviewer_agent_id": self.agent_id}
                    with timer:
                        result = self.tool_executor(tc.name, args, tc_id)
                    if isinstance(result, dict):
                        stamp_tool_result(result, timer.elapsed_ms)

                    # Inject dup warning if applicable
                    if verdict.warning and isinstance(result, dict):
                        result["_duplicate_warning"] = verdict.warning

                    tool_results.append(
                        self.adapter.make_tool_result_message(
                            tc.name, result, tool_call_id=tc_id,
                        )
                    )

                    # Emit tool result event for console/event log
                    status = "error" if (isinstance(result, dict) and result.get("status") == "error") else "success"
                    self._event_bus.emit(
                        SUB_AGENT_TOOL, agent=self.agent_id,
                        msg=f"[{self.agent_id}] {tc.name} → {status} (sync)",
                        data={
                            "task_id": sync_task_id,
                            "tool_name": tc.name,
                            "tool_args": args,
                            "tool_result": result,
                            "status": status,
                            "elapsed_ms": timer.elapsed_ms,
                        },
                    )
                    # Emit for frontend display (SSE)
                    self._event_bus.emit(
                        TOOL_RESULT, agent=self.agent_id,
                        msg=f"[{self.agent_id}] {tc.name} → {status}",
                        data={"tool_name": tc.name, "status": status, "elapsed_ms": timer.elapsed_ms},
                    )

                    # Run interception hook
                    intercept = self._on_tool_result_hook(tc.name, args, result)
                    if intercept is not None:
                        return {
                            "text": intercept,
                            "failed": False,
                            "errors": [],
                        }
                except Exception as e:
                    err_result = {"status": "error", "message": str(e)}
                    stamp_tool_result(err_result, timer.elapsed_ms)
                    tool_results.append(
                        self.adapter.make_tool_result_message(
                            tc.name, err_result, tool_call_id=tc_id,
                        )
                    )
                    collected_errors.append(f"{tc.name}: {e}")
                    self._event_bus.emit(
                        SUB_AGENT_ERROR, agent=self.agent_id, level="error",
                        msg=f"[{self.agent_id}] {tc.name} FAILED (sync): {e}",
                        data={
                            "task_id": sync_task_id,
                            "tool_name": tc.name,
                            "tool_args": args,
                            "error": str(e),
                        },
                    )

            # Process blocking calls (implicit batch_sync — parallel dispatch, wait for all)
            if blocking_calls:
                # Pop commentary, check dup verdicts, partition blocked vs executable
                cleaned_calls = []
                blocking_verdicts: list[tuple[object, DupVerdict]] = []  # (tc, verdict)
                for tc in blocking_calls:
                    raw_args = dict(tc.args) if tc.args else {}
                    commentary = raw_args.pop("commentary", None)
                    if commentary:
                        self._event_bus.emit(
                            TEXT_DELTA, agent=self.agent_id, level="info",
                            msg=f"[{self.agent_id}] {commentary}",
                            data={"text": commentary + "\n\n", "commentary": True},
                        )
                    cleaned_args = {
                        k: v for k, v in raw_args.items()
                        if k != "_sync"  # backward compat strip
                    }
                    verdict = guard.record_tool_call(tc.name, cleaned_args)
                    blocking_verdicts.append((tc, verdict))
                    if verdict.blocked:
                        # Produce synthetic result immediately
                        tc_id = getattr(tc, "id", None)
                        result = {
                            "status": "blocked",
                            "_duplicate_warning": verdict.warning,
                            "message": f"Execution skipped — duplicate call #{verdict.count}",
                        }
                        tool_results.append(
                            self.adapter.make_tool_result_message(
                                tc.name, result, tool_call_id=tc_id,
                            )
                        )
                    else:
                        cleaned_calls.append({
                            "tool": tc.name,
                            "args": cleaned_args,
                        })

                # Execute non-blocked calls via batch
                if cleaned_calls:
                    implicit_batch_args = {"calls": cleaned_calls}
                    batch_result = self._execute_batch_sync(
                        implicit_batch_args, _skip_validation=True,
                    )

                    if batch_result.get("status") == "intercepted":
                        return {
                            "text": batch_result["intercept_text"],
                            "failed": False,
                            "errors": [],
                        }

                    # Record inner tool calls for loop guard
                    inner_count = len(cleaned_calls)
                    if inner_count:
                        guard.record_calls(inner_count)

                    # Collect errors from batch results
                    if batch_result.get("status") == "error":
                        collected_errors.append(f"implicit_batch: {batch_result.get('message', '')}")
                    else:
                        for entry in batch_result.get("results", []):
                            if entry.get("status") == "error":
                                err_msg = entry.get("result", {}).get("message", "")
                                collected_errors.append(f"{entry['tool']}: {err_msg}")

                    # Map results back to individual tool_call_ids, injecting dup warnings
                    result_entries = batch_result.get("results", [])
                    for (tc_entry, verdict_entry), entry in zip(
                        [(tc, v) for tc, v in blocking_verdicts if not v.blocked],
                        result_entries,
                    ):
                        tc_id = getattr(tc_entry, "id", None)
                        if entry.get("status") == "timed_out":
                            mapped_result = {"status": "error", "message": f"{tc_entry.name} timed out"}
                        elif entry.get("status") == "error":
                            mapped_result = entry.get("result", {"status": "error", "message": "unknown error"})
                        else:
                            mapped_result = entry.get("result", {})
                        if verdict_entry.warning and isinstance(mapped_result, dict):
                            mapped_result["_duplicate_warning"] = verdict_entry.warning
                        tool_results.append(
                            self.adapter.make_tool_result_message(
                                tc_entry.name, mapped_result, tool_call_id=tc_id,
                            )
                        )

            guard.record_calls(len(response.tool_calls))

            # Feed tool results back to LLM
            response = self._llm_send(tool_results)

        final_text = "\n".join(collected_text_parts)
        failed = bool(collected_errors) and not any(
            p for p in collected_text_parts if "success" in p.lower()
        )
        return {"text": final_text, "failed": failed, "errors": collected_errors}

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_dedup_key(name: str, args: dict) -> tuple:
        """Create a hashable key for async in-flight deduplication.

        Delegates to LoopGuard._dedup_key for consistent key generation.
        """
        return LoopGuard._dedup_key(name, args)

    def _dispatch_tool(self, name: str, args: dict, tool_call_id: str | None = None) -> str:
        """Dispatch tool to background thread (async). Returns task_id.

        Deduplicates: if an identical tool (same name + args) is already
        in-flight, returns the existing task_id instead of spawning a
        duplicate thread.
        """
        proposed_key = self._tool_dedup_key(name, args)
        task_id = f"t_{uuid4().hex[:8]}"

        with self._active_tools_lock:
            # Check for in-flight duplicate
            for existing_tid, info in self._active_tools.items():
                if self._tool_dedup_key(info["name"], info["args"]) == proposed_key:
                    self._event_bus.emit(
                        DEBUG, agent=self.agent_id, level="debug",
                        msg=(f"[{self.agent_id}] Suppressed duplicate async: "
                             f"{name} (already running as {existing_tid})"),
                    )
                    return existing_tid
            self._active_tools[task_id] = {"name": name, "args": args}

        self._event_bus.emit(
            TOOL_STARTED, agent=self.agent_id, level="debug",
            msg=f"[{self.agent_id}] Async tool: {name} → {task_id}",
            data={
                "task_id": task_id,
                "tool_name": name,
                "tool_args": args,
            },
        )

        thread = threading.Thread(
            target=self._run_tool_background,
            args=(task_id, name, args, tool_call_id),
            daemon=True,
            name=f"tool-{self.agent_id}-{task_id}",
        )
        thread.start()
        return task_id

    def _deliver_tool_result(
        self, task_id: str, result: dict | None, error: str | None
    ) -> None:
        """Deliver a tool result directly to inbox or active batch collector.

        Bypasses the event bus — sub-agents own their event lifecycle and don't
        need TOOL_RESULT events on the bus (which would cause duplicates).
        """
        # Batch-aware routing: deliver to collector if active
        if task_id and self._active_batch is not None:
            if error is not None:
                if self._active_batch.deliver(task_id, None, error):
                    return
            else:
                if self._active_batch.deliver(task_id, result or {}, None):
                    return

        # Normal delivery: put in inbox
        if error is not None:
            self.inbox.put(Message(
                id=f"msg_{uuid4().hex[:12]}",
                type="tool_error",
                sender="tool_runner",
                content=error,
                reply_to=task_id,
            ))
        else:
            self.inbox.put(Message(
                id=f"msg_{uuid4().hex[:12]}",
                type="tool_result",
                sender="tool_runner",
                content=result or {},
                reply_to=task_id,
            ))

    def _run_tool_background(
        self, task_id: str, name: str, args: dict, tool_call_id: str | None
    ) -> None:
        """Execute tool in background thread, deliver result directly."""
        timer = ToolTimer()
        try:
            # Inject reviewer identity for review_memory so the orchestrator
            # knows which agent authored the review (background threads don't
            # have the orchestrator's thread-local _active_agent_name).
            if name == "review_memory":
                args = {**args, "_reviewer_agent_id": self.agent_id}
            with timer:
                result = self.tool_executor(name, args, tool_call_id)
            if isinstance(result, dict):
                stamp_tool_result(result, timer.elapsed_ms)
            self._deliver_tool_result(task_id, result, None)

            status = "error" if (isinstance(result, dict) and result.get("status") == "error") else "success"
            self._event_bus.emit(
                SUB_AGENT_TOOL, agent=self.agent_id,
                msg=f"[{self.agent_id}] {name} → {status}",
                data={
                    "task_id": task_id,
                    "tool_name": name,
                    "tool_args": args,
                    "tool_result": result,
                    "status": status,
                    "elapsed_ms": timer.elapsed_ms,
                },
            )
            # Emit for frontend display (SSE)
            self._event_bus.emit(
                TOOL_RESULT, agent=self.agent_id,
                msg=f"[{self.agent_id}] {name} → {status}",
                data={"tool_name": name, "status": status, "elapsed_ms": timer.elapsed_ms},
            )
        except Exception as e:
            self._deliver_tool_result(task_id, None, str(e))
            self._event_bus.emit(
                SUB_AGENT_ERROR, agent=self.agent_id, level="error",
                msg=f"[{self.agent_id}] {name} FAILED: {e}",
                data={
                    "task_id": task_id,
                    "tool_name": name,
                    "tool_args": args,
                    "error": str(e),
                    "elapsed_ms": timer.elapsed_ms,
                },
            )

    # ------------------------------------------------------------------
    # batch_sync execution — scatter-gather barrier
    # ------------------------------------------------------------------

    def _execute_batch_sync(self, args: dict, *, _skip_validation: bool = False) -> dict:
        """Execute a batch_sync meta-tool call.

        Dispatches all inner tools in parallel, waits for all results (with
        timeout + cancel-awareness), returns combined results.

        Args:
            _skip_validation: If True, skip tool-name validation. Used by the
                implicit batch_sync path in _process_response where tools were
                already validated by the LLM API layer.
        """
        batch_start = time.monotonic()
        calls = args.get("calls", [])
        if not calls:
            return {"status": "error", "message": "batch_sync requires non-empty 'calls' list"}

        # Reject nesting
        if self._active_batch is not None:
            return {"status": "error", "message": "batch_sync cannot be nested"}

        if not _skip_validation:
            # Validate tool names — batch_sync itself is not allowed inside
            known_names = {s.name for s in self._tool_schemas}
            for i, call in enumerate(calls):
                tool_name = call.get("tool", "")
                if tool_name == "batch_sync":
                    return {"status": "error", "message": "batch_sync cannot be nested inside batch_sync"}
                if tool_name not in known_names:
                    return {"status": "error", "message": f"Unknown tool in calls[{i}]: {tool_name!r}"}

        timeout = args.get("timeout", get_limit("agent.batch_sync_timeout"))
        collector = _BatchCollector(timeout=float(timeout), cancel_event=self._cancel_event)

        # Set active batch BEFORE dispatching (prevents race with fast executors)
        self._active_batch = collector

        task_ids: list[str] = []
        call_info: list[dict] = []  # preserves input order for output

        try:
            # Dispatch all tools in parallel
            for call in calls:
                tool_name = call.get("tool", "")
                tool_args = dict(call.get("args", {}))
                # Strip commentary from inner tool args (already emitted at batch_sync level)
                tool_args.pop("commentary", None)
                task_id = self._dispatch_tool(tool_name, tool_args)
                collector.add_task(task_id)
                task_ids.append(task_id)
                call_info.append({"tool": tool_name, "args": tool_args, "task_id": task_id})

            collector.finish_registration()

            # Block until all complete, timeout, or cancel
            batch_tool_names = [c.get("tool", "") for c in calls]
            shown, _ = trunc_items(batch_tool_names, "items.tool_names_log")
            ellipsis = "..." if len(batch_tool_names) > len(shown) else ""
            self._set_state(AgentState.PENDING,
                            reason=f"batch_sync waiting on {len(calls)} tool(s): {', '.join(shown)}{ellipsis}")
            results, errors, timed_out = collector.wait_all()
            self._set_state(AgentState.ACTIVE, reason="batch_sync complete")
        finally:
            self._active_batch = None
            # Remove batch task_ids from _active_tools
            with self._active_tools_lock:
                for tid in task_ids:
                    self._active_tools.pop(tid, None)

        # Fire _on_tool_result_hook for each completed tool (preserves interception)
        for info in call_info:
            tid = info["task_id"]
            if tid in results:
                intercept = self._on_tool_result_hook(info["tool"], info["args"], results[tid])
                if intercept is not None:
                    return {"status": "intercepted", "intercept_text": intercept}
            elif tid in errors:
                # Hook sees error results too
                err_result = {"status": "error", "message": errors[tid]}
                intercept = self._on_tool_result_hook(info["tool"], info["args"], err_result)
                if intercept is not None:
                    return {"status": "intercepted", "intercept_text": intercept}

        # Build ordered output
        completed_count = 0
        timed_out_count = 0
        result_entries = []

        for info in call_info:
            tid = info["task_id"]
            entry = {"tool": info["tool"], "args": info["args"]}
            if tid in results:
                entry["status"] = "success"
                entry["result"] = results[tid]
                completed_count += 1
            elif tid in errors:
                entry["status"] = "error"
                entry["result"] = {"message": errors[tid]}
                completed_count += 1
            else:
                entry["status"] = "timed_out"
                entry["result"] = None
                timed_out_count += 1
            result_entries.append(entry)

        # Determine overall status
        if self._cancel_event and self._cancel_event.is_set():
            status = "cancelled"
        elif timed_out_count > 0:
            status = "partial"
        else:
            status = "completed"

        batch_elapsed = int((time.monotonic() - batch_start) * 1000)
        combined = {
            "status": status,
            "results": result_entries,
            "total": len(calls),
            "completed": completed_count,
            "timed_out": timed_out_count,
        }
        stamp_tool_result(combined, batch_elapsed)
        return combined

    # ------------------------------------------------------------------
    # Event bus listener — routes tool results to inbox
    # ------------------------------------------------------------------

    def _on_event(self, event: SessionEvent) -> None:
        """Event bus listener. Route relevant tool results to our inbox.

        Batch-aware: if an active batch_sync is in progress, tool results
        for batch tasks are routed to the collector instead of the inbox.
        """
        data = event.data or {}
        if data.get("target_agent") != self.agent_id:
            return

        task_id = data.get("task_id")

        # Batch-aware routing: deliver to collector if active
        if task_id and self._active_batch is not None:
            if event.type == TOOL_RESULT:
                if self._active_batch.deliver(task_id, data.get("result", {}), None):
                    return  # routed to collector, don't put in inbox
            elif event.type == TOOL_ERROR:
                if self._active_batch.deliver(task_id, None, data.get("error", "Unknown")):
                    return  # routed to collector

        if event.type == TOOL_RESULT and task_id:
            self.inbox.put(Message(
                id=f"msg_{uuid4().hex[:12]}",
                type="tool_result",
                sender="tool_runner",
                content=data.get("result", {}),
                reply_to=task_id,
            ))
        elif event.type == TOOL_ERROR and task_id:
            self.inbox.put(Message(
                id=f"msg_{uuid4().hex[:12]}",
                type="tool_error",
                sender="tool_runner",
                content=data.get("error", "Unknown error"),
                reply_to=task_id,
            ))

    # ------------------------------------------------------------------
    # Public API — send message and wait for result
    # ------------------------------------------------------------------

    def send(self, content: str | dict, sender: str = "user") -> None:
        """Send a fire-and-forget message to the sub-agent's inbox."""
        self.inbox.put(_make_message("request", sender, content))

    def send_and_wait(
        self,
        content: str | dict,
        sender: str = "user",
        timeout: float = 300.0,
    ) -> dict:
        """Send a message and block until the sub-agent produces a result.

        Returns the result dict: {"text": ..., "failed": ..., "errors": [...]}.
        """
        reply_event = threading.Event()
        msg = _make_message("request", sender, content, reply_event=reply_event)
        self.inbox.put(msg)
        if not reply_event.wait(timeout=timeout):
            return {
                "text": f"Timeout after {timeout}s waiting for {self.agent_id}",
                "failed": True,
                "errors": ["timeout"],
            }
        return msg._reply_value or {"text": "", "failed": True, "errors": ["no reply"]}

    def _deliver_result(self, msg: Message, result: dict) -> None:
        """Deliver result to a waiting caller (if any)."""
        if msg._reply_event:
            msg._reply_value = result
            msg._reply_event.set()

    # ------------------------------------------------------------------
    # Deferred memory reviews
    # ------------------------------------------------------------------

    def _run_deferred_reviews(self, msg: Message) -> None:
        """Run deferred memory reviews after delivering the main result.

        Only executes if ``_has_deferred_reviews`` is True and memory was
        injected during this request (tracked by ``_memory_reviews_pending``
        flag set by ``_build_memory_prefix()``). Falls back to scanning for
        the ``[CONTEXT FROM LONG-TERM MEMORY]`` marker in the message content
        for backward compatibility with orchestrator-injected memory.

        The sub-agent's persistent LLM session continues after delivery, so this
        runs on the same thread — the orchestrator has already moved on.
        """
        if not self._has_deferred_reviews:
            return

        # Check flag first (set by _build_memory_prefix), then fall back to
        # string scan for legacy orchestrator-injected memory
        has_memory = self._memory_reviews_pending
        if not has_memory:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            has_memory = "[CONTEXT FROM LONG-TERM MEMORY]" in content
        if not has_memory:
            return

        self._memory_reviews_pending = False

        self._event_bus.emit(
            DEBUG, agent=self.agent_id, level="debug",
            msg=f"[{self.agent_id}] Starting deferred memory review...",
        )

        try:
            review_prompt = (
                "Your main task is complete and the result has been delivered. "
                "Now review the memories that were injected into your request. "
                "Call review_memory(memory_id, stars, comment) for 1-3 memories "
                "you found relevant (or irrelevant). Only review memories you "
                "have NOT previously reviewed.\n"
                "The comment MUST use this exact four-line format:\n"
                "(1) Rating: why this star count\n"
                "(2) Criticism: what's wrong or could be better\n"
                "(3) Suggestion: how to improve the memory\n"
                "(4) Comment: any extra observation\n\n"
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
            if hasattr(self._chat, 'interaction_id') and self._chat.interaction_id:
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
                        result = self.tool_executor(tc.name, args, getattr(tc, "id", None))
                        tool_results.append(
                            self.adapter.make_tool_result_message(
                                tc.name, result, tool_call_id=getattr(tc, "id", None),
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
                if hasattr(self._chat, 'interaction_id') and self._chat.interaction_id:
                    self._interaction_id = self._chat.interaction_id

            self._event_bus.emit(
                DEBUG, agent=self.agent_id, level="debug",
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
            "ctx_history_tokens": max(0, self._latest_input_tokens - self._system_prompt_tokens - self._tools_tokens),
            "ctx_total_tokens": self._latest_input_tokens,
        }

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _on_tool_result_hook(
        self, tool_name: str, tool_args: dict, result: dict
    ) -> str | None:
        """Hook called after each tool execution within a batch_sync.

        If this returns a non-None string, the current request processing
        returns immediately with that string as the result text.
        """
        return None

    def _post_response_hook(self, result: dict) -> None:
        """Hook called after the sub-agent finishes processing a request."""
        pass

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return sub-agent status for monitoring."""
        with self._active_tools_lock:
            active = {
                tid: {"name": info["name"]}
                for tid, info in self._active_tools.items()
            }
        return {
            "agent_id": self.agent_id,
            "state": self._state.value,
            "idle": self.is_idle,
            "queue_depth": self.inbox.qsize(),
            "active_tools": active,
            "tokens": self.get_token_usage(),
        }


# ---------------------------------------------------------------------------
# Backward compatibility aliases
# ---------------------------------------------------------------------------

Actor = SubAgent
ActorState = AgentState
