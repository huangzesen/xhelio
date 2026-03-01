"""
Structured EventBus — single source of truth for all session activity.

Replaces five parallel recording mechanisms:
1. Python logger (scattered logger.debug/info/warning/error calls)
2. OperationsLog (structured JSON for pipeline replay)
3. Chat history (adapter-specific LLM conversation turns)
4. _emit_event() callback (SSE bridge for frontend)
5. _session_events (ephemeral dicts for MemoryAgent)

Architecture:
    bus.emit() → SessionEvent → listeners[]
      ├── DebugLogListener     → Python logger FileHandler + ConsoleHandler
      ├── SSEEventListener     → SSE bridge for frontend live-log
      ├── OperationsLogListener → OperationsLog.record() for pipeline replay
      └── DisplayLogBuilder    → display_log.json for chat replay

    bus.span() → context manager that collects sub-events into a parent event
      Sub-events are dispatched live to console-only listeners, then the
      parent event (with children) is dispatched to all listeners on close.
"""

import contextvars
import json
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4


# ---- Event type constants ----

# Conversation
USER_MESSAGE = "user_message"
AGENT_RESPONSE = "agent_response"

# Tool lifecycle
TOOL_CALL = "tool_call"
TOOL_STARTED = "tool_started"      # Async tool dispatched to background
TOOL_RESULT = "tool_result"

# Data pipeline
DATA_FETCHED = "data_fetched"
DATA_COMPUTED = "data_computed"
DATA_CREATED = "data_created"
RENDER_EXECUTED = "render_executed"
MPL_RENDER_EXECUTED = "mpl_render_executed"
JSX_RENDER_EXECUTED = "jsx_render_executed"
PLOT_ACTION = "plot_action"

# Routing
DELEGATION = "delegation"
DELEGATION_DONE = "delegation_done"
SUB_AGENT_TOOL = "sub_agent_tool"
SUB_AGENT_ERROR = "sub_agent_error"

# Planning
PLAN_CREATED = "plan_created"
PLAN_TASK = "plan_task"
PLAN_COMPLETED = "plan_completed"
PROGRESS = "progress"

# LLM
THINKING = "thinking"
LLM_CALL = "llm_call"
LLM_RESPONSE = "llm_response"

# Token usage
TOKEN_USAGE = "token_usage"

# Data source access
CDF_FILE_QUERY = "cdf_file_query"
CDF_CACHE_HIT = "cdf_cache_hit"
CDF_DOWNLOAD = "cdf_download"
CDF_METADATA_SYNC = "cdf_metadata_sync"
PPI_FETCH = "ppi_fetch"

# Errors
FETCH_ERROR = "fetch_error"
HIGH_NAN = "high_nan"
CUSTOM_OP_FAILURE = "custom_op_failure"
RECOVERY = "recovery"
RENDER_ERROR = "render_error"
TOOL_ERROR = "tool_error"

# Session lifecycle
SESSION_START = "session_start"
SESSION_END = "session_end"

# Memory
MEMORY_EXTRACTION_START = "memory_extraction_start"
MEMORY_EXTRACTION_DONE = "memory_extraction_done"
MEMORY_EXTRACTION_ERROR = "memory_extraction_error"
MEMORY_ACTION = "memory_action"

# Eureka (scientific findings)
EUREKA_EXTRACTION_START = "eureka_extraction_start"
EUREKA_EXTRACTION_DONE = "eureka_extraction_done"
EUREKA_EXTRACTION_ERROR = "eureka_extraction_error"
EUREKA_FINDING = "eureka_finding"

# Pipeline registration (via MemoryAgent)
PIPELINE_REGISTERED = "pipeline_registered"

# Insight (multimodal plot analysis)
INSIGHT_RESULT = "insight_result"
INSIGHT_FEEDBACK = "insight_feedback"

# Async delegation
DELEGATION_ASYNC_STARTED = "delegation_async_started"
DELEGATION_ASYNC_COMPLETED = "delegation_async_completed"

# Control center (turnless orchestrator)
WORK_REGISTERED = "work_registered"
WORK_CANCELLED = "work_cancelled"
USER_AMENDMENT = "user_amendment"

# Short-term memory
STM_COMPACTION = "stm_compaction"

# Persistent event loop (turnless orchestrator)
TEXT_DELTA = "text_delta"
ROUND_START = "round_start"
ROUND_END = "round_end"
# Semantic aliases — a "cycle" is one sleep→sleep interval
CYCLE_START = ROUND_START
CYCLE_END = ROUND_END
SESSION_TITLE = "session_title"

# Knowledge/Bootstrap
CATALOG_SEARCH = "catalog_search"
METADATA_FETCH = "metadata_fetch"
BOOTSTRAP_PROGRESS = "bootstrap_progress"

# Agent lifecycle
AGENT_STATE_CHANGE = "agent_state_change"

# Catch-all for debug-level events that don't need a specific type
DEBUG = "debug"

# Secondary logging (console-only variants of tool lifecycle events)
TOOL_CALL_LOG = "tool_call_log"        # Console-only tool call log (secondary path)
TOOL_RESULT_LOG = "tool_result_log"    # Console-only tool result log (secondary path)
TOOL_ERROR_LOG = "tool_error_log"      # Console-only tool error from sub-agent loops
ERROR_LOG = "error_log"                # General error log (console+memory, not display)

# Data source access (user-facing variant)
CDF_DOWNLOAD_WARN = "cdf_download_warn"  # Large CDF download warning (surfaces to display)


# ---- Default tags registry ----
# Tags are split into two layers:
#   1. INFRASTRUCTURE_TAGS — WHERE events go (display, memory, console, pipeline, token)
#   2. Context tags (ctx:*) — WHO sees them in session history
# Context tags are computed from AGENT_INFORMED_REGISTRY (agent_registry.py)
# based on which tools link to each event type.

from .agent_registry import AGENT_INFORMED_REGISTRY

ALL_CTX_TAGS = frozenset({"ctx:mission", "ctx:viz_plotly", "ctx:viz_mpl", "ctx:viz_jsx", "ctx:dataops", "ctx:planner", "ctx:orchestrator"})

# ---- Infrastructure tags (non-ctx routing) ----

INFRASTRUCTURE_TAGS: dict[str, frozenset[str]] = {
    # Conversation
    USER_MESSAGE:       frozenset({"display", "memory", "console"}),
    AGENT_RESPONSE:     frozenset({"display", "memory", "console"}),
    # Tool lifecycle
    TOOL_CALL:          frozenset({"display", "memory", "console"}),
    TOOL_STARTED:       frozenset({"display", "console"}),
    TOOL_RESULT:        frozenset({"display", "memory", "console"}),
    TOOL_ERROR:         frozenset({"display", "memory", "console"}),
    TOOL_CALL_LOG:      frozenset({"console"}),
    TOOL_RESULT_LOG:    frozenset({"console"}),
    TOOL_ERROR_LOG:     frozenset(),
    ERROR_LOG:          frozenset({"memory", "console"}),
    # Data pipeline
    DATA_FETCHED:       frozenset({"display", "memory", "pipeline", "console"}),
    DATA_COMPUTED:      frozenset({"memory", "pipeline", "console"}),
    DATA_CREATED:       frozenset({"pipeline", "console"}),
    RENDER_EXECUTED:    frozenset({"display", "memory", "pipeline", "console"}),
    MPL_RENDER_EXECUTED: frozenset({"display", "memory", "pipeline", "console"}),
    JSX_RENDER_EXECUTED: frozenset({"display", "memory", "pipeline", "console"}),
    PLOT_ACTION:        frozenset({"pipeline", "console"}),
    CUSTOM_OP_FAILURE:  frozenset({"display", "memory", "pipeline", "console"}),
    # Routing
    DELEGATION:         frozenset({"display", "memory", "console"}),
    DELEGATION_DONE:    frozenset({"display", "memory", "console"}),
    DELEGATION_ASYNC_STARTED:   frozenset({"display", "console"}),
    DELEGATION_ASYNC_COMPLETED: frozenset({"display", "memory", "console"}),
    # Control center (turnless orchestrator)
    WORK_REGISTERED:    frozenset({"display", "console"}),
    WORK_CANCELLED:     frozenset({"display", "memory", "console"}),
    USER_AMENDMENT:     frozenset({"display", "memory", "console"}),
    SUB_AGENT_TOOL:     frozenset({"console"}),
    SUB_AGENT_ERROR:    frozenset({"memory", "console"}),
    # Planning
    PLAN_CREATED:       frozenset({"display", "console"}),
    PLAN_TASK:          frozenset({"display", "console"}),
    PLAN_COMPLETED:     frozenset({"display", "console"}),
    PROGRESS:           frozenset({"display", "console"}),
    # LLM
    THINKING:           frozenset({"display", "console"}),
    LLM_CALL:           frozenset({"console"}),
    LLM_RESPONSE:       frozenset({"console"}),
    # Token usage
    TOKEN_USAGE:        frozenset({"token", "console"}),
    # Session lifecycle
    SESSION_START:      frozenset({"console"}),
    SESSION_END:        frozenset({"display", "console"}),
    # Memory
    MEMORY_EXTRACTION_START: frozenset({"display", "console"}),
    MEMORY_EXTRACTION_DONE:  frozenset({"display", "console"}),
    MEMORY_EXTRACTION_ERROR: frozenset({"display", "console"}),
    MEMORY_ACTION:      frozenset({"console"}),
    PIPELINE_REGISTERED: frozenset({"display", "console"}),
    # Insight
    INSIGHT_RESULT:     frozenset({"display", "console"}),
    INSIGHT_FEEDBACK:   frozenset({"display", "memory", "console"}),
    # Eureka
    EUREKA_EXTRACTION_START: frozenset({"console"}),
    EUREKA_EXTRACTION_DONE:  frozenset({"display", "console"}),
    EUREKA_EXTRACTION_ERROR: frozenset({"console"}),
    EUREKA_FINDING:          frozenset({"display", "console"}),
    # Short-term memory
    STM_COMPACTION:     frozenset({"display", "console"}),
    # Persistent event loop (turnless orchestrator)
    TEXT_DELTA:          frozenset({"display"}),
    ROUND_START:         frozenset({"display"}),
    ROUND_END:           frozenset({"display"}),
    SESSION_TITLE:       frozenset({"display"}),
    # Errors
    FETCH_ERROR:        frozenset({"display", "memory", "console"}),
    HIGH_NAN:           frozenset({"display", "console"}),
    RECOVERY:           frozenset({"console"}),
    RENDER_ERROR:       frozenset({"display", "memory", "console"}),
    # Agent lifecycle
    AGENT_STATE_CHANGE: frozenset({"console"}),
    # Debug
    DEBUG:              frozenset({"console"}),
    # Data source access
    CDF_FILE_QUERY:     frozenset(),
    CDF_CACHE_HIT:      frozenset(),
    CDF_DOWNLOAD:       frozenset({"console"}),
    CDF_DOWNLOAD_WARN:  frozenset({"display", "console"}),
    CDF_METADATA_SYNC:  frozenset(),
    PPI_FETCH:          frozenset(),
    CATALOG_SEARCH:     frozenset(),
    METADATA_FETCH:     frozenset(),
    BOOTSTRAP_PROGRESS: frozenset(),
}

# ---- Context visibility classes ----

# Events every agent should see (user intent, errors, state changes)
UNIVERSAL_CTX_EVENTS: frozenset[str] = frozenset({
    USER_MESSAGE, AGENT_RESPONSE,
    SUB_AGENT_TOOL, SUB_AGENT_ERROR,
    CUSTOM_OP_FAILURE,
    STM_COMPACTION,
    AGENT_STATE_CHANGE,
})

# Events only orchestrator/planner should see
ROUTING_ONLY_EVENTS: frozenset[str] = frozenset({
    DELEGATION, DELEGATION_DONE,
    DELEGATION_ASYNC_STARTED, DELEGATION_ASYNC_COMPLETED,
    INSIGHT_RESULT,
    INSIGHT_FEEDBACK,
    WORK_REGISTERED, WORK_CANCELLED, USER_AMENDMENT,
    TEXT_DELTA, ROUND_START, ROUND_END, SESSION_TITLE,
})

# ---- Event-to-tool linkage for derived ctx:* tags ----
# Maps pipeline/data events to the tools that produce or consume them.
# Used by _tool_linked_ctx_tags() to derive which agents see these events.

_EVENT_TOOL_LINKS: dict[str, frozenset[str]] = {
    DATA_FETCHED:    frozenset({"fetch_data", "list_fetched_data"}),
    FETCH_ERROR:     frozenset({"fetch_data", "list_fetched_data"}),
    DATA_COMPUTED:   frozenset({"custom_operation", "list_fetched_data"}),
    DATA_CREATED:    frozenset({"store_dataframe", "list_fetched_data"}),
    RENDER_EXECUTED: frozenset({"render_plotly_json", "manage_plot"}),
    MPL_RENDER_EXECUTED: frozenset({"generate_mpl_script", "manage_mpl_output"}),
    JSX_RENDER_EXECUTED: frozenset({"generate_jsx_component", "manage_jsx_output"}),
    RENDER_ERROR:    frozenset({"render_plotly_json", "manage_plot"}),
    PLOT_ACTION:     frozenset({"manage_plot", "render_plotly_json"}),
}


def _tool_linked_ctx_tags(event_type: str) -> frozenset[str]:
    """Derive ctx:* tags from AGENT_INFORMED_REGISTRY (live query)."""
    linked = _EVENT_TOOL_LINKS.get(event_type)
    if not linked:
        return frozenset()
    return frozenset(
        ctx for ctx, tools in AGENT_INFORMED_REGISTRY.items()
        if linked & tools
    )


def _resolve_tags(event_type: str) -> frozenset[str]:
    """Resolve tags for an event type: static infrastructure + dynamic ctx:*.

    Called on every emit() so that ctx:* tags reflect the current state
    of AGENT_INFORMED_REGISTRY (which can be mutated at runtime by
    manage_tool_logs).
    """
    infra = INFRASTRUCTURE_TAGS.get(event_type, frozenset())
    if event_type in UNIVERSAL_CTX_EVENTS:
        ctx = ALL_CTX_TAGS
    elif event_type in ROUTING_ONLY_EVENTS:
        ctx = frozenset({"ctx:planner", "ctx:orchestrator"})
    else:
        ctx = _tool_linked_ctx_tags(event_type)
    return infra | ctx


# ---- SessionEvent ----

@dataclass(frozen=True)
class SessionEvent:
    """A single structured event in the session.

    Fields:
        id: Session-unique event ID (e.g. "evt_0001").
        type: Event type constant (e.g. "tool_call", "data_fetched").
        ts: ISO 8601 timestamp (UTC, millisecond precision).
        agent: Source agent name.
        level: Log level (debug/info/warning/error).
        summary: Short one-liner (<=120 chars) for sub-agent reasoning.
        details: Full context, multi-line OK.
        data: Structured machine-readable payload (unchanged).
        tags: Routing tags (infrastructure + ctx:*).
        span_id: If this event is part of a span, the span's ID.
        children: Sub-events (only on span-closing parent events).
    """
    id: str
    type: str
    ts: str
    agent: str
    level: str
    summary: str
    details: str
    data: dict
    tags: frozenset
    span_id: str = ""
    children: tuple = ()

    @property
    def msg(self) -> str:
        """Backward-compat property: returns summary."""
        return self.summary


class EventBus:
    """Per-session event bus with synchronous listener dispatch.

    Thread-safe: emit() and subscribe() use a lock.
    Supports span() context manager for grouping sub-events.
    """

    def __init__(self, session_id: str = ""):
        self._events: list[SessionEvent] = []
        self._lock = threading.Lock()
        self._listeners: list[Callable[[SessionEvent], None]] = []
        self.session_id = session_id
        self._next_event_id: int = 0
        # Thread-local span stacks: {span_id: list[dict]}
        self._active_spans: dict[str, list[dict]] = {}
        self._span_lock = threading.Lock()

    def emit(
        self,
        type: str,
        *,
        agent: str = "orchestrator",
        level: str = "debug",
        msg: str = "",
        summary: str = "",
        details: str = "",
        data: Optional[dict] = None,
        span_id: str = "",
        children: tuple = (),
    ) -> SessionEvent:
        """Create, store, and dispatch a SessionEvent.

        Tags are resolved automatically from the DEFAULT_TAGS registry
        based on event type. To route an event differently, use the
        correct event type (e.g. TOOL_CALL_LOG for console-only logging).

        Args:
            type: Event type constant (e.g. TOOL_CALL, DELEGATION).
            agent: Source agent name.
            level: Log level (debug/info/warning/error).
            msg: Deprecated — use summary instead. If summary is not
                 provided, msg is used as summary and auto-details are
                 generated from data via the formatter registry.
            summary: Short one-liner (<=120 chars).
            details: Full context, multi-line OK.
            data: Structured payload.
            span_id: Links this event to a parent span.
            children: Sub-events (only on span-closing parent events).

        Returns:
            The created SessionEvent.
        """
        effective_data = data or {}

        # Backward compat: if caller used msg= but not summary=,
        # derive summary/details from formatter or fall back to msg.
        if not summary and msg:
            # Stash msg in data so formatters can access it
            if "_msg" not in effective_data:
                effective_data = {**effective_data, "_msg": msg}
            from .event_formatters import format_event
            summary, details = format_event(type, agent, effective_data, list(children) if children else None)
            # If formatter returned empty, fall back to raw msg
            if not summary:
                from .truncation import trunc
                summary = trunc(msg, "console.summary")
            if not details:
                details = msg
        elif not summary:
            # No msg and no summary — use formatter with data only
            from .event_formatters import format_event
            summary, details = format_event(type, agent, effective_data, list(children) if children else None)

        # If inside an active span, collect as sub-event
        # and only dispatch to console listeners (live progress).
        active_span = self._get_active_span()
        if active_span is not None:
            # Assign event ID inside the lock for thread-safety
            with self._lock:
                self._next_event_id += 1
                event_id = f"evt_{self._next_event_id:04d}"
                event = SessionEvent(
                    id=event_id,
                    type=type,
                    ts=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    agent=agent,
                    level=level,
                    summary=summary,
                    details=details,
                    data=effective_data,
                    tags=_resolve_tags(type),
                    span_id=span_id,
                    children=children,
                )
                self._events.append(event)
                listeners = list(self._listeners)
            active_span.append({
                "type": event.type,
                "ts": event.ts,
                "agent": event.agent,
                "level": event.level,
                "summary": event.summary,
                "details": event.details,
                "data": event.data,
            })
            # Only dispatch to console-tagged listeners (live sub-event progress)
            for listener in listeners:
                if getattr(listener, '_console_only', False):
                    try:
                        listener(event)
                    except Exception:
                        pass
            return event

        # Normal dispatch to all listeners
        with self._lock:
            self._next_event_id += 1
            event_id = f"evt_{self._next_event_id:04d}"
            event = SessionEvent(
                id=event_id,
                type=type,
                ts=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                agent=agent,
                level=level,
                summary=summary,
                details=details,
                data=effective_data,
                tags=_resolve_tags(type),
                span_id=span_id,
                children=children,
            )
            self._events.append(event)
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                pass  # Never let a listener break the emitter
        return event

    def _get_active_span(self) -> list | None:
        """Return the innermost active span's collector, or None."""
        with self._span_lock:
            if not self._active_spans:
                return None
            # Return the most recently added span (LIFO)
            # In practice spans don't nest deeply, but support it.
            last_key = list(self._active_spans.keys())[-1]
            return self._active_spans[last_key]

    @contextmanager
    def span(
        self,
        type: str,
        *,
        agent: str = "orchestrator",
        level: str = "info",
        data: Optional[dict] = None,
    ):
        """Wrap a tool execution in a span.

        Sub-events emitted inside the span are:
        1. Streamed live to console-only listeners (so the user sees progress).
        2. Collected as children on the span-closing parent event.

        On span close, a consolidated parent event is emitted to ALL listeners
        with summary/details auto-generated from the formatter registry and
        the collected children.

        Usage:
            with bus.span(DATA_FETCHED, agent="orchestrator", data={...}) as span_id:
                # ... emit sub-events (CDF_FILE_QUERY, CDF_DOWNLOAD, etc.)
                # They will be captured and also streamed live to console.

        Yields:
            span_id (str) — a unique identifier for this span.
        """
        span_id = uuid4().hex[:12]
        collector: list[dict] = []
        with self._span_lock:
            self._active_spans[span_id] = collector
        try:
            yield span_id
        finally:
            with self._span_lock:
                self._active_spans.pop(span_id, None)

            effective_data = data or {}
            # Build summary/details from formatter registry
            from .event_formatters import format_event
            summary, details = format_event(type, agent, effective_data, collector)

            # Emit consolidated parent event with children
            self.emit(
                type,
                agent=agent,
                level=level,
                summary=summary,
                details=details,
                data=effective_data,
                span_id="",
                children=tuple(collector),
            )

    def subscribe(self, listener: Callable[[SessionEvent], None]) -> None:
        """Register a listener called synchronously on each emit()."""
        with self._lock:
            self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[SessionEvent], None]) -> None:
        """Remove a previously registered listener."""
        with self._lock:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

    def get_events(
        self,
        *,
        types: Optional[set[str]] = None,
        tags: Optional[set[str]] = None,
        since_index: int = 0,
    ) -> list[SessionEvent]:
        """Return filtered events.

        Args:
            types: If set, only return events with type in this set.
            tags: If set, only return events that have at least one matching tag.
            since_index: Skip events before this index.

        Returns:
            Filtered list of SessionEvents.
        """
        with self._lock:
            events = self._events[since_index:]
        result = []
        for e in events:
            if types and e.type not in types:
                continue
            if tags and not (e.tags & frozenset(tags)):
                continue
            result.append(e)
        return result

    def get_events_by_ids(self, ids: set[str]) -> list[SessionEvent]:
        """Return events matching the given IDs, preserving order."""
        with self._lock:
            return [e for e in self._events if e.id in ids]

    def clear(self) -> None:
        """Remove all stored events."""
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


# ---- ContextVar singleton ----

_bus_var: contextvars.ContextVar[Optional[EventBus]] = contextvars.ContextVar(
    "_bus_var", default=None
)

# Module-level fallback for code that runs before any session is created
_fallback_bus: Optional[EventBus] = None
_fallback_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Return the EventBus for the current context.

    Falls back to a module-level singleton if no context-specific bus is set.
    This allows module-level code (knowledge/, data_ops/) to emit events
    even when called outside an agent session context.
    """
    bus = _bus_var.get()
    if bus is not None:
        return bus
    # Fallback: module-level singleton (lazily created)
    global _fallback_bus
    if _fallback_bus is None:
        with _fallback_lock:
            if _fallback_bus is None:
                _fallback_bus = EventBus(session_id="<fallback>")
    return _fallback_bus


def set_event_bus(bus: EventBus) -> None:
    """Set the EventBus for the current context."""
    _bus_var.set(bus)


# ---- Listeners ----

class DebugLogListener:
    """Formats SessionEvents and writes to the Python logger.

    Preserves the existing log file format so parse_chat_entries()
    and get_recent_errors() continue to work on old + new sessions.
    """

    # Map event level strings to Python logging levels
    _LEVEL_MAP = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    # Map event type to the legacy log_tag used by parse_chat_entries.
    # Class-level constant to avoid re-creating on every event emission.
    _TYPE_TO_TAG = {
        USER_MESSAGE: "user_message",
        AGENT_RESPONSE: "agent_response",
        DELEGATION: "delegation",
        DELEGATION_DONE: "delegation_done",
        PLAN_CREATED: "plan_event",
        PLAN_COMPLETED: "plan_event",
        PLAN_TASK: "plan_task",
        PROGRESS: "progress",
        DATA_FETCHED: "data_fetched",
        THINKING: "thinking",
        TOOL_ERROR: "error",
        TOOL_ERROR_LOG: "error",
        ERROR_LOG: "error",
        FETCH_ERROR: "error",
        RENDER_ERROR: "error",
        CUSTOM_OP_FAILURE: "error",
        SUB_AGENT_ERROR: "error",
        MEMORY_EXTRACTION_START: "memory",
        MEMORY_EXTRACTION_DONE: "memory",
        MEMORY_EXTRACTION_ERROR: "memory",
        MEMORY_ACTION: "memory",
        TOKEN_USAGE: "",
    }

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def __call__(self, event: SessionEvent) -> None:
        level = self._LEVEL_MAP.get(event.level, logging.DEBUG)
        # Include log_tag in extra so _SessionFilter and file formatter
        # produce the same output as the old tagged() calls
        tag = self._TYPE_TO_TAG.get(event.type, "")
        self._logger.log(level, event.summary, extra={"log_tag": tag})


class SSEEventListener:
    """Push display/console-tagged events to an SSE bridge callback.

    Events with "display" tag drive typed payloads (tool_call, tool_result,
    thinking, memory_update) for the Activity panel and Chat.
    Events with "console" tag are emitted as log_line for the Console tab.
    Warnings/errors are always forwarded regardless of tags.
    """

    def __init__(self, callback: Callable[[dict], None]):
        self._callback = callback

    def __call__(self, event: SessionEvent) -> None:
        # Forward display-tagged or console-tagged events, plus all warnings/errors
        if ("display" not in event.tags
                and "console" not in event.tags
                and event.level not in ("warning", "error")):
            return
        try:
            sent_typed = False

            # Send typed payloads for event types the frontend handles natively
            # (these require "display" tag)
            if "display" in event.tags or event.level in ("warning", "error"):
                if event.type == THINKING:
                    self._callback({
                        "type": THINKING,
                        "text": event.data.get("text", event.summary),
                        "level": event.level,
                    })
                    sent_typed = True
                elif event.type == TOOL_CALL:
                    self._callback({
                        "type": TOOL_CALL,
                        "tool_name": event.data.get("tool_name", ""),
                        "tool_args": event.data.get("tool_args", {}),
                        "text": event.summary,
                        "level": event.level,
                        "agent": event.agent,
                    })
                    sent_typed = True
                elif event.type == TOOL_STARTED:
                    self._callback({
                        "type": TOOL_CALL,
                        "tool_name": event.data.get("tool_name", ""),
                        "tool_args": event.data.get("tool_args", {}),
                        "text": event.summary,
                        "level": event.level,
                        "agent": event.agent,
                    })
                    sent_typed = True
                elif event.type == TOOL_RESULT:
                    self._callback({
                        "type": TOOL_RESULT,
                        "tool_name": event.data.get("tool_name", ""),
                        "status": event.data.get("status", ""),
                        "text": event.summary,
                        "level": event.level,
                        "agent": event.agent,
                    })
                    sent_typed = True
                elif event.type == TOOL_ERROR:
                    self._callback({
                        "type": TOOL_RESULT,
                        "tool_name": event.data.get("tool_name", ""),
                        "status": "error",
                        "text": event.summary,
                        "level": event.level,
                        "agent": event.agent,
                    })
                    sent_typed = True
                elif event.type == MEMORY_EXTRACTION_DONE:
                    self._callback({
                        "type": "memory_update",
                        "text": event.summary,
                        "level": event.level,
                        "actions": event.data.get("actions", {}),
                    })
                    sent_typed = True
                elif event.type == RENDER_EXECUTED:
                    self._callback({
                        "type": "plot",
                        "available": True,
                    })
                    sent_typed = True
                elif event.type == MPL_RENDER_EXECUTED:
                    self._callback({
                        "type": "mpl_image",
                        "available": True,
                        "script_id": event.data.get("script_id", ""),
                        "description": event.data.get("description", ""),
                    })
                    sent_typed = True
                elif event.type == JSX_RENDER_EXECUTED:
                    self._callback({
                        "type": "jsx_component",
                        "available": True,
                        "script_id": event.data.get("script_id", ""),
                        "description": event.data.get("description", ""),
                    })
                    sent_typed = True
                elif event.type == INSIGHT_RESULT:
                    self._callback({
                        "type": "insight_result",
                        "text": event.data.get("text", event.summary),
                        "level": event.level,
                    })
                    sent_typed = True
                elif event.type == INSIGHT_FEEDBACK:
                    self._callback({
                        "type": "insight_feedback",
                        "text": event.data.get("text", event.summary),
                        "passed": event.data.get("passed", True),
                        "level": event.level,
                    })
                    sent_typed = True
                elif event.type == TEXT_DELTA:
                    payload = {
                        "type": "text_delta",
                        "text": event.data.get("text", event.msg),
                    }
                    if event.data.get("commentary"):
                        payload["commentary"] = True
                        payload["agent"] = event.agent
                    self._callback(payload)
                    sent_typed = True
                elif event.type == ROUND_START:
                    self._callback({
                        "type": "round_start",
                    })
                    sent_typed = True
                elif event.type == ROUND_END:
                    self._callback({
                        "type": "round_end",
                        "token_usage": event.data.get("token_usage", {}),
                        "round_token_usage": event.data.get("round_token_usage", {}),
                    })
                    sent_typed = True
                elif event.type == SESSION_TITLE:
                    self._callback({
                        "type": "session_title",
                        "name": event.data.get("name", ""),
                    })
                    sent_typed = True

            # Send token_usage update for live frontend token counter
            if "token" in event.tags and event.type == TOKEN_USAGE:
                self._callback({
                    "type": "token_usage",
                    "data": event.data,
                })

            # Send as log_line for the Console tab (requires "console" tag).
            # Skip if a typed payload was already sent for this event to
            # avoid duplicate callbacks for the same event.
            if "console" in event.tags and not sent_typed:
                payload: dict = {
                    "type": "log_line",
                    "text": event.summary,
                    "level": event.level,
                }
                if event.details:
                    payload["details"] = event.details
                self._callback(payload)
        except Exception:
            pass


class OperationsLogListener:
    """Populates OperationsLog from pipeline-tagged events.

    Replaces the inline get_operations_log().record() calls scattered
    throughout core.py.

    Uses a callable to resolve the OperationsLog at event time
    (supports ContextVar-based per-session instances).
    """

    def __init__(self, ops_log_getter: Callable):
        self._get_ops_log = ops_log_getter

    def __call__(self, event: SessionEvent) -> None:
        if "pipeline" not in event.tags:
            return
        d = event.data
        try:
            ops_log = self._get_ops_log()
            if event.type == DATA_FETCHED:
                ops_log.record(
                    tool="fetch_data",
                    args=d.get("args", {}),
                    outputs=d.get("outputs", []),
                    status=d.get("status", "success"),
                    error=d.get("error"),
                )
            elif event.type == DATA_COMPUTED:
                ops_log.record(
                    tool="custom_operation",
                    args=d.get("args", {}),
                    outputs=d.get("outputs", []),
                    inputs=d.get("inputs", []),
                    status=d.get("status", "success"),
                    error=d.get("error"),
                )
            elif event.type == DATA_CREATED:
                ops_log.record(
                    tool="store_dataframe",
                    args=d.get("args", {}),
                    outputs=d.get("outputs", []),
                    status=d.get("status", "success"),
                    error=d.get("error"),
                )
            # RENDER_EXECUTED recording is now done inline in core.py
            # (so the op_id can be included in the event data)
            elif event.type == PLOT_ACTION:
                ops_log.record(
                    tool="manage_plot",
                    args=d.get("args", {}),
                    outputs=d.get("outputs", []),
                    status=d.get("status", "success"),
                    error=d.get("error"),
                )
        except Exception:
            pass


class TokenLogListener:
    """Writes token usage events to a per-session JSONL file.

    Each line is a JSON object with agent, tool_context, token counts
    (current + cumulative), and api_calls.
    """

    def __init__(self, path: Path):
        self._path = path
        self._file = open(path, "a", encoding="utf-8")

    def __call__(self, event: SessionEvent) -> None:
        if "token" not in event.tags:
            return
        try:
            self._file.write(json.dumps(event.data, default=str) + "\n")
            self._file.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


class DisplayLogBuilder:
    """Builds display_log list from display-tagged events.

    Replaces parse_chat_entries() for live sessions (not for loading
    old sessions from disk — that still uses parse_chat_entries).
    """

    def __init__(self):
        self.entries: list[dict] = []

    def __call__(self, event: SessionEvent) -> None:
        if event.type == USER_MESSAGE:
            self.entries.append({
                "role": "user",
                "content": event.data.get("text", event.summary),
                "timestamp": event.ts,
            })
        elif event.type == AGENT_RESPONSE:
            self.entries.append({
                "role": "agent",
                "content": event.data.get("text", event.summary),
                "timestamp": event.ts,
            })
        elif event.type == INSIGHT_RESULT:
            self.entries.append({
                "role": "insight",
                "content": event.data.get("text", event.summary),
                "timestamp": event.ts,
            })
        elif event.type == INSIGHT_FEEDBACK:
            self.entries.append({
                "role": "insight_feedback",
                "content": event.data.get("text", event.summary),
                "passed": event.data.get("passed", True),
                "timestamp": event.ts,
            })
        elif event.type == THINKING and "display" in event.tags:
            self.entries.append({
                "role": "thinking",
                "content": event.data.get("text", event.summary),
                "timestamp": event.ts,
            })
        elif "display" in event.tags:
            self.entries.append({
                "role": "milestone",
                "content": event.summary,
                "timestamp": event.ts,
            })


class EventLogWriter:
    """Appends frontend-relevant events to a JSONL file on disk.

    Persists events with "display" or "console" tag, plus all
    warnings/errors.  Each line is a JSON object with
    {type, ts, agent, level, summary, details, data, tags}.

    The file is opened in append mode and flushed after each write,
    so it survives crashes and can be read while the session is active.
    """

    def __init__(self, path: Path):
        self._path = path
        self._file = open(path, "a", encoding="utf-8")

    def __call__(self, event: SessionEvent) -> None:
        if ("display" not in event.tags
                and "console" not in event.tags
                and event.level not in ("warning", "error")):
            return
        try:
            record = {
                "id": event.id,
                "type": event.type,
                "ts": event.ts,
                "agent": event.agent,
                "level": event.level,
                "msg": event.summary,  # backward compat key
                "summary": event.summary,
                "details": event.details,
                "data": event.data,
                "tags": sorted(event.tags),
            }
            if event.children:
                record["children"] = list(event.children)
            self._file.write(json.dumps(record, default=str) + "\n")
            self._file.flush()
        except Exception:
            pass  # Never break the emitter

    def close(self) -> None:
        """Flush and close the underlying file."""
        try:
            self._file.close()
        except Exception:
            pass


def load_event_log(path: Path) -> list[dict]:
    """Read a JSONL event log file back into a list of dicts.

    Returns an empty list if the file doesn't exist or can't be parsed.
    """
    if not path.exists():
        return []
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events
