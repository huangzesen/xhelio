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
"""

import contextvars
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional


# ---- Event type constants ----

# Conversation
USER_MESSAGE = "user_message"
AGENT_RESPONSE = "agent_response"

# Tool lifecycle
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"

# Data pipeline
DATA_FETCHED = "data_fetched"
DATA_COMPUTED = "data_computed"
DATA_CREATED = "data_created"
RENDER_EXECUTED = "render_executed"
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

# Short-term memory
STM_COMPACTION = "stm_compaction"

# Knowledge/Bootstrap
CATALOG_SEARCH = "catalog_search"
METADATA_FETCH = "metadata_fetch"
BOOTSTRAP_PROGRESS = "bootstrap_progress"

# Catch-all for debug-level events that don't need a specific type
DEBUG = "debug"

# Secondary logging (console-only variants of tool lifecycle events)
TOOL_CALL_LOG = "tool_call_log"        # Console-only tool call log (secondary path)
TOOL_RESULT_LOG = "tool_result_log"    # Console-only tool result log (secondary path)
TOOL_ERROR_LOG = "tool_error_log"      # Console-only tool error from sub-agent loops

# Data source access (user-facing variant)
CDF_DOWNLOAD_WARN = "cdf_download_warn"  # Large CDF download warning (surfaces to display)


# ---- Default tags registry ----
# Strict 1:1 mapping from event type to tags.
# Tags are fully deterministic — emit() resolves tags from this registry.

DEFAULT_TAGS: dict[str, frozenset[str]] = {
    # Conversation
    USER_MESSAGE:       frozenset({"display", "memory", "console"}),
    AGENT_RESPONSE:     frozenset({"display", "memory", "console"}),
    # Tool lifecycle
    TOOL_CALL:          frozenset({"display", "memory", "console"}),
    TOOL_RESULT:        frozenset({"display", "memory", "console"}),
    TOOL_ERROR:         frozenset({"display", "memory", "console"}),
    TOOL_CALL_LOG:      frozenset({"console"}),
    TOOL_RESULT_LOG:    frozenset({"console"}),
    TOOL_ERROR_LOG:     frozenset(),
    # Data pipeline
    DATA_FETCHED:       frozenset({"display", "memory", "pipeline", "console", "ctx:mission", "ctx:planner", "ctx:orchestrator"}),
    DATA_COMPUTED:      frozenset({"memory", "pipeline", "console", "ctx:dataops", "ctx:planner", "ctx:orchestrator"}),
    DATA_CREATED:       frozenset({"pipeline", "console"}),
    RENDER_EXECUTED:    frozenset({"display", "memory", "pipeline", "console", "ctx:viz", "ctx:planner", "ctx:orchestrator"}),
    PLOT_ACTION:        frozenset({"pipeline", "console", "ctx:viz", "ctx:planner", "ctx:orchestrator"}),
    CUSTOM_OP_FAILURE:  frozenset({"display", "memory", "pipeline", "console", "ctx:mission", "ctx:viz", "ctx:dataops", "ctx:planner", "ctx:orchestrator"}),
    # Routing
    DELEGATION:         frozenset({"display", "memory", "console", "ctx:planner", "ctx:orchestrator"}),
    DELEGATION_DONE:    frozenset({"display", "memory", "console", "ctx:planner", "ctx:orchestrator"}),
    SUB_AGENT_TOOL:     frozenset({"console", "ctx:mission", "ctx:viz", "ctx:dataops", "ctx:planner", "ctx:orchestrator"}),
    SUB_AGENT_ERROR:    frozenset({"console", "ctx:mission", "ctx:viz", "ctx:dataops", "ctx:planner", "ctx:orchestrator"}),
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
    # Short-term memory
    STM_COMPACTION:     frozenset({"display", "console", "ctx:mission", "ctx:viz", "ctx:dataops", "ctx:planner", "ctx:orchestrator"}),
    # Errors
    FETCH_ERROR:        frozenset({"display", "memory", "console", "ctx:mission", "ctx:planner", "ctx:orchestrator"}),
    HIGH_NAN:           frozenset({"display", "console"}),
    RECOVERY:           frozenset({"console"}),
    RENDER_ERROR:       frozenset({"display", "memory", "console", "ctx:viz", "ctx:planner", "ctx:orchestrator"}),
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


# ---- SessionEvent ----

@dataclass(frozen=True)
class SessionEvent:
    """A single structured event in the session."""
    type: str
    ts: str
    agent: str
    level: str
    msg: str
    data: dict
    tags: frozenset


class EventBus:
    """Per-session event bus with synchronous listener dispatch.

    Thread-safe: emit() and subscribe() use a lock.
    """

    def __init__(self, session_id: str = ""):
        self._events: list[SessionEvent] = []
        self._lock = threading.Lock()
        self._listeners: list[Callable[[SessionEvent], None]] = []
        self.session_id = session_id

    def emit(
        self,
        type: str,
        *,
        agent: str = "orchestrator",
        level: str = "debug",
        msg: str = "",
        data: Optional[dict] = None,
    ) -> SessionEvent:
        """Create, store, and dispatch a SessionEvent.

        Tags are resolved automatically from the DEFAULT_TAGS registry
        based on event type. To route an event differently, use the
        correct event type (e.g. TOOL_CALL_LOG for console-only logging).

        Args:
            type: Event type constant (e.g. TOOL_CALL, DELEGATION).
            agent: Source agent name.
            level: Log level (debug/info/warning/error).
            msg: Human-readable message.
            data: Structured payload.

        Returns:
            The created SessionEvent.
        """
        event = SessionEvent(
            type=type,
            ts=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            agent=agent,
            level=level,
            msg=msg,
            data=data or {},
            tags=DEFAULT_TAGS.get(type, frozenset()),
        )
        with self._lock:
            self._events.append(event)
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                pass  # Never let a listener break the emitter
        return event

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

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def __call__(self, event: SessionEvent) -> None:
        level = self._LEVEL_MAP.get(event.level, logging.DEBUG)
        # Include log_tag in extra so _SessionFilter and file formatter
        # produce the same output as the old tagged() calls
        tag = self._event_type_to_tag(event)
        self._logger.log(level, event.msg, extra={"log_tag": tag})

    @staticmethod
    def _event_type_to_tag(event: SessionEvent) -> str:
        """Map event type to the legacy log_tag used by parse_chat_entries."""
        # Events with "display" tag that map to known WEBUI_VISIBLE_TAGS
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
        return _TYPE_TO_TAG.get(event.type, "")


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
            # Send typed payloads for event types the frontend handles natively
            # (these require "display" tag)
            if "display" in event.tags or event.level in ("warning", "error"):
                if event.type == THINKING:
                    self._callback({
                        "type": THINKING,
                        "text": event.data.get("text", event.msg),
                        "level": event.level,
                    })
                elif event.type == TOOL_CALL:
                    self._callback({
                        "type": TOOL_CALL,
                        "tool_name": event.data.get("tool_name", ""),
                        "tool_args": event.data.get("tool_args", {}),
                        "text": event.msg,
                        "level": event.level,
                    })
                elif event.type == TOOL_RESULT:
                    self._callback({
                        "type": TOOL_RESULT,
                        "tool_name": event.data.get("tool_name", ""),
                        "status": event.data.get("status", ""),
                        "text": event.msg,
                        "level": event.level,
                    })
                elif event.type == MEMORY_EXTRACTION_DONE:
                    self._callback({
                        "type": "memory_update",
                        "text": event.msg,
                        "level": event.level,
                        "actions": event.data.get("actions", {}),
                    })

            # Send token_usage update for live frontend token counter
            if "token" in event.tags and event.type == TOKEN_USAGE:
                self._callback({
                    "type": "token_usage",
                    "data": event.data,
                })

            # Send as log_line for the Console tab (requires "console" tag)
            if "console" in event.tags:
                self._callback({
                    "type": "log_line",
                    "text": event.msg,
                    "level": event.level,
                })
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
            elif event.type == RENDER_EXECUTED:
                ops_log.record(
                    tool="render_plotly_json",
                    args=d.get("args", {}),
                    outputs=d.get("outputs", []),
                    inputs=d.get("inputs", []),
                    status=d.get("status", "success"),
                    error=d.get("error"),
                )
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
                "content": event.data.get("text", event.msg),
                "timestamp": event.ts,
            })
        elif event.type == AGENT_RESPONSE:
            self.entries.append({
                "role": "agent",
                "content": event.data.get("text", event.msg),
                "timestamp": event.ts,
            })
        elif event.type == THINKING and "display" in event.tags:
            self.entries.append({
                "role": "thinking",
                "content": event.data.get("text", event.msg),
                "timestamp": event.ts,
            })
        elif "display" in event.tags:
            self.entries.append({
                "role": "milestone",
                "content": event.msg,
                "timestamp": event.ts,
            })


class EventLogWriter:
    """Appends frontend-relevant events to a JSONL file on disk.

    Persists events with "display" or "console" tag, plus all
    warnings/errors.  Each line is a JSON object with
    {type, ts, agent, level, msg, data, tags}.

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
                "type": event.type,
                "ts": event.ts,
                "agent": event.agent,
                "level": event.level,
                "msg": event.msg,
                "data": event.data,
                "tags": sorted(event.tags),
            }
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
