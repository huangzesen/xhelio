"""
Logging configuration for Helion.

Three logging destinations, two tiers:

Tier 1 — File + Console (identical by default):
  - Full detail, no truncation
  - File: always DEBUG level
  - Console: DEBUG if --verbose, WARNING+ otherwise
  - Format: "timestamp | level | name | session_id | tag | message"
  - Config console_format options:
    - "full"   — (default) same structured format as the file handler
    - "simple" — bare messages for DEBUG/INFO, [LEVEL] prefix for WARNING+
    - "gui"    — curated tagged records only (mirrors web UI live-log sidebar)
    - "clean"  — no console output at all (file logging still active)

Tier 2 — Web UI live log (curated):
  - Only records tagged with a key in WEBUI_VISIBLE_TAGS

Log files are stored in ~/.xhelio/logs/ with one file per session.
"""

import logging
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import get_data_dir


# Log directory
LOG_DIR = get_data_dir() / "logs"

# Tags that the web UI live-log handler will display.
# To show a new category in the web UI, tag the log call with
# ``extra=tagged("my_tag")`` and add ``"my_tag"`` here.
WEBUI_VISIBLE_TAGS = frozenset({
    "delegation",       # "[Router] Delegating to X specialist"
    "delegation_done",  # "[Router] X specialist finished"
    "plan_event",       # Plan created / completed / failed
    "plan_task",        # Plan task executing / round progress
    "progress",         # Milestone updates during planning/discovery
    "data_fetched",     # "[DataOps] Stored 'label' (N points)"
    "thinking",         # "[Thinking] ..." (truncated preview)
    "error",            # log_error() — real errors with context/stack traces
})


def tagged(tag: str) -> dict:
    """Return ``extra`` dict for logger calls: ``logger.debug("...", extra=tagged("x"))``."""
    return {"log_tag": tag}


# Module-level state (shared across re-inits)
_session_filter: Optional["_SessionFilter"] = None
_current_log_file: Optional[Path] = None


class _SessionFilter(logging.Filter):
    """Injects session_id into every log record."""

    def __init__(self) -> None:
        super().__init__()
        self.session_id = ""

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = self.session_id or "-"
        if not hasattr(record, "log_tag"):
            record.log_tag = ""
        return True


class _ConsoleFormatter(logging.Formatter):
    """Console formatter: shows [LEVEL] prefix only for WARNING and above.

    DEBUG/INFO messages print bare (e.g. ``  [Gemini] Sending...``).
    WARNING/ERROR messages include the level (e.g. ``  [WARNING] ...``).
    """

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.WARNING:
            return f"  [{record.levelname}] {record.getMessage()}"
        return f"  {record.getMessage()}"


class _WebUITagFilter(logging.Filter):
    """Pass only records tagged with a key in WEBUI_VISIBLE_TAGS.

    Mirrors the curated output shown in the web UI live-log sidebar,
    but routed to the console instead.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        tag = getattr(record, "log_tag", "")
        return tag in WEBUI_VISIBLE_TAGS


def attach_log_file(session_id: str) -> None:
    """Attach a per-session file handler.

    Creates or appends to agent_{session_id}.log.
    Call this once from start_session() or load_session().
    """
    global _current_log_file
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOG_DIR / f"agent_{session_id}.log"
    _current_log_file = log_file

    logger = logging.getLogger("xhelio")
    # Remove any existing file handler (safety — e.g. after reset_session)
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]

    file_handler = logging.FileHandler(log_file, mode='a', encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(session_id)s | %(log_tag)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info(f"Session started at {datetime.now().isoformat()}")
    logger.info(f"Log file: {log_file}")


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure console-only logging for the agent.

    File handlers are attached later by ``attach_log_file()`` once a
    session ID is known.

    Args:
        verbose: If True, show DEBUG level on console; otherwise WARNING+.

    Returns:
        Configured logger instance.
    """
    global _session_filter

    # Create logger
    logger = logging.getLogger("xhelio")
    logger.setLevel(logging.DEBUG)  # Capture everything, filter at handler level

    # Clear existing handlers (in case of re-init)
    logger.handlers.clear()

    # Session filter — reuse existing instance to preserve session_id across re-inits
    if _session_filter is None:
        _session_filter = _SessionFilter()
    logger.addFilter(_session_filter)

    # Console handler — the only handler until attach_log_file() is called
    import config as _config
    console_format = _config.get("console_format", "simple")

    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(session_id)s | %(log_tag)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if console_format != "clean":
        console_handler = logging.StreamHandler(sys.stderr)

        if console_format == "gui":
            # Curated output: same tagged records as the web UI live-log sidebar
            console_handler.setLevel(logging.DEBUG)
            console_handler.addFilter(_WebUITagFilter())
            console_handler.setFormatter(_ConsoleFormatter())
        elif console_format == "simple":
            console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
            console_handler.setFormatter(_ConsoleFormatter())
        else:
            # "full" (default)
            console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
            console_handler.setFormatter(file_format)

        logger.addHandler(console_handler)
    # "clean" — no console handler at all

    return logger


def get_logger() -> logging.Logger:
    """Get the agent logger instance.

    Returns:
        The xhelio logger (creates with defaults if not configured)
    """
    logger = logging.getLogger("xhelio")
    if not logger.handlers:
        # Not configured yet, set up with defaults
        return setup_logging(verbose=False)
    return logger


def set_session_id(session_id: str) -> None:
    """Set the session ID that will be included in all subsequent log lines.

    Args:
        session_id: The session identifier (e.g. '20260209_223120_4b7103d5')
    """
    global _session_filter
    if _session_filter is None:
        # Logger not set up yet — create filter so it's ready when logging starts
        _session_filter = _SessionFilter()
    _session_filter.session_id = session_id


def log_error(
    message: str,
    exc: Optional[Exception] = None,
    context: Optional[dict] = None,
) -> None:
    """Log an error with full details including stack trace.

    Routes through the EventBus. The DebugLogListener writes to the
    Python logger (file + console), preserving backward compatibility.

    Args:
        message: Error description
        exc: Optional exception to include stack trace from
        context: Optional dict of additional context (tool name, args, etc.)
    """
    from .event_bus import get_event_bus, ERROR_LOG

    # Build detailed message for the file log
    lines = [message]

    if context:
        lines.append("Context:")
        for key, value in context.items():
            lines.append(f"  {key}: {value}")

    if exc:
        lines.append(f"Exception type: {type(exc).__name__}")
        lines.append(f"Exception message: {exc}")
        lines.append("Stack trace:")
        lines.append(traceback.format_exc())

    full_message = "\n".join(lines)

    # Emit as error log — DebugLogListener writes to file+console,
    # memory system extracts pitfalls (no SSE/display — avoids phantom tool_result)
    bus = get_event_bus()
    from .truncation import trunc
    bus.emit(ERROR_LOG, level="error", msg=full_message,
             data={"short": trunc(message, "detail.request"), "context": context or {}})


def log_tool_call(tool_name: str, tool_args: dict) -> None:
    """Log a tool call for debugging.

    Args:
        tool_name: Name of the tool being called
        tool_args: Arguments passed to the tool
    """
    from .event_bus import get_event_bus, TOOL_CALL_LOG

    args_str = str(tool_args)
    get_event_bus().emit(TOOL_CALL_LOG, level="debug",
                         msg=f"Tool call: {tool_name}({args_str})",
                         data={"tool_name": tool_name, "tool_args": tool_args})


def log_tool_result(tool_name: str, result: dict, success: bool) -> None:
    """Log a tool result.

    Args:
        tool_name: Name of the tool
        result: Result dict from the tool
        success: Whether the tool succeeded
    """
    from .event_bus import get_event_bus, TOOL_RESULT_LOG

    if success:
        get_event_bus().emit(TOOL_RESULT_LOG, level="debug",
                             msg=f"Tool result: {tool_name} -> success",
                             data={"tool_name": tool_name, "status": "success"})
    else:
        error_msg = result.get("message", "Unknown error")
        get_event_bus().emit(TOOL_RESULT_LOG, level="warning",
                             msg=f"Tool result: {tool_name} -> error: {error_msg}",
                             data={"tool_name": tool_name, "status": "error", "error": error_msg})


def log_plan_event(event: str, plan_id: str, details: Optional[str] = None) -> None:
    """Log a planning/execution event.

    Args:
        event: Event type (created, executing, completed, failed, etc.)
        plan_id: ID of the plan
        details: Optional additional details
    """
    from .event_bus import get_event_bus, PLAN_CREATED, PLAN_COMPLETED

    msg = f"Plan {event}: {plan_id[:8]}..."
    if details:
        msg += f" - {details}"

    # Map event string to event type
    event_type = PLAN_COMPLETED if event in ("completed", "failed", "cancelled") else PLAN_CREATED
    get_event_bus().emit(event_type, level="info", msg=msg,
                         data={"event": event, "plan_id": plan_id, "details": details})


def log_session_end(token_usage: dict) -> None:
    """Log session end with usage stats.

    Args:
        token_usage: Dict with input_tokens, output_tokens, api_calls
    """
    from .event_bus import get_event_bus, SESSION_END

    cached = token_usage.get('cached_tokens', 0)
    cache_str = f", cached: {cached:,}" if cached else ""
    msg = (
        f"Session ended. Tokens: {token_usage.get('total_tokens', 0):,} "
        f"(in: {token_usage.get('input_tokens', 0):,}, "
        f"out: {token_usage.get('output_tokens', 0):,}{cache_str}), "
        f"API calls: {token_usage.get('api_calls', 0)}"
    )
    get_event_bus().emit(SESSION_END, level="info", msg=msg,
                         data={"token_usage": token_usage})


def get_current_log_path() -> Path:
    """Return the path to the current session's log file."""
    if _current_log_file is not None:
        return _current_log_file
    # Fallback: find most recent log file in the directory
    logs = sorted(LOG_DIR.glob("agent_*.log"))
    if logs:
        return logs[-1]
    return LOG_DIR / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def get_log_size(path: Path) -> int:
    """Return the size of a log file in bytes, or 0 if it doesn't exist."""
    try:
        return path.stat().st_size
    except (OSError, ValueError):
        return 0


def get_recent_errors(days: int = 7, limit: int = 50) -> list[dict]:
    """Retrieve recent errors from log files.

    Args:
        days: How many days back to search
        limit: Maximum number of errors to return

    Returns:
        List of error entries with timestamp, message, and details
    """
    errors = []
    cutoff = datetime.now().timestamp() - days * 86400
    # Collect all log files, newest first
    log_files = sorted(LOG_DIR.glob("agent_*.log"), reverse=True)

    for log_file in log_files:
        if log_file.stat().st_mtime < cutoff:
            break

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                current_error = None
                for line in f:
                    if "| ERROR" in line or "| WARNING" in line:
                        if current_error:
                            errors.append(current_error)
                        # Parse the log line
                        # Format: timestamp | level | name | session_id | tag | message
                        parts = line.split(" | ", 5)
                        if len(parts) >= 6:
                            current_error = {
                                "timestamp": parts[0].strip(),
                                "level": parts[1].strip(),
                                "session_id": parts[3].strip(),
                                "message": parts[5].strip(),
                                "details": [],
                            }
                    elif current_error and line.startswith("  "):
                        # Continuation of error details
                        current_error["details"].append(line.rstrip())

                if current_error:
                    errors.append(current_error)

        except Exception:
            continue

        if len(errors) >= limit:
            break

    return errors[:limit]


def print_recent_errors(days: int = 7, limit: int = 10) -> None:
    """Print recent errors to console for review.

    Args:
        days: How many days back to search
        limit: Maximum number of errors to show
    """
    errors = get_recent_errors(days=days, limit=limit)

    if not errors:
        print(f"No errors found in the last {days} days.")
        return

    print(f"Recent errors (last {days} days, showing up to {limit}):")
    print("-" * 60)

    for i, error in enumerate(errors, 1):
        print(f"\n{i}. [{error['timestamp']}] {error['level']}")
        print(f"   {error['message']}")
        if error["details"]:
            for detail in error["details"][:5]:  # Limit detail lines
                print(f"   {detail}")
            if len(error["details"]) > 5:
                print(f"   ... and {len(error['details']) - 5} more lines")

    print("-" * 60)
    print(f"Full logs available at: {LOG_DIR}")


# ---- Log-file chat extraction ----

# Matches a timestamped log line:
#   "2026-02-17 02:23:55 | INFO     | helion | session_id | tag | message"
_LOG_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| "  # group 1: timestamp
    r"\S+\s*\| "                                      # level (e.g. "INFO    ")
    r"\S+ \| "                                        # logger name
    r"(\S+) \| "                                      # group 2: session_id
    r"(.*)$"                                          # group 3: rest (tag | msg  OR  msg)
)

# Tags whose log messages are surfaced in the chat box as milestones.
# Derived from WEBUI_VISIBLE_TAGS minus "error" (errors shown separately).
_MILESTONE_TAGS = WEBUI_VISIBLE_TAGS - {"error"}


def _parse_tag_and_message(rest: str) -> tuple[str, str]:
    """Split ``"tag | message"`` from the tail of a log line.

    Format: ``" tag | message"`` or ``"  | message"`` (empty tag).
    """
    tag, _, msg = rest.partition(" | ")
    return tag.strip(), msg


def parse_chat_entries(
    log_path: Path, session_id: str, *, tail_lines: int = 0,
) -> list[dict]:
    """Extract chat entries from a log file for display in the UI.

    Returns ``[User]``, ``[Agent]``, and tagged milestone entries as a flat
    list — the single dataset that drives the chat box for both live polling
    and replay.

    Args:
        log_path: Path to the agent log file.
        session_id: Only include entries for this session.
        tail_lines: If > 0, only parse the last *N* lines of the file
            (for efficient live polling).  0 means read the whole file.

    Returns:
        List of dicts with keys ``role`` (``"user"``/``"agent"``/``"milestone"``),
        ``content``, and ``timestamp``.  Chronological order.
        Returns ``[]`` if the file doesn't exist.
    """
    if not log_path.exists():
        return []

    entries: list[dict] = []
    current: dict | None = None  # entry being accumulated (multi-line)

    def _flush():
        nonlocal current
        if current is not None:
            entries.append(current)
            current = None

    # Read lines (optionally only the tail)
    if tail_lines > 0:
        with open(log_path, "rb") as fb:
            fb.seek(0, 2)  # end
            size = fb.tell()
            # Over-read to be safe, then take last N lines
            chunk = min(size, tail_lines * 300)
            fb.seek(max(0, size - chunk))
            raw = fb.read().decode("utf-8", errors="replace")
        lines = raw.splitlines()[-tail_lines:]
        # Skip orphaned continuation lines at the start of the tail window
        while lines and not _LOG_LINE_RE.match(lines[0]):
            lines.pop(0)
    else:
        with open(log_path, "r", encoding="utf-8") as fh:
            lines = [l.rstrip("\n") for l in fh]

    for line in lines:
        m = _LOG_LINE_RE.match(line)
        if m:
            ts, sid, rest = m.group(1), m.group(2), m.group(3)

            if sid != session_id:
                _flush()
                continue

            tag, msg = _parse_tag_and_message(rest)

            if msg.startswith("[User] "):
                _flush()
                current = {
                    "role": "user",
                    "content": msg[7:],
                    "timestamp": ts,
                }
            elif msg.startswith("[Agent] "):
                _flush()
                current = {
                    "role": "agent",
                    "content": msg[8:],
                    "timestamp": ts,
                }
            elif tag in _MILESTONE_TAGS:
                _flush()
                if tag == "thinking":
                    # Thinking entries are multi-line — accumulate continuations
                    current = {
                        "role": "milestone",
                        "content": msg,
                        "timestamp": ts,
                    }
                else:
                    entries.append({
                        "role": "milestone",
                        "content": msg,
                        "timestamp": ts,
                    })
            else:
                # Not a chat entry — flush any open entry so that
                # subsequent continuation lines don't get appended to it.
                _flush()
        else:
            # Continuation line (no timestamp prefix)
            if current is not None:
                current["content"] += "\n" + line

    _flush()
    return entries
