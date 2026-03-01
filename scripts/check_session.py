#!/usr/bin/env python3
"""
Session diagnostic tool for xhelio.

Reads a session's event log, metadata, operations, and data index to produce
a structured report of errors, orphaned delegations, unpaired tool calls,
data integrity issues, timeline anomalies, retry loops, and token usage.

Usage:
    python scripts/check_session.py <session_id>                 # Check one session, print to stdout
    python scripts/check_session.py <session_id> --save <dir>    # Check and save report to <dir>/
    python scripts/check_session.py --batch <n> --save <dir>     # Check last N sessions, skip already-checked
    python scripts/check_session.py --list                       # List recent sessions
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

SESSIONS_DIR = Path.home() / ".xhelio" / "sessions"

# Event types that represent errors
ERROR_EVENT_TYPES = {
    "tool_error", "sub_agent_error", "fetch_error",
    "render_error", "custom_op_failure",
}

# Thresholds
SLOW_TOOL_THRESHOLD_S = 60
SLOW_LLM_THRESHOLD_S = 30
RETRY_LOOP_THRESHOLD = 3


def get_tool_name(event: dict) -> str:
    """Extract tool name from an event, handling all known data layouts.

    tool_call / tool_result / sub_agent_tool / sub_agent_error:
        data.tool_name
    tool_error:
        data.context.tool_name  (fallback: data.tool_name)
    """
    data = event.get("data", {})
    name = data.get("tool_name")
    if not name:
        ctx = data.get("context")
        if isinstance(ctx, dict):
            name = ctx.get("tool_name")
    return name or ""


def parse_list_field(value) -> list:
    """Parse a field that may be a list or a stringified list (e.g. from repr)."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.startswith("["):
        try:
            return json.loads(value.replace("'", '"'))
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def parse_ts(ts_str: str) -> datetime | None:
    """Parse ISO 8601 timestamp string to datetime."""
    if not ts_str:
        return None
    try:
        # Handle various ISO formats (with/without timezone)
        ts_str = ts_str.replace("+00:00", "+0000").replace("Z", "+0000")
        if "+" in ts_str[10:]:
            # Has timezone
            return datetime.strptime(ts_str[:26] + ts_str[-5:], "%Y-%m-%dT%H:%M:%S.%f%z")
        elif "T" in ts_str:
            return datetime.strptime(ts_str[:26], "%Y-%m-%dT%H:%M:%S.%f")
        else:
            return datetime.strptime(ts_str[:19], "%Y-%m-%dT%H:%M:%S")
    except (ValueError, IndexError):
        try:
            return datetime.fromisoformat(ts_str)
        except ValueError:
            return None


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def load_json(path: Path) -> dict | list | None:
    """Load a JSON file, return None if missing."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt_tokens(n: int) -> str:
    """Format token count with K suffix."""
    if n >= 1000:
        return f"{n / 1000:.1f}K"
    return str(n)


def fmt_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m{secs:.0f}s"


# ── Section: Overview ──────────────────────────────────────────────────────

def section_overview(
    session_id: str,
    metadata: dict | None,
    events: list[dict],
    data_index: dict | None,
    token_events: list[dict],
) -> str:
    lines = ["## Overview\n"]

    if metadata:
        model = metadata.get("model", "unknown")
        created = metadata.get("created_at", "?")[:19]
        updated = metadata.get("updated_at", "?")[:19]
        turns = metadata.get("turn_count", "?")
        name = metadata.get("name", "")
        lines.append(f"- **Model**: {model}")
        if name:
            lines.append(f"- **Name**: {name}")
        lines.append(f"- **Created**: {created}")
        lines.append(f"- **Updated**: {updated}")
        lines.append(f"- **Turns**: {turns}")
    else:
        lines.append("- **Metadata**: not found")

    # Event distribution
    type_counts = Counter(e.get("type", "?") for e in events)
    total = len(events)
    top_types = ", ".join(
        f"{c} {t}" for t, c in type_counts.most_common(6)
    )
    lines.append(f"- **Events**: {total} ({top_types}, ...)")

    # Token usage from metadata
    if metadata and "token_usage" in metadata:
        tu = metadata["token_usage"]
        parts = []
        parts.append(f"{fmt_tokens(tu.get('input_tokens', 0))} in")
        parts.append(f"{fmt_tokens(tu.get('output_tokens', 0))} out")
        if tu.get("thinking_tokens", 0) > 0:
            parts.append(f"{fmt_tokens(tu['thinking_tokens'])} think")
        if tu.get("cached_tokens", 0) > 0:
            parts.append(f"{fmt_tokens(tu['cached_tokens'])} cached")
        parts.append(f"{tu.get('api_calls', '?')} API calls")
        lines.append(f"- **Tokens**: {' / '.join(parts)}")

    # Data labels
    if data_index is not None:
        n_labels = len(data_index)
        if n_labels <= 8:
            label_list = ", ".join(data_index.keys())
        else:
            label_list = ", ".join(list(data_index.keys())[:6]) + f", ... (+{n_labels - 6} more)"
        lines.append(f"- **Data labels**: {n_labels} ({label_list})")
    else:
        lines.append("- **Data labels**: no _index.json found")

    # Session directory path for manual inspection
    lines.append(f"- **Session dir**: `~/.xhelio/sessions/{session_id}/`")

    return "\n".join(lines) + "\n"


# ── Section: User Messages ─────────────────────────────────────────────

def section_user_messages(events: list[dict]) -> str:
    """Show user messages to provide context for what the session was about."""
    messages = []
    for i, e in enumerate(events, 1):
        if e.get("type") == "user_message":
            data = e.get("data", {})
            text = data.get("text") or e.get("msg", "")
            # Strip [User] prefix if present
            if text.startswith("[User] "):
                text = text[7:]
            messages.append((i, e.get("ts", "?")[:19], text))

    if not messages:
        return "## User Messages\n\nNo user messages found.\n"

    lines = ["## User Messages\n"]
    for idx, (ln, ts, text) in enumerate(messages, 1):
        lines.append(f"{idx}. **[line {ln}]** {text}")

    return "\n".join(lines) + "\n"


# ── Section: Errors ────────────────────────────────────────────────────────

def _format_error_detail(line_num: int, event: dict) -> str:
    """Format a single error event with full context for investigation."""
    data = event.get("data", {})
    etype = event.get("type", "?")
    agent = event.get("agent", "?")
    ts = event.get("ts", "?")[:23]
    tool = data.get("tool_name") or get_tool_name(event) or ""

    parts = [f"### Line {line_num} — `{etype}` ({agent})"]
    parts.append(f"- **Time**: {ts}")
    if tool:
        parts.append(f"- **Tool**: `{tool}`")

    # Full error message
    error_msg = data.get("error") or event.get("msg", "")
    parts.append(f"- **Error**: {error_msg}")

    # Show code that caused the error (custom_op_failure)
    args = data.get("args")
    if isinstance(args, dict) and "code" in args:
        code = args["code"]
        parts.append(f"- **Code**:")
        parts.append(f"  ```python\n  {code}\n  ```")
    elif isinstance(args, dict) and args:
        # Show relevant args (skip very large ones)
        compact = {k: v for k, v in args.items()
                   if not isinstance(v, (dict, list)) or len(str(v)) < 200}
        if compact:
            parts.append(f"- **Args**: `{json.dumps(compact, default=str)}`")

    # Show inputs/outputs
    inputs = data.get("inputs")
    if inputs:
        labels = parse_list_field(inputs) if isinstance(inputs, str) else inputs
        if labels:
            parts.append(f"- **Inputs**: {', '.join(f'`{l}`' for l in labels)}")

    return "\n".join(parts)


def section_errors(events: list[dict]) -> str:
    """Find all error events with full context for investigation."""
    errors = []
    for i, e in enumerate(events, 1):
        is_error = (
            e.get("level") in ("error", "warning")
            or e.get("type") in ERROR_EVENT_TYPES
        )
        if is_error:
            errors.append((i, e))

    if not errors:
        return "## Errors (0)\n\nNo errors or warnings found.\n"

    # Group repeated error messages
    msg_occurrences: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for line_num, e in errors:
        error_msg = e.get("data", {}).get("error") or e.get("msg", "")
        # Normalize for grouping (first 120 chars)
        key = error_msg[:120]
        msg_occurrences[key].append((line_num, e))

    # Separate unique and repeated
    unique_errors = []
    repeated_groups = []
    for key, occurrences in msg_occurrences.items():
        if len(occurrences) >= RETRY_LOOP_THRESHOLD:
            repeated_groups.append((key, occurrences))
        else:
            unique_errors.extend(occurrences)

    unique_errors.sort(key=lambda x: x[0])

    lines = [f"## Errors ({len(errors)})\n"]

    # Detailed listing for each unique error
    for ln, e in unique_errors:
        lines.append(_format_error_detail(ln, e))
        lines.append("")

    # Repeated errors — show detail for first occurrence, then list the rest
    if repeated_groups:
        lines.append("### Repeated Errors\n")
        for key, occurrences in repeated_groups:
            first_ln, first_e = occurrences[0]
            lines.append(_format_error_detail(first_ln, first_e))
            other_lines = [str(ln) for ln, _ in occurrences[1:]]
            lines.append(
                f"\n*Repeated {len(occurrences)}x total "
                f"(also at lines {', '.join(other_lines)})*\n"
            )

    return "\n".join(lines) + "\n"


# ── Section: Orphaned Delegations ──────────────────────────────────────────

def _extract_delegation_target(event: dict) -> str:
    """Extract the target agent name from a delegation event.

    Handles multiple msg formats observed across sessions:
        '[Router] Delegating to VOYAGER1 specialist'
        'Delegating to [Router] Delegating to PSP specialist'  (double-wrapped)
        '[Router] VOYAGER1 specialist finished'
        '[Router] Visualization specialist finished finished'   (double-wrapped)

    Also checks data._msg which has the cleaner format.
    """
    data = event.get("data", {})
    # Try explicit data fields
    agent = data.get("target_agent") or data.get("agent")
    if agent:
        return agent

    # Prefer data._msg (cleaner), fall back to event msg
    msg = data.get("_msg") or event.get("msg", "")

    # "Delegating to X specialist" — extract X specialist
    m = re.search(r"Delegating to (\w[\w\s-]* specialist)", msg)
    if m:
        return m.group(1)

    # "[Router] X specialist finished" — extract X specialist
    m = re.search(r"(\w[\w\s-]* specialist)\s+finished", msg)
    if m:
        return m.group(1)

    # Broader fallback: "Delegating to ..."
    m = re.search(r"Delegating to (.+?)(?:\s*$)", msg)
    if m:
        return m.group(1)

    return event.get("agent", "?")


def section_orphaned_delegations(events: list[dict]) -> str:
    """Find delegations without matching delegation_done."""
    # Track sync delegations
    open_delegations: dict[str, tuple[int, dict]] = {}  # key -> (line, event)
    # Track async delegations
    open_async: dict[str, tuple[int, dict]] = {}

    for i, e in enumerate(events, 1):
        etype = e.get("type", "")

        if etype == "delegation":
            agent = _extract_delegation_target(e)
            key = f"sync:{agent}:{e.get('ts', '')}"
            open_delegations[key] = (i, e)

        elif etype == "delegation_done":
            agent = _extract_delegation_target(e)
            # Find matching open delegation — match by agent name
            matched_key = None
            for k in open_delegations:
                if f"sync:{agent}:" in k:
                    matched_key = k
                    break
            if matched_key:
                del open_delegations[matched_key]

        elif etype == "delegation_async_started":
            agent = _extract_delegation_target(e)
            key = f"async:{agent}:{e.get('ts', '')}"
            open_async[key] = (i, e)

        elif etype == "delegation_async_completed":
            agent = _extract_delegation_target(e)
            matched_key = None
            for k in open_async:
                if f"async:{agent}:" in k:
                    matched_key = k
                    break
            if matched_key:
                del open_async[matched_key]

    orphans = list(open_delegations.values()) + list(open_async.values())

    if not orphans:
        return "## Orphaned Delegations (0)\n\nAll delegations have matching completions.\n"

    lines = [f"## Orphaned Delegations ({len(orphans)})\n"]
    lines.append("Delegations that started but never received a `delegation_done` event.\n")
    for idx, (ln, e) in enumerate(sorted(orphans, key=lambda x: x[0]), 1):
        agent = _extract_delegation_target(e)
        ts = e.get("ts", "?")[:23]
        etype = e.get("type", "?")
        msg = e.get("msg", "")
        lines.append(f"### {idx}. {agent} (line {ln})")
        lines.append(f"- **Type**: `{etype}`")
        lines.append(f"- **Started**: {ts}")
        lines.append(f"- **Message**: {msg}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ── Section: Unpaired Tool Calls ───────────────────────────────────────────

def section_unpaired_tool_calls(events: list[dict]) -> str:
    """Find tool_call events without matching tool_result or tool_error."""
    open_calls: dict[str, tuple[int, dict]] = {}  # tool_name:ts -> (line, event)

    for i, e in enumerate(events, 1):
        etype = e.get("type", "")
        data = e.get("data", {})

        if etype == "tool_call":
            tool = get_tool_name(e)
            # Use tool name + timestamp as key
            key = f"{tool}:{e.get('ts', '')}"
            open_calls[key] = (i, e)

        elif etype in ("tool_result", "tool_error"):
            tool = get_tool_name(e)
            # Find matching — match by tool name (most recent)
            matched_key = None
            for k in reversed(list(open_calls.keys())):
                if k.startswith(f"{tool}:"):
                    matched_key = k
                    break
            if matched_key:
                del open_calls[matched_key]

    if not open_calls:
        return "## Unpaired Tool Calls (0)\n\nAll tool calls have matching results.\n"

    lines = [f"## Unpaired Tool Calls ({len(open_calls)})\n"]
    lines.append("| # | Line | Tool | Agent | Called At |")
    lines.append("|---|------|------|-------|-----------|")
    for idx, (ln, e) in enumerate(sorted(open_calls.values(), key=lambda x: x[0]), 1):
        tool = get_tool_name(e) or "?"
        agent = e.get("agent", "?")
        ts = e.get("ts", "?")[:23]
        lines.append(f"| {idx} | {ln} | {tool} | {agent} | {ts} |")

    return "\n".join(lines) + "\n"


# ── Section: Data Integrity ────────────────────────────────────────────────

def section_data_integrity(
    events: list[dict],
    data_index: dict | None,
    operations: list | None,
) -> str:
    """Check data label references against the data store index."""
    if data_index is None:
        return "## Data Integrity\n\nNo data/_index.json found — skipping.\n"

    stored_labels = set(data_index.keys())
    issues = []

    # Collect all data labels referenced in events
    referenced_labels: dict[str, list[int]] = defaultdict(list)  # label -> [line_nums]

    for i, e in enumerate(events, 1):
        etype = e.get("type", "")
        data = e.get("data", {})

        if etype == "data_fetched":
            for label in data.get("outputs", []):
                referenced_labels[label].append(i)
        elif etype == "data_computed":
            for label in data.get("outputs", []):
                referenced_labels[label].append(i)
            for label in data.get("inputs", []):
                referenced_labels[label].append(i)
        elif etype == "render_executed":
            # render_executed stores labels in data.inputs (may be stringified list)
            for label in parse_list_field(data.get("inputs", [])):
                referenced_labels[label].append(i)
            # Also check args.data_labels if present
            args = data.get("args", {})
            if isinstance(args, dict):
                for label in parse_list_field(args.get("data_labels", [])):
                    referenced_labels[label].append(i)
            elif isinstance(args, str) and "data_label" in args:
                # args may be stringified dict — try to extract data_labels
                try:
                    parsed = json.loads(args.replace("'", '"'))
                    if isinstance(parsed, dict):
                        fg = parsed.get("figure_json", {})
                        if isinstance(fg, dict):
                            for trace in fg.get("data", []):
                                dl = trace.get("data_label")
                                if dl:
                                    referenced_labels[dl].append(i)
                except (json.JSONDecodeError, ValueError, AttributeError):
                    pass

    # Check operations.json too
    if operations:
        for op in operations:
            for label in op.get("outputs", []):
                referenced_labels[label].append(-1)
            for label in op.get("inputs", []):
                referenced_labels[label].append(-1)

    # Find labels referenced but not in store
    missing_from_store = {}
    for label, line_nums in referenced_labels.items():
        if label not in stored_labels:
            missing_from_store[label] = line_nums

    # Find labels in store but never referenced in events
    unreferenced = stored_labels - set(referenced_labels.keys())

    lines = ["## Data Integrity\n"]

    if missing_from_store:
        lines.append("### Missing Data Labels\n")
        lines.append("These labels are referenced in events/operations but not found in `data/_index.json`:\n")
        for label, line_nums in missing_from_store.items():
            refs = ", ".join(str(ln) for ln in line_nums if ln > 0)
            # Try to find a close match in the store
            close_matches = [
                sl for sl in stored_labels
                if label.split(".")[0] in sl or sl.split(".")[0] in label
            ]
            match_hint = ""
            if close_matches:
                match_hint = f" (similar in store: {', '.join(f'`{m}`' for m in close_matches[:3])})"
            if refs:
                issues.append(f"- `{label}` — referenced at lines {refs}{match_hint}")
            else:
                issues.append(f"- `{label}` — referenced in operations.json{match_hint}")
        lines.extend(issues)
        lines.append("")

    if not missing_from_store:
        lines.append("All referenced data labels present in store.\n")

    # Fetch outputs check
    fetch_outputs = set()
    for e in events:
        if e.get("type") == "data_fetched":
            for label in e.get("data", {}).get("outputs", []):
                fetch_outputs.add(label)
    fetch_missing = fetch_outputs - stored_labels
    if fetch_missing:
        lines.append("### Fetched but Not Persisted\n")
        for label in sorted(fetch_missing):
            lines.append(f"- `{label}`")
        lines.append("")
    elif fetch_outputs:
        lines.append("All fetch outputs present in store.\n")

    # Show what IS in the store for reference
    if stored_labels:
        lines.append("### Labels in Store\n")
        for label in sorted(stored_labels):
            entry = data_index.get(label, {})
            source = entry.get("source", "?")
            desc = entry.get("description", "")[:60]
            lines.append(f"- `{label}` ({source}) — {desc}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ── Section: Timeline Anomalies ────────────────────────────────────────────

def section_timeline(events: list[dict]) -> str:
    """Detect out-of-order timestamps, slow tool calls, and slow LLM calls."""
    anomalies = []

    # Check for out-of-order timestamps
    prev_ts = None
    prev_line = 0
    for i, e in enumerate(events, 1):
        ts = parse_ts(e.get("ts", ""))
        if ts and prev_ts:
            # Allow naive/aware comparison by stripping tzinfo if mixed
            try:
                if ts < prev_ts:
                    anomalies.append(
                        f"- **Out-of-order**: line {i} ({e.get('ts', '')[:23]}) "
                        f"is before line {prev_line} ({events[prev_line - 1].get('ts', '')[:23]})"
                    )
            except TypeError:
                pass  # Mixed tz-aware/naive — skip comparison
        if ts:
            prev_ts = ts
            prev_line = i

    # Find slow tool calls (tool_call -> tool_result pairs)
    open_tool_calls: dict[str, tuple[int, datetime, str]] = {}  # key -> (line, ts, tool_name)

    for i, e in enumerate(events, 1):
        etype = e.get("type", "")
        ts = parse_ts(e.get("ts", ""))

        if etype == "tool_call" and ts:
            tool = get_tool_name(e) or "?"
            key = f"{tool}:{e.get('ts', '')}"
            open_tool_calls[key] = (i, ts, tool)

        elif etype in ("tool_result", "tool_error") and ts:
            tool = get_tool_name(e) or "?"
            matched_key = None
            for k in reversed(list(open_tool_calls.keys())):
                if k.startswith(f"{tool}:"):
                    matched_key = k
                    break
            if matched_key:
                start_line, start_ts, start_tool = open_tool_calls.pop(matched_key)
                try:
                    duration = (ts - start_ts).total_seconds()
                except TypeError:
                    continue
                if duration > SLOW_TOOL_THRESHOLD_S:
                    anomalies.append(
                        f"- **Slow tool call**: `{start_tool}` took {fmt_duration(duration)} "
                        f"(line {start_line} → {i})"
                    )

    # Find slow LLM calls
    for i, e in enumerate(events, 1):
        if e.get("type") == "llm_call":
            data = e.get("data", {})
            duration = data.get("duration_s") or data.get("duration")
            if duration and float(duration) > SLOW_LLM_THRESHOLD_S:
                agent = e.get("agent", "?")
                anomalies.append(
                    f"- **Slow LLM call**: {agent} took {fmt_duration(float(duration))} (line {i})"
                )

    if not anomalies:
        return "## Timeline\n\nNo anomalies detected.\n"

    lines = [f"## Timeline ({len(anomalies)} anomalies)\n"]
    lines.extend(anomalies)
    return "\n".join(lines) + "\n"


# ── Section: Retry Loops ───────────────────────────────────────────────────

def section_retry_loops(events: list[dict]) -> str:
    """Detect repeated tool calls with similar args and repeated error messages."""
    findings = []

    # Track tool calls by tool name with their args
    # Deduplicate by timestamp — dual-logging (sub-agent + executor) emits the
    # same logical call twice at the same timestamp, which is not a retry.
    tool_call_groups: dict[str, list[tuple[int, str, str]]] = defaultdict(list)  # tool -> [(line, ts, args_json)]
    for i, e in enumerate(events, 1):
        if e.get("type") == "tool_call":
            data = e.get("data", {})
            tool = get_tool_name(e) or "?"
            ts = e.get("ts", "")
            args_json = json.dumps(data.get("tool_args", data.get("args", {})), sort_keys=True)
            tool_call_groups[tool].append((i, ts, args_json))

    for tool, calls in tool_call_groups.items():
        # Deduplicate: collapse calls with same args at same timestamp
        seen: dict[tuple[str, str], int] = {}  # (ts, args) -> first line
        deduped: list[tuple[int, str]] = []  # (line, args_json)
        for ln, ts, args_json in calls:
            key = (ts, args_json)
            if key not in seen:
                seen[key] = ln
                deduped.append((ln, args_json))

        if len(deduped) >= RETRY_LOOP_THRESHOLD:
            # Group by identical args
            arg_groups: dict[str, list[int]] = defaultdict(list)
            for ln, args_json in deduped:
                arg_groups[args_json].append(ln)

            for args_json, line_nums in arg_groups.items():
                if len(line_nums) >= RETRY_LOOP_THRESHOLD:
                    lines_str = ", ".join(str(ln) for ln in line_nums[:10])
                    if len(line_nums) > 10:
                        lines_str += ", ..."
                    findings.append(
                        f"- `{tool}` called {len(line_nums)}x with identical args "
                        f"(lines {lines_str})"
                    )

    # Check for repeated error messages
    error_msgs: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(events, 1):
        if e.get("level") in ("error", "warning") or e.get("type") in ERROR_EVENT_TYPES:
            # Normalize message for grouping (strip variable parts like timestamps)
            msg = e.get("msg", "")[:120]
            error_msgs[msg].append(i)

    for msg, line_nums in error_msgs.items():
        if len(line_nums) >= RETRY_LOOP_THRESHOLD:
            lines_str = ", ".join(str(ln) for ln in line_nums[:10])
            if len(line_nums) > 10:
                lines_str += ", ..."
            short_msg = msg[:80]
            findings.append(
                f"- Error repeated {len(line_nums)}x: \"{short_msg}\" "
                f"(lines {lines_str})"
            )

    if not findings:
        return "## Retry Loops\n\nNo retry loops detected.\n"

    lines = [f"## Retry Loops ({len(findings)})\n"]
    lines.extend(findings)
    return "\n".join(lines) + "\n"


# ── Prompt Analysis: Shared Helpers ────────────────────────────────────────

# Issue marker prefix used by prompt analysis sections.
# section_overview scans for this to count prompt-quality issues.
_PROMPT_ISSUE_MARKER = "**[P"

# Patterns for intent detection
NEGATION_PATTERNS = [
    (re.compile(r"don'?t\s+use\s+pipeline", re.I), "no_pipeline"),
    (re.compile(r"from\s+scratch", re.I), "no_pipeline"),
    (re.compile(r"don'?t\s+use\s+(\w+)", re.I), "negation"),
    (re.compile(r"without\s+(\w+)", re.I), "negation"),
    (re.compile(r"do\s*n(?:ot|'t)\s+(\w+)", re.I), "negation"),
]

VIZ_KEYWORDS = {"show", "plot", "visualize", "display", "figure", "chart", "graph"}
DATA_KEYWORDS = {"fetch", "get", "download", "load", "retrieve"}

# Known mission names for matching
MISSION_NAMES = {
    "ace", "wind", "psp", "parker", "soho", "stereo", "voyager",
    "voyager1", "voyager2", "cassini", "juno", "ulysses", "ibex",
    "maven", "new horizons", "messenger", "dawn", "galileo",
    "pioneer", "themis", "van allen", "dscovr", "solar orbiter",
    "solo", "bepi", "bepicolombo", "juice", "psyche", "lucy",
    "europa clipper", "mro", "insight", "lro",
}

ACKNOWLEDGMENT_KEYWORDS = {
    "error", "fail", "failed", "couldn't", "could not", "unable",
    "issue", "problem", "sorry", "unfortunately", "not available",
    "not found", "missing", "exception", "cannot",
}


def _group_events_by_turn(events: list[dict]) -> list[dict]:
    """Split events into turns, each bounded by user_message → agent_response.

    Returns a list of dicts:
        {
            "user": (line, event) | None,
            "events": [(line, event), ...],
            "response": (line, event) | None,
            "cycle": int | None,
        }
    """
    turns = []
    current_turn = None

    for i, e in enumerate(events, 1):
        etype = e.get("type", "")

        if etype == "user_message":
            # Close any open turn
            if current_turn is not None:
                turns.append(current_turn)
            cycle = e.get("data", {}).get("cycle")
            current_turn = {
                "user": (i, e),
                "events": [],
                "response": None,
                "cycle": cycle,
            }

        elif current_turn is not None:
            if etype == "agent_response":
                current_turn["response"] = (i, e)
                current_turn["events"].append((i, e))
                turns.append(current_turn)
                current_turn = None
            else:
                current_turn["events"].append((i, e))

    # Close any trailing turn without a response
    if current_turn is not None:
        turns.append(current_turn)

    return turns


def _extract_delegation_info(events: list[dict]) -> list[tuple[int, str, str, int | None]]:
    """Extract all delegation tool calls.

    Returns list of (line, target_agent, request_text, cycle).
    """
    delegations = []
    for i, e in enumerate(events, 1):
        if e.get("type") != "delegation":
            continue
        agent = _extract_delegation_target(e)
        data = e.get("data", {})
        # Get the request text from the delegation message or data
        request_text = data.get("request") or data.get("_msg") or e.get("msg", "")
        cycle = data.get("cycle")
        delegations.append((i, agent, request_text, cycle))
    return delegations


def _word_set(text: str) -> set[str]:
    """Extract a set of lowercase words from text."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _word_overlap(text1: str, text2: str) -> float:
    """Compute word overlap ratio between two texts (Jaccard similarity)."""
    words1 = _word_set(text1)
    words2 = _word_set(text2)
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


# ── Section 10: Intent Tracking ───────────────────────────────────────────

def section_intent_tracking(events: list[dict]) -> str:
    """Detect user intents that were not fulfilled by the agent."""
    turns = _group_events_by_turn(events)
    findings = []

    for turn in turns:
        if turn["user"] is None:
            continue
        user_line, user_event = turn["user"]
        user_data = user_event.get("data", {})
        user_text = (user_data.get("text") or user_event.get("msg", "")).lower()
        if user_text.startswith("[user] "):
            user_text = user_text[7:]

        # Collect event types in this turn
        turn_types = set()
        turn_tool_names = set()
        for _, te in turn["events"]:
            turn_types.add(te.get("type", ""))
            tn = get_tool_name(te)
            if tn:
                turn_tool_names.add(tn.lower())

        response_text = ""
        if turn["response"]:
            resp_line, resp_event = turn["response"]
            response_text = (resp_event.get("data", {}).get("text") or
                             resp_event.get("msg", "")).lower()

        # --- Check 1: Visualization requested but not produced ---
        user_words = set(user_text.split())
        if user_words & VIZ_KEYWORDS:
            has_render = "render_executed" in turn_types
            has_plot_ref = any(
                kw in response_text
                for kw in ("figure", "plot", "chart", "here is", "visualization")
            )
            if not has_render and not has_plot_ref:
                findings.append(
                    f"- **[P1: Viz not produced]** Turn at line {user_line}: "
                    f"User asked to visualize data but no `render_executed` event "
                    f"and response does not reference a figure.\n"
                    f"  User: \"{user_text[:120]}\""
                )

        # --- Check 2: Explicit instruction ignored ---
        for pattern, intent_type in NEGATION_PATTERNS:
            m = pattern.search(user_text)
            if not m:
                continue

            if intent_type == "no_pipeline":
                # Check if pipeline tools were used
                pipeline_tools = {"execute_pipeline", "run_pipeline", "pipeline"}
                if turn_tool_names & pipeline_tools:
                    findings.append(
                        f"- **[P2: Instruction ignored]** Turn at line {user_line}: "
                        f"User said \"{m.group(0)}\" but pipeline tools were used.\n"
                        f"  Tools used: {', '.join(sorted(turn_tool_names & pipeline_tools))}"
                    )
            elif intent_type == "negation":
                negated_word = m.group(1).lower() if m.lastindex else ""
                if negated_word and negated_word in turn_tool_names:
                    findings.append(
                        f"- **[P2: Instruction ignored]** Turn at line {user_line}: "
                        f"User said \"{m.group(0)}\" but `{negated_word}` was used anyway."
                    )

        # --- Check 3: Data request not fulfilled ---
        if user_words & DATA_KEYWORDS:
            has_data = "data_fetched" in turn_types or "data_computed" in turn_types
            if not has_data:
                # Only flag if there's no delegation (delegation might handle it)
                has_delegation = "delegation" in turn_types
                if not has_delegation:
                    findings.append(
                        f"- **[P3: Data not fetched]** Turn at line {user_line}: "
                        f"User requested data but no `data_fetched` or `data_computed` "
                        f"event in this turn.\n"
                        f"  User: \"{user_text[:120]}\""
                    )

    if not findings:
        return "## Intent Tracking\n\nAll detected user intents appear to have been addressed.\n"

    lines = [f"## Intent Tracking ({len(findings)} issues)\n"]
    lines.extend(findings)
    return "\n".join(lines) + "\n"


# ── Section 11: Delegation Analysis ──────────────────────────────────────

def section_delegation_analysis(events: list[dict]) -> str:
    """Detect redundant delegations, state tracking failures, and delegation loops."""
    findings = []
    delegations = _extract_delegation_info(events)

    # --- Check 1: Delegation loops (same agent 3+ times per cycle) ---
    # Group delegations by (target_agent, cycle)
    by_agent_cycle: dict[tuple[str, int | None], list[tuple[int, str]]] = defaultdict(list)
    for line, agent, request, cycle in delegations:
        by_agent_cycle[(agent, cycle)].append((line, request))

    for (agent, cycle), group in by_agent_cycle.items():
        if len(group) < RETRY_LOOP_THRESHOLD:
            continue

        lines_str = ", ".join(str(ln) for ln, _ in group)
        cycle_str = f" in cycle {cycle}" if cycle is not None else ""
        findings.append(
            f"- **[P4: Delegation loop]** `{agent}` delegated to "
            f"{len(group)}x{cycle_str} (lines {lines_str})."
        )

        # Check for overlapping request text (redundant delegations)
        requests_text = [(ln, req) for ln, req in group]
        overlap_pairs = []
        for i_idx in range(len(requests_text)):
            for j_idx in range(i_idx + 1, len(requests_text)):
                overlap = _word_overlap(requests_text[i_idx][1], requests_text[j_idx][1])
                if overlap > 0.5:
                    overlap_pairs.append(
                        (requests_text[i_idx][0], requests_text[j_idx][0], overlap)
                    )
        if overlap_pairs:
            for ln1, ln2, ov in overlap_pairs[:3]:
                findings.append(
                    f"  - Redundant: lines {ln1} and {ln2} have "
                    f"{ov:.0%} word overlap in request text."
                )

    # --- Check 2: State tracking failures ---
    # Find data_fetched events, then look for subsequent delegations that claim
    # data was NOT fetched
    fetched_labels: dict[str, int] = {}  # label -> line where fetched
    for i, e in enumerate(events, 1):
        if e.get("type") == "data_fetched":
            for label in e.get("data", {}).get("outputs", []):
                fetched_labels[label] = i

    contradiction_phrases = [
        "not fetched", "not actually fetch", "did not fetch",
        "empty", "no data", "failed to fetch", "NOT fetched",
        "did NOT", "haven't fetched", "hasn't fetched",
    ]

    for line, agent, request, cycle in delegations:
        request_lower = request.lower()
        for phrase in contradiction_phrases:
            if phrase.lower() in request_lower:
                # Check if data WAS actually fetched before this delegation
                data_before = {
                    label: fline for label, fline in fetched_labels.items()
                    if fline < line
                }
                if data_before:
                    labels_str = ", ".join(
                        f"`{l}` (line {ln})" for l, ln in
                        sorted(data_before.items(), key=lambda x: x[1])[:5]
                    )
                    findings.append(
                        f"- **[P5: State tracking failure]** Line {line}: "
                        f"Delegation to `{agent}` claims \"{phrase}\" but "
                        f"data was fetched: {labels_str}."
                    )
                break

    if not findings:
        return "## Delegation Analysis\n\nNo delegation issues detected.\n"

    lines_out = [f"## Delegation Analysis ({len(findings)} issues)\n"]
    lines_out.extend(findings)
    return "\n".join(lines_out) + "\n"


# ── Section 12: Silent Failures ───────────────────────────────────────────

def section_silent_failures(events: list[dict]) -> str:
    """Detect tool errors that the agent response doesn't acknowledge."""
    turns = _group_events_by_turn(events)
    findings = []

    for turn in turns:
        if turn["user"] is None:
            continue
        user_line = turn["user"][0]

        # Collect errors in this turn (expanded coverage beyond ERROR_EVENT_TYPES)
        turn_errors = []
        for _, te in turn["events"]:
            etype = te.get("type", "")
            data = te.get("data", {})

            is_error = (
                te.get("level") in ("error", "warning")
                or etype in ERROR_EVENT_TYPES
                or (etype == "sub_agent_tool" and data.get("status") == "error")
            )
            if is_error:
                error_msg = data.get("error") or te.get("msg", "")
                tool = get_tool_name(te) or etype
                turn_errors.append((tool, error_msg))

        if not turn_errors:
            continue

        # Check response for acknowledgment
        response_text = ""
        response_line = None
        if turn["response"]:
            response_line, resp_event = turn["response"]
            response_text = (resp_event.get("data", {}).get("text") or
                             resp_event.get("msg", "")).lower()

        has_acknowledgment = any(
            kw in response_text for kw in ACKNOWLEDGMENT_KEYWORDS
        )

        # --- Check 1: Unacknowledged tool errors ---
        if not has_acknowledgment and turn_errors:
            error_summary = ", ".join(
                f"`{tool}`" for tool, _ in turn_errors[:5]
            )
            if len(turn_errors) > 5:
                error_summary += f", ... (+{len(turn_errors) - 5} more)"
            resp_preview = response_text[:100] if response_text else "(no response)"
            findings.append(
                f"- **[P6: Silent errors]** Turn at line {user_line}: "
                f"{len(turn_errors)} error(s) from {error_summary} "
                f"but response doesn't acknowledge failure.\n"
                f"  Response: \"{resp_preview}\""
            )

        # --- Check 2: Uninformative error response ---
        if turn_errors and response_text and len(response_text) < 100:
            if not has_acknowledgment:
                findings.append(
                    f"- **[P7: Uninformative response]** Turn at line {user_line}: "
                    f"{len(turn_errors)} errors occurred but response is only "
                    f"{len(response_text)} chars without error keywords.\n"
                    f"  Response: \"{response_text}\""
                )

        # --- Check 3: Cascading failures ---
        # Check if a fetch error label matches labels in subsequent errors
        fetch_error_labels = set()
        for tool, msg in turn_errors:
            if "fetch" in tool.lower():
                # Try to extract label from error message
                label_match = re.search(r"label[=: ]+['\"]?(\S+)", msg)
                if label_match:
                    fetch_error_labels.add(label_match.group(1).strip("'\""))

        if fetch_error_labels:
            cascade_count = 0
            for tool, msg in turn_errors:
                if "fetch" not in tool.lower():
                    for label in fetch_error_labels:
                        if label in msg:
                            cascade_count += 1
                            break
            if cascade_count > 0:
                findings.append(
                    f"- **[P8: Cascading failure]** Turn at line {user_line}: "
                    f"Fetch errors on labels {', '.join(f'`{l}`' for l in fetch_error_labels)} "
                    f"caused {cascade_count} downstream error(s)."
                )

    if not findings:
        return "## Silent Failures\n\nAll errors were properly acknowledged in responses.\n"

    lines = [f"## Silent Failures ({len(findings)} issues)\n"]
    lines.extend(findings)
    return "\n".join(lines) + "\n"


# ── Section 13: Response Quality ──────────────────────────────────────────

def section_response_quality(events: list[dict]) -> str:
    """Detect response quality issues: unaddressed requests, false state claims."""
    turns = _group_events_by_turn(events)
    findings = []

    for turn in turns:
        if turn["user"] is None:
            continue
        user_line, user_event = turn["user"]
        user_data = user_event.get("data", {})
        user_text = user_data.get("text") or user_event.get("msg", "")
        if user_text.startswith("[User] "):
            user_text = user_text[7:]
        user_text_lower = user_text.lower()

        if not turn["response"]:
            continue
        response_line, resp_event = turn["response"]
        response_text = (resp_event.get("data", {}).get("text") or
                         resp_event.get("msg", ""))
        response_lower = response_text.lower()

        # --- Check 1: Key nouns from user message absent from response ---
        # Extract mission names mentioned by user
        mentioned_missions = []
        for name in MISSION_NAMES:
            if name in user_text_lower:
                mentioned_missions.append(name)

        if mentioned_missions:
            missing = [
                m for m in mentioned_missions
                if m not in response_lower
            ]
            # Only flag if ALL mentioned missions are missing (suggests
            # the response is completely off-topic)
            if missing and len(missing) == len(mentioned_missions):
                findings.append(
                    f"- **[P9: Unaddressed request]** Turn at line {user_line}: "
                    f"User mentioned {', '.join(mentioned_missions)} but "
                    f"response doesn't reference any of them.\n"
                    f"  Response preview: \"{response_text[:120]}\""
                )

        # --- Check 2: Agent claims false state ---
        # "data was not fetched" / "not produced" when events show otherwise
        false_state_patterns = [
            (r"(?:data|results?)\s+(?:was|were)\s+not\s+(?:fetched|found|available)",
             "data_fetched"),
            (r"visualization[:\s]+not\s+produced", "render_executed"),
            (r"no\s+data\s+(?:was\s+)?(?:fetched|found|available)", "data_fetched"),
        ]

        turn_event_types = set(te.get("type", "") for _, te in turn["events"])

        for pattern, expected_event in false_state_patterns:
            if re.search(pattern, response_lower):
                if expected_event in turn_event_types:
                    findings.append(
                        f"- **[P10: False state claim]** Turn at line {user_line}: "
                        f"Response claims \"{re.search(pattern, response_lower).group(0)}\" "
                        f"but `{expected_event}` events exist in this turn."
                    )

        # --- Check 3: Thinking-action mismatch ---
        thinking_text = ""
        for _, te in turn["events"]:
            if te.get("type") == "thinking":
                thinking_text += " " + (te.get("data", {}).get("text") or
                                        te.get("msg", ""))

        if thinking_text:
            thinking_lower = thinking_text.lower()

            # Check: thinking mentions a specific action but tool calls don't match
            # e.g., thinking says "plot RTN" but render uses GSE
            thinking_tools = set()
            actual_tools = set()

            for _, te in turn["events"]:
                if te.get("type") == "tool_call":
                    actual_tools.add(get_tool_name(te).lower())

            # Check for frame mismatches in thinking vs tool args
            frame_pattern = re.compile(r"\b(rtn|gse|gsm|hci|hee|eclipj2000|j2000)\b", re.I)
            thinking_frames = set(m.group(0).upper() for m in frame_pattern.finditer(thinking_text))

            if thinking_frames:
                for _, te in turn["events"]:
                    if te.get("type") == "tool_call":
                        args = te.get("data", {}).get("tool_args", {})
                        args_str = json.dumps(args).upper()
                        actual_frames = set(m.group(0) for m in frame_pattern.finditer(args_str))
                        if actual_frames and thinking_frames and not (thinking_frames & actual_frames):
                            findings.append(
                                f"- **[P11: Thinking-action mismatch]** Turn at line {user_line}: "
                                f"Thinking mentions frame(s) {thinking_frames} but tool call "
                                f"uses {actual_frames}."
                            )

    if not findings:
        return "## Response Quality\n\nNo response quality issues detected.\n"

    lines = [f"## Response Quality ({len(findings)} issues)\n"]
    lines.extend(findings)
    return "\n".join(lines) + "\n"


# ── Section: Token Usage (Detailed) ────────────────────────────────────────


def _pct(part: int, whole: int) -> str:
    """Format a percentage, returning '—' if whole is 0."""
    if whole == 0:
        return "—"
    return f"{part / whole * 100:.1f}%"


def _agent_table(agent_stats: dict[str, dict]) -> list[str]:
    """Render an agent breakdown table from {agent: {input,output,thinking,cached,calls}}."""
    sorted_agents = sorted(agent_stats.items(), key=lambda x: x[1]["input"], reverse=True)
    lines = []
    lines.append("| Agent | Input | Output | Thinking | Cached | Cache% | API Calls |")
    lines.append("|-------|-------|--------|----------|--------|--------|-----------|")

    total = {"input": 0, "output": 0, "thinking": 0, "cached": 0, "calls": 0}
    for agent, s in sorted_agents:
        cache_rate = _pct(s["cached"], s["input"])
        lines.append(
            f"| {agent} | {fmt_tokens(s['input'])} | "
            f"{fmt_tokens(s['output'])} | {fmt_tokens(s['thinking'])} | "
            f"{fmt_tokens(s['cached'])} | {cache_rate} | {s['calls']} |"
        )
        for k in total:
            total[k] += s[k]

    cache_rate = _pct(total["cached"], total["input"])
    lines.append(
        f"| **Total** | **{fmt_tokens(total['input'])}** | "
        f"**{fmt_tokens(total['output'])}** | **{fmt_tokens(total['thinking'])}** | "
        f"**{fmt_tokens(total['cached'])}** | **{cache_rate}** | **{total['calls']}** |"
    )
    return lines, total


def section_token_usage(events: list[dict], session_dir: Path) -> str:
    """Detailed token usage analysis: per-agent, per-round, per-call timeline, and cost."""
    # ── Collect token_usage events (per-agent per-call) ──
    token_file = session_dir / "token_usage.jsonl"
    if token_file.exists():
        token_records = load_jsonl(token_file)
    else:
        token_records = [
            e.get("data", {})
            for e in events
            if e.get("type") == "token_usage"
        ]

    # ── Collect round_end events (per-round totals) ──
    round_events = [e for e in events if e.get("type") == "round_end"]

    # ── Collect all token_usage events with timestamps for timeline ──
    token_timeline = [
        e for e in events if e.get("type") == "token_usage"
    ]

    has_agent_data = bool(token_records)
    has_round_data = any("token_usage" in e.get("data", {}) for e in round_events)

    if not has_agent_data and not has_round_data:
        return "## Token Usage\n\nNo token usage data found.\n"

    lines = ["## Token Usage\n"]

    # ── 1. Session totals from metadata ──
    metadata = load_json(session_dir / "metadata.json")
    if metadata and "token_usage" in metadata:
        tu = metadata["token_usage"]
        inp = tu.get("input_tokens", 0)
        out = tu.get("output_tokens", 0)
        think = tu.get("thinking_tokens", 0)
        cached = tu.get("cached_tokens", 0)
        total = tu.get("total_tokens", 0)
        calls = tu.get("api_calls", 0)

        lines.append("### Session Totals\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Input tokens | {fmt_tokens(inp)} |")
        lines.append(f"| Output tokens | {fmt_tokens(out)} |")
        lines.append(f"| Thinking tokens | {fmt_tokens(think)} |")
        lines.append(f"| Cached tokens | {fmt_tokens(cached)} |")
        lines.append(f"| Total tokens | {fmt_tokens(total)} |")
        lines.append(f"| API calls | {calls} |")
        lines.append(f"| Cache hit rate | {_pct(cached, inp)} |")
        if inp > 0:
            lines.append(f"| Output/Input ratio | {_pct(out, inp)} |")
        if calls > 0:
            lines.append(f"| Avg input/call | {fmt_tokens(inp // calls)} |")
            lines.append(f"| Avg output/call | {fmt_tokens(out // calls)} |")
        lines.append("")

    # ── 2. Per-round breakdown ──
    if has_round_data:
        lines.append("### Per-Round Breakdown\n")
        lines.append("| Round | Input | Output | Thinking | Cached | Cache% | Calls | Cumulative In |")
        lines.append("|-------|-------|--------|----------|--------|--------|-------|---------------|")

        for i, re_ in enumerate(round_events, 1):
            d = re_.get("data", {})
            rtu = d.get("round_token_usage", {})
            cum = d.get("token_usage", {})

            r_in = rtu.get("input_tokens", 0)
            r_out = rtu.get("output_tokens", 0)
            r_think = rtu.get("thinking_tokens", 0)
            r_cached = rtu.get("cached_tokens", 0)
            r_calls = rtu.get("api_calls", 0)
            cum_in = cum.get("input_tokens", 0)

            cache_rate = _pct(r_cached, r_in)
            lines.append(
                f"| {i} | {fmt_tokens(r_in)} | {fmt_tokens(r_out)} | "
                f"{fmt_tokens(r_think)} | {fmt_tokens(r_cached)} | {cache_rate} | "
                f"{r_calls} | {fmt_tokens(cum_in)} |"
            )

        # Show growth pattern
        if len(round_events) >= 2:
            first_round = round_events[0].get("data", {}).get("round_token_usage", {})
            last_round = round_events[-1].get("data", {}).get("round_token_usage", {})
            first_in = first_round.get("input_tokens", 0)
            last_in = last_round.get("input_tokens", 0)
            if first_in > 0:
                growth = last_in / first_in
                if growth > 1.5:
                    lines.append(f"\n**Context growth**: Round 1 → Round {len(round_events)}: "
                                 f"{fmt_tokens(first_in)} → {fmt_tokens(last_in)} input "
                                 f"({growth:.1f}x growth)")
                elif growth < 0.5:
                    lines.append(f"\n**Context shrink**: Round 1 → Round {len(round_events)}: "
                                 f"{fmt_tokens(first_in)} → {fmt_tokens(last_in)} input "
                                 f"({growth:.1f}x)")
        lines.append("")

    # ── 3. Per-agent breakdown ──
    if has_agent_data:
        agent_stats: dict[str, dict] = defaultdict(
            lambda: {"input": 0, "output": 0, "thinking": 0, "cached": 0, "calls": 0}
        )
        for rec in token_records:
            agent = rec.get("agent_name") or rec.get("agent") or "unknown"
            agent_stats[agent]["input"] += rec.get("input_tokens", 0)
            agent_stats[agent]["output"] += rec.get("output_tokens", 0)
            agent_stats[agent]["thinking"] += rec.get("thinking_tokens", 0)
            agent_stats[agent]["cached"] += rec.get("cached_tokens", 0)
            agent_stats[agent]["calls"] += 1

        lines.append("### Per-Agent Breakdown\n")
        table_lines, agent_total = _agent_table(agent_stats)
        lines.extend(table_lines)

        # Compare sub-agent total to session total
        if metadata and "token_usage" in metadata:
            session_in = metadata["token_usage"].get("input_tokens", 0)
            tracked_in = agent_total["input"]
            if session_in > 0 and tracked_in > 0:
                orchestrator_in = session_in - tracked_in
                if orchestrator_in > 0:
                    lines.append(f"\n**Orchestrator (inferred)**: {fmt_tokens(orchestrator_in)} input tokens "
                                 f"({_pct(orchestrator_in, session_in)} of session total) — "
                                 f"not tracked in per-agent token_usage events.")
        lines.append("")

    # ── 4. Per-call timeline ──
    if token_timeline:
        lines.append("### Per-Call Timeline\n")
        lines.append("Chronological token_usage events showing each sub-agent API call:\n")
        lines.append("| # | Time | Agent | Context | Input | Output | Thinking | Cached | Cum. Input |")
        lines.append("|---|------|-------|---------|-------|--------|----------|--------|------------|")

        for i, e in enumerate(token_timeline, 1):
            d = e.get("data", {})
            ts = e.get("ts", "?")
            if len(ts) > 19:
                ts = ts[11:19]  # Extract HH:MM:SS
            agent = d.get("agent_name", e.get("agent", "?"))
            ctx = d.get("tool_context", "?")
            inp = d.get("input_tokens", 0)
            out = d.get("output_tokens", 0)
            think = d.get("thinking_tokens", 0)
            cached = d.get("cached_tokens", 0)
            cum_in = d.get("cumulative_input", 0)

            lines.append(
                f"| {i} | {ts} | {agent} | {ctx} | "
                f"{fmt_tokens(inp)} | {fmt_tokens(out)} | {fmt_tokens(think)} | "
                f"{fmt_tokens(cached)} | {fmt_tokens(cum_in)} |"
            )
        lines.append("")

    # ── 5. Efficiency analysis ──
    lines.append("### Efficiency Analysis\n")
    efficiency_notes = []

    if metadata and "token_usage" in metadata:
        tu = metadata["token_usage"]
        inp = tu.get("input_tokens", 0)
        out = tu.get("output_tokens", 0)
        cached = tu.get("cached_tokens", 0)
        calls = tu.get("api_calls", 0)

        # Cache efficiency
        if inp > 0:
            cache_pct = cached / inp * 100
            if cache_pct >= 50:
                efficiency_notes.append(f"- **Good cache utilization**: {cache_pct:.0f}% of input tokens were cached")
            elif cache_pct >= 20:
                efficiency_notes.append(f"- **Moderate cache utilization**: {cache_pct:.0f}% of input tokens were cached")
            elif cache_pct > 0:
                efficiency_notes.append(f"- **Low cache utilization**: Only {cache_pct:.0f}% of input tokens were cached — "
                                        "consider whether context is being reused effectively")
            else:
                efficiency_notes.append("- **No caching**: 0% cache hit rate — all tokens were fresh inputs")

        # Output density (how much useful output per input)
        if inp > 0:
            ratio = out / inp * 100
            if ratio < 1:
                efficiency_notes.append(f"- **Low output density**: {ratio:.1f}% output/input ratio — "
                                        "large context with minimal generation")

        # API call efficiency
        if calls > 0 and inp > 0:
            avg_in = inp // calls
            if avg_in > 50000:
                efficiency_notes.append(f"- **Large average context**: {fmt_tokens(avg_in)}/call — "
                                        "context window usage is high")

    # Round growth analysis
    if len(round_events) >= 2:
        round_inputs = []
        for re_ in round_events:
            rtu = re_.get("data", {}).get("round_token_usage", {})
            round_inputs.append(rtu.get("input_tokens", 0))

        if len(round_inputs) >= 2 and round_inputs[0] > 0:
            growth_factor = round_inputs[-1] / round_inputs[0]
            if growth_factor < 0.5:
                efficiency_notes.append(
                    f"- **Context reduced across rounds**: {growth_factor:.1f}x — "
                    "later rounds used less context (good)")
            elif growth_factor > 2.0:
                efficiency_notes.append(
                    f"- **Context grew across rounds**: {growth_factor:.1f}x — "
                    "conversation history may be inflating context")

    # Sub-agent overhead
    if has_agent_data and metadata and "token_usage" in metadata:
        session_in = metadata["token_usage"].get("input_tokens", 0)
        agent_in = sum(s["input"] for s in agent_stats.values()) if has_agent_data else 0
        if session_in > 0 and agent_in > 0:
            sub_agent_pct = agent_in / session_in * 100
            if sub_agent_pct > 30:
                efficiency_notes.append(
                    f"- **Sub-agent overhead**: {sub_agent_pct:.0f}% of total input went to sub-agents")

    if efficiency_notes:
        lines.extend(efficiency_notes)
    else:
        lines.append("No notable efficiency issues detected.")
    lines.append("")

    return "\n".join(lines) + "\n"


# ── Main ───────────────────────────────────────────────────────────────────

def check_session(session_id: str) -> str:
    """Run all diagnostic checks on a session and return the report."""
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        return f"Error: Session directory not found: {session_dir}"

    # Load data files
    events_path = session_dir / "events.jsonl"
    if not events_path.exists():
        return f"Error: events.jsonl not found in {session_dir}"

    events = load_jsonl(events_path)
    metadata = load_json(session_dir / "metadata.json")
    operations = load_json(session_dir / "operations.json")
    data_index = load_json(session_dir / "data" / "_index.json")

    # Collect token_usage events for overview
    token_events = [e for e in events if e.get("type") == "token_usage"]

    # Build report
    sections = []
    sections.append(f"# Session Check: {session_id}\n")
    sections.append(section_overview(session_id, metadata, events, data_index, token_events))
    sections.append(section_user_messages(events))
    sections.append(section_errors(events))
    sections.append(section_orphaned_delegations(events))
    sections.append(section_unpaired_tool_calls(events))
    sections.append(section_data_integrity(events, data_index, operations))
    sections.append(section_timeline(events))
    sections.append(section_retry_loops(events))
    sections.append(section_intent_tracking(events))
    sections.append(section_delegation_analysis(events))
    sections.append(section_silent_failures(events))
    sections.append(section_response_quality(events))
    sections.append(section_token_usage(events, session_dir))

    # Post-hoc: count prompt issues and inject into overview
    report = "\n".join(sections)
    prompt_issue_count = report.count(_PROMPT_ISSUE_MARKER)
    if prompt_issue_count > 0:
        # Insert prompt issues line after the last overview bullet
        insert_marker = f"- **Session dir**: `~/.xhelio/sessions/{session_id}/`"
        report = report.replace(
            insert_marker,
            f"{insert_marker}\n- **Prompt issues**: {prompt_issue_count}",
            1,
        )

    return report


def list_sessions(n: int = 20) -> str:
    """List recent sessions with basic info."""
    if not SESSIONS_DIR.exists():
        return f"Error: Sessions directory not found: {SESSIONS_DIR}"

    sessions = sorted(
        (d for d in SESSIONS_DIR.iterdir() if d.is_dir()),
        reverse=True,
    )
    if not sessions:
        return "No sessions found."

    lines = ["# Recent Sessions\n"]
    lines.append("| # | Session ID | Model | Turns | Created |")
    lines.append("|---|-----------|-------|-------|---------|")

    for idx, session_dir in enumerate(sessions[:n], 1):
        meta_path = session_dir / "metadata.json"
        if meta_path.exists():
            meta = load_json(meta_path)
            model = (meta.get("model", "?") if meta else "?")[:30]
            turns = meta.get("turn_count", "?") if meta else "?"
            created = (meta.get("created_at", "?") if meta else "?")[:19]
        else:
            model, turns, created = "?", "?", "?"
        lines.append(f"| {idx} | `{session_dir.name}` | {model} | {turns} | {created} |")

    return "\n".join(lines)


def get_session_token_summary(session_id: str) -> dict | None:
    """Extract token usage summary from a session's metadata and events.

    Returns a dict with session_id, metadata totals, and per-agent breakdown,
    or None if the session doesn't exist.
    """
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        return None

    metadata = load_json(session_dir / "metadata.json")
    if not metadata:
        return None

    tu = metadata.get("token_usage", {})
    result = {
        "session_id": session_id,
        "name": metadata.get("name", ""),
        "model": metadata.get("model", "?"),
        "turns": metadata.get("turn_count", 0),
        "created": metadata.get("created_at", "?")[:19],
        "input_tokens": tu.get("input_tokens", 0),
        "output_tokens": tu.get("output_tokens", 0),
        "thinking_tokens": tu.get("thinking_tokens", 0),
        "cached_tokens": tu.get("cached_tokens", 0),
        "total_tokens": tu.get("total_tokens", 0),
        "api_calls": tu.get("api_calls", 0),
        "agents": {},
    }

    # Per-agent breakdown from events
    events_path = session_dir / "events.jsonl"
    if events_path.exists():
        events = load_jsonl(events_path)
        for e in events:
            if e.get("type") == "token_usage":
                data = e.get("data", {})
                agent = data.get("agent_name") or data.get("agent") or "unknown"
                if agent not in result["agents"]:
                    result["agents"][agent] = {
                        "input": 0, "output": 0, "thinking": 0,
                        "cached": 0, "calls": 0,
                    }
                stats = result["agents"][agent]
                stats["input"] += data.get("input_tokens", 0)
                stats["output"] += data.get("output_tokens", 0)
                stats["thinking"] += data.get("thinking_tokens", 0)
                stats["cached"] += data.get("cached_tokens", 0)
                stats["calls"] += 1

    return result


def build_token_usage_report(summaries: list[dict]) -> str:
    """Build a cross-session token usage report from session summaries."""
    lines = ["# Token Usage Report\n"]
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    lines.append(f"*Sessions: {len(summaries)} (newest first)*\n")

    # ── Per-session detailed breakdown ──────────────────────────────────
    grand = {"input": 0, "output": 0, "thinking": 0, "cached": 0, "total": 0, "calls": 0, "turns": 0}

    for s in summaries:
        sid = s["session_id"]
        name = s["name"] or "Untitled"
        lines.append(f"## `{sid}`\n")
        lines.append(f"- **Name**: {name}")
        lines.append(f"- **Model**: {s['model']}")
        lines.append(f"- **Created**: {s['created']}")
        lines.append(f"- **Turns**: {s['turns']}")
        lines.append(
            f"- **Totals**: {fmt_tokens(s['input_tokens'])} in / "
            f"{fmt_tokens(s['output_tokens'])} out / "
            f"{fmt_tokens(s['thinking_tokens'])} think / "
            f"{fmt_tokens(s['cached_tokens'])} cached / "
            f"{fmt_tokens(s['total_tokens'])} total / "
            f"{s['api_calls']} API calls"
        )

        if s["agents"]:
            lines.append("")
            lines.append("| Agent | Input | Output | Thinking | Cached | API Calls |")
            lines.append("|-------|-------|--------|----------|--------|-----------|")
            sorted_agents = sorted(s["agents"].items(), key=lambda x: x[1]["input"], reverse=True)
            for agent, stats in sorted_agents:
                lines.append(
                    f"| {agent} | {fmt_tokens(stats['input'])} | "
                    f"{fmt_tokens(stats['output'])} | {fmt_tokens(stats['thinking'])} | "
                    f"{fmt_tokens(stats['cached'])} | {stats['calls']} |"
                )
        lines.append("")

        grand["input"] += s["input_tokens"]
        grand["output"] += s["output_tokens"]
        grand["thinking"] += s["thinking_tokens"]
        grand["cached"] += s["cached_tokens"]
        grand["total"] += s["total_tokens"]
        grand["calls"] += s["api_calls"]
        grand["turns"] += s["turns"]

    # ── Grand total summary ─────────────────────────────────────────────
    lines.append("## Grand Total\n")
    lines.append(f"- **Sessions**: {len(summaries)}")
    lines.append(f"- **Turns**: {grand['turns']}")
    lines.append(
        f"- **Tokens**: {fmt_tokens(grand['input'])} in / "
        f"{fmt_tokens(grand['output'])} out / "
        f"{fmt_tokens(grand['thinking'])} think / "
        f"{fmt_tokens(grand['cached'])} cached / "
        f"{fmt_tokens(grand['total'])} total"
    )
    lines.append(f"- **API Calls**: {grand['calls']}")

    # ── Aggregate per-agent breakdown ───────────────────────────────────
    agent_totals: dict[str, dict] = defaultdict(
        lambda: {"input": 0, "output": 0, "thinking": 0, "cached": 0, "calls": 0}
    )
    for s in summaries:
        for agent, stats in s["agents"].items():
            for k in ("input", "output", "thinking", "cached", "calls"):
                agent_totals[agent][k] += stats[k]

    if agent_totals:
        lines.append("\n## Aggregate by Agent (across all sessions)\n")
        lines.append("| Agent | Input | Output | Thinking | Cached | API Calls |")
        lines.append("|-------|-------|--------|----------|--------|-----------|")

        sorted_agents = sorted(agent_totals.items(), key=lambda x: x[1]["input"], reverse=True)
        at = {"input": 0, "output": 0, "thinking": 0, "cached": 0, "calls": 0}
        for agent, stats in sorted_agents:
            lines.append(
                f"| {agent} | {fmt_tokens(stats['input'])} | "
                f"{fmt_tokens(stats['output'])} | {fmt_tokens(stats['thinking'])} | "
                f"{fmt_tokens(stats['cached'])} | {stats['calls']} |"
            )
            for k in at:
                at[k] += stats[k]

        lines.append(
            f"| **Total** | **{fmt_tokens(at['input'])}** | "
            f"**{fmt_tokens(at['output'])}** | **{fmt_tokens(at['thinking'])}** | "
            f"**{fmt_tokens(at['cached'])}** | **{at['calls']}** |"
        )

    return "\n".join(lines) + "\n"


def report_filename(session_id: str) -> str:
    """Return the report filename for a session ID."""
    return f"{session_id}.md"


def get_recent_session_ids(n: int) -> list[str]:
    """Return the N most recent session IDs (sorted newest first)."""
    if not SESSIONS_DIR.exists():
        return []
    dirs = sorted(
        (d for d in SESSIONS_DIR.iterdir() if d.is_dir()),
        key=lambda d: d.name,
        reverse=True,
    )
    return [d.name for d in dirs[:n]]


def already_checked(session_id: str, save_dir: "str | Path") -> bool:
    """Return True if a report already exists for this session."""
    save_dir = Path(save_dir)
    return (save_dir / report_filename(session_id)).exists()


def save_report(report: str, session_id: str, save_dir: "str | Path") -> Path:
    """Save a report to the save directory. Returns the path written."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / report_filename(session_id)
    path.write_text(report)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Session diagnostic tool for xhelio",
    )
    parser.add_argument(
        "session_id", nargs="?", default=None,
        help="Session ID to check (omit for --batch or --list mode)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List recent sessions",
    )
    parser.add_argument(
        "--save", metavar="DIR", type=Path, default=None,
        help="Save reports to DIR/<session_id>.md",
    )
    parser.add_argument(
        "--batch", metavar="N", type=int, default=None,
        help="Check the last N sessions (skips already-checked unless --force)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-check sessions even if a report already exists",
    )

    args = parser.parse_args()

    if args.list:
        print(list_sessions())
        return

    # ── Batch mode ──────────────────────────────────────────────────────
    if args.batch is not None:
        if args.save is None:
            parser.error("--batch requires --save <dir>")
        session_ids = get_recent_session_ids(args.batch)
        if not session_ids:
            print("No sessions found.")
            return

        checked = 0
        skipped = 0
        all_session_ids = []  # track all (checked + skipped) for token report
        for sid in session_ids:
            all_session_ids.append(sid)
            if not args.force and already_checked(sid, args.save):
                skipped += 1
                continue
            report = check_session(sid)
            path = save_report(report, sid, args.save)
            print(f"Saved: {path}")
            checked += 1

        # Always regenerate the token usage report for all sessions in the batch
        token_summaries = []
        for sid in all_session_ids:
            summary = get_session_token_summary(sid)
            if summary:
                token_summaries.append(summary)

        if token_summaries:
            token_report = build_token_usage_report(token_summaries)
            token_path = args.save / "_token_usage.md"
            token_path.write_text(token_report)
            print(f"Saved: {token_path}")

        print(f"\nDone: {checked} checked, {skipped} skipped (already exist).")
        if skipped > 0:
            print("Use --force to re-check skipped sessions.")
        return

    # ── Single session mode ─────────────────────────────────────────────
    if args.session_id is None:
        parser.error("Provide a session_id, or use --batch N, or --list")

    report = check_session(args.session_id)

    if args.save:
        path = save_report(report, args.session_id, args.save)
        print(report)
        print(f"\nReport saved to: {path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
