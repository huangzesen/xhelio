"""agent/truncation.py — Central truncation registry.

Every truncation limit in the codebase lives here as a named constant.
Config.json overrides via ``"truncation"`` (text limits) and
``"truncation_items"`` (item count limits).  Setting a limit to ``0``
disables truncation for that key.

Public API:
    trunc(text, limit_name)        — truncate text, append "..." if cut
    trunc_items(items, limit_name) — truncate list, return (list, total)
    join_labels(labels, limit_name)— join + truncate
    get_limit(name)                — raw lookup (int)
    get_item_limit(name)           — raw lookup (int)
    reload()                       — re-read config overrides
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Default limits — text character counts
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, int] = {
    # Console summaries
    "console.summary":          300,
    "console.summary.inner":    500,
    "console.error":            500,
    "console.error.short":      500,
    "console.error.log":        500,
    "console.args":             500,
    "console.args.value":       300,
    "console.outcome":          500,
    "console.memory_content":   500,
    "console.query":            500,
    # Detail sections (shown in expandable console details)
    "detail.request":          2000,
    "detail.code":             5000,
    "detail.code.short":       2000,
    "detail.text":              500,
    # History blocks (event feed, compaction, etc.)
    "history.result":           500,
    "history.error":            500,
    "history.error.short":      500,
    "history.error.brief":      500,
    "history.summary":          500,
    "history.summary.short":    500,
    "history.args":             500,
    "history.task_result":      500,
    # Memory agent
    "memory.user_text":         500,
    "memory.agent_text":        500,
    "memory.error":             500,
    "memory.code":              500,
    "memory.msg":               500,
    "memory.feedback":          500,
    # Inline / misc
    "inline.preview":           120,
    "inline.debug":             500,
    # API / routes
    "api.session_preview":      500,
    # Feed compaction
    "feed.details":             500,
    # LLM context content
    "context.document":         50000,
    "context.dataset_docs":      4000,
    "context.turn_text":          300,
    "context.turn_text.discovery": 500,
    "context.turn_text.inline":   200,
    "context.discovery_search":   500,
    "context.param_description":   60,
    "context.docstring_summary":  200,
    "context.mission_example":     57,
    "context.task_outcome_error": 100,
    "context.tool_args_sanitize": 200,
    "context.session_preview":     40,
    # Output token limits
    "output.reflection_tokens":   100,
    "output.inline_tokens":       100,
}

# ---------------------------------------------------------------------------
# Default limits — item counts
# ---------------------------------------------------------------------------

ITEM_DEFAULTS: dict[str, int] = {
    "items.tool_args":            3,
    "items.labels":               3,
    "items.labels.expanded":     10,
    "items.variables":           15,
    "items.child_events":         5,
    "items.outputs":              5,
    "items.columns":              6,
    "items.error_details":        5,
    "items.tool_names_log":    1000,
    "items.datasets":             3,
    "items.conversation_turns":  20,
    "items.existing_summaries":  20,
    "items.data_labels":         10,
    "items.pipeline_steps":       3,
    # LLM context item limits
    "items.parameters":          10,
    "items.catalog_functions":    8,
    "items.mission_keywords":     2,
    "items.check_events":       200,
    "items.query_event_log":    100,
    "items.data_preview_rows":   50,
    "items.data_preview_xr":     10,
    "items.data_sample_points":  20,
    "items.ops_log_reflection":  15,
    "items.error_summary":        3,
    "items.conversation_window": 20,
    "items.follow_up_turns":      6,
    "items.inline_turns":         4,
    "items.sessions_shown":      10,
    "items.api_data_preview":    10,
    "items.api_input_history":  200,
}


# ---------------------------------------------------------------------------
# Runtime state — overrides from config.json
# ---------------------------------------------------------------------------

_text_overrides: dict[str, int] = {}
_item_overrides: dict[str, int] = {}


def reload() -> None:
    """Re-read config.json overrides for truncation limits.

    Called by ``config.reload_config()`` and at import time.
    """
    global _text_overrides, _item_overrides
    import config
    _text_overrides = config.get("truncation", {})
    _item_overrides = config.get("truncation_items", {})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_limit(name: str) -> int:
    """Return the effective text character limit for *name*.

    Raises ``KeyError`` if *name* is not in DEFAULTS (catches typos).
    Config override of ``0`` means "no truncation" — returned as 0.
    """
    if name not in DEFAULTS:
        raise KeyError(f"Unknown truncation limit: {name!r}")
    override = _text_overrides.get(name)
    if override is not None:
        return int(override)
    return DEFAULTS[name]


def get_item_limit(name: str) -> int:
    """Return the effective item count limit for *name*.

    Raises ``KeyError`` if *name* is not in ITEM_DEFAULTS.
    """
    if name not in ITEM_DEFAULTS:
        raise KeyError(f"Unknown item limit: {name!r}")
    override = _item_overrides.get(name)
    if override is not None:
        return int(override)
    return ITEM_DEFAULTS[name]


def trunc(text: str, limit_name: str) -> str:
    """Truncate *text* to the named limit, appending ``"..."`` if cut.

    A limit of ``0`` (from config override) disables truncation.
    """
    n = get_limit(limit_name)
    if n == 0 or len(text) <= n:
        return text
    return text[: n - 3] + "..."


def trunc_items(items: list, limit_name: str) -> tuple[list, int]:
    """Return ``(truncated_list, total_count)`` for a named item limit.

    A limit of ``0`` returns the full list.
    """
    total = len(items)
    n = get_item_limit(limit_name)
    if n == 0 or total <= n:
        return items, total
    return items[:n], total


def join_labels(labels: list, limit_name: str) -> str:
    """Join labels with ``", "`` and truncate the resulting string.

    Returns ``"?"`` for empty lists.
    """
    if not labels:
        return "(none)"
    text = ", ".join(str(l) for l in labels)
    return trunc(text, limit_name)


# ---------------------------------------------------------------------------
# Helpers for child event summaries (replaces _child_summaries)
# ---------------------------------------------------------------------------

def child_summaries(children: list, limit_name: str = "items.child_events") -> str:
    """Format child event summaries for detail sections.

    Uses the named item limit to cap the number of children shown.
    """
    if not children:
        return ""
    shown, total = trunc_items(children, limit_name)
    lines = []
    for c in shown:
        s = c.get("summary", c.get("msg", ""))
        if s:
            lines.append(f"  - {s}")
    if total > len(shown):
        lines.append(f"  ... and {total - len(shown)} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Initialize overrides at import time
# ---------------------------------------------------------------------------

try:
    reload()
except Exception:
    pass  # config may not be loadable yet (e.g., during testing)
