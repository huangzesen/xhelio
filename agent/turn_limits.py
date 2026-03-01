"""agent/turn_limits.py — Central turn limits registry.

Every agent loop limit in the codebase lives here as a named constant.
Config.json overrides via ``"turn_limits"``.

Public API:
    get_limit(name)  — lookup (int), KeyError on typo
    reload()         — re-read config overrides
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Default limits
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, int] = {
    # Orchestrator main loop
    "orchestrator.max_iterations":      200,
    "orchestrator.max_total_calls":     400,
    # Orchestrator task execution
    "orchestrator.task.max_iterations":  10,
    "orchestrator.task.max_total_calls": 20,
    # Orchestrator duplicate call cooldown
    "orchestrator.dup_free_passes":       2,
    "orchestrator.dup_hard_block":        8,
    # Sub-agent conversational (SubAgent._handle_request)
    "sub_agent.max_iterations":          20,
    "sub_agent.max_total_calls":         40,
    # Sub-agent duplicate call cooldown
    "sub_agent.dup_free_passes":          3,
    "sub_agent.dup_hard_block":          10,
    # Sub-agent task execution
    "sub_agent.task.max_iterations":     12,
    "sub_agent.task.max_total_calls":    25,
    # Think phase (shared by viz, dataops, planner)
    "think.max_iterations":              12,
    "think.max_total_calls":             25,
    # Planner max planning rounds
    "planner.max_rounds":                10,
    # Sub-agent batch_sync default timeout (seconds)
    "agent.batch_sync_timeout":         120,
}

# ---------------------------------------------------------------------------
# Runtime state — overrides from config.json
# ---------------------------------------------------------------------------

_overrides: dict[str, int] = {}


def reload() -> None:
    """Re-read config.json overrides for turn limits.

    Called by ``config.reload_config()`` and at import time.
    """
    global _overrides
    import config
    _overrides = config.get("turn_limits", {})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_limit(name: str) -> int:
    """Return the effective turn limit for *name*.

    Raises ``KeyError`` if *name* is not in DEFAULTS (catches typos).
    """
    if name not in DEFAULTS:
        raise KeyError(f"Unknown turn limit: {name!r}")
    override = _overrides.get(name)
    if override is not None:
        return int(override)
    return DEFAULTS[name]


# ---------------------------------------------------------------------------
# Initialize overrides at import time
# ---------------------------------------------------------------------------

try:
    reload()
except Exception:
    pass  # config may not be loadable yet (e.g., during testing)
