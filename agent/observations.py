"""Structured observation summaries for tool results.

Generates human-readable observation strings that are injected into tool
result dicts before they are sent back to the LLM.  This helps the model
reason about what happened and what to do next — especially on errors.
"""

from __future__ import annotations

from .logging import get_logger
from .truncation import trunc

logger = get_logger()


def generate_observation(tool_name: str, tool_args: dict, result: dict) -> str:
    """Build a concise, human-readable observation for a tool result.

    Args:
        tool_name: The tool that was called (e.g. ``"xhelio__assets"``).
        tool_args: The arguments passed to the tool.
        result: The result dict returned by the tool executor.

    Returns:
        A one- or two-sentence observation string.
    """
    status = result.get("status", "unknown")

    # ---- Error path (with reflection hints) ----
    if status == "error":
        return _error_observation(tool_name, tool_args, result)

    # ---- Success / other statuses ----
    handler = _TOOL_HANDLERS.get(tool_name)
    if handler:
        try:
            return handler(tool_args, result)
        except Exception as e:
            logger.debug("Observation handler failed for %s: %s", tool_name, e)

    # Delegate tools
    if tool_name.startswith("delegate_to_"):
        return _obs_delegation(tool_args, result)

    # Generic fallback
    return f"{tool_name} completed successfully."


# ---------------------------------------------------------------------------
# Tool-specific success handlers
# ---------------------------------------------------------------------------

def _obs_run_code(args: dict, result: dict) -> str:
    label = result.get("label", "result")
    num_points = result.get("num_points")
    units = result.get("units", "")
    parts = [f"Computed '{label}'"]
    if num_points:
        parts[0] += f" ({num_points:,} points"
        if units:
            parts[0] += f", {units}"
        parts[0] += ")"
    elif units:
        parts[0] += f" ({units})"
    parts[0] += "."
    return " ".join(parts)


def _obs_render_plotly(args: dict, result: dict) -> str:
    base = "Plot rendered successfully."
    feedback = result.get("insight_feedback")
    if feedback:
        return f"{base}\n\n--- Automatic Figure Review ---\n{feedback}"
    return base


def _obs_manage_plot(args: dict, result: dict) -> str:
    action = args.get("action", "update")
    return f"Plot {action} completed successfully."


def _obs_assets(args: dict, result: dict) -> str:
    action = args.get("action", "list")
    if action == "list":
        assets = result.get("assets", [])
        if not assets:
            return "No assets in session."
        kinds: dict[str, list[str]] = {}
        for a in assets:
            kinds.setdefault(a.get("kind", "unknown"), []).append(
                a.get("name", a.get("asset_id", "?"))
            )
        parts = [f"{k}: {', '.join(v)}" for k, v in kinds.items()]
        return f"{len(assets)} asset(s) — {'; '.join(parts)}"
    elif action == "status":
        plot = result.get("plot", {})
        data = result.get("data", {})
        ops = result.get("operations_count", 0)
        return f"Plot: {plot.get('state', 'none')}, Data: {data.get('total_entries', 0)} entries, Ops: {ops}"
    elif action == "restore_plot":
        return result.get("message", "Plot restored.")
    return f"assets({action}) completed."


def _obs_delegation(args: dict, result: dict) -> str:
    # Cancelled work unit
    if result.get("status") == "cancelled":
        agent = result.get("agent_name", "sub-agent")
        duration = result.get("duration_s", "")
        dur_str = f" (was running {duration}s)" if duration else ""
        return f"Work cancelled: {agent}{dur_str}."

    # When an operation log is present (async delegation), summarize steps
    op_log = result.get("operation_log")
    if op_log:
        steps = []
        for entry in op_log:
            tool = entry.get("tool", "?")
            status = entry.get("status", "?")
            outputs = entry.get("outputs", [])
            if status == "success" and outputs:
                steps.append(f"{tool}=ok({','.join(outputs[:2])})")
            elif status == "success":
                steps.append(f"{tool}=ok")
            else:
                err = entry.get("error", "")
                short_err = trunc(err, "console.args.value") if err else "failed"
                steps.append(f"{tool}=FAILED({short_err})")
        duration = result.get("duration_s", "")
        dur_str = f" [{duration}s]" if duration else ""
        return f"Delegation completed{dur_str}. Steps: {'; '.join(steps)}"

    text = result.get("result", "")
    if text:
        preview = trunc(text.replace("\n", " "), "inline.preview")
        return f"Sub-agent completed: {preview}"
    return "Sub-agent completed."


def _obs_manage_workers(args: dict, result: dict) -> str:
    action = args.get("action", "list")
    if action == "cancel":
        status = result.get("status", "unknown")
        if status == "error":
            return f"Cancel failed: {result.get('message', 'unknown error')}"
        cancelled = result.get("cancelled")
        if cancelled is not None:
            return f"Cancelled {cancelled} work unit(s)."
        return "Work unit cancelled." if status == "ok" else f"Cancel result: {status}."
    # list
    units = result.get("work_units", [])
    if not units:
        return "No active work units."
    descs = []
    for u in units:
        name = u.get("agent_name", u.get("agent_type", "?"))
        elapsed = u.get("elapsed_s", "?")
        descs.append(f"{name} ({elapsed}s)")
    return f"{len(units)} active: {', '.join(descs)}."


_TOOL_HANDLERS = {
    "xhelio__run_code": _obs_run_code,
    "xhelio__render_plotly_json": _obs_render_plotly,
    "xhelio__manage_plot": _obs_manage_plot,
    "xhelio__assets": _obs_assets,
    "manage_workers": _obs_manage_workers,
}


# ---------------------------------------------------------------------------
# Error observations with reflection hints
# ---------------------------------------------------------------------------

_ERROR_HINTS = {
    "xhelio__run_code": (
        "Check variable names with assets. "
        "Verify the code syntax and ensure referenced columns exist."
    ),
    "xhelio__render_plotly_json": (
        "Check that the data_labels reference existing data in memory "
        "(use assets). Verify the Plotly JSON structure."
    ),
    "xhelio__manage_plot": (
        "Verify the plot exists and the action parameters are correct."
    ),
}


def _error_observation(tool_name: str, tool_args: dict, result: dict) -> str:
    """Build an error observation with a reflection hint."""
    msg = result.get("message", result.get("error", "unknown error"))

    # Delegation errors
    if tool_name.startswith("delegate_to_"):
        hint = (
            "The user's request is NOT complete. "
            "Try: (1) different parameters or time range, "
            "(2) an alternative dataset or instrument, or "
            "(3) handle the operation directly with your own tools. "
            "Do NOT say 'Done'."
        )
        return f"FAILED: {msg}. {hint}"

    hint = _ERROR_HINTS.get(tool_name, "Consider a different approach or different parameters.")
    return f"FAILED: {msg}. Consider: {hint}"


# =============================================================================
# Registry protocol adapter
# =============================================================================


class _ObservationRegistryAdapter:
    name = "observations"
    description = "Per-tool observation generators for LLM feedback"

    def get(self, key: str):
        return _TOOL_HANDLERS.get(key)

    def list_all(self) -> dict:
        return dict(_TOOL_HANDLERS)


OBSERVATION_REGISTRY = _ObservationRegistryAdapter()
from agent.registry_protocol import register_registry  # noqa: E402
register_registry(OBSERVATION_REGISTRY)
