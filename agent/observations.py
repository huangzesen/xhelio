"""Structured observation summaries for tool results.

Generates human-readable observation strings that are injected into tool
result dicts before they are sent back to the LLM.  This helps the model
reason about what happened and what to do next — especially on errors.
"""

from __future__ import annotations

from .truncation import trunc, trunc_items


def generate_observation(tool_name: str, tool_args: dict, result: dict) -> str:
    """Build a concise, human-readable observation for a tool result.

    Args:
        tool_name: The tool that was called (e.g. ``"fetch_data"``).
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
        except Exception:
            pass

    # Delegate tools
    if tool_name.startswith("delegate_to_"):
        return _obs_delegation(tool_args, result)

    # SPICE tools
    if tool_name in _SPICE_TOOLS:
        return _obs_spice(tool_name, tool_args, result)

    # Generic fallback
    return f"{tool_name} completed successfully."


# ---------------------------------------------------------------------------
# Tool-specific success handlers
# ---------------------------------------------------------------------------

def _obs_fetch_data(args: dict, result: dict) -> str:
    label = result.get("label", args.get("dataset_id", "?"))
    if result.get("status") == "already_loaded":
        return f"Already in memory: {label}"
    num_points = result.get("num_points")
    columns = result.get("columns", [])
    units = result.get("units", "")
    nan_pct = result.get("nan_percentage")

    parts = [f"Fetched {num_points:,} points of {label}" if num_points else f"Fetched {label}"]
    if columns:
        shown_cols, _ = trunc_items(columns, "items.columns")
        parts[0] += f" ({', '.join(shown_cols)}"
        if units:
            parts[0] += f"; {units}"
        parts[0] += ")"
    elif units:
        parts[0] += f" ({units})"
    parts[0] += "."
    if nan_pct is not None and nan_pct > 0:
        parts.append(f"{nan_pct:.1f}% NaN.")
    if result.get("quality_warning"):
        parts.append(result["quality_warning"])
    return " ".join(parts)


def _obs_search_datasets(args: dict, result: dict) -> str:
    count = result.get("count", 0)
    query = args.get("query") or args.get("keywords") or "?"
    if count == 0:
        return f"No datasets found for '{query}'."
    return f"Found {count} matching dataset(s) for '{query}'."


def _obs_custom_operation(args: dict, result: dict) -> str:
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


def _obs_list_fetched_data(args: dict, result: dict) -> str:
    entries = result.get("entries", [])
    if not entries:
        return "No data in memory."
    return f"{len(entries)} data entries in memory."


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


def _obs_spice(tool_name: str, args: dict, result: dict) -> str:
    sc = args.get("spacecraft", "?")
    if tool_name == "get_spacecraft_position":
        r_au = result.get("r_au")
        observer = args.get("observer", "SUN")
        if r_au is not None:
            return f"Got {sc} position at {r_au:.3f} AU from {observer}."
        return f"Got {sc} position."
    if tool_name == "get_spacecraft_trajectory":
        n = result.get("num_points", "?")
        return f"Got {sc} trajectory ({n} points)."
    if tool_name == "get_spacecraft_velocity":
        n = result.get("num_points", "?")
        return f"Got {sc} velocity ({n} points)."
    if tool_name == "compute_distance":
        t1 = args.get("target1", "?")
        t2 = args.get("target2", "?")
        min_au = result.get("min_distance_au")
        if min_au is not None:
            return f"Distance {t1}–{t2}: min {min_au:.3f} AU."
        return f"Computed distance {t1}–{t2}."
    if tool_name == "transform_coordinates":
        return f"Coordinate transform completed ({args.get('from_frame', '?')} → {args.get('to_frame', '?')})."
    if tool_name == "list_spice_missions":
        return "Listed available SPICE missions."
    if tool_name == "list_coordinate_frames":
        return "Listed available coordinate frames."
    if tool_name == "manage_kernels":
        action = args.get("action", "?")
        return f"Kernel {action} completed."
    return f"{tool_name} completed successfully."


_SPICE_TOOLS: set[str] = set()


def register_spice_tool_names(names: list[str]) -> None:
    """Register dynamically discovered SPICE tool names for observation routing."""
    _SPICE_TOOLS.update(names)


def _obs_list_active_work(args: dict, result: dict) -> str:
    units = result.get("work_units", [])
    if not units:
        return "No active work units."
    descs = []
    for u in units:
        name = u.get("agent_name", u.get("agent_type", "?"))
        elapsed = u.get("elapsed_s", "?")
        descs.append(f"{name} ({elapsed}s)")
    return f"{len(units)} active: {', '.join(descs)}."


def _obs_cancel_work(args: dict, result: dict) -> str:
    status = result.get("status", "unknown")
    if status == "error":
        return f"Cancel failed: {result.get('message', 'unknown error')}"
    cancelled = result.get("cancelled")
    if cancelled is not None:
        return f"Cancelled {cancelled} work unit(s)."
    return "Work unit cancelled." if status == "ok" else f"Cancel result: {status}."


_TOOL_HANDLERS = {
    "fetch_data": _obs_fetch_data,
    "search_datasets": _obs_search_datasets,
    "custom_operation": _obs_custom_operation,
    "render_plotly_json": _obs_render_plotly,
    "manage_plot": _obs_manage_plot,
    "list_fetched_data": _obs_list_fetched_data,
    "list_active_work": _obs_list_active_work,
    "cancel_work": _obs_cancel_work,
}


# ---------------------------------------------------------------------------
# Error observations with reflection hints
# ---------------------------------------------------------------------------

_ERROR_HINTS = {
    "fetch_data": (
        "Try search_datasets to find the correct dataset/parameter ID, "
        "or check the time range."
    ),
    "custom_operation": (
        "Check variable names with list_fetched_data. "
        "Verify the code syntax and ensure referenced columns exist."
    ),
    "render_plotly_json": (
        "Check that the data_labels reference existing data in memory "
        "(use list_fetched_data). Verify the Plotly JSON structure."
    ),
    "manage_plot": (
        "Verify the plot exists and the action parameters are correct."
    ),
    "search_datasets": (
        "Try broader or different keywords."
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
