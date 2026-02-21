"""Structured observation summaries for tool results.

Generates human-readable observation strings that are injected into tool
result dicts before they are sent back to the LLM.  This helps the model
reason about what happened and what to do next — especially on errors.
"""

from __future__ import annotations


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
        parts[0] += f" ({', '.join(columns[:6])}"
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
    return "Plot rendered successfully."


def _obs_manage_plot(args: dict, result: dict) -> str:
    action = args.get("action", "update")
    return f"Plot {action} completed successfully."


def _obs_list_fetched_data(args: dict, result: dict) -> str:
    entries = result.get("entries", [])
    if not entries:
        return "No data in memory."
    return f"{len(entries)} data entries in memory."


def _obs_delegation(args: dict, result: dict) -> str:
    text = result.get("result", "")
    if text:
        preview = text[:120].replace("\n", " ")
        if len(text) > 120:
            preview += "..."
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


_SPICE_TOOLS = {
    "get_spacecraft_position",
    "get_spacecraft_trajectory",
    "get_spacecraft_velocity",
    "compute_distance",
    "transform_coordinates",
    "list_spice_missions",
    "list_coordinate_frames",
    "manage_kernels",
}


_TOOL_HANDLERS = {
    "fetch_data": _obs_fetch_data,
    "search_datasets": _obs_search_datasets,
    "custom_operation": _obs_custom_operation,
    "render_plotly_json": _obs_render_plotly,
    "manage_plot": _obs_manage_plot,
    "list_fetched_data": _obs_list_fetched_data,
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
        hint = "Consider handling this directly instead of delegating."
        return f"FAILED: {msg}. Consider: {hint}"

    hint = _ERROR_HINTS.get(tool_name, "Consider a different approach or different parameters.")
    return f"FAILED: {msg}. Consider: {hint}"
