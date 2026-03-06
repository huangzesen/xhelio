"""SPICE ephemeris tool handlers (MCP passthrough + auto-store)."""

from __future__ import annotations
from typing import TYPE_CHECKING

import pandas as pd

from agent.event_bus import get_event_bus, DEBUG
from agent.logging import get_logger, log_error
from agent.mcp_client import get_spice_client
from data_ops.store import get_store, DataEntry

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent

logger = get_logger()


def _handle_spice_tool(tool_name: str, tool_args: dict) -> dict:
    """Route a SPICE tool call through the MCP client.

    For data-producing calls (responses containing a "data" key),
    extracts full data and stores it in DataStore for plotting.
    """
    try:
        client = get_spice_client()
    except Exception as e:
        log_error(f"SPICE MCP connection failed: {e}", e)
        return {"status": "error", "message": f"SPICE MCP server unavailable: {e}"}

    try:
        result = client.call_tool(tool_name, tool_args)
    except Exception as e:
        log_error(f"SPICE MCP call failed for {tool_name}: {e}", e)
        # Try one reconnect
        try:
            client = get_spice_client()
            result = client.call_tool(tool_name, tool_args)
        except Exception as e2:
            log_error(f"SPICE MCP retry failed for {tool_name}: {e2}", e2)
            return {"status": "error", "message": f"SPICE MCP call failed: {e2}"}

    if result.get("status") == "error":
        return result

    # Detect data-producing responses: if the result has a "data" key
    # with a list of records, store it in the DataStore for plotting.
    records = result.get("data")
    if isinstance(records, list) and records:
        try:
            df = pd.DataFrame(records)
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], format="ISO8601")
                df = df.set_index("time")

            sc_name = (
                tool_args.get("spacecraft") or tool_args.get("target1") or "unknown"
            )
            sc_name = sc_name.upper().replace(" ", "_")
            # Derive a descriptive suffix from the tool name
            suffix = (
                tool_name.replace("get_spacecraft_", "")
                .replace("get_", "")
                .replace("compute_", "")
            )
            label = f"SPICE.{sc_name}_{suffix}"
            # Guess units from columns
            cols = list(df.columns)
            if any("km_s" in c for c in cols):
                units = "km/s"
            elif any("au" in c.lower() for c in cols):
                units = "AU"
            else:
                units = "km"
            store = get_store()
            entry = DataEntry(
                label=label,
                data=df,
                units=units,
                description=f"SPICE {suffix} of {sc_name} rel. {tool_args.get('observer', tool_args.get('target2', 'SUN'))}",
                source="spice",
            )
            store.put(entry)
            result["label"] = label
            result["note"] = f"Stored as '{label}' — use render_plotly_json to plot."

            # Remove full data from result returned to LLM (it only needs
            # the summary + label; full data is in the DataStore)
            del result["data"]
        except Exception as e:
            get_event_bus().emit(
                DEBUG,
                level="warning",
                msg=f"[SPICE] Failed to store {tool_name} data: {e}",
            )
            # Non-fatal: the summary result is still returned

    return result


def _make_spice_handler(tool_name: str):
    """Create a closure that routes a specific SPICE tool through MCP."""

    def handler(orch: "OrchestratorAgent", tool_args: dict) -> dict:
        return _handle_spice_tool(tool_name, tool_args)

    return handler


def register_spice_handlers(names: list[str]) -> None:
    """Register SPICE tool handlers in TOOL_REGISTRY after MCP discovery."""
    from agent.tool_handlers import TOOL_REGISTRY

    for name in names:
        if name not in TOOL_REGISTRY:
            TOOL_REGISTRY[name] = _make_spice_handler(name)
