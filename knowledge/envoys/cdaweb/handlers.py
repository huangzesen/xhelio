"""Tool handlers for cdaweb envoy kind."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent


def handle_fetch_data_cdaweb(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    """Fetch data from CDAWeb — delegates to the shared fetch handler."""
    from agent.tool_handlers.data_ops import handle_fetch_data
    return handle_fetch_data(orch, tool_args)


def handle_browse_parameters(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    """Browse all parameters for one or more datasets from the local metadata cache."""
    from knowledge.metadata_client import get_dataset_info, get_dataset_time_range
    from agent.event_bus import DEBUG

    # Accept single dataset_id or multiple dataset_ids
    dataset_ids: list[str] = []
    if tool_args.get("dataset_ids"):
        dataset_ids = tool_args["dataset_ids"]
    elif tool_args.get("dataset_id"):
        dataset_ids = [tool_args["dataset_id"]]

    if not dataset_ids:
        return {
            "status": "error",
            "message": "Missing required parameter: dataset_id or dataset_ids",
        }

    results: dict[str, dict] = {}
    for ds_id in dataset_ids:
        try:
            info = get_dataset_info(ds_id)
            params = info.get("parameters", [])
            entry: dict = {"parameters": params}
        except Exception as e:
            orch._event_bus.emit(
                DEBUG,
                level="debug",
                msg=f"[Metadata] Could not load parameters for {ds_id}: {e}",
            )
            entry = {"parameters": [], "error": str(e)}

        time_range = get_dataset_time_range(ds_id)
        if time_range:
            entry["time_range"] = time_range

        results[ds_id] = entry

    # Flatten for single-dataset calls
    if len(results) == 1:
        ds_id, entry = next(iter(results.items()))
        return {"status": "success", "dataset_id": ds_id, **entry}

    return {"status": "success", "datasets": results}
