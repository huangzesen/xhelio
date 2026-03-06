"""Discovery tool handlers."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent


def handle_search_datasets(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from knowledge.catalog import search_by_keywords
    from knowledge.metadata_client import list_cached_datasets
    from agent.event_bus import DEBUG

    query = tool_args.get("query")
    if not query:
        return {
            "status": "error",
            "message": "Missing required parameter: query",
        }
    orch._event_bus.emit(DEBUG, level="debug", msg=f"[Catalog] Searching for: {query}")
    result = search_by_keywords(query)
    if not result:
        orch._event_bus.emit(DEBUG, level="debug", msg="[Catalog] No matches found.")
        return {"status": "success", "message": "No matching datasets found."}

    orch._event_bus.emit(DEBUG, level="debug", msg="[Catalog] Found matches.")
    # Enrich datasets with time ranges from _index.json
    mission_id = result.get("mission")
    date_lookup: dict = {}
    if mission_id:
        index = list_cached_datasets(mission_id)
        if index:
            date_lookup = {ds["id"]: ds for ds in index.get("datasets", [])}
    enriched = []
    for ds_id in result.get("datasets", []):
        info = date_lookup.get(ds_id, {})
        enriched.append({
            "id": ds_id,
            "start_date": info.get("start_date", ""),
            "stop_date": info.get("stop_date", ""),
        })
    result["datasets"] = enriched
    return {"status": "success", **result}


def handle_list_parameters(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from knowledge.metadata_client import list_parameters, get_dataset_time_range
    from agent.event_bus import DEBUG

    dataset_id = tool_args.get("dataset_id")
    if not dataset_id:
        return {
            "status": "error",
            "message": "Missing required parameter: dataset_id",
        }
    try:
        from data_ops.fetch_cdf import list_cdf_variables

        cdf_vars = list_cdf_variables(dataset_id)
        # list_cdf_variables already logs the count — no duplicate here
        result = {"status": "success", "parameters": cdf_vars}
    except Exception as e:
        orch._event_bus.emit(
            DEBUG,
            level="debug",
            msg=f"[CDF] Could not list variables for {dataset_id}: {e}, using metadata cache",
        )
        params = list_parameters(dataset_id)
        result = {"status": "success", "parameters": params}

    # Include availability time range so the agent knows valid date bounds
    time_range = get_dataset_time_range(dataset_id)
    if time_range:
        result["time_range"] = time_range
        start = time_range.get("start") or "?"
        stop = time_range.get("stop") or "?"
        orch._event_bus.emit(
            DEBUG,
            level="debug",
            msg=f"[CDF] {dataset_id} available: {start} to {stop}",
        )

    return result



def handle_browse_datasets(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from knowledge.metadata_client import browse_datasets
    from knowledge.mission_loader import load_mission as _load_mission
    from knowledge.catalog import SPACECRAFT, classify_instrument_type

    mission_id = tool_args.get("mission_id")
    if not mission_id:
        return {
            "status": "error",
            "message": "Missing required parameter: mission_id",
        }
    try:
        _load_mission(mission_id)
    except FileNotFoundError:
        pass
    datasets = browse_datasets(mission_id)
    if datasets is None:
        return {
            "status": "error",
            "message": f"No dataset index for '{mission_id}'.",
        }

    sc = SPACECRAFT.get(mission_id, {})
    ds_to_instrument = {}
    for inst_id, inst in sc.get("instruments", {}).items():
        kws = inst.get("keywords", [])
        for ds_id in inst.get("datasets", []):
            ds_to_instrument[ds_id] = {"instrument": inst_id, "keywords": kws}

    for ds in datasets:
        info = ds_to_instrument.get(ds["id"], {})
        ds["instrument"] = info.get("instrument", "")
        ds["type"] = classify_instrument_type(info.get("keywords", []))

    return {
        "status": "success",
        "mission_id": mission_id,
        "dataset_count": len(datasets),
        "datasets": datasets,
    }


def handle_list_missions(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from knowledge.metadata_client import list_missions

    missions = list_missions()
    return {"status": "success", "missions": missions, "count": len(missions)}


def handle_get_dataset_docs(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from knowledge.metadata_client import get_dataset_docs

    dataset_id = tool_args.get("dataset_id")
    if not dataset_id:
        return {
            "status": "error",
            "message": "Missing required parameter: dataset_id",
        }
    docs = get_dataset_docs(dataset_id)
    if docs.get("documentation"):
        return {"status": "success", **docs}
    else:
        result = {
            "status": "partial" if docs.get("contact") else "error",
            "dataset_id": docs.get("dataset_id", dataset_id),
            "message": "Could not fetch documentation.",
        }
        if docs.get("contact"):
            result["contact"] = docs["contact"]
        if docs.get("resource_url"):
            result["resource_url"] = docs["resource_url"]
        return result


def handle_search_full_catalog(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from knowledge.catalog_search import search_catalog as search_full_catalog
    from agent.event_bus import DEBUG

    query = tool_args.get("query")
    if not query:
        return {
            "status": "error",
            "message": "Missing required parameter: query",
        }
    max_results = int(tool_args.get("max_results", 20))
    orch._event_bus.emit(
        DEBUG, level="debug", msg=f"[Catalog] Full catalog search: {query}"
    )
    results = search_full_catalog(query, max_results=max_results)
    if results:
        return {
            "status": "success",
            "query": query,
            "count": len(results),
            "datasets": results,
            "note": "Use fetch_data with any dataset ID above. Use list_parameters to see available parameters.",
        }
    else:
        return {
            "status": "success",
            "query": query,
            "count": 0,
            "datasets": [],
            "message": f"No datasets found matching '{query}'. Try broader search terms.",
        }


def handle_web_search(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from agent.event_bus import DEBUG

    query = tool_args.get("query")
    if not query:
        return {
            "status": "error",
            "message": "Missing required parameter: query",
        }
    orch._event_bus.emit(DEBUG, level="debug", msg=f"[Search] Query: {query}")
    return orch._web_search(query)
