"""Discovery tool handlers."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent


def handle_browse_tools(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from agent.tool_catalog import get_browse_result

    category = tool_args.get("category")
    agent_ctx = f"ctx:{orch._current_agent_type}"
    result = get_browse_result(category=category, agent_context=agent_ctx)
    return {"status": "success", **result}


def handle_load_tools(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from agent.tool_catalog import resolve_tools, META_TOOL_NAMES
    from agent.truncation import trunc_items
    from agent.event_bus import DEBUG

    requested = tool_args.get("tools", [])
    if not requested:
        return {
            "status": "error",
            "message": "Missing required parameter: tools",
        }
    agent_ctx = f"ctx:{orch._current_agent_type}"
    new_names = resolve_tools(requested, agent_context=agent_ctx)

    orch._loaded_tool_names.update(new_names)
    loaded_schemas = [
        s
        for s in orch._all_tool_schemas
        if s.name in orch._loaded_tool_names and s.name not in META_TOOL_NAMES
    ]
    orch._tool_schemas = list(orch._meta_tool_schemas) + loaded_schemas
    orch._token_decomp_dirty = True
    if orch.chat is not None:
        orch.chat.update_tools(orch._tool_schemas)
    total_active = len(orch._tool_schemas)

    _shown_names, _total_names = trunc_items(sorted(new_names), "items.tool_names_log")
    who = orch._current_agent_type
    orch._event_bus.emit(
        DEBUG,
        level="info",
        msg=f"[ToolStore] {who} loaded {len(new_names)} tools: {', '.join(_shown_names)}{'...' if len(new_names) > len(_shown_names) else ''}",
    )
    return {
        "status": "success",
        "loaded": sorted(new_names),
        "total_active": total_active,
    }


def handle_search_datasets(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from knowledge.catalog import search_by_keywords
    from agent.event_bus import DEBUG

    query = tool_args.get("query")
    if not query:
        return {
            "status": "error",
            "message": "Missing required parameter: query",
        }
    orch._event_bus.emit(DEBUG, level="debug", msg=f"[Catalog] Searching for: {query}")
    result = search_by_keywords(query)
    if result:
        orch._event_bus.emit(DEBUG, level="debug", msg="[Catalog] Found matches.")
        return {"status": "success", **result}
    else:
        orch._event_bus.emit(DEBUG, level="debug", msg="[Catalog] No matches found.")
        return {"status": "success", "message": "No matching datasets found."}


def handle_list_parameters(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from knowledge.metadata_client import list_parameters
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
        orch._event_bus.emit(
            DEBUG,
            level="debug",
            msg=f"[CDF] Listed {len(cdf_vars)} data variables for {dataset_id}",
        )
        return {"status": "success", "parameters": cdf_vars}
    except Exception as e:
        orch._event_bus.emit(
            DEBUG,
            level="debug",
            msg=f"[CDF] Could not list variables for {dataset_id}: {e}, using metadata cache",
        )
        params = list_parameters(dataset_id)
        return {"status": "success", "parameters": params}


def handle_get_data_availability(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from knowledge.metadata_client import get_dataset_time_range

    dataset_id = tool_args.get("dataset_id")
    if not dataset_id:
        return {
            "status": "error",
            "message": "Missing required parameter: dataset_id",
        }
    time_range = get_dataset_time_range(dataset_id)
    if time_range is None:
        return {
            "status": "error",
            "message": f"Could not fetch availability for '{dataset_id}'.",
        }
    return {
        "status": "success",
        "dataset_id": dataset_id,
        "start": time_range.get("start"),
        "stop": time_range.get("stop"),
    }


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


def handle_google_search(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from agent.event_bus import DEBUG

    query = tool_args.get("query")
    if not query:
        return {
            "status": "error",
            "message": "Missing required parameter: query",
        }
    orch._event_bus.emit(DEBUG, level="debug", msg=f"[Search] Query: {query}")
    return orch._google_search(query)
