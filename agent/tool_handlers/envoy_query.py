"""Handler for the envoy_query tool — unified envoy capability discovery.

Replaces list_parameters, browse_datasets, list_missions, search_datasets
at the orchestrator level with a single tool that traverses each envoy's
static JSON tree via dot-separated paths + regex search.

Design doc: docs/plans/2026-03-08-envoy-query-design.md
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent


def get_envoy_kind(envoy_id: str) -> str:
    """Get the kind for an envoy from the registry."""
    from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY
    return ENVOY_KIND_REGISTRY.get_kind(envoy_id)


def get_all_missions() -> dict[str, dict]:
    """Return the full MISSIONS dict from catalog."""
    from knowledge.catalog import MISSIONS
    return dict(MISSIONS)


def load_mission_json(envoy_id: str) -> dict:
    """Load full mission JSON for an envoy."""
    from knowledge.mission_loader import load_mission
    return load_mission(envoy_id)


def fetch_parameters(dataset_id: str) -> list[dict]:
    """Fetch parameters for a CDAWeb/PPI dataset (lazy leaf)."""
    from knowledge.metadata_client import list_parameters
    params = list_parameters(dataset_id)
    return params


def _is_simple_dict(d: dict) -> bool:
    """Check if a dict contains only simple (non-container) values."""
    return all(
        isinstance(v, (str, int, float, bool, type(None)))
        for v in d.values()
    )


def _summarize_dict_entry(entry: dict) -> dict:
    """Create a summary for a single dict entry (instrument, dataset, etc.)."""
    summary: dict[str, Any] = {}
    if "name" in entry:
        summary["name"] = entry["name"]
    if "description" in entry:
        desc = entry["description"]
        summary["description"] = desc[:120] + "..." if len(desc) > 120 else desc
    if "datasets" in entry:
        summary["dataset_count"] = len(entry["datasets"])
    if "keywords" in entry:
        summary["keywords"] = entry["keywords"]
    if "start_date" in entry:
        summary["start_date"] = entry["start_date"]
    if "stop_date" in entry:
        summary["stop_date"] = entry["stop_date"]
    if "module" in entry:
        summary["module"] = entry["module"]
    return summary if summary else {"type": "object"}


def _summarize_node(node: Any, path: str = "") -> dict:
    """Summarize a JSON node: return scalars + child key listing."""
    if not isinstance(node, dict):
        return {"value": node}

    # Check if the entire node is a container of entries (dict-of-dicts)
    # e.g. instruments = {FIELDS/MAG: {...}, SWEAP: {...}}
    # or datasets = {PSP_FLD...: {...}, ...}
    non_private_values = {k: v for k, v in node.items() if not k.startswith("_")}
    if non_private_values and all(isinstance(v, dict) for v in non_private_values.values()):
        children = {}
        for ck, cv in non_private_values.items():
            children[ck] = _summarize_dict_entry(cv)
        return {"status": "success", "children": children}

    result: dict[str, Any] = {"status": "success"}
    children: dict[str, Any] = {}

    for key, value in node.items():
        if key.startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            result[key] = value
        elif isinstance(value, list):
            if all(isinstance(v, (str, int, float)) for v in value):
                result[key] = value
            elif all(isinstance(v, dict) for v in value):
                # List of dicts — summarize each by name
                child_summary = {}
                for i, item in enumerate(value):
                    item_key = item.get("name", str(i))
                    child_summary[item_key] = {
                        k: v for k, v in item.items()
                        if isinstance(v, (str, int, float, bool)) and k != "name"
                    }
                children[key] = child_summary
            else:
                result[key] = value
        elif isinstance(value, dict):
            # Check if this is a container of named entries (dict-of-dicts)
            non_private = {k: v for k, v in value.items() if not k.startswith("_")}
            if non_private and all(isinstance(v, dict) for v in non_private.values()):
                # Container of entries — summarize each as a child
                child_summary = {}
                for ck, cv in non_private.items():
                    child_summary[ck] = _summarize_dict_entry(cv)
                children[key] = child_summary
            elif _is_simple_dict(value):
                # Simple value dict — include directly (e.g. parameters: {input: "..."})
                result[key] = value
            else:
                # Mixed complex dict — summarize as a single child entry
                children[key] = _summarize_dict_entry(value)

    if children:
        result["children"] = children

    return result


def _split_path(path: str) -> list[str]:
    """Split dot-separated path into segments."""
    return path.split(".") if path else []


def _traverse(tree: dict, path: str) -> tuple[Any, str | None]:
    """Walk the tree by dot-separated path.
    Returns (node, error_message). If error_message is not None, node is None.
    """
    segments = _split_path(path)
    node = tree
    traversed = []

    for seg in segments:
        if isinstance(node, dict):
            if seg in node:
                node = node[seg]
                traversed.append(seg)
            else:
                available = [k for k in node.keys() if not k.startswith("_")]
                return None, (
                    f"Key '{seg}' not found at '{'.'.join(traversed) or 'root'}'. "
                    f"Available keys: {available}"
                )
        elif isinstance(node, list):
            found = None
            for item in node:
                if isinstance(item, dict) and item.get("name") == seg:
                    found = item
                    break
            if found is not None:
                node = found
                traversed.append(seg)
            else:
                names = [item.get("name", str(i)) for i, item in enumerate(node) if isinstance(item, dict)]
                return None, (
                    f"Item '{seg}' not found in list at '{'.'.join(traversed)}'. "
                    f"Available: {names}"
                )
        else:
            return None, (
                f"Cannot traverse into scalar value at '{'.'.join(traversed)}'"
            )

    return node, None


def _search_tree(
    tree: dict,
    pattern: re.Pattern,
    envoy_id: str,
    path_prefix: str = "",
    max_matches: int = 50,
) -> list[dict]:
    """Recursively search all string values in tree, return matches with paths."""
    matches: list[dict] = []

    for key, value in tree.items():
        if key.startswith("_"):
            continue
        current_path = f"{path_prefix}.{key}" if path_prefix else key

        if isinstance(value, str):
            if pattern.search(value):
                matches.append({
                    "envoy": envoy_id,
                    "path": current_path,
                    "field": key,
                    "value": value[:200] + "..." if len(value) > 200 else value,
                })
                if len(matches) >= max_matches:
                    return matches
        elif isinstance(value, dict):
            matches.extend(_search_tree(value, pattern, envoy_id, current_path, max_matches - len(matches)))
            if len(matches) >= max_matches:
                return matches
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    item_key = item.get("name", str(i))
                    item_path = f"{current_path}.{item_key}"
                    matches.extend(_search_tree(item, pattern, envoy_id, item_path, max_matches - len(matches)))
                elif isinstance(item, str) and pattern.search(item):
                    matches.append({
                        "envoy": envoy_id,
                        "path": current_path,
                        "field": key,
                        "value": item,
                    })
                if len(matches) >= max_matches:
                    return matches

    return matches


def handle_envoy_query(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    """Handle envoy_query tool calls."""
    envoy_id = tool_args.get("envoy")
    path = tool_args.get("path")
    search = tool_args.get("search")

    # --- Mode 3: Search ---
    if search:
        try:
            pattern = re.compile(search)
        except re.error as e:
            return {"status": "error", "message": f"Invalid regex pattern: {e}"}

        all_matches: list[dict] = []

        if envoy_id:
            try:
                mission = load_mission_json(envoy_id)
            except (FileNotFoundError, KeyError):
                return {"status": "error", "message": f"Envoy '{envoy_id}' not found."}
            all_matches = _search_tree(mission, pattern, envoy_id)
        else:
            missions = get_all_missions()
            for mid, mdata in missions.items():
                try:
                    full = load_mission_json(mid)
                except (FileNotFoundError, KeyError):
                    continue
                all_matches.extend(_search_tree(full, pattern, mid, max_matches=50 - len(all_matches)))
                if len(all_matches) >= 50:
                    break

        return {"status": "success", "matches": all_matches, "count": len(all_matches)}

    # --- Mode 1: List all envoys ---
    if not envoy_id:
        missions = get_all_missions()
        envoys = []
        for mid, mdata in sorted(missions.items()):
            envoys.append({
                "id": mid,
                "name": mdata.get("name", mid),
                "kind": get_envoy_kind(mid),
                "description": mdata.get("profile", {}).get("description", ""),
            })
        return {"status": "success", "envoys": envoys, "count": len(envoys)}

    # --- Mode 2: Navigate ---
    try:
        mission = load_mission_json(envoy_id)
    except (FileNotFoundError, KeyError):
        return {"status": "error", "message": f"Envoy '{envoy_id}' not found."}

    if not path:
        from knowledge.prompt_builder import _mission_to_markdown
        catalog = _mission_to_markdown(mission, simplified=True)
        return {"status": "success", "id": envoy_id, "catalog": catalog}

    node, error = _traverse(mission, path)
    if error:
        return {"status": "error", "message": error}

    # Check if this is a dataset leaf — fetch parameters lazily.
    # Heuristic: a dict with both "start_date" and "description" is a dataset node.
    # This matches CDAWeb/PPI dataset entries which always have these fields.
    fetched_params = None
    if isinstance(node, dict) and "start_date" in node and "description" in node:
        segments = _split_path(path)
        dataset_id = segments[-1] if segments else ""
        if dataset_id and not node.get("parameters"):
            try:
                params = fetch_parameters(dataset_id)
                if params:
                    fetched_params = params
            except Exception:
                pass  # Parameters unavailable — return static metadata only

    if isinstance(node, dict):
        summary = _summarize_node(node)
        if fetched_params is not None:
            summary["parameters"] = fetched_params
        return summary
    elif isinstance(node, list):
        # List of named dicts — summarize as children keyed by name
        if node and all(isinstance(item, dict) for item in node):
            child_summary = {}
            for i, item in enumerate(node):
                item_key = item.get("name", str(i))
                child_summary[item_key] = {
                    k: v for k, v in item.items()
                    if isinstance(v, (str, int, float, bool)) and k != "name"
                }
            return {"status": "success", "children": child_summary, "count": len(child_summary)}
        return {"status": "success", "items": node, "count": len(node)}
    else:
        return {"status": "success", "value": node}
