"""Tool catalog — browse-and-load tool store for all agents.

Provides a "supermarket" of tools that the model browses on first turn
(via ``browse_tools``) and selectively loads (via ``load_tools``).

The catalog is **built dynamically** from the ``TOOLS`` list in ``tools.py``.
When new tools are registered (e.g., SPICE tools from MCP), they
automatically appear in ``browse_tools`` output.

Categories mirror the groupings in ``agent_registry.py``.
"""

from __future__ import annotations

from .tools import TOOLS


# ---------------------------------------------------------------------------
# Category registry — maps each tool name to its category
# ---------------------------------------------------------------------------

# Category definitions: {category_name: summary}
CATEGORY_SUMMARIES: dict[str, str] = {
    "delegation": "Route work to specialist sub-agents (mission, viz, data_ops, extraction, insight, planning)",
    "discovery": "Web search and mission listing for real-world context (solar events, space weather, spacecraft info)",
    "mission_data": "Search datasets, list parameters, inspect data availability, browse mission catalogs",
    "data_ops": "Fetch data, run custom computations, describe/preview/save data",
    "visualization": "Render Plotly figures, manage plot panels, restore/export plots",
    "conversation": "Ask user for clarification, get session assets",
    "memory": "Search discoveries, recall past memories, review/rate memories",
    "pipeline": "Save, run, search, inspect, modify, and execute data pipelines",
    "document": "Read documents (PDF, images, etc.)",
    "spice": "SPICE ephemeris — spacecraft positions, distances, coordinate transforms",
    "session": "Event feed, tool log management, active work tracking",
    "data_extraction": "Store DataFrames from text/document extraction",
}

# Tool → category mapping
_TOOL_CATEGORIES: dict[str, str] = {
    # delegation
    "delegate_to_mission": "delegation",
    "delegate_to_viz_plotly": "delegation",
    "delegate_to_viz_mpl": "delegation",
    "delegate_to_data_ops": "delegation",
    "delegate_to_data_extraction": "delegation",
    "delegate_to_insight": "delegation",
    "request_planning": "delegation",
    # discovery (web search + mission listing — always pre-loaded)
    "google_search": "discovery",
    "list_missions": "discovery",
    # mission_data (dataset-level operations — loaded on demand)
    "search_datasets": "mission_data",
    "list_parameters": "mission_data",
    "get_data_availability": "mission_data",
    "browse_datasets": "mission_data",
    "get_dataset_docs": "mission_data",
    "search_full_catalog": "mission_data",
    # data_ops
    "fetch_data": "data_ops",
    "list_fetched_data": "data_ops",
    "custom_operation": "data_ops",
    "describe_data": "data_ops",
    "preview_data": "data_ops",
    "save_data": "data_ops",
    "search_function_docs": "data_ops",
    "get_function_docs": "data_ops",
    # visualization
    "render_plotly_json": "visualization",
    "manage_plot": "visualization",
    "restore_plot": "visualization",
    "get_session_assets": "visualization",
    # conversation
    "ask_clarification": "conversation",
    # memory
    "recall_memories": "memory",
    "review_memory": "memory",
    # pipeline
    "get_pipeline_info": "pipeline",
    "modify_pipeline_node": "pipeline",
    "execute_pipeline": "pipeline",
    "save_pipeline": "pipeline",
    "run_pipeline": "pipeline",
    "search_pipelines": "pipeline",
    # document
    "read_document": "document",
    # data_extraction
    "store_dataframe": "data_extraction",
    # session
    "check_events": "session",
    "get_event_details": "session",
    "manage_tool_logs": "session",
    "list_active_work": "session",
    "cancel_work": "session",
}


def _get_tool_by_name(name: str) -> dict | None:
    """Look up a tool schema dict by name from the TOOLS list."""
    for t in TOOLS:
        if t["name"] == name:
            return t
    return None


def _categorize_tool(name: str) -> str:
    """Return the category for a tool.

    Dynamically registered tools (e.g. from MCP) that aren't in the
    static mapping default to ``"spice"`` (most dynamic tools come from
    the SPICE MCP server). A debug log is emitted for unmapped tools.
    """
    cat = _TOOL_CATEGORIES.get(name)
    if cat is None:
        import logging
        logging.getLogger("xhelio").debug("Unmapped tool '%s' defaulting to 'spice' category", name)
        return "spice"
    return cat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_browse_result(
    category: str | None = None,
    agent_context: str | None = None,
) -> dict:
    """Build the full browse result — descriptions + parameter info.

    This is returned as the ``browse_tools`` function result.
    The model reads this once; it enters server-side history and is
    never re-sent.

    Args:
        category: If provided, filter to tools in this category only.
        agent_context: If provided (e.g., ``"ctx:mission"``), each tool
            gets an ``"access"`` annotation:
            - ``"call"`` — tool is in the agent's call set
            - ``"informed"`` — agent sees event logs from this tool
            - ``"available"`` — exists in the system but not in scope

    Returns:
        Dict with ``categories`` (list of category info dicts).
    """
    # Resolve call/informed sets for the requesting agent
    call_set: frozenset[str] = frozenset()
    informed_set: frozenset[str] = frozenset()
    if agent_context:
        from .agent_registry import AGENT_CALL_REGISTRY, AGENT_INFORMED_REGISTRY
        call_set = AGENT_CALL_REGISTRY.get(agent_context, frozenset())
        informed_set = AGENT_INFORMED_REGISTRY.get(agent_context)

    # Group tools by category
    grouped: dict[str, list[dict]] = {}
    for tool in TOOLS:
        cat = _categorize_tool(tool["name"])
        if category and cat != category:
            continue
        grouped.setdefault(cat, []).append(tool)

    categories = []
    for cat_name, cat_tools in grouped.items():
        summary = CATEGORY_SUMMARIES.get(cat_name, "")
        tool_entries = []
        for t in cat_tools:
            entry: dict = {
                "name": t["name"],
                "description": t["description"],
            }
            # Include parameter schema for the model to understand signatures
            params = t.get("parameters", {})
            props = params.get("properties", {})
            required = params.get("required", [])
            if props:
                param_info = {}
                for pname, pschema in props.items():
                    info = {"type": pschema.get("type", "string")}
                    if pschema.get("description"):
                        info["description"] = pschema["description"]
                    if pschema.get("enum"):
                        info["enum"] = pschema["enum"]
                    if pname in required:
                        info["required"] = True
                    param_info[pname] = info
                entry["parameters"] = param_info
            # Annotate access level when agent_context is provided
            if agent_context:
                name = t["name"]
                if name in call_set:
                    entry["access"] = "call"
                elif name in informed_set:
                    entry["access"] = "informed"
                else:
                    entry["access"] = "available"
            tool_entries.append(entry)

        categories.append({
            "category": cat_name,
            "summary": summary,
            "tools": tool_entries,
        })

    return {"categories": categories}


def resolve_tools(
    names: list[str],
    agent_context: str | None = None,
) -> list[str]:
    """Resolve category names and/or individual tool names to a flat tool name list.

    Args:
        names: Mix of category names (e.g., ``"delegation"``) and
            individual tool names (e.g., ``"fetch_data"``).
        agent_context: If provided (e.g., ``"ctx:mission"``), filter the
            resolved list to only tools the agent can call. This prevents
            a sub-agent from loading tools outside its scope.

    Returns:
        Flat list of individual tool names (deduplicated).
    """
    result: set[str] = set()
    # Build reverse mapping: category → tool names
    cat_to_tools: dict[str, list[str]] = {}
    for tool in TOOLS:
        cat = _categorize_tool(tool["name"])
        cat_to_tools.setdefault(cat, []).append(tool["name"])

    for name in names:
        if name in cat_to_tools:
            # It's a category name — expand to all tools in that category
            result.update(cat_to_tools[name])
        else:
            # Assume it's an individual tool name
            result.add(name)

    # Filter to callable tools when agent_context is provided
    if agent_context:
        from .agent_registry import AGENT_CALL_REGISTRY
        call_set = AGENT_CALL_REGISTRY.get(agent_context, frozenset())
        result = result & call_set

    return sorted(result)


# Meta-tool names that are always active (never need loading)
META_TOOL_NAMES = frozenset({"browse_tools", "load_tools", "ask_clarification"})

# Default tool categories pre-loaded at orchestrator init (no browse+load needed)
DEFAULT_TOOL_CATEGORIES = ["delegation", "discovery", "memory", "session", "data_ops", "visualization"]
# Individual tools loaded outside their category
DEFAULT_EXTRA_TOOLS: list[str] = []
