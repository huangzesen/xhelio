"""agent/agent_registry.py — Single source of truth for agent tool access.

Centralizes all tool-name lists that were previously expressed as
category+extra pairs across individual agent files.  Two derived registries
are computed at import time:

    AGENT_CALL_REGISTRY     — tools the agent can invoke (for tool schema injection)
    AGENT_INFORMED_REGISTRY — tools whose logs the agent should see in history
                              (call tools + informed tools, for ctx:* tag
                              derivation in event_bus.py and tool-explanation
                              logic in core.py).  Now a mutable singleton
                              (InformedRegistry) that can be mutated at runtime
                              and persisted to disk.

Import chain: agent_registry → tools (zero imports). No circular deps.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .tools import get_tool_schemas

# ── Agent tool configurations ──
# Each agent defines two tiers:
#   Call tools:     tools it can invoke (flat name list)
#   Informed tools: tools whose logs it should see in history (explicit list)

# --- Orchestrator ---
ORCHESTRATOR_TOOLS = [
    # discovery (always pre-loaded)
    "google_search", "list_missions",
    # mission_data (loaded on demand)
    "search_datasets", "list_parameters",
    "get_data_availability", "browse_datasets",
    "get_dataset_docs", "search_full_catalog",
    # conversation
    "ask_clarification", "get_session_assets",
    # routing
    "delegate_to_mission", "delegate_to_viz_plotly",
    "delegate_to_viz_mpl", "delegate_to_viz_jsx",
    "delegate_to_data_ops", "delegate_to_data_extraction",
    "delegate_to_insight", "request_planning",
    # document
    "read_document",
    # memory
    "recall_memories", "review_memory",
    # data_export
    "save_data",
    # pipeline
    "get_pipeline_info", "modify_pipeline_node", "execute_pipeline",
    # pipeline_ops
    "save_pipeline", "run_pipeline", "search_pipelines",
    # extras (were in ORCHESTRATOR_EXTRA_TOOLS)
    "list_fetched_data", "preview_data", "restore_plot",
    # self-curation
    "manage_tool_logs",
    # event feed (pull-based session context)
    "check_events", "get_event_details",
    # control center (turnless orchestrator)
    "list_active_work", "cancel_work",
    # tool store (browse-and-load meta-tools)
    "browse_tools", "load_tools",
]
ORCHESTRATOR_INFORMED_TOOLS = [
    "render_plotly_json",    # know when plots are rendered/failed
    "manage_plot",           # know about plot actions
    "generate_mpl_script",   # know when MPL scripts are executed
    "generate_jsx_component",  # know when JSX components are compiled
    "custom_operation",      # know what computations were done
    "fetch_data",            # know what data was fetched
]

# --- Mission ---
MISSION_TOOLS = [
    # discovery
    "list_missions",
    # mission_data
    "search_datasets", "list_parameters",
    "get_data_availability", "browse_datasets",
    "get_dataset_docs", "search_full_catalog",
    # data_ops_fetch
    "fetch_data",
    # conversation
    "ask_clarification", "get_session_assets",
    # spice — populated dynamically via register_spice_tools()
    # extras (were in MISSION_EXTRA_TOOLS)
    "list_fetched_data", "review_memory",
    # self-curation
    "manage_tool_logs",
    # event feed (pull-based session context)
    "check_events", "get_event_details",
]

# --- SPICE Mission ---
# Used by MissionAgent when mission_id == "SPICE".
# SPICE missions don't use CDAWeb fetch/discovery — only SPICE tools.
SPICE_TOOLS = [
    # conversation
    "ask_clarification", "get_session_assets",
    # extras
    "list_fetched_data", "review_memory",
    # self-curation
    "manage_tool_logs",
    # event feed
    "check_events", "get_event_details",
    # spice — populated dynamically via register_spice_tools()
]
MISSION_INFORMED_TOOLS = [
    "custom_operation",      # know what computations were done on fetched data
    "render_plotly_json",    # know if data was already plotted
]

# --- Visualization (Plotly) ---
VIZ_PLOTLY_TOOLS = [
    # visualization
    "render_plotly_json", "manage_plot",
    # extras (were in VIZ_EXTRA_TOOLS)
    "list_fetched_data", "review_memory",
    # self-curation
    "manage_tool_logs",
    # event feed (pull-based session context)
    "check_events", "get_event_details",
]
VIZ_PLOTLY_INFORMED_TOOLS = [
    "fetch_data",            # know what data is available to plot
    "custom_operation",      # know what computed labels exist
]

# --- Visualization (MPL) ---
VIZ_MPL_TOOLS = [
    # visualization
    "generate_mpl_script", "manage_mpl_output",
    # extras
    "list_fetched_data", "review_memory",
    # self-curation
    "manage_tool_logs",
    # event feed (pull-based session context)
    "check_events", "get_event_details",
]
VIZ_MPL_INFORMED_TOOLS = [
    "fetch_data",            # know what data is available to plot
    "custom_operation",      # know what computed labels exist
]

# --- Visualization (JSX/Recharts) ---
VIZ_JSX_TOOLS = [
    # visualization
    "generate_jsx_component", "manage_jsx_output",
    # extras
    "list_fetched_data", "review_memory",
    # self-curation
    "manage_tool_logs",
    # event feed (pull-based session context)
    "check_events", "get_event_details",
]
VIZ_JSX_INFORMED_TOOLS = [
    "fetch_data",            # know what data is available to plot
    "custom_operation",      # know what computed labels exist
]

# --- DataOps ---
DATAOPS_TOOLS = [
    # data_ops_compute
    "custom_operation", "describe_data", "preview_data",
    # conversation
    "ask_clarification", "get_session_assets",
    # extras (were in DATAOPS_EXTRA_TOOLS)
    "list_fetched_data", "search_function_docs", "get_function_docs",
    "review_memory",
    # self-curation
    "manage_tool_logs",
    # event feed (pull-based session context)
    "check_events", "get_event_details",
]
DATAOPS_INFORMED_TOOLS = [
    "fetch_data",            # know what raw data is available to transform
]

# --- Planner ---
PLANNER_TOOLS = [
    # mission-level routing
    "list_missions",
    # research
    "google_search",
    # memory check
    "list_fetched_data",
    # self-curation
    "manage_tool_logs",
    # event feed (pull-based session context)
    "check_events", "get_event_details",
    # tool store (can load discovery tools if needed)
    "browse_tools", "load_tools",
]
PLANNER_INFORMED_TOOLS = [
    "fetch_data",            # know what data has been fetched
    "custom_operation",      # know what computations succeeded/failed
    "render_plotly_json",    # know what was plotted (Plotly)
    "manage_plot",           # know about plot actions
    "generate_mpl_script",   # know when MPL scripts are executed
    "manage_mpl_output",     # know about MPL output actions
    "generate_jsx_component",  # know when JSX components are compiled
    "manage_jsx_output",     # know about JSX output actions
]

# --- DataExtraction ---
EXTRACTION_TOOLS = [
    # data_extraction
    "store_dataframe",
    # document
    "read_document",
    # conversation
    "ask_clarification", "get_session_assets",
    # extras (were in EXTRACTION_EXTRA_TOOLS)
    "list_fetched_data", "review_memory",
    # self-curation
    "manage_tool_logs",
    # event feed (pull-based session context)
    "check_events", "get_event_details",
]
EXTRACTION_INFORMED_TOOLS: list[str] = []

# --- Think-phase tool configs (not used for registry, just centralized here) ---
VIZ_THINK_TOOLS = [
    "list_fetched_data", "describe_data", "preview_data", "custom_operation",
]
DATAOPS_THINK_TOOLS = [
    # function_docs
    "search_function_docs", "get_function_docs",
    # extras (were in DATAOPS_THINK_EXTRA_TOOLS)
    "list_fetched_data", "preview_data", "describe_data",
]


# ── Sync tools — execute inline in agents (no async dispatch) ──
# These are fast, read-only tools that complete near-instantly.
# When an agent calls one of these, the result is returned inline
# in the same _process_response() loop iteration, avoiding the
# async dispatch → inbox wake → new LLM round feedback loop.
SYNC_TOOLS: frozenset[str] = frozenset({
    # discovery (fast metadata lookups — must be sync to avoid async polling loop)
    "search_datasets",
    "list_parameters",
    "get_data_availability",
    "browse_datasets",
    "list_missions",
    "get_dataset_docs",
    "search_full_catalog",
    # session state queries
    "list_fetched_data",
    "check_events",
    "get_event_details",
    "get_session_assets",
    # tool store meta-tools
    "browse_tools",
    "load_tools",
    # function docs (local lookup)
    "search_function_docs",
    "get_function_docs",
    # conversation
    "ask_clarification",
    # memory
    "review_memory",
    # self-curation
    "manage_tool_logs",
    # data inspection (reads from in-memory store)
    "describe_data",
    "preview_data",
    # mpl output management (fast file reads)
    "manage_mpl_output",
    # jsx output management (fast file reads)
    "manage_jsx_output",
})


# ── Derived registries ──

def _resolve_tools(tools: list[str]) -> frozenset[str]:
    """Convert a tool name list into a frozenset."""
    return frozenset(tools)


# Tools the agent can CALL (used for tool schema injection)
AGENT_CALL_REGISTRY: dict[str, frozenset[str]] = {
    "ctx:orchestrator": _resolve_tools(ORCHESTRATOR_TOOLS),
    "ctx:mission":      _resolve_tools(MISSION_TOOLS),
    "ctx:viz_plotly":    _resolve_tools(VIZ_PLOTLY_TOOLS),
    "ctx:viz_mpl":      _resolve_tools(VIZ_MPL_TOOLS),
    "ctx:viz_jsx":      _resolve_tools(VIZ_JSX_TOOLS),
    "ctx:dataops":      _resolve_tools(DATAOPS_TOOLS),
    "ctx:planner":      _resolve_tools(PLANNER_TOOLS),
    "ctx:extraction":   _resolve_tools(EXTRACTION_TOOLS),
}


# ── Mutable InformedRegistry ──

# Static defaults for informed tools (call tools + informed-only tools)
_INFORMED_DEFAULTS: list[tuple[str, frozenset[str], list[str]]] = [
    ("ctx:orchestrator", AGENT_CALL_REGISTRY["ctx:orchestrator"], ORCHESTRATOR_INFORMED_TOOLS),
    ("ctx:mission",      AGENT_CALL_REGISTRY["ctx:mission"],      MISSION_INFORMED_TOOLS),
    ("ctx:viz_plotly",   AGENT_CALL_REGISTRY["ctx:viz_plotly"],   VIZ_PLOTLY_INFORMED_TOOLS),
    ("ctx:viz_mpl",      AGENT_CALL_REGISTRY["ctx:viz_mpl"],      VIZ_MPL_INFORMED_TOOLS),
    ("ctx:viz_jsx",      AGENT_CALL_REGISTRY["ctx:viz_jsx"],      VIZ_JSX_INFORMED_TOOLS),
    ("ctx:dataops",      AGENT_CALL_REGISTRY["ctx:dataops"],      DATAOPS_INFORMED_TOOLS),
    ("ctx:planner",      AGENT_CALL_REGISTRY["ctx:planner"],      PLANNER_INFORMED_TOOLS),
    ("ctx:extraction",   AGENT_CALL_REGISTRY["ctx:extraction"],   EXTRACTION_INFORMED_TOOLS),
]


class InformedRegistry:
    """Mutable registry of which tools each agent sees logs for.

    Thread-safe. Starts from static defaults, can be mutated at runtime
    by manage_tool_logs, and persisted to disk.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # Start from static defaults (call tools + informed tools)
        self._registry: dict[str, set[str]] = {
            ctx: set(call_tools | frozenset(informed))
            for ctx, call_tools, informed in _INFORMED_DEFAULTS
        }
        self._changelog: list[dict] = []

    def get(self, ctx: str) -> frozenset[str]:
        """Return current informed tools for an agent context."""
        with self._lock:
            return frozenset(self._registry.get(ctx, set()))

    def keys(self):
        """Return context keys (matches dict.keys() interface)."""
        with self._lock:
            return list(self._registry.keys())

    def items(self) -> list[tuple[str, frozenset[str]]]:
        """Snapshot of all (ctx, tools) pairs."""
        with self._lock:
            return [(ctx, frozenset(tools)) for ctx, tools in self._registry.items()]

    def add(self, ctx: str, tool_name: str, reasoning: str) -> bool:
        """Add a tool to an agent's informed set. Returns False if already present."""
        with self._lock:
            tools = self._registry.get(ctx)
            if tools is None or tool_name in tools:
                return False
            tools.add(tool_name)
            self._changelog.append({
                "action": "add", "ctx": ctx, "tool": tool_name,
                "reasoning": reasoning,
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            return True

    def drop(self, ctx: str, tool_name: str, reasoning: str) -> tuple[bool, str]:
        """Drop a tool from informed set. Cannot drop call tools.

        Returns (success, error_message).
        """
        call_tools = AGENT_CALL_REGISTRY.get(ctx, frozenset())
        if tool_name in call_tools:
            return False, f"Cannot drop {tool_name} — it's a callable tool for {ctx}"
        with self._lock:
            tools = self._registry.get(ctx)
            if tools is None or tool_name not in tools:
                return False, f"{tool_name} not in informed set for {ctx}"
            tools.discard(tool_name)
            self._changelog.append({
                "action": "drop", "ctx": ctx, "tool": tool_name,
                "reasoning": reasoning,
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            return True, ""

    def save(self, path: Path):
        """Persist current state + changelog to JSON."""
        with self._lock:
            data = {
                "overrides": {
                    ctx: sorted(tools - set(AGENT_CALL_REGISTRY.get(ctx, frozenset())))
                    for ctx, tools in self._registry.items()
                },
                "changelog": self._changelog,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path):
        """Load persisted overrides (additive on top of defaults)."""
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        overrides = data.get("overrides", {})
        self._changelog = data.get("changelog", [])
        with self._lock:
            for ctx, extra_tools in overrides.items():
                if ctx in self._registry:
                    self._registry[ctx].update(extra_tools)


# Module-level singleton (replaces the old frozen dict)
AGENT_INFORMED_REGISTRY = InformedRegistry()


def register_spice_tools(names: list[str]) -> None:
    """Register dynamically discovered SPICE tool names.

    Adds the tool names to MISSION_TOOLS and SPICE_TOOLS, and rebuilds
    the mission AGENT_CALL_REGISTRY entry. The orchestrator does NOT
    get SPICE tools — it delegates to the SPICE mission agent instead.
    The orchestrator's InformedRegistry is updated so it sees SPICE logs.

    Safe to call multiple times — skips names already present.
    """
    for name in names:
        if name not in MISSION_TOOLS:
            MISSION_TOOLS.append(name)
        if name not in SPICE_TOOLS:
            SPICE_TOOLS.append(name)

    # Rebuild mission call registry (NOT orchestrator — SPICE is a mission)
    AGENT_CALL_REGISTRY["ctx:mission"] = _resolve_tools(MISSION_TOOLS)

    # Update the informed registry so log routing picks up the new tools.
    # Orchestrator sees SPICE logs (informed) but cannot call them.
    with AGENT_INFORMED_REGISTRY._lock:
        for name in names:
            AGENT_INFORMED_REGISTRY._registry.setdefault("ctx:orchestrator", set()).add(name)
            AGENT_INFORMED_REGISTRY._registry.setdefault("ctx:mission", set()).add(name)

    # Update the per-mission tool registry: SPICE tools go to both groups
    # (cdaweb missions can also use SPICE for ephemeris)
    MISSION_TOOL_REGISTRY.add_tools_to_group("spice", names)
    MISSION_TOOL_REGISTRY.add_tools_to_group("cdaweb", names)


# ── Per-Mission Tool Registry ──

# Base tools shared by ALL mission agents regardless of group.
# Extracted from the intersection of MISSION_TOOLS and SPICE_TOOLS.
MISSION_BASE_TOOLS: list[str] = [
    # conversation
    "ask_clarification", "get_session_assets",
    # extras
    "list_fetched_data", "review_memory",
    # self-curation
    "manage_tool_logs",
    # event feed
    "check_events", "get_event_details",
]

# CDAWeb-specific tools (discovery + fetch) — added on top of base for
# missions that use CDAWeb data (i.e. most missions).
_CDAWEB_GROUP_TOOLS: list[str] = [
    "search_datasets", "list_parameters",
    "get_data_availability", "browse_datasets", "list_missions",
    "get_dataset_docs", "search_full_catalog",
    "fetch_data",
]

# SPICE-specific tools — starts empty, populated by register_spice_tools().
_SPICE_GROUP_TOOLS: list[str] = []


class MissionToolRegistry:
    """Thread-safe registry mapping mission_id → tool list via groups.

    Groups define sets of additional tools beyond MISSION_BASE_TOOLS:
      - "cdaweb": CDAWeb discovery + fetch (default for most missions)
      - "spice":  SPICE ephemeris tools (for the SPICE pseudo-mission)
      - (future groups trivial to add)

    Missions are mapped to a group explicitly via _mission_to_group;
    unmapped missions default to "cdaweb".
    """

    def __init__(self):
        self._lock = threading.Lock()
        # group → additional tools beyond base
        self._group_tools: dict[str, list[str]] = {
            "cdaweb": list(_CDAWEB_GROUP_TOOLS),
            "spice": list(_SPICE_GROUP_TOOLS),
        }
        # mission_id → group (explicit overrides; unmapped defaults to "cdaweb")
        self._mission_to_group: dict[str, str] = {
            "SPICE": "spice",
        }
        # Missions that have had agents created (for dynamic informed set)
        self._active_missions: set[str] = set()

    def get_group(self, mission_id: str) -> str:
        """Resolve the tool group for a mission. Defaults to 'cdaweb'."""
        return self._mission_to_group.get(mission_id, "cdaweb")

    def get_tools(self, mission_id: str) -> list[str]:
        """Return base + group tools for a mission.

        The returned list is a fresh copy safe to mutate.
        """
        group = self.get_group(mission_id)
        with self._lock:
            group_tools = list(self._group_tools.get(group, []))
        return list(MISSION_BASE_TOOLS) + group_tools

    def add_tools_to_group(self, group: str, names: list[str]) -> None:
        """Add tool names to a group. Skips names already present.

        Used by register_spice_tools() to populate groups dynamically.
        """
        with self._lock:
            tools = self._group_tools.setdefault(group, [])
            for name in names:
                if name not in tools:
                    tools.append(name)

    def mark_active(self, mission_id: str) -> bool:
        """Mark a mission as having an active agent.

        Returns True if this is a newly activated mission (first time),
        False if it was already active.
        """
        with self._lock:
            if mission_id in self._active_missions:
                return False
            self._active_missions.add(mission_id)
            return True

    def get_active_mission_tools_union(self) -> frozenset[str]:
        """Return the union of tools from all active missions.

        Used by the orchestrator to know which tools are currently
        available across all active mission agents.
        """
        with self._lock:
            missions = set(self._active_missions)
        all_tools: set[str] = set()
        for mid in missions:
            all_tools.update(self.get_tools(mid))
        return frozenset(all_tools)

    def clear_active(self) -> None:
        """Reset the active missions set. Called on agent teardown."""
        with self._lock:
            self._active_missions.clear()


# Module-level singleton
MISSION_TOOL_REGISTRY = MissionToolRegistry()
