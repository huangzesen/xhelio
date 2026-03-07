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

# ── JSON Registry Loading ──

_REGISTRY_PATH = Path(__file__).parent / "tool_registry.json"


def _load_registry() -> dict:
    """Load and validate the tool registry JSON at import time."""
    data = json.loads(_REGISTRY_PATH.read_text())
    if data.get("version") != 1:
        raise ValueError(f"tool_registry.json: unknown version {data.get('version')}")
    for name, cfg in data.get("agents", {}).items():
        if not isinstance(cfg.get("call"), list) or not isinstance(
            cfg.get("informed"), list
        ):
            raise ValueError(
                f"tool_registry.json: agent '{name}' missing 'call' or 'informed' list"
            )
    for group, tools in data.get("envoy_groups", {}).items():
        if not isinstance(tools, list):
            raise ValueError(
                f"tool_registry.json: envoy_groups['{group}'] must be a list"
            )
    return data


_REGISTRY = _load_registry()


# ── Agent tool configurations (derived from JSON) ──

ORCHESTRATOR_TOOLS: list[str] = list(_REGISTRY["agents"]["orchestrator"]["call"])
ORCHESTRATOR_INFORMED_TOOLS: list[str] = list(
    _REGISTRY["agents"]["orchestrator"]["informed"]
)
ENVOY_TOOLS: list[str] = list(_REGISTRY["agents"]["envoy"]["call"])
ENVOY_INFORMED_TOOLS: list[str] = list(_REGISTRY["agents"]["envoy"]["informed"])
VIZ_PLOTLY_TOOLS: list[str] = list(_REGISTRY["agents"]["viz_plotly"]["call"])
VIZ_PLOTLY_INFORMED_TOOLS: list[str] = list(
    _REGISTRY["agents"]["viz_plotly"]["informed"]
)
VIZ_MPL_TOOLS: list[str] = list(_REGISTRY["agents"]["viz_mpl"]["call"])
VIZ_MPL_INFORMED_TOOLS: list[str] = list(_REGISTRY["agents"]["viz_mpl"]["informed"])
VIZ_JSX_TOOLS: list[str] = list(_REGISTRY["agents"]["viz_jsx"]["call"])
VIZ_JSX_INFORMED_TOOLS: list[str] = list(_REGISTRY["agents"]["viz_jsx"]["informed"])
DATAOPS_TOOLS: list[str] = list(_REGISTRY["agents"]["dataops"]["call"])
DATAOPS_INFORMED_TOOLS: list[str] = list(_REGISTRY["agents"]["dataops"]["informed"])
PLANNER_TOOLS: list[str] = list(_REGISTRY["agents"]["planner"]["call"])
PLANNER_INFORMED_TOOLS: list[str] = list(_REGISTRY["agents"]["planner"]["informed"])
DATA_IO_TOOLS: list[str] = list(_REGISTRY["agents"]["data_io"]["call"])
DATA_IO_INFORMED_TOOLS: list[str] = list(
    _REGISTRY["agents"]["data_io"]["informed"]
)
EUREKA_TOOLS: list[str] = list(_REGISTRY["agents"]["eureka"]["call"])
EUREKA_INFORMED_TOOLS: list[str] = list(_REGISTRY["agents"]["eureka"]["informed"])

# Mission groups (derived from JSON)
ENVOY_BASE_TOOLS: list[str] = list(_REGISTRY["envoy_groups"]["base"])
_CDAWEB_GROUP_TOOLS: list[str] = list(_REGISTRY["envoy_groups"]["cdaweb"])


# ── Derived registries ──


def _resolve_tools(tools: list[str]) -> frozenset[str]:
    """Convert a tool name list into a frozenset."""
    return frozenset(tools)


# Tools the agent can CALL (for tool schema injection and permission gate)
AGENT_CALL_REGISTRY: dict[str, frozenset[str]] = {}
for _json_name, _ctx_key in [
    ("orchestrator", "ctx:orchestrator"),
    ("envoy", "ctx:envoy"),
    ("viz_plotly", "ctx:viz_plotly"),
    ("viz_mpl", "ctx:viz_mpl"),
    ("viz_jsx", "ctx:viz_jsx"),
    ("dataops", "ctx:dataops"),
    ("planner", "ctx:planner"),
    ("data_io", "ctx:data_io"),
    ("eureka", "ctx:eureka"),
]:
    _cfg = _REGISTRY["agents"][_json_name]
    AGENT_CALL_REGISTRY[_ctx_key] = frozenset(_cfg["call"])


# ── Mutable InformedRegistry ──

# Static defaults for informed tools (call tools + informed-only tools)
_INFORMED_DEFAULTS: list[tuple[str, frozenset[str], list[str]]] = [
    (
        "ctx:orchestrator",
        AGENT_CALL_REGISTRY["ctx:orchestrator"],
        ORCHESTRATOR_INFORMED_TOOLS,
    ),
    ("ctx:envoy", AGENT_CALL_REGISTRY["ctx:envoy"], ENVOY_INFORMED_TOOLS),
    (
        "ctx:viz_plotly",
        AGENT_CALL_REGISTRY["ctx:viz_plotly"],
        VIZ_PLOTLY_INFORMED_TOOLS,
    ),
    ("ctx:viz_mpl", AGENT_CALL_REGISTRY["ctx:viz_mpl"], VIZ_MPL_INFORMED_TOOLS),
    ("ctx:viz_jsx", AGENT_CALL_REGISTRY["ctx:viz_jsx"], VIZ_JSX_INFORMED_TOOLS),
    ("ctx:dataops", AGENT_CALL_REGISTRY["ctx:dataops"], DATAOPS_INFORMED_TOOLS),
    ("ctx:planner", AGENT_CALL_REGISTRY["ctx:planner"], PLANNER_INFORMED_TOOLS),
    (
        "ctx:data_io",
        AGENT_CALL_REGISTRY["ctx:data_io"],
        DATA_IO_INFORMED_TOOLS,
    ),
    (
        "ctx:eureka",
        AGENT_CALL_REGISTRY["ctx:eureka"],
        EUREKA_INFORMED_TOOLS,
    ),
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
            self._changelog.append(
                {
                    "action": "add",
                    "ctx": ctx,
                    "tool": tool_name,
                    "reasoning": reasoning,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            )
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
            self._changelog.append(
                {
                    "action": "drop",
                    "ctx": ctx,
                    "tool": tool_name,
                    "reasoning": reasoning,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            )
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

    Adds the tool names to ORCHESTRATOR_TOOLS, ENVOY_TOOLS, and rebuilds
    both call registries. The orchestrator gets SPICE tools directly (no
    delegation needed). Only the SPICE envoy gets them injected via the
    "spice" tool group.

    Also adds SPICE tools to EnvoyAgent._PARALLEL_SAFE_TOOLS (all SPICE
    tools are read-only ephemeris lookups, safe to parallelize).

    Safe to call multiple times — skips names already present.
    """
    for name in names:
        if name not in ORCHESTRATOR_TOOLS:
            ORCHESTRATOR_TOOLS.append(name)
        if name not in ENVOY_TOOLS:
            ENVOY_TOOLS.append(name)

    # Rebuild both call registries
    AGENT_CALL_REGISTRY["ctx:orchestrator"] = _resolve_tools(ORCHESTRATOR_TOOLS)
    AGENT_CALL_REGISTRY["ctx:envoy"] = _resolve_tools(ENVOY_TOOLS)

    # Update the informed registry so log routing picks up the new tools.
    with AGENT_INFORMED_REGISTRY._lock:
        for name in names:
            AGENT_INFORMED_REGISTRY._registry.setdefault("ctx:orchestrator", set()).add(
                name
            )
            AGENT_INFORMED_REGISTRY._registry.setdefault("ctx:envoy", set()).add(name)

    # Add to spice group so only SPICE envoy gets these tools
    ENVOY_TOOL_REGISTRY.add_tools_to_group("spice", names)

    # Add SPICE tools to EnvoyAgent's parallel-safe set
    from .envoy_agent import EnvoyAgent

    EnvoyAgent._PARALLEL_SAFE_TOOLS.update(names)


# ── Per-Mission Tool Registry ──


class EnvoyToolRegistry:
    """Thread-safe registry mapping mission_id → tool list via groups.

    Groups define sets of additional tools beyond ENVOY_BASE_TOOLS:
      - "cdaweb": CDAWeb discovery + fetch (default for most missions)
      - (future groups trivial to add)

    Missions are mapped to a group explicitly via _mission_to_group;
    unmapped missions default to "cdaweb".
    """

    def __init__(self):
        self._lock = threading.Lock()
        # group → additional tools beyond base (from JSON)
        self._group_tools: dict[str, list[str]] = {
            group: list(tools)
            for group, tools in _REGISTRY["envoy_groups"].items()
            if group != "base"
        }
        # mission_id → group (explicit overrides; unmapped defaults to "cdaweb")
        self._mission_to_group: dict[str, str] = dict(
            _REGISTRY["envoy_group_assignments"]
        )
        self._default_group: str = _REGISTRY["envoy_default_group"]
        # Missions that have had agents created (for dynamic informed set)
        self._active_missions: set[str] = set()

    def get_group(self, mission_id: str) -> str:
        """Resolve the tool group for a mission. Defaults to configured default."""
        return self._mission_to_group.get(mission_id, self._default_group)

    def get_tools(self, mission_id: str) -> list[str]:
        """Return base + group tools for a mission.

        The returned list is a fresh copy safe to mutate.
        """
        group = self.get_group(mission_id)
        with self._lock:
            group_tools = list(self._group_tools.get(group, []))
        return list(ENVOY_BASE_TOOLS) + group_tools

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

    def clear_active(self) -> None:
        """Reset the active missions set. Called on agent teardown."""
        with self._lock:
            self._active_missions.clear()


# Module-level singleton
ENVOY_TOOL_REGISTRY = EnvoyToolRegistry()
