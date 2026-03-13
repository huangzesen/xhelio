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
    if data.get("version") not in (1, 2):
        raise ValueError(f"tool_registry.json: unknown version {data.get('version')}")
    for name, cfg in data.get("agents", {}).items():
        if not isinstance(cfg.get("call"), list) or not isinstance(
            cfg.get("informed"), list
        ):
            raise ValueError(
                f"tool_registry.json: agent '{name}' missing 'call' or 'informed' list"
            )
    return data


_REGISTRY = _load_registry()


# ── Agent tool configurations (derived from JSON) ──

ORCHESTRATOR_TOOLS: list[str] = list(_REGISTRY["agents"]["orchestrator"]["call"])
ORCHESTRATOR_INFORMED_TOOLS: list[str] = list(
    _REGISTRY["agents"]["orchestrator"]["informed"]
)


def _collect_envoy_tools() -> list[str]:
    """Collect all tool names across all envoy kinds for permission gating.

    Discovers kinds dynamically from knowledge/envoys/ directories that
    contain an __init__.py. Uses _load_kind_module() which bootstraps
    knowledge.envoys without triggering knowledge/__init__.py's heavy imports.
    """
    from pathlib import Path
    from agent.envoy_kinds.registry import _load_kind_module

    envoys_dir = Path(__file__).resolve().parent.parent / "knowledge" / "envoys"
    all_tools: set[str] = set()
    if not envoys_dir.exists():
        return list(all_tools)
    for kind_dir in sorted(envoys_dir.iterdir()):
        if not kind_dir.is_dir() or kind_dir.name.startswith("_"):
            continue
        if not (kind_dir / "__init__.py").exists():
            continue
        try:
            mod = _load_kind_module(kind_dir.name)
            all_tools.update(t["name"] for t in mod.TOOLS)
            all_tools.update(mod.GLOBAL_TOOLS)
        except (ValueError, ImportError):
            continue
    return list(all_tools)


ENVOY_TOOLS: list[str] = _collect_envoy_tools()
ENVOY_INFORMED_TOOLS: list[str] = []
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
DATA_IO_TOOLS: list[str] = list(_REGISTRY["agents"]["data_io"]["call"])
DATA_IO_INFORMED_TOOLS: list[str] = list(
    _REGISTRY["agents"]["data_io"]["informed"]
)
EUREKA_TOOLS: list[str] = list(_REGISTRY["agents"]["eureka"]["call"])
EUREKA_INFORMED_TOOLS: list[str] = list(_REGISTRY["agents"]["eureka"]["informed"])
MEMORY_TOOLS: list[str] = list(_REGISTRY["agents"]["memory"]["call"])
MEMORY_INFORMED_TOOLS: list[str] = list(_REGISTRY["agents"]["memory"]["informed"])



# ── Context tag constants — single source of truth for ctx:* strings ──

CTX_ORCHESTRATOR = "ctx:orchestrator"
CTX_ENVOY = "ctx:envoy"
CTX_VIZ_PLOTLY = "ctx:viz_plotly"
CTX_VIZ_MPL = "ctx:viz_mpl"
CTX_VIZ_JSX = "ctx:viz_jsx"
CTX_DATAOPS = "ctx:dataops"
CTX_DATA_IO = "ctx:data_io"
CTX_EUREKA = "ctx:eureka"
CTX_MEMORY = "ctx:memory"


# ── Derived registries ──


def _resolve_tools(tools: list[str]) -> frozenset[str]:
    """Convert a tool name list into a frozenset."""
    return frozenset(tools)


# Tools the agent can CALL (for tool schema injection and permission gate)
AGENT_CALL_REGISTRY: dict[str, frozenset[str]] = {}
for _json_name, _ctx_key in [
    ("orchestrator", CTX_ORCHESTRATOR),
    ("viz_plotly", CTX_VIZ_PLOTLY),
    ("viz_mpl", CTX_VIZ_MPL),
    ("viz_jsx", CTX_VIZ_JSX),
    ("dataops", CTX_DATAOPS),
    ("data_io", CTX_DATA_IO),
    ("eureka", CTX_EUREKA),
    ("memory", CTX_MEMORY),
]:
    _cfg = _REGISTRY["agents"][_json_name]
    AGENT_CALL_REGISTRY[_ctx_key] = frozenset(_cfg["call"])

# Envoy call registry built from kind modules (not JSON)
AGENT_CALL_REGISTRY[CTX_ENVOY] = frozenset(ENVOY_TOOLS)

# Action-level permissions per tool (e.g. assets → ["list", "status"])
AGENT_PERMISSIONS: dict[str, dict[str, list[str]]] = {}
for _json_name, _ctx_key in [
    ("orchestrator", CTX_ORCHESTRATOR),
    ("viz_plotly", CTX_VIZ_PLOTLY),
    ("viz_mpl", CTX_VIZ_MPL),
    ("viz_jsx", CTX_VIZ_JSX),
    ("dataops", CTX_DATAOPS),
    ("data_io", CTX_DATA_IO),
    ("eureka", CTX_EUREKA),
    ("memory", CTX_MEMORY),
]:
    _cfg = _REGISTRY["agents"][_json_name]
    _perms = _cfg.get("permissions", {})
    if _perms:
        AGENT_PERMISSIONS[_ctx_key] = _perms


# ── Mutable InformedRegistry ──

# Static defaults for informed tools (call tools + informed-only tools)
_INFORMED_DEFAULTS: list[tuple[str, frozenset[str], list[str]]] = [
    (CTX_ORCHESTRATOR, AGENT_CALL_REGISTRY[CTX_ORCHESTRATOR], ORCHESTRATOR_INFORMED_TOOLS),
    (CTX_ENVOY, AGENT_CALL_REGISTRY[CTX_ENVOY], ENVOY_INFORMED_TOOLS),
    (CTX_VIZ_PLOTLY, AGENT_CALL_REGISTRY[CTX_VIZ_PLOTLY], VIZ_PLOTLY_INFORMED_TOOLS),
    (CTX_VIZ_MPL, AGENT_CALL_REGISTRY[CTX_VIZ_MPL], VIZ_MPL_INFORMED_TOOLS),
    (CTX_VIZ_JSX, AGENT_CALL_REGISTRY[CTX_VIZ_JSX], VIZ_JSX_INFORMED_TOOLS),
    (CTX_DATAOPS, AGENT_CALL_REGISTRY[CTX_DATAOPS], DATAOPS_INFORMED_TOOLS),
    (CTX_DATA_IO, AGENT_CALL_REGISTRY[CTX_DATA_IO], DATA_IO_INFORMED_TOOLS),
    (CTX_EUREKA, AGENT_CALL_REGISTRY[CTX_EUREKA], EUREKA_INFORMED_TOOLS),
    (CTX_MEMORY, AGENT_CALL_REGISTRY[CTX_MEMORY], MEMORY_INFORMED_TOOLS),
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





# =============================================================================
# Registry protocol adapter
# =============================================================================


class _AgentCallRegistryAdapter:
    name = "agents.tool_access"
    description = "Per-agent-context tool call permissions"

    def get(self, key: str):
        return AGENT_CALL_REGISTRY.get(key)

    def list_all(self) -> dict:
        return dict(AGENT_CALL_REGISTRY)


AGENT_CALL_PROTOCOL_REGISTRY = _AgentCallRegistryAdapter()
from agent.registry_protocol import register_registry  # noqa: E402
register_registry(AGENT_CALL_PROTOCOL_REGISTRY)

