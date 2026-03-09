"""Envoy kind registry — maps missions to kinds and resolves tools at runtime."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    pass

# Mission → kind mapping. Missions not listed here default to DEFAULT_KIND.
MISSION_KINDS: dict[str, str] = {
    "CASSINI_PPI": "ppi",
    "GALILEO": "ppi",
    "INSIGHT": "ppi",
    "JUNO_PPI": "ppi",
    "LRO": "ppi",
    "LUNAR-PROSPECTOR": "ppi",
    "MAVEN_PPI": "ppi",
    "MESSENGER_PPI": "ppi",
    "MEX": "ppi",
    "MGS": "ppi",
    "NEW-HORIZONS_PPI": "ppi",
    "PIONEER_PPI": "ppi",
    "PIONEER-VENUS_PPI": "ppi",
    "ULYSSES_PPI": "ppi",
    "VEX": "ppi",
    "VOYAGER1_PPI": "ppi",
    "VOYAGER2_PPI": "ppi",
    "SPICE": "spice",
}
DEFAULT_KIND = "cdaweb"


class _KnowledgeStub:
    """Minimal stand-in for the ``knowledge`` package.

    Inserted into ``sys.modules["knowledge"]`` during early bootstrap to
    break the circular import chain:
    ``agent_registry → knowledge.__init__ → metadata_client → event_bus → agent_registry``

    On first attribute access (e.g., ``from knowledge import load_mission``),
    the stub replaces itself with the real ``knowledge`` module by executing
    ``knowledge/__init__.py`` normally.
    """

    def __init__(self, path: str, init_file: str):
        import importlib.util
        self.__path__ = [path]
        self.__file__ = init_file
        self.__package__ = "knowledge"
        self.__name__ = "knowledge"
        # Set __spec__ so the import machinery treats this as a valid package
        self.__spec__ = importlib.util.spec_from_file_location(
            "knowledge",
            init_file,
            submodule_search_locations=[path],
        )
        self.__loader__ = self.__spec__.loader if self.__spec__ else None
        self._is_stub = True
        self._replacing = False

    def _replace(self):
        """Execute the real __init__.py, replacing this stub in sys.modules."""
        if self._replacing:
            return  # Prevent re-entrant replacement
        self._replacing = True
        import importlib
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "knowledge",
            self.__file__,
            submodule_search_locations=self.__path__,
        )
        if spec and spec.loader:
            real_mod = importlib.util.module_from_spec(spec)
            sys.modules["knowledge"] = real_mod
            spec.loader.exec_module(real_mod)

    def __getattr__(self, name: str):
        # The import machinery probes dunder attrs (__spec__, __loader__, etc.)
        # before the module is fully initialized. Let those raise AttributeError
        # so import doesn't break, UNLESS someone is doing a real import like
        # `from knowledge import load_mission`.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        self._replace()
        import sys
        real = sys.modules.get("knowledge")
        if real is not self:
            return getattr(real, name)
        raise AttributeError(name)


def _load_kind_module(kind: str):
    """Lazily import a kind module by name from knowledge.envoys.

    Bootstraps parent packages minimally to avoid triggering
    knowledge/__init__.py's heavy imports (which cause circular deps
    when called during agent_registry init).
    """
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    _knowledge_dir = Path(__file__).resolve().parent.parent.parent / "knowledge"

    # Ensure 'knowledge' is in sys.modules — but if it hasn't been imported
    # yet, register a lazy stub instead of executing __init__.py (which
    # triggers metadata_client → event_bus → agent_registry circular dep).
    # The stub auto-replaces on first real attribute access.
    if "knowledge" not in sys.modules:
        stub = _KnowledgeStub(
            str(_knowledge_dir),
            str(_knowledge_dir / "__init__.py"),
        )
        sys.modules["knowledge"] = stub

    # knowledge.envoys is lightweight — safe to import normally.
    if "knowledge.envoys" not in sys.modules:
        envoys_dir = _knowledge_dir / "envoys"
        spec = importlib.util.spec_from_file_location(
            "knowledge.envoys",
            envoys_dir / "__init__.py",
            submodule_search_locations=[str(envoys_dir)],
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["knowledge.envoys"] = mod
            spec.loader.exec_module(mod)

    module_name = f"knowledge.envoys.{kind}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ValueError(f"Unknown envoy kind: {kind}")


class EnvoyKindRegistry:
    """Thread-safe registry for envoy kinds."""

    def __init__(self):
        self._lock = threading.Lock()
        self._mission_kinds: dict[str, str] = dict(MISSION_KINDS)
        self._active_missions: set[str] = set()

    def get_kind(self, mission_id: str) -> str:
        """Resolve the kind for a mission. Defaults to DEFAULT_KIND."""
        return self._mission_kinds.get(mission_id, DEFAULT_KIND)

    def get_function_schemas(self, mission_id: str) -> list:
        """Return FunctionSchema objects for LLM adapters."""
        from agent.tools import _inject_commentary

        kind = self.get_kind(mission_id)
        mod = _load_kind_module(kind)

        # Augment kind-specific schemas with commentary parameter
        kind_schemas = [_inject_commentary(t) for t in mod.TOOLS]

        # Convert kind schemas to FunctionSchema
        from agent.llm.base import FunctionSchema
        kind_fn_schemas = [
            FunctionSchema(
                name=ts["name"],
                description=ts.get("description", ""),
                parameters=ts["parameters"],
            )
            for ts in kind_schemas
        ]

        # Get global function schemas by name
        from agent.tools import get_function_schemas as _get_fn_schemas
        global_schemas = []
        if mod.GLOBAL_TOOLS:
            global_schemas = _get_fn_schemas(names=mod.GLOBAL_TOOLS)

        return kind_fn_schemas + global_schemas

    def get_tool_names(self, mission_id: str) -> list[str]:
        """Return all tool names for a mission (kind + global)."""
        kind = self.get_kind(mission_id)
        mod = _load_kind_module(kind)
        kind_names = [t["name"] for t in mod.TOOLS]
        return kind_names + list(mod.GLOBAL_TOOLS)

    def get_handler(self, tool_name: str, mission_id: str) -> Callable | None:
        """Look up handler for a tool. Checks kind HANDLERS first."""
        kind = self.get_kind(mission_id)
        mod = _load_kind_module(kind)
        handler = mod.HANDLERS.get(tool_name)
        if handler:
            return handler
        # Fall back to global TOOL_REGISTRY for global tools
        from agent.tool_handlers import TOOL_REGISTRY
        return TOOL_REGISTRY.get(tool_name)

    def register_mission(self, mission_id: str, kind: str) -> None:
        """Register a mission into a kind. Skips if already assigned."""
        with self._lock:
            if mission_id not in self._mission_kinds:
                self._mission_kinds[mission_id] = kind

    def unregister_mission(self, mission_id: str) -> None:
        """Remove a mission from the kind registry."""
        with self._lock:
            self._mission_kinds.pop(mission_id, None)

    def mark_active(self, mission_id: str) -> bool:
        """Mark a mission as having an active agent. Returns True if newly activated."""
        with self._lock:
            if mission_id in self._active_missions:
                return False
            self._active_missions.add(mission_id)
            return True

    def clear_active(self) -> None:
        """Reset the active missions set."""
        with self._lock:
            self._active_missions.clear()

    def add_tools_to_kind(self, kind: str, tools: list[dict], handlers: dict) -> None:
        """Dynamically add tools to a kind (used by SPICE MCP discovery)."""
        with self._lock:
            mod = _load_kind_module(kind)
            existing_names = {t["name"] for t in mod.TOOLS}
            for tool in tools:
                if tool["name"] not in existing_names:
                    mod.TOOLS.append(tool)
                    existing_names.add(tool["name"])
            mod.HANDLERS.update(handlers)

    def register_handlers_globally(self) -> None:
        """Push all kind handlers into the global TOOL_REGISTRY.

        Call this once at startup so _execute_tool can dispatch kind tools
        without special-casing. Auto-discovers kind directories from disk.
        """
        from agent.tool_handlers import TOOL_REGISTRY
        from knowledge.mission_loader import _ENVOYS_DIR
        import logging
        logger = logging.getLogger("xhelio")
        if not _ENVOYS_DIR.exists():
            return
        for kind_dir in _ENVOYS_DIR.iterdir():
            if not kind_dir.is_dir() or kind_dir.name.startswith("_"):
                continue
            kind_name = kind_dir.name
            try:
                mod = _load_kind_module(kind_name)
                TOOL_REGISTRY.update(mod.HANDLERS)
            except (ValueError, ImportError) as exc:
                logger.warning("Failed to load kind %s handlers: %s", kind_name, exc)


def _discover_runtime_kinds(envoys_dir=None) -> dict[str, str]:
    """Scan knowledge/envoys/ for runtime-created kinds.

    Returns a dict of envoy_id → kind for any kind directory that has
    both __init__.py and a .json file with an "id" field.
    Prebuilt kinds (cdaweb, ppi, spice) are skipped since they are
    already in MISSION_KINDS.
    """
    import json
    from pathlib import Path
    if envoys_dir is None:
        # Compute directly instead of importing from knowledge.mission_loader
        # to avoid circular import: event_bus → agent_registry → here →
        # knowledge.__init__ → metadata_client → event_bus
        envoys_dir = Path(__file__).resolve().parent.parent.parent / "knowledge" / "envoys"

    discovered: dict[str, str] = {}
    if not envoys_dir.exists():
        return discovered

    for kind_dir in envoys_dir.iterdir():
        if not kind_dir.is_dir() or kind_dir.name.startswith("_"):
            continue
        # Skip prebuilt kinds
        if kind_dir.name in ("cdaweb", "ppi", "spice"):
            continue
        if not (kind_dir / "__init__.py").exists():
            continue
        # Look for JSON files that define envoy IDs
        for json_file in kind_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text())
                envoy_id = data.get("id")
                if envoy_id and envoy_id not in MISSION_KINDS:
                    discovered[envoy_id] = kind_dir.name
            except (json.JSONDecodeError, OSError):
                continue

    return discovered


# Module-level singleton
ENVOY_KIND_REGISTRY = EnvoyKindRegistry()

# Auto-discover runtime-created kinds from disk at import time
_discovered = _discover_runtime_kinds()
MISSION_KINDS.update(_discovered)
ENVOY_KIND_REGISTRY._mission_kinds.update(_discovered)
