"""Envoy kind registry — maps missions to kinds and resolves tools at runtime."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    pass

# Mission → kind mapping. Empty — no prebuilt kinds.
MISSION_KINDS: dict[str, str] = {}
DEFAULT_KIND = ""  # No default — must be explicitly registered


def _load_kind_module(kind: str):
    """Import a kind module from knowledge.envoys."""
    import importlib
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
        """Resolve the kind for a mission."""
        kind = self._mission_kinds.get(mission_id)
        if kind is None:
            raise KeyError(f"No envoy kind registered for '{mission_id}'")
        return kind

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

    def register_handlers_globally(self) -> None:
        """Push all kind handlers into the global TOOL_REGISTRY."""
        from agent.tool_handlers import TOOL_REGISTRY
        from pathlib import Path
        import logging
        logger = logging.getLogger("xhelio")
        envoys_dir = Path(__file__).resolve().parent.parent.parent / "knowledge" / "envoys"
        if not envoys_dir.exists():
            return
        for kind_dir in envoys_dir.iterdir():
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
    a kind.json manifest (new codegen path) or __init__.py + .json (legacy).
    """
    import json
    import logging
    from pathlib import Path
    logger = logging.getLogger("xhelio")

    if envoys_dir is None:
        envoys_dir = Path(__file__).resolve().parent.parent.parent / "knowledge" / "envoys"

    discovered: dict[str, str] = {}
    if not envoys_dir.exists():
        return discovered

    for kind_dir in envoys_dir.iterdir():
        if not kind_dir.is_dir() or kind_dir.name.startswith("_"):
            continue

        manifest_path = kind_dir / "kind.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                envoy_id = manifest.get("envoy_id", "")
                if not envoy_id or envoy_id in MISSION_KINDS:
                    continue
                # Regenerate .py files if missing (restart recovery)
                if not (kind_dir / "__init__.py").exists():
                    from agent.envoy_kinds.codegen import generate_kind_files
                    generate_kind_files(kind_dir, manifest)
                    logger.info("Regenerated .py files for runtime kind %s", kind_dir.name)
                discovered[envoy_id] = kind_dir.name
            except Exception as e:
                logger.warning("Failed to load runtime kind %s: %s", kind_dir.name, e)
            continue

        # Legacy fallback: __init__.py + *.json with "id" field
        if not (kind_dir / "__init__.py").exists():
            continue
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
