"""Handlers for envoy lifecycle management (add, list, remove)."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent


def _get_package_envoy_dir() -> Path:
    """Return the directory for user-defined package envoy JSONs."""
    return Path(__file__).parent.parent.parent / "knowledge" / "missions" / "packages"


def handle_add_envoy(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    """Introspect a Python package and return its API surface for review.

    Args (in tool_args):
        package_name: Python import path (e.g., "pfsspy", "sunpy.map")
    """
    package_name = tool_args.get("package_name", "")
    if not package_name:
        return {"status": "error", "message": "package_name is required"}

    try:
        import importlib
        mod = importlib.import_module(package_name)
    except ImportError as e:
        return {
            "status": "error",
            "message": f"Cannot import '{package_name}': {e}. "
                       f"Make sure the package is installed in the virtualenv.",
        }

    # Introspect public API
    api_surface = []
    for name in sorted(dir(mod)):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name, None)
        if obj is None:
            continue
        if callable(obj):
            try:
                sig = str(inspect.signature(obj))
            except (ValueError, TypeError):
                sig = "(...)"
            doc = inspect.getdoc(obj) or ""
            if len(doc) > 300:
                doc = doc[:300] + "..."
            api_surface.append({
                "name": name,
                "type": "function" if inspect.isfunction(obj) else
                        "class" if inspect.isclass(obj) else "callable",
                "signature": f"{package_name}.{name}{sig}",
                "docstring": doc,
            })
        elif inspect.ismodule(obj):
            api_surface.append({
                "name": name,
                "type": "submodule",
                "signature": f"{package_name}.{name}",
                "docstring": (inspect.getdoc(obj) or "")[:300],
            })

    if not api_surface:
        return {
            "status": "error",
            "message": f"Package '{package_name}' has no public API to expose.",
        }

    return {
        "status": "success",
        "action": "review_api",
        "package": package_name,
        "api_surface": api_surface,
        "instructions": (
            "Review the API surface above. Discuss with the user which functions "
            "to expose as tools for this envoy. Once agreed, call save_envoy with "
            "the finalized configuration."
        ),
    }


def handle_save_envoy(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    """Save a finalized package envoy definition to disk.

    Args (in tool_args):
        envoy_id: Unique ID for the envoy (e.g., "PFSS")
        envoy_name: Human-readable name (optional, defaults to envoy_id)
        description: What this envoy does
        imports: List of {"import_path": str, "sandbox_alias": str}
        functions: List of {"name": str, "description": str, "signature": str, ...}
        keywords: List of search keywords (optional)
    """
    envoy_id = tool_args.get("envoy_id", "")
    if not envoy_id:
        return {"status": "error", "message": "envoy_id is required"}
    if not re.match(r'^[a-zA-Z0-9_-]+$', envoy_id):
        return {"status": "error", "message": "envoy_id must contain only letters, digits, underscores, and hyphens"}

    imports = tool_args.get("imports", [])
    functions = tool_args.get("functions", [])
    if not imports:
        return {"status": "error", "message": "At least one import is required"}

    envoy_json = {
        "id": envoy_id.upper(),
        "name": tool_args.get("envoy_name", envoy_id),
        "type": "package",
        "keywords": tool_args.get("keywords", [envoy_id.lower()]),
        "profile": {
            "description": tool_args.get("description", ""),
        },
        "sandbox": {
            "imports": imports,
            "functions": functions,
        },
        "instruments": {},
    }

    pkg_dir = _get_package_envoy_dir()
    pkg_dir.mkdir(parents=True, exist_ok=True)
    file_path = pkg_dir / f"{envoy_id.lower()}.json"
    file_path.write_text(json.dumps(envoy_json, indent=2))

    # Register in the envoy group system
    from agent.agent_registry import ENVOY_TOOL_REGISTRY
    ENVOY_TOOL_REGISTRY.register_mission(envoy_id.upper(), "package")

    # Clear mission cache so the new envoy is picked up
    from knowledge.mission_loader import _mission_cache
    cache_key = envoy_id.lower().replace("-", "_")
    _mission_cache.pop(cache_key, None)

    return {
        "status": "success",
        "message": f"Envoy '{envoy_id}' saved to {file_path}. "
                   f"It is now available for delegation.",
        "envoy_id": envoy_id.upper(),
        "file": str(file_path),
    }


def handle_list_envoys(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    """List all user-defined package envoys."""
    pkg_dir = _get_package_envoy_dir()
    envoys = []
    if pkg_dir.exists():
        for f in sorted(pkg_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                envoys.append({
                    "id": data.get("id", f.stem.upper()),
                    "name": data.get("name", f.stem),
                    "description": data.get("profile", {}).get("description", ""),
                    "imports": [
                        imp["import_path"]
                        for imp in data.get("sandbox", {}).get("imports", [])
                    ],
                    "num_functions": len(data.get("sandbox", {}).get("functions", [])),
                })
            except (json.JSONDecodeError, OSError):
                continue

    return {"status": "success", "envoys": envoys, "count": len(envoys)}


def handle_remove_envoy(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    """Remove a user-defined package envoy."""
    envoy_id = tool_args.get("envoy_id", "")
    if not envoy_id:
        return {"status": "error", "message": "envoy_id is required"}

    pkg_dir = _get_package_envoy_dir()
    file_path = pkg_dir / f"{envoy_id.lower()}.json"

    if not file_path.exists():
        return {"status": "error", "message": f"Envoy '{envoy_id}' not found at {file_path}"}

    file_path.unlink()

    # Remove from envoy group system
    from agent.agent_registry import ENVOY_TOOL_REGISTRY
    ENVOY_TOOL_REGISTRY.unregister_mission(envoy_id.upper())

    # Clear mission cache
    from knowledge.mission_loader import _mission_cache
    cache_key = envoy_id.lower().replace("-", "_")
    _mission_cache.pop(cache_key, None)

    # Stop the running agent if it exists
    agent_id = f"EnvoyAgent[{envoy_id.upper()}]"
    if hasattr(orch, '_sub_agents_lock'):
        with orch._sub_agents_lock:
            agent = orch._sub_agents.pop(agent_id, None)
        if agent:
            agent.stop(timeout=2.0)

    return {
        "status": "success",
        "message": f"Envoy '{envoy_id}' removed.",
    }
