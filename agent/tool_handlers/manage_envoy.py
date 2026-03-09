"""Handler for manage_envoy — runtime envoy kind creation and removal."""

from __future__ import annotations

import importlib
import logging
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent

logger = logging.getLogger("xhelio")

_ENVOYS_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge" / "envoys"
_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge" / "prompts"

# Prebuilt kinds that cannot be removed
_PREBUILT_KINDS = frozenset({"cdaweb", "ppi", "spice"})

# Default global tools every envoy gets
_DEFAULT_GLOBAL_TOOLS = [
    "ask_clarification",
    "manage_session_assets",
    "list_fetched_data",
    "review_memory",
    "events",
    "run_code",
]


def handle_manage_envoy(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    """Dispatch to create or remove based on action."""
    action = tool_args.get("action")
    if action == "create":
        return _create_envoy(orch, tool_args)
    elif action == "remove":
        return _remove_envoy(orch, tool_args)
    else:
        return {"status": "error", "message": f"Unknown action: {action}"}


def _create_envoy(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    """Create a new envoy kind on disk and register it."""
    kind = tool_args["kind"].lower()
    envoy_id = tool_args["envoy_id"].upper()
    source = tool_args.get("source", "")
    source_type = tool_args.get("source_type", "package")
    tools = tool_args.get("tools", [])

    kind_dir = _ENVOYS_DIR / kind
    if kind_dir.exists():
        return {"status": "error", "message": f"Kind '{kind}' already exists"}

    # Build handlers.py content
    handler_lines = [
        f'"""Auto-generated handlers for the {kind} envoy kind."""',
        "",
        "from __future__ import annotations",
        "from typing import TYPE_CHECKING",
        "",
        "if TYPE_CHECKING:",
        "    from agent.core import OrchestratorAgent",
        "",
    ]
    for tool in tools:
        code = tool.get("handler_code", "")
        if code:
            handler_lines.append(code)
            handler_lines.append("")

    # Build __init__.py content
    tool_names = [t["name"] for t in tools]
    handler_imports = ", ".join(f"handle_{name}" for name in tool_names)
    handler_map = ", ".join(f'"{name}": handle_{name}' for name in tool_names)

    init_lines = [
        f'"""Runtime-created envoy kind: {kind} (source: {source})."""',
        "",
    ]
    if handler_imports:
        init_lines.append(f"from .handlers import {handler_imports}")
        init_lines.append("")

    # TOOLS list
    init_lines.append("TOOLS: list[dict] = [")
    for tool in tools:
        init_lines.append("    {")
        init_lines.append(f'        "name": "{tool["name"]}",')
        init_lines.append(f'        "description": """{tool.get("description", "")}""",')
        init_lines.append(f'        "parameters": {tool.get("parameters", {"type": "object", "properties": {}})},')
        init_lines.append("    },")
    init_lines.append("]")
    init_lines.append("")

    # HANDLERS dict
    if handler_map:
        init_lines.append(f"HANDLERS: dict = {{{handler_map}}}")
    else:
        init_lines.append("HANDLERS: dict = {}")
    init_lines.append("")

    # GLOBAL_TOOLS
    init_lines.append(f"GLOBAL_TOOLS: list[str] = {_DEFAULT_GLOBAL_TOOLS}")
    init_lines.append("")

    # Write to disk
    try:
        kind_dir.mkdir(parents=True)
        (kind_dir / "__init__.py").write_text("\n".join(init_lines))
        (kind_dir / "handlers.py").write_text("\n".join(handler_lines))
    except Exception as e:
        # Clean up on failure
        if kind_dir.exists():
            shutil.rmtree(kind_dir)
        return {"status": "error", "message": f"Failed to write kind files: {e}"}

    # Validate the module imports cleanly
    module_name = f"knowledge.envoys.{kind}"
    try:
        if module_name in sys.modules:
            del sys.modules[module_name]
        mod = importlib.import_module(module_name)
    except Exception as e:
        shutil.rmtree(kind_dir)
        return {"status": "error", "message": f"Kind module failed to import: {e}"}

    # Register in the kind registry
    from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY
    ENVOY_KIND_REGISTRY.register_mission(envoy_id, kind)

    # Register handlers globally
    from agent.tool_handlers import TOOL_REGISTRY
    TOOL_REGISTRY.update(mod.HANDLERS)

    # Generate a default prompt template
    prompt_dir = _PROMPTS_DIR / f"envoy_{kind}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "role.md").write_text(
        f"## {envoy_id} Data Access\n\n"
        f"You access data via the `{source}` {'MCP server' if source_type == 'mcp' else 'Python package'}.\n\n"
        + "\n".join(f"- Use `{t['name']}` — {t.get('description', '')}" for t in tools)
        + "\n"
    )

    # Generate envoy JSON if possible
    try:
        from knowledge.generate_envoy_json import from_package, from_mcp
        from knowledge.mission_loader import clear_cache

        if source_type == "mcp":
            from_mcp(
                tool_schemas=tools,
                package_info={"name": source, "version": "unknown", "doc": ""},
                envoy_id=envoy_id,
                output_dir=kind_dir,
            )
        elif source:
            from_package(
                package_name=source,
                envoy_id=envoy_id,
                output_dir=kind_dir,
            )
        clear_cache()
    except Exception as e:
        logger.warning("Failed to generate envoy JSON for %s: %s", envoy_id, e)

    logger.info("Created envoy kind '%s' for envoy %s", kind, envoy_id)
    return {
        "status": "success",
        "message": f"Created envoy kind '{kind}' with {len(tools)} tools. Envoy {envoy_id} is ready for delegation.",
        "kind": kind,
        "envoy_id": envoy_id,
        "tools": tool_names,
    }


def _remove_envoy(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    """Remove a runtime-created envoy kind."""
    kind = tool_args["kind"].lower()
    envoy_id = tool_args["envoy_id"].upper()

    if kind in _PREBUILT_KINDS:
        return {
            "status": "error",
            "message": f"Cannot remove prebuilt kind '{kind}'. Only runtime-created kinds can be removed.",
        }

    kind_dir = _ENVOYS_DIR / kind
    if not kind_dir.exists():
        return {"status": "error", "message": f"Kind '{kind}' not found"}

    # Unregister from kind registry
    from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY
    ENVOY_KIND_REGISTRY.unregister_mission(envoy_id)

    # Remove from sys.modules
    module_name = f"knowledge.envoys.{kind}"
    sys.modules.pop(module_name, None)

    # Delete from disk
    shutil.rmtree(kind_dir)

    # Remove prompt template if exists
    prompt_dir = _PROMPTS_DIR / f"envoy_{kind}"
    if prompt_dir.exists():
        shutil.rmtree(prompt_dir)

    # Clear mission loader cache
    try:
        from knowledge.mission_loader import clear_cache
        clear_cache()
    except Exception:
        pass

    logger.info("Removed envoy kind '%s' (envoy %s)", kind, envoy_id)
    return {
        "status": "success",
        "message": f"Removed envoy kind '{kind}' and envoy {envoy_id}.",
    }
