"""Handler for manage_envoy — runtime envoy kind creation and removal."""

from __future__ import annotations

import importlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.tool_caller import ToolCaller
    from agent.tool_context import ToolContext

logger = logging.getLogger("xhelio")

_ENVOYS_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge" / "envoys"
_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge" / "prompts"

# Prebuilt kinds that cannot be removed (none currently)
_PREBUILT_KINDS: frozenset[str] = frozenset()

# Default global tools every envoy gets
_DEFAULT_GLOBAL_TOOLS = [
    "xhelio__assets",
    "xhelio__review_memory",
    "xhelio__events",
    "xhelio__run_code",
]


def handle_manage_envoy(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    """Dispatch to create or remove based on action."""
    action = tool_args.get("action")
    if action == "create":
        return _create_envoy(ctx, tool_args)
    elif action == "remove":
        return _remove_envoy(ctx, tool_args)
    else:
        return {"status": "error", "message": f"Unknown action: {action}"}


def _create_envoy(ctx: "ToolContext", tool_args: dict) -> dict:
    """Create a new envoy kind on disk and register it."""
    kind = tool_args["kind"].lower()
    envoy_id = tool_args["envoy_id"].upper()
    source = tool_args.get("source", "")
    source_type = tool_args.get("source_type", "package")
    tools = tool_args.get("tools", [])

    # LLM may pass tools as a JSON string instead of a list
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return {"status": "error", "message": "tools must be a JSON array of tool objects"}
    if not isinstance(tools, list):
        return {"status": "error", "message": f"tools must be a list, got {type(tools).__name__}"}

    kind_dir = _ENVOYS_DIR / kind
    if kind_dir.exists():
        return {"status": "error", "message": f"Kind '{kind}' already exists"}

    # Filter to valid tools (have both name and handler_code)
    valid_tools = [t for t in tools if t.get("name") and t.get("handler_code")]
    tool_names = [t["name"] for t in valid_tools]

    # Validate each handler_code before writing anything to disk
    from data_ops.sandbox import validate_code_blocklist
    for tool in valid_tools:
        code = tool["handler_code"]
        name = tool["name"]
        try:
            compile(code, f"{name}.py", "exec")
        except SyntaxError as e:
            return {"status": "error", "message": f"Syntax error in handler_code for '{name}': {e}"}
        violations = validate_code_blocklist(code)
        if violations:
            return {"status": "error", "message": f"Unsafe handler_code for '{name}': {'; '.join(violations)}"}

    # Determine kind type: "mcp" or "codegen" (default)
    kind_type = tool_args.get("type", "codegen")

    # Build the manifest
    manifest = {
        "kind": kind,
        "envoy_id": envoy_id,
        "source": source,
        "source_type": source_type,
        "type": kind_type,
        "global_tools": list(_DEFAULT_GLOBAL_TOOLS),
        "tools": [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                "handler_code": t.get("handler_code", ""),
                "pipeline_relevant": t.get("pipeline_relevant", False),
            }
            for t in valid_tools
        ],
    }

    try:
        kind_dir.mkdir(parents=True)
        # Persist manifest as JSON (source of truth for restart recovery)
        (kind_dir / "kind.json").write_text(json.dumps(manifest, indent=2))
    except Exception as e:
        if kind_dir.exists():
            shutil.rmtree(kind_dir)
        return {"status": "error", "message": f"Failed to write kind files: {e}"}

    from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY
    from agent.tool_handlers import TOOL_REGISTRY

    if kind_type == "mcp":
        # MCP path: register tools via make_mcp_handler (no codegen)
        from agent.tool_handlers import register_envoy_tools
        tool_names_registered = register_envoy_tools(kind, manifest["tools"])
        ENVOY_KIND_REGISTRY.register_mission(envoy_id, kind)
    else:
        # Codegen path: generate .py files and import
        from agent.envoy_kinds.codegen import generate_kind_files
        try:
            generate_kind_files(kind_dir, manifest)
        except Exception as e:
            shutil.rmtree(kind_dir)
            return {"status": "error", "message": f"Failed to generate kind files: {e}"}

        # Validate the module imports cleanly
        module_name = f"knowledge.envoys.{kind}"
        try:
            sys.modules.pop(module_name, None)
            sys.modules.pop(f"{module_name}.handlers", None)
            mod = importlib.import_module(module_name)
        except Exception as e:
            shutil.rmtree(kind_dir)
            return {"status": "error", "message": f"Kind module failed to import: {e}"}

        ENVOY_KIND_REGISTRY.register_mission(envoy_id, kind)
        TOOL_REGISTRY.update(mod.HANDLERS)

    # Generate a default prompt template
    prompt_dir = _PROMPTS_DIR / f"envoy_{kind}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "role.md").write_text(
        f"## {envoy_id} Data Access\n\n"
        f"You access data via the `{source}` {'MCP server' if source_type == 'mcp' else 'Python package'}.\n\n"
        + "\n".join(f"- Use `{t['name']}` — {t.get('description', '')}" for t in valid_tools)
        + "\n"
    )

    logger.info("Created envoy kind '%s' for envoy %s", kind, envoy_id)
    return {
        "status": "success",
        "message": f"Created envoy kind '{kind}' with {len(valid_tools)} tools. Envoy {envoy_id} is ready for delegation.",
        "kind": kind,
        "envoy_id": envoy_id,
        "tools": tool_names,
    }


def _remove_envoy(ctx: "ToolContext", tool_args: dict) -> dict:
    """Remove a runtime-created envoy kind."""
    kind = tool_args.get("kind", "").lower()
    envoy_id = tool_args["envoy_id"].upper()

    # LLM may omit kind — look it up from the registry
    if not kind:
        from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY
        kind = ENVOY_KIND_REGISTRY._mission_kinds.get(envoy_id, "")
        if not kind:
            return {"status": "error", "message": f"Cannot determine kind for envoy '{envoy_id}'"}

    if kind in _PREBUILT_KINDS:
        return {
            "status": "error",
            "message": f"Cannot remove prebuilt kind '{kind}'. Only runtime-created kinds can be removed.",
        }

    kind_dir = _ENVOYS_DIR / kind
    if not kind_dir.exists():
        return {"status": "error", "message": f"Kind '{kind}' not found"}

    # Clean up TOOL_REGISTRY before losing the module/files
    from agent.tool_handlers import TOOL_REGISTRY

    # Read kind.json to determine type and tool names
    kind_json_path = kind_dir / "kind.json"
    kind_type = "codegen"
    if kind_json_path.exists():
        try:
            manifest = json.loads(kind_json_path.read_text())
            kind_type = manifest.get("type", "codegen")
            if kind_type == "mcp":
                # MCP tools are registered as kind:tool_name
                from data_ops.dag import PIPELINE_TOOLS
                for tool in manifest.get("tools", []):
                    namespaced = f"{kind}:{tool.get('name', '')}"
                    TOOL_REGISTRY.pop(namespaced, None)
                    PIPELINE_TOOLS.discard(namespaced)
        except (json.JSONDecodeError, KeyError):
            pass

    if kind_type == "codegen":
        module_name = f"knowledge.envoys.{kind}"
        mod = sys.modules.get(module_name)
        if mod and hasattr(mod, "HANDLERS"):
            for tool_name in mod.HANDLERS:
                TOOL_REGISTRY.pop(tool_name, None)

    # Unregister from kind registry
    from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY
    ENVOY_KIND_REGISTRY.unregister_mission(envoy_id)

    # Remove package and submodule from sys.modules (codegen path only)
    module_name = f"knowledge.envoys.{kind}"
    sys.modules.pop(module_name, None)
    sys.modules.pop(f"{module_name}.handlers", None)

    # Delete from disk
    shutil.rmtree(kind_dir)

    # Remove prompt template if exists
    prompt_dir = _PROMPTS_DIR / f"envoy_{kind}"
    if prompt_dir.exists():
        shutil.rmtree(prompt_dir)

    logger.info("Removed envoy kind '%s' (envoy %s)", kind, envoy_id)
    return {
        "status": "success",
        "message": f"Removed envoy kind '{kind}' and envoy {envoy_id}.",
    }
