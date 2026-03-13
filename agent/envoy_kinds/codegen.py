"""Generate Python source files for runtime envoy kinds from structured JSON data.

Instead of interpolating LLM output into Python source via string concatenation
(which caused 12+ bugs), this module generates .py files from a validated JSON
manifest. Data flows through json.dumps() (safe), and handler_code blocks are
written verbatim after compile() + blocklist validation.
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_kind_files(kind_dir: Path, manifest: dict) -> None:
    """Generate __init__.py and handlers.py from a kind.json manifest.

    Args:
        kind_dir: Directory to write files into (must already exist).
        manifest: Parsed kind.json with keys: kind, envoy_id, source,
                  source_type, tools (list of tool dicts with name,
                  description, parameters, handler_code).
    """
    tools = manifest.get("tools", [])
    valid_tools = [t for t in tools if t.get("name") and t.get("handler_code")]
    all_named_tools = [t for t in tools if t.get("name")]

    _write_handlers(kind_dir / "handlers.py", manifest["kind"], valid_tools)
    _write_init(kind_dir / "__init__.py", manifest, all_named_tools, valid_tools)


def _write_handlers(path: Path, kind: str, valid_tools: list[dict]) -> None:
    """Write handlers.py with handler_code blocks and wrappers."""
    lines = [
        f'"""Auto-generated handlers for the {kind} envoy kind."""',
        "# codegen_version: 2  # ToolContext signature",
        "",
    ]

    for tool in valid_tools:
        name = tool["name"]
        code = tool["handler_code"]
        lines.append(code.rstrip())
        lines.append("")
        # Generate handle_{name} wrapper unless handler_code already defines one
        if not _defines_handle_function(code, name):
            lines.append(f"def handle_{name}(ctx, tool_args):")
            lines.append(f"    return {name}(**tool_args)")
            lines.append("")

    path.write_text("\n".join(lines))


def _write_init(path: Path, manifest: dict, all_named_tools: list[dict],
                valid_tools: list[dict]) -> None:
    """Write __init__.py with TOOLS, HANDLERS, GLOBAL_TOOLS."""
    kind = manifest["kind"]
    source = manifest.get("source", "")
    global_tools = manifest.get("global_tools", [])

    tool_names = [t["name"] for t in valid_tools]
    handler_imports = ", ".join(f"handle_{n}" for n in tool_names)

    lines = [
        f'"""Runtime-created envoy kind: {kind} (source: {source})."""',
        "",
    ]
    if handler_imports:
        lines.append(f"from .handlers import {handler_imports}")
        lines.append("")

    # TOOLS list — use json.dumps for safe serialization of LLM-provided data
    lines.append("TOOLS: list[dict] = [")
    for tool in all_named_tools:
        tool_dict = {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        }
        # json.dumps each tool dict, then indent it
        serialized = json.dumps(tool_dict, indent=4)
        # Indent each line by 4 spaces
        indented = "\n".join("    " + line for line in serialized.splitlines())
        lines.append(indented + ",")
    lines.append("]")
    lines.append("")

    # HANDLERS dict
    if tool_names:
        handler_map = ", ".join(f'"{n}": handle_{n}' for n in tool_names)
        lines.append(f"HANDLERS: dict = {{{handler_map}}}")
    else:
        lines.append("HANDLERS: dict = {}")
    lines.append("")

    # GLOBAL_TOOLS — use json.dumps for safe serialization
    lines.append(f"GLOBAL_TOOLS: list[str] = {json.dumps(global_tools)}")
    lines.append("")

    path.write_text("\n".join(lines))


def _defines_handle_function(code: str, name: str) -> bool:
    """Check if handler_code already defines a handle_{name} function."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == f"handle_{name}":
            return True
    return False
