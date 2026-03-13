"""Sandbox package management handler — list, install, and add packages."""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from agent.tool_caller import ToolCaller
    from agent.tool_context import ToolContext

# Matches valid pip package specifiers: name, extras, version constraints.
# Rejects flags (--index-url, --extra-index-url, etc.) and shell metacharacters.
_SAFE_PIP_NAME = re.compile(
    r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?(\[[A-Za-z0-9,._-]+\])?"
    r"([><=!~]{1,2}[A-Za-z0-9.*]+([, ]*[><=!~]{1,2}[A-Za-z0-9.*]+)*)?$"
)


def _get_auto_install() -> bool:
    """Check if sandbox.auto_install is enabled in config."""
    import config
    return bool(config.get("sandbox.auto_install", False))


def _register_package(
    import_path: str,
    sandbox_alias: str,
    description: str,
    catalog_submodules: list[str] | None = None,
    required: bool = False,
) -> None:
    """Register a package in the sandbox registry."""
    from data_ops.custom_ops import add_package_to_registry
    add_package_to_registry(
        import_path=import_path,
        sandbox_alias=sandbox_alias,
        description=description,
        catalog_submodules=catalog_submodules or [],
        required=required,
    )


def handle_manage_sandbox_packages(
    ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None
) -> dict:
    """List, install, or add packages in the computation sandbox."""
    action = tool_args.get("action", "")

    if action == "list":
        return _list_packages()
    elif action == "install":
        return _install_package(ctx, tool_args)
    elif action == "add":
        return _add_package(ctx, tool_args)
    else:
        return {"status": "error", "message": f"Unknown action: {action}"}


def _list_packages() -> dict:
    """List all packages currently in the sandbox."""
    from data_ops.custom_ops import SANDBOX_REGISTRY

    packages = []
    # Core packages (always available)
    packages.append({"alias": "pd", "import_path": "pandas", "description": "Data manipulation", "required": True})
    packages.append({"alias": "np", "import_path": "numpy", "description": "Numerical computing", "required": True})
    packages.append({"alias": "xr", "import_path": "xarray", "description": "N-dimensional arrays", "required": True})

    # Registry packages
    for pkg in SANDBOX_REGISTRY["packages"]:
        packages.append({
            "alias": pkg["sandbox_alias"],
            "import_path": pkg["import_path"],
            "description": pkg["description"],
            "required": pkg["required"],
            "catalog_submodules": pkg.get("catalog_submodules", []),
        })

    return {"status": "ok", "packages": packages}


def _install_package(ctx: "ToolContext", tool_args: dict) -> dict:
    """pip install + verify import + register in sandbox."""
    pip_name = tool_args.get("pip_name", "")
    import_path = tool_args.get("import_path", "")
    sandbox_alias = tool_args.get("sandbox_alias", "")
    description = tool_args.get("description", "")
    catalog_submodules = tool_args.get("catalog_submodules", [])

    if not pip_name or not import_path or not sandbox_alias:
        return {
            "status": "error",
            "message": "pip_name, import_path, and sandbox_alias are all required for 'install'.",
        }

    if not _SAFE_PIP_NAME.match(pip_name):
        return {
            "status": "error",
            "message": f"Invalid pip_name '{pip_name}'. Must be a valid package name "
                       f"(e.g., 'scikit-learn', 'numpy>=1.20'). Flags and shell "
                       f"metacharacters are not allowed.",
        }

    # Permission gate (skipped if auto_install is enabled)
    if not _get_auto_install():
        command = f"pip install {pip_name}"
        request_id = f"perm-{uuid4().hex[:12]}"
        perm = ctx.request_permission(
            request_id=request_id,
            action="install_package",
            description=f"Install '{pip_name}' ({description}). "
                        f"Import path: {import_path}, sandbox alias: {sandbox_alias}",
            command=command,
        )
        if not perm["approved"]:
            return {
                "status": "denied",
                "message": f"User denied installation of '{pip_name}': {perm['reason']}",
            }

    # Run pip install
    python_path = sys.executable
    try:
        result = subprocess.run(
            [python_path, "-m", "pip", "install", pip_name],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": f"pip install {pip_name} timed out after 120 seconds.",
        }

    if result.returncode != 0:
        return {
            "status": "error",
            "message": f"pip install {pip_name} failed:\n{result.stderr}",
        }

    # Verify import works
    try:
        importlib.import_module(import_path)
    except ImportError as e:
        return {
            "status": "error",
            "message": f"Package installed but import '{import_path}' failed: {e}",
        }

    # Register in sandbox
    _register_package(
        import_path=import_path,
        sandbox_alias=sandbox_alias,
        description=description,
        catalog_submodules=catalog_submodules,
    )

    return {
        "status": "installed",
        "message": f"Successfully installed '{pip_name}' and registered as "
                   f"'{sandbox_alias}' in the sandbox.",
        "sandbox_alias": sandbox_alias,
    }


def _add_package(ctx: "ToolContext", tool_args: dict) -> dict:
    """Add an already-installed package to the sandbox (requires permission)."""
    import_path = tool_args.get("import_path", "")
    sandbox_alias = tool_args.get("sandbox_alias", "")
    description = tool_args.get("description", "")
    catalog_submodules = tool_args.get("catalog_submodules", [])

    if not import_path or not sandbox_alias:
        return {
            "status": "error",
            "message": "import_path and sandbox_alias are required for 'add' action.",
        }

    # Verify import works
    try:
        importlib.import_module(import_path)
    except ImportError:
        return {
            "status": "error",
            "message": f"Package '{import_path}' is not installed. "
                       f"Use action='install' to install it first.",
        }

    # Permission gate (skipped if auto_install is enabled)
    if not _get_auto_install():
        request_id = f"perm-{uuid4().hex[:12]}"
        perm = ctx.request_permission(
            request_id=request_id,
            action="modify_sandbox",
            description=f"Add '{import_path}' (alias: '{sandbox_alias}') to the "
                        f"computation sandbox. This modifies sandbox_registry.json on disk.",
            command=f"Add to sandbox: {import_path} as {sandbox_alias}",
        )
        if not perm["approved"]:
            return {
                "status": "denied",
                "message": f"User denied adding '{import_path}' to sandbox.",
            }

    _register_package(
        import_path=import_path,
        sandbox_alias=sandbox_alias,
        description=description,
        catalog_submodules=catalog_submodules,
    )

    return {
        "status": "added",
        "message": f"Added '{import_path}' as '{sandbox_alias}' to the sandbox.",
        "sandbox_alias": sandbox_alias,
    }
