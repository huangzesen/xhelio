"""Sandbox package management handler."""

from __future__ import annotations
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent


def handle_manage_sandbox_packages(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    """List or add packages in the computation sandbox."""
    action = tool_args.get("action", "")

    if action == "list":
        return _list_packages()
    elif action == "add":
        return _add_package(orch, tool_args)
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


def _add_package(orch: "OrchestratorAgent", tool_args: dict) -> dict:
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
        import importlib
        importlib.import_module(import_path)
    except ImportError:
        return {
            "status": "error",
            "message": f"Package '{import_path}' is not installed. Use install_package to install it first.",
        }

    # Ask permission (modifies disk)
    request_id = f"perm-{uuid4().hex[:12]}"
    perm = orch.request_permission(
        request_id=request_id,
        action="modify_sandbox",
        description=f"Add '{import_path}' (alias: '{sandbox_alias}') to the computation sandbox. "
                    f"This modifies sandbox_registry.json on disk.",
        command=f"Add to sandbox: {import_path} as {sandbox_alias}",
    )

    if not perm["approved"]:
        return {
            "status": "denied",
            "message": f"User denied adding '{import_path}' to sandbox.",
        }

    from data_ops.custom_ops import add_package_to_registry
    add_package_to_registry(
        import_path=import_path,
        sandbox_alias=sandbox_alias,
        description=description,
        catalog_submodules=catalog_submodules,
    )

    return {
        "status": "added",
        "message": f"Added '{import_path}' as '{sandbox_alias}' to the sandbox. "
                   f"Data ops and viz agents can now use it.",
        "sandbox_alias": sandbox_alias,
    }
