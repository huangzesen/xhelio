"""Package installation handler — pip install + sandbox registration."""

from __future__ import annotations

import re
import subprocess
import sys
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent

# Matches valid pip package specifiers: name, extras, version constraints.
# Rejects flags (--index-url, --extra-index-url, etc.) and shell metacharacters.
_SAFE_PIP_NAME = re.compile(
    r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?(\[[A-Za-z0-9,._-]+\])?"
    r"([><=!~]{1,2}[A-Za-z0-9.*]+([, ]*[><=!~]{1,2}[A-Za-z0-9.*]+)*)?$"
)


def handle_install_package(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    """Install a pip package after user approval, then register in sandbox."""
    pip_name = tool_args.get("pip_name", "")
    import_path = tool_args.get("import_path", "")
    sandbox_alias = tool_args.get("sandbox_alias", "")
    description = tool_args.get("description", "")
    catalog_submodules = tool_args.get("catalog_submodules", [])

    if not pip_name or not import_path or not sandbox_alias:
        return {
            "status": "error",
            "message": "pip_name, import_path, and sandbox_alias are all required.",
        }

    if not _SAFE_PIP_NAME.match(pip_name):
        return {
            "status": "error",
            "message": f"Invalid pip_name '{pip_name}'. Must be a valid package name "
                       f"(e.g., 'scikit-learn', 'numpy>=1.20'). Flags and shell "
                       f"metacharacters are not allowed.",
        }

    # Ask user for permission
    command = f"pip install {pip_name}"
    request_id = f"perm-{uuid4().hex[:12]}"
    perm = orch.request_permission(
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

    # Run pip install using the project's venv python
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
        import importlib
        importlib.import_module(import_path)
    except ImportError as e:
        return {
            "status": "error",
            "message": f"Package installed but import '{import_path}' failed: {e}",
        }

    # Register in sandbox
    from data_ops.custom_ops import add_package_to_registry
    add_package_to_registry(
        import_path=import_path,
        sandbox_alias=sandbox_alias,
        description=description,
        catalog_submodules=catalog_submodules,
        required=False,
    )

    return {
        "status": "installed",
        "message": f"Successfully installed '{pip_name}' and registered as '{sandbox_alias}' in the sandbox. "
                   f"Data ops and viz agents can now use it as '{sandbox_alias}' in their code.",
        "sandbox_alias": sandbox_alias,
    }
