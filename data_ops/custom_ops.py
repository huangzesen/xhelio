"""
Custom pandas operation executor with AST-based safety validation.

Allows the LLM to generate arbitrary pandas/numpy code that operates on
a DataFrame. Code is validated via AST analysis to block dangerous
constructs (imports, exec, file I/O, dunder access) and executed in a
restricted namespace with only df, pd, and np available.

Security policy is defined in ``sandbox_registry.json`` (same directory).
"""

import ast
import builtins
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# Load sandbox policy from JSON registry
# ---------------------------------------------------------------------------

def _load_sandbox_registry() -> dict:
    """Load sandbox_registry.json from the same directory as this module."""
    registry_path = Path(__file__).parent / "sandbox_registry.json"
    with open(registry_path, "r") as f:
        return json.load(f)


SANDBOX_REGISTRY: dict = _load_sandbox_registry()

# Derive frozensets from registry
_SAFE_BUILTINS = frozenset(SANDBOX_REGISTRY["builtins"]["safe"])
_DANGEROUS_BUILTINS = frozenset(SANDBOX_REGISTRY["builtins"]["dangerous"])

# Flatten blocked_attrs.any_object categories into a single frozenset
_BLOCKED_ATTRS_ANY = frozenset(
    attr
    for category, items in SANDBOX_REGISTRY["blocked_attrs"]["any_object"].items()
    if category != "_comment"
    for attr in items
)

# Flatten blocked_attrs.module_only categories into a single frozenset
_BLOCKED_ATTRS_MODULE = frozenset(
    attr
    for category, items in SANDBOX_REGISTRY["blocked_attrs"]["module_only"].items()
    if category != "_comment"
    for attr in items
)

# Module names in the sandbox: core (pd, np, xr) + all package aliases
_MODULE_NAMES = frozenset(
    {"pd", "np", "xr"}
    | {pkg["sandbox_alias"] for pkg in SANDBOX_REGISTRY["packages"]}
)

# Allowed variable names in the sandbox namespace
_ALLOWED_NAMES = frozenset(_MODULE_NAMES | {"df", "result"})

# Submodules to catalog in function_catalog.py (single source of truth)
CATALOGED_SUBMODULES: list[str] = [
    submod
    for pkg in SANDBOX_REGISTRY["packages"]
    for submod in pkg["catalog_submodules"]
]


def reload_sandbox_registry() -> None:
    """Reload sandbox_registry.json and rebuild all derived constants.

    Called after install_package or manage_sandbox_packages modifies the JSON.
    """
    global SANDBOX_REGISTRY, _SAFE_BUILTINS, _DANGEROUS_BUILTINS
    global _BLOCKED_ATTRS_ANY, _BLOCKED_ATTRS_MODULE, _MODULE_NAMES
    global _ALLOWED_NAMES, CATALOGED_SUBMODULES

    SANDBOX_REGISTRY = _load_sandbox_registry()

    _SAFE_BUILTINS = frozenset(SANDBOX_REGISTRY["builtins"]["safe"])
    _DANGEROUS_BUILTINS = frozenset(SANDBOX_REGISTRY["builtins"]["dangerous"])

    _BLOCKED_ATTRS_ANY = frozenset(
        attr
        for category, items in SANDBOX_REGISTRY["blocked_attrs"]["any_object"].items()
        if category != "_comment"
        for attr in items
    )
    _BLOCKED_ATTRS_MODULE = frozenset(
        attr
        for category, items in SANDBOX_REGISTRY["blocked_attrs"]["module_only"].items()
        if category != "_comment"
        for attr in items
    )
    _MODULE_NAMES = frozenset(
        {"pd", "np", "xr"}
        | {pkg["sandbox_alias"] for pkg in SANDBOX_REGISTRY["packages"]}
    )
    _ALLOWED_NAMES = frozenset(_MODULE_NAMES | {"df", "result"})
    CATALOGED_SUBMODULES = [
        submod
        for pkg in SANDBOX_REGISTRY["packages"]
        for submod in pkg["catalog_submodules"]
    ]


def add_package_to_registry(
    import_path: str,
    sandbox_alias: str,
    description: str,
    catalog_submodules: list[str] | None = None,
    required: bool = False,
) -> None:
    """Add a package to sandbox_registry.json and hot-reload.

    Args:
        import_path: Python import path (e.g., 'sklearn').
        sandbox_alias: Alias in sandbox namespace (e.g., 'sklearn').
        description: Package description.
        catalog_submodules: Submodules to catalog for function search.
        required: Whether the package is required (True) or optional (False).
    """
    registry_path = Path(__file__).parent / "sandbox_registry.json"
    with open(registry_path, "r") as f:
        data = json.load(f)

    # Check if already registered
    for pkg in data["packages"]:
        if pkg["import_path"] == import_path:
            return  # Already registered

    data["packages"].append({
        "import_path": import_path,
        "sandbox_alias": sandbox_alias,
        "required": required,
        "description": description,
        "catalog_submodules": catalog_submodules or [],
    })

    with open(registry_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    reload_sandbox_registry()


def validate_code(
    code: str,
    require_result: bool = True,
    extra_module_names: frozenset[str] | None = None,
) -> list[str]:
    """Validate pandas code for safety using AST analysis.

    Args:
        code: Python code string to validate.
        require_result: If True (default), require ``result = ...`` assignment.
            Set to False for code that mutates objects in place (e.g., Plotly figures).
        extra_module_names: Additional module aliases (from per-envoy sandbox
            imports) to include in module-level attribute blocking.

    Returns:
        List of violation descriptions. Empty list means code is safe.
    """
    # Build effective blocklists from config (hot-reloadable)
    try:
        import config as _cfg
        allowed = set(_cfg.get("sandbox.allowed_attrs", []) or [])
        extra = set(_cfg.get("sandbox.extra_blocked_attrs", []) or [])
    except Exception:
        allowed, extra = set(), set()

    blocked_any = (_BLOCKED_ATTRS_ANY - allowed) | extra
    blocked_module = _BLOCKED_ATTRS_MODULE - allowed

    # Effective module names: static registry + per-envoy extras
    effective_module_names = _MODULE_NAMES | extra_module_names if extra_module_names else _MODULE_NAMES

    violations = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    has_result_assignment = False

    for node in ast.walk(tree):
        # Block imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            violations.append("Imports are not allowed")

        # Block dangerous builtins
        if isinstance(node, ast.Name) and node.id in _DANGEROUS_BUILTINS:
            violations.append(f"Dangerous builtin '{node.id}' is not allowed")

        # Block dunder attribute access (e.g., __class__, __dict__)
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            violations.append(f"Dunder attribute access '{node.attr}' is not allowed")

        # Block dangerous attributes (two-tier — see blocklist comments above)
        if isinstance(node, ast.Attribute):
            if node.attr in blocked_any:
                violations.append(
                    f"Attribute '{node.attr}' is not allowed (file I/O / "
                    f"deserialization)"
                )
            elif node.attr in blocked_module:
                # Only block when receiver is a known module name
                if (isinstance(node.value, ast.Name)
                        and node.value.id in effective_module_names):
                    violations.append(
                        f"Module-level '{node.value.id}.{node.attr}' is not "
                        f"allowed (code execution / deserialization)"
                    )

        # Block global/nonlocal
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            violations.append("global/nonlocal statements are not allowed")

        # Block async constructs
        if isinstance(node, (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await)):
            violations.append("Async constructs are not allowed")

        # Track result assignment
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "result":
                    has_result_assignment = True
        if isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "result":
                has_result_assignment = True

    if require_result and not has_result_assignment:
        violations.append("Code must assign to 'result'")

    return violations


def _build_sandbox_namespace(extra_imports: list[dict] | None = None) -> dict:
    """Build the sandbox namespace with all allowed scientific packages.

    Core packages (pd, np, xr) are always present. Additional packages
    are loaded from the registry — required packages raise on import
    failure, optional packages are silently skipped.

    Args:
        extra_imports: Optional list of per-envoy package dicts, each with
            ``import_path`` and ``sandbox_alias`` keys.  Missing packages
            are silently skipped (all are optional).
    """
    ns: dict = {
        "pd": pd, "np": np, "xr": xr,
        "result": None,
    }
    # Import packages from registry
    for pkg in SANDBOX_REGISTRY["packages"]:
        import_path = pkg["import_path"]
        alias = pkg["sandbox_alias"]
        try:
            import importlib
            ns[alias] = importlib.import_module(import_path)
        except ImportError:
            if pkg["required"]:
                raise
    # Per-envoy extra imports (all optional — user-specified packages)
    if extra_imports:
        import importlib as _il
        for pkg in extra_imports:
            try:
                ns[pkg["sandbox_alias"]] = _il.import_module(pkg["import_path"])
            except ImportError:
                pass
    return ns


def _execute_in_sandbox(code: str, namespace: dict) -> object:
    """Execute code in a sandboxed namespace and return the 'result' value.

    Builds safe builtins, runs exec(), extracts result, then cleans up.

    Args:
        code: Python code to execute.
        namespace: Dict with variables available to the code (must include 'result': None).

    Returns:
        The value of 'result' after execution.

    Raises:
        RuntimeError: If code execution fails.
    """
    safe_builtins = {name: getattr(builtins, name) for name in _SAFE_BUILTINS if hasattr(builtins, name)}
    # Allow __import__ at runtime so that internal lazy imports (e.g., np.fft,
    # scipy.signal) work.  The AST validator already blocks user code from
    # calling __import__ directly (it's in _DANGEROUS_BUILTINS).
    safe_builtins["__import__"] = builtins.__import__

    # Use a single dict as globals (no separate locals) so that pd/np/etc.
    # are visible inside lambdas and nested functions.  When exec() receives
    # separate globals and locals, closures can only capture globals — causing
    # NameError for names that live only in locals.
    namespace["__builtins__"] = safe_builtins
    try:
        exec(code, namespace)
    except Exception as e:
        raise RuntimeError(f"Execution error: {type(e).__name__}: {e}") from e

    result = namespace.get("result")

    del namespace
    gc.collect()

    return result


# =============================================================================
# Registry protocol adapter
# =============================================================================


class _SandboxProtocolRegistryAdapter:
    name = "sandbox.packages"
    description = "Allowed packages and security policy for code sandbox"

    def get(self, key: str):
        for pkg in SANDBOX_REGISTRY.get("packages", []):
            if pkg.get("sandbox_alias") == key:
                return pkg
        return None

    def list_all(self) -> dict:
        return {pkg["sandbox_alias"]: pkg for pkg in SANDBOX_REGISTRY.get("packages", [])}


SANDBOX_PROTOCOL_REGISTRY = _SandboxProtocolRegistryAdapter()
from agent.registry_protocol import register_registry  # noqa: E402
register_registry(SANDBOX_PROTOCOL_REGISTRY)
