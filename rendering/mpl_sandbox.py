"""
Matplotlib script utilities — AST validation and data label extraction.

Provides safety validation for LLM-generated matplotlib scripts (allowlist-based)
and regex extraction of data labels from load_data() calls.

Script execution is handled by data_ops/sandbox.py (execute_sandboxed).
"""

from __future__ import annotations

import ast
import re


# ---- Data Label Extraction ----

_LOAD_DATA_PATTERN = re.compile(r'load_data\(\s*["\']([^"\']+)["\']\s*\)')


def extract_data_labels(code: str) -> list[str]:
    """Extract data labels from load_data("label") calls in matplotlib scripts.

    Args:
        code: Python code string to scan.

    Returns:
        List of unique data labels found (order preserved).
    """
    seen = set()
    labels = []
    for match in _LOAD_DATA_PATTERN.finditer(code):
        label = match.group(1)
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


# ---- AST Validation ----

# Allowed import modules (top-level names)
_ALLOWED_IMPORTS = frozenset({
    "matplotlib", "mpl_toolkits",
    "numpy", "pandas", "scipy", "xarray",
    "datetime", "math", "colorsys", "textwrap",
    "pathlib", "json", "os",  # os.path only — os.system blocked via attrs
    "collections", "itertools", "functools",
    "warnings", "re", "copy", "time",
})

# Blocked import modules
_BLOCKED_IMPORTS = frozenset({
    "socket", "http", "urllib", "requests",
    "subprocess", "ctypes", "multiprocessing",
    "importlib", "builtins", "pickle", "marshal",
    "shlex", "signal", "shutil",
    "tempfile", "webbrowser", "code", "codeop",
    "pty", "fcntl", "termios", "resource",
    "asyncio", "concurrent", "threading",
    "xmlrpc", "ftplib", "smtplib", "poplib",
    "imaplib", "telnetlib", "nntplib",
})

# Blocked builtins
_BLOCKED_BUILTINS = frozenset({
    "exec", "eval", "compile", "__import__",
    "globals", "locals", "vars",
    "breakpoint", "exit", "quit", "input",
    "getattr", "setattr", "delattr",
    "open",  # file I/O
    "memoryview", "type",
})

# Blocked attribute names (on any object)
_BLOCKED_ATTRS = frozenset({
    "system", "popen", "spawn",
    # File I/O
    "read_csv", "read_excel", "read_json", "read_pickle",
    "to_pickle", "to_csv", "to_excel", "to_json",
    "savez", "savez_compressed", "loadtxt", "savetxt",
    "genfromtxt", "fromfile", "tofile", "memmap",
    # subprocess / os
    "run", "call", "check_output", "check_call",
    "Popen",
})


def validate_mpl_script(code: str) -> list[str]:
    """Validate a matplotlib script for safety using AST analysis.

    Args:
        code: Python code string to validate.

    Returns:
        List of violation descriptions. Empty list means code is safe.
    """
    violations = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".")[0]
                if top_level in _BLOCKED_IMPORTS:
                    violations.append(
                        f"Blocked import: '{alias.name}' (module '{top_level}' is not allowed)"
                    )
                elif top_level not in _ALLOWED_IMPORTS:
                    violations.append(
                        f"Unknown import: '{alias.name}' — only matplotlib, numpy, pandas, "
                        f"scipy, xarray, and standard library math/datetime are allowed"
                    )

        if isinstance(node, ast.ImportFrom):
            if node.module:
                top_level = node.module.split(".")[0]
                if top_level in _BLOCKED_IMPORTS:
                    violations.append(
                        f"Blocked import: 'from {node.module}' (module '{top_level}' is not allowed)"
                    )
                elif top_level not in _ALLOWED_IMPORTS:
                    violations.append(
                        f"Unknown import: 'from {node.module}' — only matplotlib, numpy, pandas, "
                        f"scipy, xarray, and standard library math/datetime are allowed"
                    )

        # Block dangerous builtins
        if isinstance(node, ast.Name) and node.id in _BLOCKED_BUILTINS:
            violations.append(f"Blocked builtin: '{node.id}'")

        # Block dunder attribute access
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                violations.append(
                    f"Dunder attribute access '{node.attr}' is not allowed"
                )
            if node.attr in _BLOCKED_ATTRS:
                violations.append(
                    f"Blocked attribute: '{node.attr}' (system access / I/O)"
                )

        # Block global/nonlocal
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            violations.append("global/nonlocal statements are not allowed")

        # Block async constructs
        if isinstance(node, (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await)):
            violations.append("Async constructs are not allowed")

    return violations
