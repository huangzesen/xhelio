"""
Matplotlib script sandbox — AST validation and subprocess execution.

Validates LLM-generated matplotlib scripts for safety, wraps them with
data-loading helpers and headless backend config, then executes in a
subprocess. Output images are saved to the session's mpl_outputs directory.

Modeled on data_ops/custom_ops.py:validate_code for the AST validation pattern.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import logging

logger = logging.getLogger("xhelio")


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


# ---- Script Wrapper ----

_SCRIPT_PREAMBLE = textwrap.dedent("""\
    # === Auto-generated wrapper — do not edit above this line ===
    import sys
    import json
    import warnings
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    # ---- Data helpers ----
    _DATA_DIR = Path({data_dir_repr})
    _OUTPUT_PATH = Path({output_path_repr})
    _LABELS_INDEX = json.loads({labels_json_repr})

    def load_data(label: str) -> pd.DataFrame:
        \"\"\"Load a DataFrame from the session data store by label.\"\"\"
        h = _LABELS_INDEX.get(label)
        if h is None:
            available = list(_LABELS_INDEX.keys())
            raise KeyError(
                f"Label '{{label}}' not found. Available labels: {{available}}"
            )
        data_path = _DATA_DIR / h / "data.pkl"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found for label '{{label}}': {{data_path}}")
        return pd.read_pickle(data_path)

    def load_meta(label: str) -> dict:
        \"\"\"Load metadata dict for a label.\"\"\"
        h = _LABELS_INDEX.get(label)
        if h is None:
            raise KeyError(f"Label '{{label}}' not found in store")
        meta_path = _DATA_DIR / h / "meta.json"
        if not meta_path.exists():
            return {{}}
        with open(meta_path) as f:
            return json.load(f)

    def available_labels() -> list[str]:
        \"\"\"Return all available data labels.\"\"\"
        return list(_LABELS_INDEX.keys())

    # === User script starts below ===
""")

_SCRIPT_EPILOGUE = textwrap.dedent("""\

    # === Auto-generated epilogue ===
    plt.savefig(str(_OUTPUT_PATH), dpi=150, bbox_inches="tight")
    plt.close("all")
""")


def build_script_wrapper(
    user_code: str,
    data_dir: Path,
    output_path: Path,
    labels_index: dict[str, str],
) -> str:
    """Wrap user code with data-loading helpers and matplotlib backend setup.

    Args:
        user_code: The LLM-generated matplotlib script.
        data_dir: Path to the session's data directory (contains _labels.json and hash dirs).
        output_path: Path where the output image should be saved.
        labels_index: Mapping of label → hash folder name.

    Returns:
        Complete Python script ready for subprocess execution.
    """
    labels_json = json.dumps(labels_index)
    preamble = _SCRIPT_PREAMBLE.format(
        data_dir_repr=repr(str(data_dir)),
        output_path_repr=repr(str(output_path)),
        labels_json_repr=repr(labels_json),
    )
    return preamble + user_code + _SCRIPT_EPILOGUE


# ---- Sandbox Result ----

@dataclass
class MplSandboxResult:
    """Result of executing a matplotlib script in a subprocess."""
    success: bool
    output_path: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    script_path: Optional[str] = None
    exit_code: int = -1


# ---- Script Execution ----

def run_mpl_script(
    code: str,
    data_dir: Path,
    output_dir: Path,
    labels_index: dict[str, str],
    script_id: str,
    timeout: float = 60.0,
) -> MplSandboxResult:
    """Validate and execute a matplotlib script in a subprocess.

    1. Validate via validate_mpl_script()
    2. Build wrapped script via build_script_wrapper()
    3. Save to {output_dir}/../mpl_scripts/{script_id}.py
    4. Execute in subprocess with venv Python
    5. Return MplSandboxResult

    Args:
        code: The LLM-generated matplotlib code.
        data_dir: Path to session data directory.
        output_dir: Path to mpl_outputs directory.
        labels_index: Mapping of label → hash folder name.
        script_id: Unique ID for this script execution.
        timeout: Max execution time in seconds.

    Returns:
        MplSandboxResult with execution outcome.
    """
    # Validate
    violations = validate_mpl_script(code)
    if violations:
        return MplSandboxResult(
            success=False,
            stderr="Script validation failed:\n" + "\n".join(f"  - {v}" for v in violations),
            exit_code=-1,
        )

    # Prepare directories
    output_dir = Path(output_dir)
    scripts_dir = output_dir.parent / "mpl_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{script_id}.png"
    script_path = scripts_dir / f"{script_id}.py"

    # Build and save wrapped script
    wrapped = build_script_wrapper(code, data_dir, output_path, labels_index)
    script_path.write_text(wrapped, encoding="utf-8")

    # Find Python executable (prefer venv)
    python_exe = _find_python()

    # Execute in subprocess with minimal environment
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
    }

    # Use a temporary MPLCONFIGDIR to avoid matplotlib cache issues
    with tempfile.TemporaryDirectory(prefix="mpl_") as mpl_config:
        env["MPLCONFIGDIR"] = mpl_config

        try:
            result = subprocess.run(
                [python_exe, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            return MplSandboxResult(
                success=result.returncode == 0 and output_path.exists(),
                output_path=str(output_path) if output_path.exists() else None,
                stdout=result.stdout,
                stderr=result.stderr,
                script_path=str(script_path),
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return MplSandboxResult(
                success=False,
                stderr=f"Script execution timed out after {timeout} seconds",
                script_path=str(script_path),
                exit_code=-1,
            )
        except Exception as e:
            return MplSandboxResult(
                success=False,
                stderr=f"Failed to execute script: {e}",
                script_path=str(script_path),
                exit_code=-1,
            )


def _find_python() -> str:
    """Find the appropriate Python executable, preferring venv."""
    # Check if we're in a venv already
    venv_python = Path(sys.prefix) / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)

    # Check for project venv
    project_root = Path(__file__).resolve().parent.parent
    project_venv = project_root / "venv" / "bin" / "python"
    if project_venv.exists():
        return str(project_venv)

    # Fallback to sys.executable
    return sys.executable
