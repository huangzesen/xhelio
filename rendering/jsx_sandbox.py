"""
JSX/Recharts component sandbox — pattern validation and esbuild compilation.

Validates LLM-generated JSX/TSX code for safety using pattern-based checks,
wraps it with a data context provider, compiles via esbuild subprocess, and
serializes DataEntry DataFrames to JSON for the component to consume.

Modeled on rendering/mpl_sandbox.py for the validation + subprocess pattern.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import logging

logger = logging.getLogger("xhelio")


# ---- Pattern-based Validation ----

# Blocked patterns — these indicate dangerous browser API access
_BLOCKED_PATTERNS: list[tuple[str, str]] = [
    (r'\bfetch\s*\(', "Network access via fetch() is not allowed"),
    (r'\bXMLHttpRequest\b', "Network access via XMLHttpRequest is not allowed"),
    (r'\bwindow\.location\b', "Accessing window.location is not allowed"),
    (r'\bdocument\.cookie\b', "Accessing document.cookie is not allowed"),
    (r'\beval\s*\(', "eval() is not allowed"),
    (r'\bnew\s+Function\b', "new Function() is not allowed"),
    (r'\bimport\s*\(', "Dynamic import() is not allowed"),
    (r'\brequire\s*\(', "require() is not allowed"),
    (r'__proto__', "Accessing __proto__ is not allowed"),
    (r'\blocalStorage\b', "localStorage access is not allowed"),
    (r'\bsessionStorage\b', "sessionStorage access is not allowed"),
    (r'\bWebSocket\b', "WebSocket is not allowed"),
    (r'\bWorker\b', "Worker is not allowed"),
    (r'\bSharedWorker\b', "SharedWorker is not allowed"),
    (r'\bServiceWorker\b', "ServiceWorker is not allowed"),
    (r'\bdocument\.write\b', "document.write is not allowed"),
    (r'\bdocument\.createElement\b', "document.createElement is not allowed"),
    (r'\binnerHTML\b', "innerHTML is not allowed"),
    (r'\bouterHTML\b', "outerHTML is not allowed"),
    (r'\bpostMessage\b', "postMessage is not allowed (reserved for iframe communication)"),
]

# Allowed import sources (bare specifiers only)
_ALLOWED_IMPORTS = frozenset({"react", "recharts"})

# Regex to find import sources: import ... from "source" or import "source"
_IMPORT_PATTERN = re.compile(
    r'''import\s+(?:.*?\s+from\s+)?['"]([^'"]+)['"]''',
    re.MULTILINE,
)

# Regex to find useData("label") calls
_USE_DATA_PATTERN = re.compile(
    r'''useData\s*\(\s*['"]([^'"]+)['"]\s*\)''',
)


def validate_jsx_code(code: str) -> list[str]:
    """Validate JSX/TSX code for safety using pattern-based checks.

    Args:
        code: JSX/TSX code string to validate.

    Returns:
        List of violation descriptions. Empty list means code is safe.
    """
    violations = []

    # Check blocked patterns
    for pattern, message in _BLOCKED_PATTERNS:
        if re.search(pattern, code):
            violations.append(message)

    # Check imports — only react and recharts allowed
    for match in _IMPORT_PATTERN.finditer(code):
        source = match.group(1)
        # Allow relative imports within the component (shouldn't happen but harmless)
        if source.startswith("."):
            continue
        # Extract the package name (e.g., "recharts/lib/foo" -> "recharts")
        pkg_name = source.split("/")[0]
        if pkg_name not in _ALLOWED_IMPORTS:
            violations.append(
                f"Blocked import: '{source}' — only 'react' and 'recharts' are allowed"
            )

    # Must have an export default
    if not re.search(r'\bexport\s+default\b', code):
        violations.append(
            "Component must have an 'export default' statement"
        )

    return violations


def extract_data_labels(code: str) -> list[str]:
    """Extract data labels referenced via useData("label") calls.

    Args:
        code: JSX/TSX code to scan.

    Returns:
        List of unique data labels found (order preserved).
    """
    seen = set()
    labels = []
    for match in _USE_DATA_PATTERN.finditer(code):
        label = match.group(1)
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


# ---- JSX Wrapper ----

_DATA_PROVIDER_WRAPPER = textwrap.dedent("""\
    // === Auto-generated data provider wrapper ===
    import React, { createContext, useContext } from "react";

    // Data context — populated by the iframe host
    declare global {{
      interface Window {{
        __XHELIO_DATA__: Record<string, any[]>;
      }}
    }}

    const DataContext = createContext<Record<string, any[]>>(
      (typeof window !== "undefined" && window.__XHELIO_DATA__) || {{}}
    );

    export function useData(label: string): any[] {{
      const data = useContext(DataContext);
      return data[label] || [];
    }}

    export function useAllLabels(): string[] {{
      const data = useContext(DataContext);
      return Object.keys(data);
    }}

    // === User component starts below ===
    {user_code}
""")


def build_jsx_wrapper(user_code: str) -> str:
    """Wrap LLM-generated code with data context provider hooks.

    The wrapper injects:
    - DataContext with __XHELIO_DATA__ from the iframe window global
    - useData(label) hook to access data by label
    - useAllLabels() hook to list available labels

    Args:
        user_code: The LLM-generated JSX/TSX component code.

    Returns:
        Complete TSX source ready for esbuild compilation.
    """
    return _DATA_PROVIDER_WRAPPER.format(user_code=user_code)


# ---- Data Serialization ----

def serialize_data_for_jsx(
    store,
    labels: list[str],
    max_points: int = 10_000,
) -> dict[str, list[dict]]:
    """Convert DataEntry DataFrames to JSON-serializable dicts.

    Uses stride decimation to honor max_points (same approach as
    the Plotly renderer).

    Args:
        store: The DataStore instance.
        labels: List of data labels to serialize.
        max_points: Maximum data points per label before decimation.

    Returns:
        Dict mapping label -> list of row dicts (JSON-serializable).
    """
    import numpy as np

    result = {}
    for label in labels:
        entry = store.get(label)
        if entry is None:
            continue
        df = entry.df
        if df is None or df.empty:
            result[label] = []
            continue

        # Stride decimation
        n = len(df)
        if n > max_points and max_points > 0:
            stride = max(1, n // max_points)
            df = df.iloc[::stride]

        # Convert to list of row dicts
        rows = []
        for idx, row in df.iterrows():
            row_dict: dict = {}
            # Include index (often timestamps)
            if hasattr(idx, 'isoformat'):
                row_dict["_time"] = idx.isoformat()
            else:
                row_dict["_index"] = idx if not isinstance(idx, (np.integer, np.floating)) else idx.item()

            for col in df.columns:
                val = row[col]
                if hasattr(val, 'item'):
                    val = val.item()
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    val = None
                row_dict[str(col)] = val
            rows.append(row_dict)

        result[label] = rows
    return result


# ---- Compilation Result ----

@dataclass
class JsxCompileResult:
    """Result of JSX compilation pipeline."""
    success: bool
    output_path: Optional[str] = None
    data_path: Optional[str] = None
    stderr: str = ""
    script_path: Optional[str] = None
    exit_code: int = -1


# ---- esbuild Compilation ----

def _find_esbuild() -> str:
    """Locate the esbuild binary.

    Search order:
    1. project frontend/node_modules/.bin/esbuild
    2. project root node_modules/.bin/esbuild
    3. System PATH (via shutil.which)

    Returns:
        Path to esbuild binary.

    Raises:
        FileNotFoundError: If esbuild cannot be found.
    """
    project_root = Path(__file__).resolve().parent.parent

    # Check frontend node_modules
    frontend_esbuild = project_root / "frontend" / "node_modules" / ".bin" / "esbuild"
    if frontend_esbuild.exists():
        return str(frontend_esbuild)

    # Check root node_modules
    root_esbuild = project_root / "node_modules" / ".bin" / "esbuild"
    if root_esbuild.exists():
        return str(root_esbuild)

    # Check system PATH
    system_esbuild = shutil.which("esbuild")
    if system_esbuild:
        return system_esbuild

    raise FileNotFoundError(
        "esbuild not found. Install it via: npm install esbuild "
        "(in project root or frontend directory), or install globally: npm install -g esbuild"
    )


def compile_jsx(
    code: str,
    output_path: Path,
    timeout: float = 30.0,
) -> JsxCompileResult:
    """Compile JSX/TSX code to a browser-ready ES module via esbuild.

    Args:
        code: Complete TSX source (already wrapped with data provider).
        output_path: Path for the compiled .js output.
        timeout: Maximum compilation time in seconds.

    Returns:
        JsxCompileResult with compilation outcome.
    """
    try:
        esbuild = _find_esbuild()
    except FileNotFoundError as e:
        return JsxCompileResult(
            success=False,
            stderr=str(e),
            exit_code=-1,
        )

    # Write TSX to a temp file for esbuild input
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tsx", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        input_path = f.name

    try:
        result = subprocess.run(
            [
                esbuild,
                input_path,
                "--bundle",
                "--format=esm",
                "--jsx=automatic",
                "--loader:.tsx=tsx",
                "--external:react",
                "--external:react-dom",
                "--external:recharts",
                f"--outfile={output_path}",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return JsxCompileResult(
            success=result.returncode == 0,
            output_path=str(output_path) if result.returncode == 0 else None,
            stderr=result.stderr,
            exit_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return JsxCompileResult(
            success=False,
            stderr=f"esbuild compilation timed out after {timeout} seconds",
            exit_code=-1,
        )
    except Exception as e:
        return JsxCompileResult(
            success=False,
            stderr=f"Failed to run esbuild: {e}",
            exit_code=-1,
        )
    finally:
        # Clean up temp file
        try:
            os.unlink(input_path)
        except OSError:
            pass


# ---- Full Pipeline ----

def run_jsx_pipeline(
    code: str,
    store,
    output_dir: Path,
    script_id: str,
    timeout: float = 30.0,
    max_points: int = 10_000,
) -> JsxCompileResult:
    """Full validate → wrap → compile → serialize pipeline.

    1. Validate JSX code for security violations
    2. Extract data labels from useData() calls
    3. Build wrapper with data provider hooks
    4. Compile via esbuild
    5. Serialize referenced data to JSON
    6. Save source TSX for debugging

    Args:
        code: LLM-generated JSX/TSX component code.
        store: The DataStore instance.
        output_dir: Directory for compiled output (session's jsx_outputs/).
        script_id: Unique ID for this compilation.
        timeout: Max esbuild compilation time.
        max_points: Max data points per label before decimation.

    Returns:
        JsxCompileResult with paths to compiled bundle and data JSON.
    """
    # 1. Validate
    violations = validate_jsx_code(code)
    if violations:
        return JsxCompileResult(
            success=False,
            stderr="JSX validation failed:\n" + "\n".join(f"  - {v}" for v in violations),
            exit_code=-1,
        )

    # 2. Extract data labels
    labels = extract_data_labels(code)

    # 3. Prepare directories
    output_dir = Path(output_dir)
    scripts_dir = output_dir.parent / "jsx_scripts"
    output_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Save original source
    script_path = scripts_dir / f"{script_id}.tsx"
    script_path.write_text(code, encoding="utf-8")

    # 4. Wrap and compile
    wrapped = build_jsx_wrapper(code)
    bundle_path = output_dir / f"{script_id}.js"
    result = compile_jsx(wrapped, bundle_path, timeout=timeout)
    result.script_path = str(script_path)

    if not result.success:
        return result

    # 5. Serialize data
    if labels:
        data = serialize_data_for_jsx(store, labels, max_points=max_points)
    else:
        data = {}

    data_path = output_dir / f"{script_id}.data.json"
    data_path.write_text(json.dumps(data), encoding="utf-8")
    result.data_path = str(data_path)

    return result
