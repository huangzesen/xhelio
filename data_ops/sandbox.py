"""
Blocklist-only sandbox for LLM-generated Python code.

Security model — three defense layers:
1. Subprocess isolation — code runs in a separate process, killed on timeout
2. Restricted open() — file I/O confined to work_dir via path resolution
3. Threat-category blocklist — prevents imports/builtins/attrs that could
   escape the subprocess or cause damage outside work_dir

The blocklist is organized by threat category so it's self-documenting and
auditable. Each category explains *why* something is blocked, not just *what*.

Code runs in a subprocess with a timeout. Print output is captured.
File I/O is restricted to the work_dir via os.chdir() + restricted open().
"""

import ast
import builtins as _builtins_mod
import io
import multiprocessing
import os
import sys
import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# Threat-category blocklist
# ---------------------------------------------------------------------------
# Organized by threat type. Each category has optional keys:
#   "imports"  — blocked module names (top-level)
#   "builtins" — blocked builtin function names
#   "attrs"    — blocked attribute names (on any object)

THREAT_CATEGORIES: dict[str, dict[str, list[str]]] = {
    # Network access is allowed — subprocess isolation + timeout + env-var
    # scrubbing provide sufficient protection.  Agents need to fetch remote
    # data (PFSS maps, CDF files, etc.) inside run_code.
    "process_spawn": {
        "imports": ["subprocess", "pty", "multiprocessing"],
        "attrs": [
            # os.system / os.popen
            "system", "popen",
            # os.spawn* family
            "spawn", "spawnl", "spawnle", "spawnlp", "spawnlpe",
            "spawnv", "spawnve", "spawnvp", "spawnvpe",
            # os.fork
            "fork", "forkpty",
            # os.exec* family
            "execl", "execle", "execlp", "execlpe",
            "execv", "execve", "execvp", "execvpe",
            # subprocess attrs (defense in depth — import already blocked)
            "Popen",
            # os.kill — could kill parent process
            "kill", "killpg", "getpid", "getppid",
        ],
    },
    "code_injection": {
        "builtins": ["exec", "eval", "compile", "__import__"],
        # pickle/marshal enable arbitrary code execution via deserialization
        "imports": ["pickle", "marshal"],
    },
    "ffi": {
        "imports": ["ctypes", "_ctypes", "cffi"],
    },
    "loader_manipulation": {
        "imports": ["importlib"],
    },
    "filesystem_damage": {
        "attrs": [
            "rmtree", "unlink", "remove", "rmdir",
        ],
    },
    "information_leak": {
        # os.environ exposes API keys and secrets
        "attrs": ["environ"],
    },
    "reflection": {
        # Dynamic attribute access can bypass attr-level blocks
        "builtins": ["getattr", "setattr", "delattr"],
    },
    "subprocess_hang": {
        # These hang or prematurely terminate the subprocess
        "builtins": ["breakpoint", "exit", "quit", "input"],
    },
}

# Flatten categories into runtime sets
BLOCKED_IMPORTS: set[str] = set()
BLOCKED_BUILTINS: set[str] = set()
BLOCKED_ATTRS: set[str] = set()

for _cat in THREAT_CATEGORIES.values():
    BLOCKED_IMPORTS.update(_cat.get("imports", []))
    BLOCKED_BUILTINS.update(_cat.get("builtins", []))
    BLOCKED_ATTRS.update(_cat.get("attrs", []))


# ---------------------------------------------------------------------------
# AST validation
# ---------------------------------------------------------------------------

def validate_code_blocklist(code: str) -> list[str]:
    """Validate code against the blocklist using AST inspection.

    Returns a list of violation descriptions. Empty list = safe.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax/parse error: {e}"]

    violations = []

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in BLOCKED_IMPORTS or alias.name in BLOCKED_IMPORTS:
                    violations.append(f"Blocked import: {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in BLOCKED_IMPORTS or node.module in BLOCKED_IMPORTS:
                    violations.append(f"Blocked import: {node.module}")

        # Check builtin calls
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BLOCKED_BUILTINS:
                violations.append(f"Blocked builtin: {func.id}()")

        # Check attribute access
        if isinstance(node, ast.Attribute):
            # Block dunder attribute access (sandbox escape vector)
            if node.attr.startswith("__") and node.attr.endswith("__"):
                violations.append(
                    f"Dunder attribute access: .{node.attr}"
                )
            # Block dangerous attrs
            if node.attr in BLOCKED_ATTRS:
                violations.append(f"Blocked attribute: .{node.attr}")

    return violations


# ---------------------------------------------------------------------------
# Dynamic prompt generation
# ---------------------------------------------------------------------------

def build_sandbox_rules_prompt() -> str:
    """Generate a markdown section describing sandbox restrictions for LLM prompts.

    Built dynamically from THREAT_CATEGORIES so prompts never drift
    from actual enforcement.
    """
    lines = [
        "## Sandbox Restrictions",
        "",
        "Your code runs in a sandboxed subprocess. The following are blocked:",
        "",
    ]
    for cat_name, cat in THREAT_CATEGORIES.items():
        label = cat_name.replace("_", " ").title()
        items = []
        for key in ("imports", "builtins", "attrs"):
            items.extend(cat.get(key, []))
        lines.append(f"- **{label}:** `{'`, `'.join(items)}`")
    lines.append("")
    lines.append(
        "Dunder attribute access (`__class__`, `__bases__`, etc.) is also blocked."
    )
    lines.append("")
    lines.append(
        "Everything else is allowed — `os`, `sys`, `shutil`, `threading`, etc. are fine."
    )
    lines.append(
        "Use `os.path` for path operations, `shutil.copy` for file copies within the sandbox."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subprocess sandbox
# ---------------------------------------------------------------------------

_SENSITIVE_ENV_PATTERNS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")


def _run_in_subprocess(code: str, work_dir: str, pipe,
                       output_vars: list[str] | None = None):
    """Target function for the sandbox subprocess."""
    try:
        # Defense in depth: clear sensitive env vars so sandbox code
        # cannot leak secrets even if the attr block on 'environ' is bypassed
        for key in list(os.environ.keys()):
            if any(s in key.upper() for s in _SENSITIVE_ENV_PATTERNS):
                del os.environ[key]

        # Restrict working directory
        os.chdir(work_dir)

        # Capture stdout
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured

        # Restrict open() to work_dir
        original_open = open
        work_dir_resolved = Path(os.path.realpath(work_dir))

        def restricted_open(file, *args, **kwargs):
            resolved = Path(os.path.realpath(os.path.join(work_dir, str(file))))
            try:
                resolved.relative_to(work_dir_resolved)
            except ValueError:
                raise PermissionError(
                    f"Access denied: {file} is outside the sandbox directory"
                )
            return original_open(str(resolved), *args, **kwargs)

        # Filtered builtins — only safe names are exposed in the namespace.
        # This is an independent enforcement layer from the AST blocklist.
        _SAFE_BUILTINS = {
            "abs", "all", "any", "bin", "bool", "bytes", "callable", "chr",
            "complex", "dict", "dir", "divmod", "enumerate", "filter", "float",
            "format", "frozenset", "hasattr", "hash", "hex", "id",
            "int", "isinstance", "issubclass", "iter", "len", "list", "map",
            "max", "min", "next", "object", "oct", "ord", "pow", "print",
            "range", "repr", "reversed", "round", "set", "slice", "sorted",
            "str", "sum", "super", "tuple", "type", "zip",
            "True", "False", "None", "NotImplemented", "Ellipsis",
            "ArithmeticError", "AssertionError", "AttributeError",
            "BufferError", "EOFError", "FileNotFoundError",
            "FloatingPointError", "GeneratorExit", "ImportError",
            "IndexError", "KeyError", "KeyboardInterrupt", "LookupError",
            "MemoryError", "NameError", "NotImplementedError", "OSError",
            "OverflowError", "PermissionError", "RecursionError",
            "ReferenceError", "RuntimeError", "StopIteration",
            "StopAsyncIteration", "SyntaxError", "SystemExit", "TypeError",
            "UnboundLocalError", "UnicodeDecodeError", "UnicodeEncodeError",
            "UnicodeError", "UnicodeTranslationError", "ValueError",
            "ZeroDivisionError", "Exception", "BaseException",
            "ConnectionError", "FileExistsError", "IsADirectoryError",
            "NotADirectoryError", "TimeoutError", "ProcessLookupError",
            "ChildProcessError", "BrokenPipeError", "ConnectionAbortedError",
            "ConnectionRefusedError", "ConnectionResetError",
            # Warning hierarchy — needed by warnings.filterwarnings(category=...)
            "Warning", "UserWarning", "DeprecationWarning", "FutureWarning",
            "RuntimeWarning", "SyntaxWarning", "ResourceWarning",
            "PendingDeprecationWarning", "ImportWarning", "UnicodeWarning",
            "BytesWarning", "EncodingWarning",
        }
        safe_builtins = {
            name: getattr(_builtins_mod, name)
            for name in _SAFE_BUILTINS
            if hasattr(_builtins_mod, name)
        }
        # Wrap __import__ to block imports of forbidden modules at runtime.
        # Needed because internal lazy imports (np.fft, scipy.signal) use
        # __import__, but we must prevent sandbox code from importing blocked
        # modules via aliases like `f = __import__; f("os")`.
        _real_import = _builtins_mod.__import__

        def _guarded_import(name, *args, **kwargs):
            top = name.split(".")[0]
            if top in BLOCKED_IMPORTS or name in BLOCKED_IMPORTS:
                raise ImportError(f"Import blocked in sandbox: {name}")
            return _real_import(name, *args, **kwargs)

        safe_builtins["__import__"] = _guarded_import
        namespace = {"__builtins__": safe_builtins, "open": restricted_open}

        try:
            exec(compile(code, "<sandbox>", "exec"), namespace)  # noqa: S102
        except Exception:
            captured.write(traceback.format_exc())
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()

        # Multi-output: write each requested variable to a typed file
        results = {}
        for var_name in (output_vars or []):
            if var_name not in namespace:
                results[var_name] = {"status": "missing"}
                continue
            val = namespace[var_name]
            try:
                import pandas as _pd
                if isinstance(val, _pd.DataFrame):
                    val.to_parquet(os.path.join(work_dir, f"{var_name}.parquet"))
                    results[var_name] = {"status": "ok", "format": "parquet"}
                    continue
            except Exception:
                pass
            try:
                import xarray as _xr
                if isinstance(val, (_xr.DataArray, _xr.Dataset)):
                    val.to_netcdf(os.path.join(work_dir, f"{var_name}.nc"))
                    results[var_name] = {"status": "ok", "format": "nc"}
                    continue
            except Exception:
                pass
            # Fallback: JSON
            try:
                import json
                with original_open(os.path.join(work_dir, f"{var_name}.json"), "w") as f:
                    json.dump(val, f, default=str)
                results[var_name] = {"status": "ok", "format": "json"}
            except Exception:
                results[var_name] = {"status": "missing"}

        pipe.send(("ok", output, results))

    except Exception:
        pipe.send(("error", traceback.format_exc(), {}))


def execute_sandboxed(
    code: str,
    work_dir: Path,
    timeout: int = 30,
    extra_imports: list[str] | None = None,
    output_vars: list[str] | None = None,
) -> tuple[str, dict[str, dict]]:
    """Execute code in a sandboxed subprocess.

    Args:
        code: Python code to execute.
        work_dir: Directory for file I/O (code runs with cwd set here).
        timeout: Maximum execution time in seconds.
        extra_imports: Additional import statements to prepend (for envoy packages).
        output_vars: Variable names to extract from the namespace after execution.
            Each variable is written to a typed file (parquet, netCDF, or JSON).

    Returns:
        Tuple of (captured_output, results_dict).
        results_dict maps variable names to
        {"status": "ok", "format": "parquet"|"nc"|"json"} or
        {"status": "missing"}.

    Raises:
        TimeoutError: If execution times out.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Prepend extra imports if provided (validate each one first)
    if extra_imports:
        for imp in extra_imports:
            violations = validate_code_blocklist(imp)
            if violations:
                return f"Unsafe extra_import: {'; '.join(violations)}", {}
        code = "\n".join(extra_imports) + "\n" + code

    parent_conn, child_conn = multiprocessing.Pipe()
    proc = multiprocessing.Process(
        target=_run_in_subprocess,
        args=(code, str(work_dir), child_conn, output_vars or []),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
            proc.join()
        raise TimeoutError(f"Code execution timed out after {timeout}s")

    if parent_conn.poll():
        status, output, results = parent_conn.recv()
        if status == "error":
            return output, {}
        return output, results
    else:
        return "Execution completed with no output", {}
