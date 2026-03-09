"""
Blocklist-only sandbox for LLM-generated Python code.

Security model: allow everything except explicitly blocked operations.
Blocked: os/subprocess/shutil/sys/signal/socket/http/ctypes/threading imports,
         eval/exec/compile/__import__ builtins, and dangerous attributes.

Code runs in a subprocess with a timeout. Print output is captured.
File I/O is restricted to the work_dir via os.chdir().
"""

import ast
import io
import json
import multiprocessing
import os
import sys
import traceback
from pathlib import Path

# Load blocklist from JSON config next to this module
_BLOCKLIST_PATH = Path(__file__).parent / "sandbox_blocklist.json"
with open(_BLOCKLIST_PATH, "r", encoding="utf-8") as _f:
    _BLOCKLIST = json.load(_f)

BLOCKED_IMPORTS: set[str] = set(_BLOCKLIST["blocked_imports"])
BLOCKED_BUILTINS: set[str] = set(_BLOCKLIST["blocked_builtins"])
BLOCKED_ATTRS: set[str] = set(_BLOCKLIST["blocked_attrs"])


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
            elif isinstance(func, ast.Attribute) and func.attr in BLOCKED_ATTRS:
                violations.append(f"Blocked attribute: .{func.attr}()")

    return violations


def _run_in_subprocess(code: str, work_dir: str, pipe):
    """Target function for the sandbox subprocess."""
    try:
        # Restrict working directory
        os.chdir(work_dir)

        # Capture stdout
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured

        # Restrict open() to work_dir
        original_open = open
        work_dir_resolved = os.path.realpath(work_dir)

        def restricted_open(file, *args, **kwargs):
            resolved = os.path.realpath(os.path.join(work_dir, str(file)))
            if not resolved.startswith(work_dir_resolved):
                raise PermissionError(
                    f"Access denied: {file} is outside the sandbox directory"
                )
            return original_open(resolved, *args, **kwargs)

        namespace = {"__builtins__": __builtins__, "open": restricted_open}

        try:
            exec(compile(code, "<sandbox>", "exec"), namespace)  # noqa: S102
        except Exception:
            captured.write(traceback.format_exc())
        finally:
            sys.stdout = old_stdout

        result = namespace.get("result", None)
        output = captured.getvalue()

        # Only return serializable results
        try:
            json.dumps(result, default=str)
            pipe.send(("ok", output, result))
        except (TypeError, ValueError):
            # Result not JSON-serializable — try to convert
            pipe.send(("ok", output, str(result)))

    except Exception:
        pipe.send(("error", traceback.format_exc(), None))


def execute_sandboxed(
    code: str,
    work_dir: Path,
    timeout: int = 30,
    extra_imports: list[str] | None = None,
) -> tuple[str, object]:
    """Execute code in a sandboxed subprocess.

    Args:
        code: Python code to execute.
        work_dir: Directory for file I/O (code runs with cwd set here).
        timeout: Maximum execution time in seconds.
        extra_imports: Additional import statements to prepend (for envoy packages).

    Returns:
        Tuple of (captured_output, result_value_or_None).

    Raises:
        RuntimeError: If execution times out.
        TimeoutError: If execution times out.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Prepend extra imports if provided
    if extra_imports:
        code = "\n".join(extra_imports) + "\n" + code

    parent_conn, child_conn = multiprocessing.Pipe()
    proc = multiprocessing.Process(
        target=_run_in_subprocess,
        args=(code, str(work_dir), child_conn),
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
        status, output, result = parent_conn.recv()
        if status == "error":
            return output, None
        return output, result
    else:
        return "Execution completed with no output", None
