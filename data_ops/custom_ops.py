"""
Custom pandas operation executor with AST-based safety validation.

Allows the LLM to generate arbitrary pandas/numpy code that operates on
a DataFrame. Code is validated via AST analysis to block dangerous
constructs (imports, exec, file I/O, dunder access) and executed in a
restricted namespace with only df, pd, and np available.
"""

import ast
import builtins
import gc

import numpy as np
import pandas as pd
import xarray as xr


# Builtins that are safe to use in custom operations
_SAFE_BUILTINS = frozenset({
    "abs", "bool", "dict", "enumerate", "float", "int", "len", "list",
    "max", "min", "print", "range", "round", "sorted", "str", "sum",
    "tuple", "zip", "True", "False", "None", "isinstance",
})

# Builtins that are explicitly dangerous
_DANGEROUS_BUILTINS = frozenset({
    "exec", "eval", "compile", "open", "__import__", "getattr", "setattr",
    "delattr", "globals", "locals", "vars", "dir", "breakpoint", "exit",
    "quit", "input", "memoryview", "classmethod", "staticmethod", "super",
    "property", "type",
})

# Names allowed in the execution namespace
_ALLOWED_NAMES = frozenset({"df", "pd", "np", "xr", "scipy", "pywt", "result"})

# ---------------------------------------------------------------------------
# Sandbox attribute blocklists
#
# Two tiers of blocking, each easy to modify:
#
#   _BLOCKED_ATTRS_ANY      — blocked on ANY object (x.read_csv, df.to_pickle …)
#                             Use for names that are dangerous regardless of the
#                             receiver (I/O, deserialization, system access).
#
#   _BLOCKED_ATTRS_MODULE   — blocked ONLY when the receiver is a known module
#                             name (pd, np, xr, scipy, pywt).  Use for names
#                             that are dangerous as module-level calls but
#                             legitimate as DataFrame/DataArray methods
#                             (e.g., df.eval() vs pd.eval()).
#
# To allow a previously-blocked method, move or remove it from the
# appropriate set.  To block a new method, add it.  Changes take effect
# immediately — no restart needed (module is re-imported per request).
# ---------------------------------------------------------------------------

# Blocked on ALL objects — I/O, deserialization, system access
_BLOCKED_ATTRS_ANY = frozenset({
    # pandas I/O
    "read_csv", "read_excel", "read_json", "read_html", "read_xml",
    "read_parquet", "read_feather", "read_orc", "read_stata", "read_sas",
    "read_spss", "read_pickle", "read_table", "read_fwf", "read_clipboard",
    "read_sql", "read_sql_table", "read_sql_query", "read_gbq", "read_hdf",
    "to_pickle", "to_csv", "to_excel", "to_json", "to_html", "to_xml",
    "to_parquet", "to_feather", "to_orc", "to_stata", "to_hdf",
    # numpy I/O
    "save", "savez", "savez_compressed", "loadtxt", "savetxt",
    "genfromtxt", "fromfile", "tofile", "memmap",
    # xarray I/O
    "open_dataset", "open_dataarray", "open_mfdataset", "open_zarr",
    "load_dataset", "load_dataarray",
    # subprocess / os / system access
    "system", "popen",
    # pandas internal I/O module
    "io",
    # ctypes / low-level
    "ctypeslib",
})

# Blocked ONLY on module objects (pd, np, xr, scipy, pywt) — these are
# legitimate DataFrame/DataArray methods but dangerous as module-level calls
# because they can execute arbitrary strings or create runtime callables that
# bypass AST validation.
_BLOCKED_ATTRS_MODULE = frozenset({
    # pd.eval / pd.query execute arbitrary expression strings
    "eval", "query",
    # np.load can unpickle — dangerous at module level
    "load",
    # np.frompyfunc / np.vectorize wrap arbitrary callables
    "frompyfunc", "vectorize",
})

# The set of module-level variable names in the sandbox namespace
_MODULE_NAMES = frozenset({"pd", "np", "xr", "scipy", "pywt"})


def validate_code(code: str, require_result: bool = True) -> list[str]:
    """Validate pandas code for safety using AST analysis.

    Args:
        code: Python code string to validate.
        require_result: If True (default), require ``result = ...`` assignment.
            Set to False for code that mutates objects in place (e.g., Plotly figures).

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
                        and node.value.id in _MODULE_NAMES):
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


def _validate_result(
    result: object, require_timeseries: bool = False,
) -> pd.DataFrame | xr.DataArray:
    """Validate that the sandbox result is a DataFrame or xarray DataArray.

    - Series → converted to single-column DataFrame
    - xarray DataArray with ``time`` dim → returned as-is (any dimensionality)
    - DataFrame → accepted; if ``require_timeseries`` is True, must have DatetimeIndex
    - Other types → error

    Args:
        result: The value produced by sandbox execution.
        require_timeseries: If True, enforce that DataFrame results have a
            DatetimeIndex. Used when all source data is timeseries.

    Returns:
        Validated DataFrame or DataArray.

    Raises:
        ValueError: If result is None, wrong type, or fails timeseries check.
    """
    if result is None:
        raise ValueError("Code did not assign a value to 'result'")

    if isinstance(result, pd.Series):
        result = result.to_frame(name="value")

    # xarray DataArray: accept if it has a time dimension
    if isinstance(result, xr.DataArray):
        if "time" in result.dims:
            return result
        dims = dict(result.sizes)
        raise ValueError(
            f"Result is an xarray DataArray with dims {dims} but no 'time' "
            f"dimension. The result must have a 'time' dimension. "
            f"Use .rename() if the time dimension has a different name, "
            f"or convert to DataFrame with .to_pandas()."
        )

    if not isinstance(result, pd.DataFrame):
        raise ValueError(
            f"Result must be a DataFrame, Series, or xarray DataArray, "
            f"got {type(result).__name__}"
        )

    if require_timeseries and not isinstance(result.index, pd.DatetimeIndex):
        raise ValueError(
            f"Result must have a DatetimeIndex (got {type(result.index).__name__}). "
            f"All source data is timeseries, so the result should preserve "
            f"the time axis. Use df.index for the time values."
        )

    return result


def execute_multi_source_operation(
    sources: dict[str, pd.DataFrame | xr.DataArray],
    code: str,
    source_timeseries: dict[str, bool] | None = None,
) -> pd.DataFrame | xr.DataArray:
    """Execute validated pandas/xarray code with multiple sources.

    Each source is available in the sandbox by its key:
    - ``df_SUFFIX`` for pandas DataFrames
    - ``da_SUFFIX`` for xarray DataArrays

    The first DataFrame source is also aliased as ``df`` for backward
    compatibility.  ``xr`` (xarray) is always available in the namespace.

    Args:
        sources: Mapping of variable names to DataFrames or DataArrays.
        code: Validated Python code that assigns to 'result'.
        source_timeseries: Optional map of variable name → is_timeseries flag.
            When provided, only coerce to DatetimeIndex for timeseries sources.
            When None, all DataFrame sources are coerced (backward compat).

    Returns:
        Result DataFrame or xarray DataArray.

    Raises:
        RuntimeError: If code execution fails.
        ValueError: If result is not a valid type or missing time axis.
    """
    import scipy
    import pywt
    namespace = {"pd": pd, "np": np, "xr": xr, "scipy": scipy, "pywt": pywt, "result": None}
    first_df_key = None
    for key, data in sources.items():
        if isinstance(data, xr.DataArray):
            namespace[key] = data.copy()
        else:
            df = data.copy()
            # Only coerce to DatetimeIndex for timeseries sources
            is_ts = source_timeseries.get(key, True) if source_timeseries else True
            if is_ts and not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            namespace[key] = df
            if first_df_key is None:
                first_df_key = key
                namespace["df"] = df

    # Enforce DatetimeIndex on result only when all sources are timeseries
    if source_timeseries is not None:
        all_ts = all(source_timeseries.get(k, True) for k in sources)
    else:
        all_ts = True  # backward compat: assume timeseries
    result = _execute_in_sandbox(code, namespace)
    return _validate_result(result, require_timeseries=all_ts)


def validate_result(
    result: pd.DataFrame | xr.DataArray,
    sources: dict[str, pd.DataFrame | xr.DataArray],
) -> list[str]:
    """Validate a computation result against its sources for common pitfalls.

    For DataFrame results, checks:
    1. NaN-to-zero: result has zeros where sources have NaN (skipna issue)
    2. Row count anomaly: result has significantly more rows than largest source
    3. Constant output: result is constant when sources are not

    For DataArray results, only checks row count anomaly.

    Args:
        result: The computation result (DataFrame or DataArray).
        sources: The source DataFrames/DataArrays used (keyed by sandbox variable name).

    Returns:
        List of warning strings. Empty list means no issues detected.
    """
    warnings = []

    # For DataArray results, only do row-count check
    if isinstance(result, xr.DataArray):
        if sources:
            def _source_len(s):
                return s.sizes["time"] if isinstance(s, xr.DataArray) else len(s)
            result_len = result.sizes["time"]
            max_source_rows = max(_source_len(s) for s in sources.values())
            if max_source_rows > 0 and result_len > max_source_rows * 1.1:
                warnings.append(
                    f"Result has {result_len} time steps vs largest source "
                    f"{max_source_rows} — unexpected expansion"
                )
        return warnings

    # DataFrame-specific checks below
    result_df = result

    # Separate DataFrame sources (skip DataArray sources for DataFrame-specific checks)
    df_sources = {k: v for k, v in sources.items() if isinstance(v, pd.DataFrame)}

    # Check 1: NaN-to-zero — zeros in result coinciding with NaN in sources
    result_zeros = (result_df == 0.0)
    for var_name, src_df in df_sources.items():
        src_nan_times = src_df.index[src_df.isna().any(axis=1)]
        if len(src_nan_times) == 0:
            continue
        overlap = result_df.index.intersection(src_nan_times)
        if len(overlap) == 0:
            continue
        zero_at_nan = result_zeros.loc[overlap].any(axis=1).sum()
        if zero_at_nan > 0:
            warnings.append(
                f"Result has {zero_at_nan} zeros coinciding with NaN in "
                f"source '{var_name}' — possible skipna issue"
            )

    # Check 2: Row count anomaly
    if sources:
        def _source_len(s):
            return s.sizes["time"] if isinstance(s, xr.DataArray) else len(s)
        max_source_rows = max(_source_len(s) for s in sources.values())
        if max_source_rows > 0 and len(result_df) > max_source_rows * 1.1:
            warnings.append(
                f"Result has {len(result_df)} rows vs largest source "
                f"{max_source_rows} — unexpected expansion"
            )

    # Check 3: Constant output from non-constant sources
    for col in result_df.columns:
        col_data = result_df[col].dropna()
        if len(col_data) < 2:
            continue
        if col_data.std() == 0:
            # Check if any source is non-constant
            any_source_varies = any(
                src_df.std().max() > 0 for src_df in df_sources.values()
            )
            if any_source_varies:
                warnings.append(
                    f"Result column '{col}' is constant "
                    f"(value={col_data.iloc[0]}) — possible fill value "
                    f"or collapsed computation"
                )

    return warnings


def run_multi_source_operation(
    sources: dict[str, pd.DataFrame | xr.DataArray],
    code: str,
    source_timeseries: dict[str, bool] | None = None,
) -> tuple[pd.DataFrame | xr.DataArray, list[str]]:
    """Validate code, execute with multiple sources, then validate result.

    Convenience function combining code validation, multi-source execution,
    and result validation.

    Args:
        sources: Mapping of variable names to source DataFrames/DataArrays.
        code: Python code that operates on named variables and assigns to 'result'.
        source_timeseries: Optional map of variable name → is_timeseries flag.
            Passed through to ``execute_multi_source_operation``.

    Returns:
        Tuple of (result DataFrame or DataArray, list of warning strings).

    Raises:
        ValueError: If code validation fails or result is invalid.
        RuntimeError: If execution fails.
    """
    violations = validate_code(code)
    if violations:
        raise ValueError(
            "Code validation failed:\n" + "\n".join(f"  - {v}" for v in violations)
        )
    result = execute_multi_source_operation(sources, code, source_timeseries=source_timeseries)
    warnings = validate_result(result, sources)
    return result, warnings


def execute_custom_operation(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Execute validated pandas code in a restricted namespace.

    Args:
        df: Input DataFrame (will be copied to prevent mutation).
        code: Validated Python code that assigns to 'result'.

    Returns:
        Result DataFrame or xarray DataArray.

    Raises:
        RuntimeError: If code execution fails.
        ValueError: If result is not a DataFrame/Series/DataArray.
    """
    import scipy
    import pywt
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    result = _execute_in_sandbox(code, {"df": df, "pd": pd, "np": np, "xr": xr, "scipy": scipy, "pywt": pywt, "result": None})
    return _validate_result(result)


def run_custom_operation(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Validate and execute a custom pandas operation.

    Convenience function that combines validation and execution.

    Args:
        df: Input DataFrame.
        code: Python code that operates on 'df' and assigns to 'result'.

    Returns:
        Result DataFrame.

    Raises:
        ValueError: If validation fails or result is invalid.
        RuntimeError: If execution fails.
    """
    violations = validate_code(code)
    if violations:
        raise ValueError(
            "Code validation failed:\n" + "\n".join(f"  - {v}" for v in violations)
        )
    return execute_custom_operation(df, code)


def execute_dataframe_creation(code: str) -> pd.DataFrame:
    """Execute validated pandas code to create a DataFrame from scratch.

    Unlike execute_custom_operation(), there is no input DataFrame — the code
    constructs data using pd and np only.  Used by the store_dataframe tool to
    turn text data (event catalogs, search results) into stored DataFrames.

    Args:
        code: Validated Python code that assigns to 'result'.

    Returns:
        Result DataFrame (any index type).

    Raises:
        RuntimeError: If code execution fails.
        ValueError: If result is not a DataFrame/Series/DataArray.
    """
    result = _execute_in_sandbox(code, {"pd": pd, "np": np, "xr": xr, "result": None})
    return _validate_result(result)


def run_dataframe_creation(code: str) -> pd.DataFrame:
    """Validate and execute code that creates a DataFrame from scratch.

    Convenience function that combines validation and execution.

    Args:
        code: Python code that constructs data using pd/np and assigns to 'result'.

    Returns:
        Result DataFrame.

    Raises:
        ValueError: If validation fails or result is invalid.
        RuntimeError: If execution fails.
    """
    violations = validate_code(code)
    if violations:
        raise ValueError(
            "Code validation failed:\n" + "\n".join(f"  - {v}" for v in violations)
        )
    return execute_dataframe_creation(code)
