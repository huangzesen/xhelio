"""
Auto-generated searchable catalog of scientific computing functions.

Catalogs functions from whitelisted scipy submodules and pywt at import time.
Provides keyword search over function names and summaries, plus full docstring
retrieval for individual functions.

Used by the DataOps agent's think phase to research APIs before writing code.
"""

import importlib
import inspect
import re


# Packages and submodules to catalog (most relevant for heliophysics timeseries)
_CATALOGED_PACKAGES = [
    "scipy.signal",
    "scipy.fft",
    "scipy.interpolate",
    "scipy.stats",
    "scipy.integrate",
    "pywt",
]

# Maximum docstring length returned by get_function_docstring()
_MAX_DOCSTRING_LENGTH = 3000


def _build_function_catalog() -> dict[str, list[dict]]:
    """Build the function catalog from whitelisted packages.

    Returns:
        Dict mapping package path to list of function entries:
        ``{package_path: [{name, sandbox_call, summary}, ...]}``
    """
    catalog = {}

    for pkg_path in _CATALOGED_PACKAGES:
        try:
            mod = importlib.import_module(pkg_path)
        except ImportError:
            continue

        entries = []
        for name, obj in inspect.getmembers(mod):
            # Skip private, test, and non-callable
            if name.startswith("_") or "test" in name.lower():
                continue
            if not callable(obj):
                continue
            # Skip classes, modules, and other non-function objects
            if inspect.isclass(obj) or inspect.ismodule(obj):
                continue

            # First line of docstring as summary
            doc = inspect.getdoc(obj)
            summary = ""
            if doc:
                first_line = doc.split("\n")[0].strip()
                from agent.truncation import trunc
                summary = trunc(first_line, "context.docstring_summary")

            # Build the sandbox call path
            sandbox_call = f"{pkg_path}.{name}"

            entries.append({
                "name": name,
                "sandbox_call": sandbox_call,
                "summary": summary,
            })

        if entries:
            catalog[pkg_path] = entries

    return catalog


# Build catalog at import time
FUNCTION_CATALOG: dict[str, list[dict]] = _build_function_catalog()


def search_functions(
    query: str, package: str | None = None, max_results: int = 15
) -> list[dict]:
    """Search the function catalog by keyword.

    Uses regex word-boundary matching on function names and summaries.
    Falls back to substring matching if no word-boundary matches found.

    Args:
        query: Search keyword(s).
        package: Optional package path to restrict search (e.g., "scipy.signal").
        max_results: Maximum number of results to return.

    Returns:
        List of matching function entries with package, name, sandbox_call, summary.
    """
    if not query:
        return []

    # Determine which packages to search
    if package and package in FUNCTION_CATALOG:
        search_scope = {package: FUNCTION_CATALOG[package]}
    else:
        search_scope = FUNCTION_CATALOG

    # Split query into words for matching
    words = query.lower().split()

    # Phase 1: word-boundary matching
    results = []
    for pkg_path, entries in search_scope.items():
        for entry in entries:
            search_text = f"{entry['name']} {entry['summary']}".lower()
            if all(re.search(rf"\b{re.escape(w)}", search_text) for w in words):
                results.append({
                    "package": pkg_path,
                    **entry,
                })

    # Phase 2: fallback to substring matching
    if not results:
        for pkg_path, entries in search_scope.items():
            for entry in entries:
                search_text = f"{entry['name']} {entry['summary']}".lower()
                if all(w in search_text for w in words):
                    results.append({
                        "package": pkg_path,
                        **entry,
                    })

    return results[:max_results]


def get_function_docstring(package: str, function_name: str) -> dict:
    """Get the full docstring for a specific function.

    Args:
        package: Package path (e.g., "scipy.signal").
        function_name: Function name (e.g., "butter").

    Returns:
        Dict with package, name, sandbox_call, docstring, and signature.
        On error, returns dict with error message.
    """
    try:
        mod = importlib.import_module(package)
    except ImportError:
        return {"error": f"Package '{package}' not available"}

    obj = getattr(mod, function_name, None)
    if obj is None or not callable(obj):
        return {"error": f"Function '{function_name}' not found in '{package}'"}

    docstring = inspect.getdoc(obj) or "No documentation available."
    if len(docstring) > _MAX_DOCSTRING_LENGTH:
        docstring = docstring[:_MAX_DOCSTRING_LENGTH] + "\n... (truncated)"

    # Get signature
    try:
        sig = str(inspect.signature(obj))
    except (ValueError, TypeError):
        sig = "(...)"

    return {
        "package": package,
        "name": function_name,
        "sandbox_call": f"{package}.{function_name}",
        "signature": f"{function_name}{sig}",
        "docstring": docstring,
    }


def get_function_index_summary() -> str:
    """Get a compact text summary of the function catalog for prompt injection.

    Returns:
        Multi-line string listing each package with function count and top names.
    """
    lines = ["Available function libraries:"]
    from agent.truncation import get_item_limit
    func_limit = get_item_limit("items.catalog_functions")
    for pkg_path, entries in FUNCTION_CATALOG.items():
        top_names = [e["name"] for e in entries[:func_limit]]
        names_str = ", ".join(top_names)
        if len(entries) > func_limit:
            names_str += f", ... ({len(entries)} total)"
        lines.append(f"- {pkg_path}: {names_str}")
    return "\n".join(lines)
