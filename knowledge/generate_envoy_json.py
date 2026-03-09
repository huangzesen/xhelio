"""Generic envoy JSON generator from MCP tool schemas or Python package introspection.

Produces envoy JSON files compatible with the mission catalog format used by
``knowledge/mission_loader.py``. Two entry points:

- ``from_mcp()`` — generates from MCP tool schemas + package metadata
- ``from_package()`` — generates from Python package introspection

Both use an LLM call to produce structured instrument/dataset groupings.
If the LLM call fails, a valid fallback JSON is produced with raw descriptions.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("xhelio")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def from_mcp(
    *,
    tool_schemas: list[dict],
    package_info: dict,
    envoy_id: str,
    output_dir: Path,
) -> Path:
    """Generate envoy JSON from MCP tool schemas.

    Args:
        tool_schemas: List of MCP tool schema dicts (each has ``name``,
            ``description``, ``inputSchema``).
        package_info: Dict with ``name``, ``version``, ``doc`` keys describing
            the source package.
        envoy_id: Envoy identifier (e.g. ``"SPICE"``).
        output_dir: Directory to write the JSON file into.

    Returns:
        Path to the generated (or cached) JSON file.
    """
    version = package_info.get("version", "unknown")
    output_path = output_dir / f"{envoy_id.lower()}.json"

    # Version caching — skip regeneration if version matches
    # When version is "unknown", always regenerate (can't cache without version)
    if version != "unknown" and output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
            if existing.get("_generator_version") == version:
                logger.debug(
                    "Envoy JSON for %s already up-to-date (v%s), skipping",
                    envoy_id, version,
                )
                return output_path
        except (json.JSONDecodeError, OSError):
            pass  # Corrupt file — regenerate

    # Build raw metadata for LLM prompt
    raw_metadata = {
        "package": package_info,
        "tools": tool_schemas,
    }

    # Try LLM-assisted generation
    llm_result = _call_llm(_build_llm_prompt(raw_metadata))

    # Validate: instruments must be a dict of dicts
    llm_instruments = llm_result.get("instruments") if llm_result else None
    if isinstance(llm_instruments, dict) and all(
        isinstance(v, dict) for v in llm_instruments.values()
    ):
        envoy_data = {
            "id": envoy_id,
            "name": llm_result.get("name", envoy_id),
            "keywords": llm_result.get("keywords", []),
            "profile": {
                "description": llm_result.get("description", package_info.get("doc", "")),
                "coordinate_systems": [],
                "typical_cadence": "",
                "data_caveats": [],
                "analysis_patterns": [],
            },
            "instruments": llm_instruments,
            "_generator_version": version,
            "_generator_source": "mcp",
        }
    else:
        # Fallback — no LLM or LLM returned empty/invalid result
        logger.info(
            "LLM generation failed for %s, using MCP fallback", envoy_id,
        )
        envoy_data = {
            "id": envoy_id,
            "name": envoy_id,
            "keywords": [],
            "profile": {
                "description": package_info.get("doc", ""),
                "coordinate_systems": [],
                "typical_cadence": "",
                "data_caveats": [],
                "analysis_patterns": [],
            },
            "instruments": _fallback_instruments_from_mcp(tool_schemas),
            "_generator_version": version,
            "_generator_source": "mcp",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(envoy_data, indent=2) + "\n")
    logger.info("Generated envoy JSON: %s", output_path)
    return output_path


def from_package(
    *,
    package_name: str,
    envoy_id: str,
    output_dir: Path,
) -> Path:
    """Generate envoy JSON from Python package introspection.

    Args:
        package_name: Importable Python package name.
        envoy_id: Envoy identifier.
        output_dir: Directory to write the JSON file into.

    Returns:
        Path to the generated (or cached) JSON file.
    """
    pkg_info = _introspect_package(package_name)
    version = pkg_info.get("version", "unknown")
    output_path = output_dir / f"{envoy_id.lower()}.json"

    # Version caching — skip when version is "unknown" (can't cache reliably)
    if version != "unknown" and output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
            if existing.get("_generator_version") == version:
                logger.debug(
                    "Envoy JSON for %s already up-to-date (v%s), skipping",
                    envoy_id, version,
                )
                return output_path
        except (json.JSONDecodeError, OSError):
            pass

    raw_metadata = {
        "package": pkg_info,
        "functions": pkg_info.get("functions", []),
    }

    llm_result = _call_llm(_build_llm_prompt(raw_metadata))

    # Validate: instruments must be a dict of dicts
    llm_instruments = llm_result.get("instruments") if llm_result else None
    if isinstance(llm_instruments, dict) and all(
        isinstance(v, dict) for v in llm_instruments.values()
    ):
        envoy_data = {
            "id": envoy_id,
            "name": llm_result.get("name", envoy_id),
            "keywords": llm_result.get("keywords", []),
            "profile": {
                "description": llm_result.get("description", pkg_info.get("doc", "")),
                "coordinate_systems": [],
                "typical_cadence": "",
                "data_caveats": [],
                "analysis_patterns": [],
            },
            "instruments": llm_instruments,
            "_generator_version": version,
            "_generator_source": "package",
        }
    else:
        logger.info(
            "LLM generation failed for %s, using package fallback", envoy_id,
        )
        envoy_data = {
            "id": envoy_id,
            "name": envoy_id,
            "keywords": [],
            "profile": {
                "description": pkg_info.get("doc", ""),
                "coordinate_systems": [],
                "typical_cadence": "",
                "data_caveats": [],
                "analysis_patterns": [],
            },
            "instruments": _fallback_instruments_from_package(pkg_info),
            "_generator_version": version,
            "_generator_source": "package",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(envoy_data, indent=2) + "\n")
    logger.info("Generated envoy JSON: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _call_llm(prompt: str) -> dict:
    """Call the LLM to generate structured envoy content.

    Returns a dict with keys like ``name``, ``description``, ``keywords``,
    ``instruments``. Returns an empty dict on any failure.
    """
    try:
        import config as _config
        from agent.llm.service import LLMService

        provider, model, base_url = _config.resolve_agent_model("inline")
        service = LLMService(provider=provider, model=model, base_url=base_url)
        response = service.generate(
            prompt,
            system_prompt=(
                "You are a technical documentation writer. Generate structured "
                "JSON for a tool catalog. Return ONLY valid JSON, no markdown "
                "fencing or explanation."
            ),
            json_schema={"type": "object"},
            tracked=False,
        )
        return json.loads(response.text.strip())
    except Exception:
        logger.warning("LLM call failed for envoy JSON generation", exc_info=True)
        return {}


def _build_llm_prompt(raw_metadata: dict) -> str:
    """Build the LLM prompt for envoy JSON generation."""
    pkg = raw_metadata.get("package", {})
    tools = raw_metadata.get("tools", [])
    functions = raw_metadata.get("functions", [])

    parts = [
        "Given the following package/tool metadata, generate a structured JSON "
        "catalog for an envoy agent.\n",
    ]

    if pkg:
        parts.append(f"Package: {pkg.get('name', 'unknown')}")
        parts.append(f"Version: {pkg.get('version', 'unknown')}")
        if pkg.get("doc"):
            parts.append(f"Description: {pkg['doc']}")
        parts.append("")

    if tools:
        parts.append("MCP Tools:")
        for t in tools:
            schema_str = ""
            if t.get("inputSchema", {}).get("properties"):
                props = t["inputSchema"]["properties"]
                params = ", ".join(
                    f"{k}: {v.get('type', 'any')}" for k, v in props.items()
                )
                schema_str = f" ({params})"
            parts.append(f"  - {t['name']}{schema_str}: {t.get('description', '')}")
        parts.append("")

    if functions:
        parts.append("Python Functions:")
        for f in functions:
            parts.append(
                f"  - {f['name']}{f.get('signature', '')}: {f.get('doc', '')}"
            )
        parts.append("")

    parts.append(
        "Return JSON with these keys:\n"
        '- "name": human-readable name for this tool suite\n'
        '- "description": one-sentence description\n'
        '- "keywords": list of search keywords\n'
        '- "instruments": dict of instrument groups, each with:\n'
        '  - "name": group name\n'
        '  - "keywords": list of keywords\n'
        '  - "datasets": dict mapping tool/function names to '
        '{"description": "..."}\n'
        "\n"
        "Group related tools into logical instrument categories. "
        "Every tool/function must appear in exactly one group."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Package introspection
# ---------------------------------------------------------------------------


def _introspect_package(package_name: str) -> dict:
    """Introspect a Python package to extract metadata and public functions.

    Returns a dict with ``name``, ``version``, ``doc``, and ``functions`` keys.
    """
    try:
        mod = importlib.import_module(package_name)
    except ImportError:
        logger.warning("Cannot import package %r for introspection", package_name)
        return {
            "name": package_name,
            "version": "unknown",
            "doc": "",
            "functions": [],
        }

    version = getattr(mod, "__version__", "unknown")
    doc = inspect.getdoc(mod) or ""

    functions = []
    for name, obj in inspect.getmembers(mod):
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) or inspect.isbuiltin(obj):
            try:
                sig = str(inspect.signature(obj))
            except (ValueError, TypeError):
                sig = "(...)"
            functions.append({
                "name": name,
                "signature": sig,
                "doc": inspect.getdoc(obj) or "",
            })

    return {
        "name": package_name,
        "version": version,
        "doc": doc,
        "functions": functions,
    }


# ---------------------------------------------------------------------------
# Fallback instrument builders (no LLM)
# ---------------------------------------------------------------------------


def _fallback_instruments_from_mcp(tool_schemas: list[dict]) -> dict:
    """Build minimal instruments from raw MCP tool schemas without LLM.

    Groups all tools under a single "Tools" instrument group.
    """
    datasets: dict[str, dict[str, Any]] = {}
    for tool in tool_schemas:
        name = tool.get("name", "unknown")
        datasets[name] = {
            "description": tool.get("description", ""),
        }

    if not datasets:
        return {}

    return {
        "Tools": {
            "name": "Tools",
            "keywords": [],
            "datasets": datasets,
        }
    }


def _fallback_instruments_from_package(package_info: dict) -> dict:
    """Build minimal instruments from raw package info without LLM.

    Groups all functions under a single "Functions" instrument group.
    """
    functions = package_info.get("functions", [])
    datasets: dict[str, dict[str, Any]] = {}
    for func in functions:
        name = func.get("name", "unknown")
        datasets[name] = {
            "description": func.get("doc", ""),
        }

    if not datasets:
        return {}

    return {
        "Functions": {
            "name": "Functions",
            "keywords": [],
            "datasets": datasets,
        }
    }
