"""
Per-mission JSON loader with caching.

Loads mission data from knowledge/missions/*.json files.
Provides a clean API for accessing mission metadata, routing tables,
and dataset information without loading everything into memory upfront.

Supports override files in ``{data_dir}/mission_overrides/`` that are
deep-merged on top of the auto-generated base JSON at load time.
Override files are sparse patches — only fields that differ from the
auto-generated base need to be present.  Generic recursive deep-merge:
dicts merge recursively, everything else replaces.
"""

import json
from pathlib import Path

from ._utils import deep_merge


def _emit_debug(*, agent: str = "MissionLoader", level: str = "debug", msg: str = "") -> None:
    """Lazy import to avoid circular dependency (knowledge → agent → knowledge)."""
    from agent.event_bus import get_event_bus, DEBUG
    get_event_bus().emit(DEBUG, agent=agent, level=level, msg=msg)

# Directory containing per-mission JSON files
_MISSIONS_DIR = Path(__file__).parent / "missions"

# Source-specific subdirectories
_CDAWEB_DIR = _MISSIONS_DIR / "cdaweb"
_PPI_DIR = _MISSIONS_DIR / "ppi"

# All source directories to scan for mission JSONs
_SOURCE_DIRS = [_CDAWEB_DIR, _PPI_DIR]

# Module-level cache: mission_id (lowercase) -> parsed dict
_mission_cache: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Override support
# ---------------------------------------------------------------------------


def _get_overrides_dir() -> Path:
    """Return the directory for mission override files.

    Uses ``config.get_data_dir() / "mission_overrides"``.  Evaluated lazily
    because ``get_data_dir()`` depends on env/config not available at import
    time.
    """
    from config import get_data_dir
    return get_data_dir() / "mission_overrides"


def _load_override(cache_key: str) -> dict | None:
    """Load an override file for the given mission cache key.

    Returns:
        Parsed dict, or ``None`` if the file is missing, unreadable, or
        contains invalid JSON.
    """
    path = _get_overrides_dir() / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            _emit_debug(level="warning", msg=f"Override file {path} is not a JSON object; ignoring")
            return None
        return data
    except json.JSONDecodeError as exc:
        _emit_debug(level="warning", msg=f"Malformed JSON in override file {path}: {exc}")
        return None
    except OSError as exc:
        _emit_debug(level="warning", msg=f"Cannot read override file {path}: {exc}")
        return None


# Backward-compatible alias — canonical implementation is in _utils.py
_deep_merge = deep_merge


def _save_override(cache_key: str, data: dict) -> None:
    """Write an override dict to disk.

    Creates the overrides directory if it doesn't exist.
    """
    overrides_dir = _get_overrides_dir()
    overrides_dir.mkdir(parents=True, exist_ok=True)
    path = overrides_dir / f"{cache_key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    _emit_debug(msg=f"Saved mission override: {path}")


def update_mission_override(cache_key: str, patch: dict) -> dict:
    """Read-modify-write a mission override file.

    Loads the existing override (if any), deep-merges *patch* into it,
    writes the result back, and invalidates the mission cache so the
    next ``load_mission()`` call picks up the change.

    Args:
        cache_key: Mission cache key (e.g. ``"psp"``, ``"ace"``).
        patch: Sparse dict to merge into the override.

    Returns:
        The full override dict after merging.
    """
    existing = _load_override(cache_key) or {}
    _deep_merge(existing, patch)
    _save_override(cache_key, existing)
    # Invalidate cache so next load picks up the change
    _mission_cache.pop(cache_key, None)
    return existing


def load_mission(mission_id: str) -> dict:
    """Load a single mission's JSON data, with caching.

    Scans all source directories for a JSON whose ``"id"`` field matches
    *mission_id* (case-insensitive).  Each source directory is independent
    — no cross-source merging.

    Args:
        mission_id: Mission identifier (e.g., "PSP", "ACE", "VOYAGER1_PPI").
                    Case-insensitive; the JSON's "id" field is canonical.

    Returns:
        Parsed mission dict from the JSON file.

    Raises:
        FileNotFoundError: If no JSON file with a matching "id" is found.
        json.JSONDecodeError: If a JSON file is malformed.
    """
    cache_key = mission_id.lower().replace("-", "_")

    if cache_key not in _mission_cache:
        # First try direct filename match (fast path)
        # For PPI missions with _PPI suffix, the stem is the part before _ppi
        stem = cache_key.removesuffix("_ppi") if cache_key.endswith("_ppi") else cache_key

        # Scan all source dirs for a JSON whose "id" matches mission_id
        found_path = None
        for source_dir in _SOURCE_DIRS:
            path = source_dir / f"{stem}.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("id", "").lower() == cache_key:
                    found_path = path
                    break

        if found_path is None:
            # Fallback: scan all JSONs for matching "id" field
            for source_dir in _SOURCE_DIRS:
                if not source_dir.exists():
                    continue
                for path in source_dir.glob("*.json"):
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("id", "").lower() == cache_key:
                        found_path = path
                        break
                if found_path is not None:
                    break

        if found_path is None:
            checked = [str(d) for d in _SOURCE_DIRS]
            raise FileNotFoundError(
                f"No mission file found with id='{mission_id}' "
                f"(checked {', '.join(checked)})"
            )

        # Apply user overrides before caching
        override = _load_override(cache_key)
        if override is not None:
            _deep_merge(data, override)
        _mission_cache[cache_key] = data

    return _mission_cache[cache_key]


def _load_missions_from_source_dirs() -> dict[str, dict]:
    """Shared implementation: scan source dirs and return missions dict.

    Does NOT trigger bootstrap — only reads existing JSON files on disk.
    Returns empty dict if no JSON files exist.
    """
    result = {}
    for source_dir in _SOURCE_DIRS:
        if not source_dir.exists():
            continue
        for filepath in sorted(source_dir.glob("*.json")):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            mission_id = data.get("id", filepath.stem.upper())
            cache_key = mission_id.lower().replace("-", "_")

            if cache_key not in _mission_cache:
                # Apply user overrides before caching
                override = _load_override(cache_key)
                if override is not None:
                    _deep_merge(data, override)
                _mission_cache[cache_key] = data

            result[_mission_cache[cache_key]["id"]] = _mission_cache[cache_key]
    return result


def load_all_missions() -> dict[str, dict]:
    """Load all mission JSON files, keyed by canonical mission ID.

    Iterates each source directory independently — each JSON file becomes
    its own entry.  CDAWeb and PPI missions with the same stem are NOT
    merged; they appear as separate entries (e.g. ``"VOYAGER1"`` and
    ``"VOYAGER1_PPI"``).

    On first run (no JSON files exist), triggers full auto-download
    from CDAWeb (catalog + parameter metadata via Master CDF).

    Returns:
        Dict mapping mission ID (from JSON "id" field) to mission data.
        Example: {"PSP": {...}, "VOYAGER1": {...}, "VOYAGER1_PPI": {...}}
    """
    from .bootstrap import ensure_missions_populated
    ensure_missions_populated()

    return _load_missions_from_source_dirs()


def load_all_missions_from_disk() -> dict[str, dict]:
    """Load mission JSON files that already exist on disk.

    Identical to ``load_all_missions()`` but **skips**
    ``ensure_missions_populated()``.  Returns an empty dict if no
    JSON files are present (fresh install).
    """
    return _load_missions_from_source_dirs()


def get_mission_ids() -> list[str]:
    """Get all known mission IDs from available JSON files.

    Returns:
        Sorted list of canonical mission IDs (e.g., ["ACE", "DSCOVR", ...]).
    """
    missions = load_all_missions()
    return sorted(missions.keys())


def _append_spice_routing_entry(table: list[dict]) -> None:
    """Append the SPICE ephemeris entry to a routing table."""
    table.append({
        "id": "SPICE",
        "name": "SPICE Ephemeris",
        "capabilities": ["coordinate transforms", "distance", "position", "trajectory", "velocity"],
    })


def _classify_keyword(kw: str) -> str | None:
    """Map an instrument keyword to a higher-level capability, or None."""
    if kw in ("magnetic", "field", "mag", "b-field", "bfield", "imf",
               "mfi", "fgm", "impact", "magnetometer"):
        return "magnetic field"
    if kw in ("plasma", "solar wind", "proton", "density", "velocity",
              "temperature", "ion", "electron", "sweap", "swa",
              "swe", "faraday", "plastic", "fpi"):
        return "plasma"
    if kw in ("particle", "energetic", "cosmic ray"):
        return "energetic particles"
    if kw in ("electric", "e-field"):
        return "electric field"
    if kw in ("radio", "wave", "plasma wave"):
        return "radio/plasma waves"
    if kw in ("index", "indices", "sym-h", "geomagnetic", "dst", "kp", "ae"):
        return "geomagnetic indices"
    if kw in ("ephemeris", "orbit", "attitude", "position"):
        return "ephemeris"
    if kw in ("composition", "charge state"):
        return "composition"
    if kw in ("coronagraph", "heliograph"):
        return "coronagraph"
    if kw in ("imaging", "remote sensing"):
        return "imaging"
    return None


def _build_routing_entries(missions: dict) -> list[dict]:
    """Build routing table entries from a missions dict.

    Shared implementation for :func:`get_routing_table` and
    :func:`get_routing_table_from_disk`.
    """
    table = []
    for mission_id, mission in missions.items():
        capabilities: set[str] = set()
        for inst in mission.get("instruments", {}).values():
            for kw in inst.get("keywords", []):
                cap = _classify_keyword(kw)
                if cap is not None:
                    capabilities.add(cap)
        table.append({
            "id": mission_id,
            "name": mission["name"],
            "capabilities": sorted(capabilities),
        })
    _append_spice_routing_entry(table)
    return table


def get_routing_table() -> list[dict]:
    """Build a slim routing table for the main agent's system prompt.

    Each entry has the mission ID, name, and a list of capability keywords
    derived from instrument keywords (deduplicated).

    Returns:
        List of dicts: [{"id": "PSP", "name": "Parker Solar Probe",
                         "capabilities": ["magnetic field", "plasma"]}, ...]
    """
    return _build_routing_entries(load_all_missions())


def get_routing_table_from_disk() -> list[dict]:
    """Build a routing table from mission JSON files already on disk.

    Same as ``get_routing_table()`` but uses ``load_all_missions_from_disk()``
    instead of ``load_all_missions()``, so it never triggers bootstrap.
    Returns an empty list if no JSON files exist.
    """
    return _build_routing_entries(load_all_missions_from_disk())


def get_mission_datasets(mission_id: str) -> list[str]:
    """Get all dataset IDs for a mission.

    Args:
        mission_id: Mission identifier (e.g., "PSP").

    Returns:
        List of dataset ID strings.
    """
    mission = load_mission(mission_id)
    dataset_ids = []
    for inst in mission.get("instruments", {}).values():
        for ds_id in inst.get("datasets", {}):
            dataset_ids.append(ds_id)
    return dataset_ids


def clear_cache():
    """Clear the mission cache. Useful for testing."""
    _mission_cache.clear()
