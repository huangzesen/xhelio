"""
CDAWeb parameter metadata client.

Fetches parameter info dynamically and filters to 1D plottable parameters
(scalars and small vectors with size <= 3).

Uses a three-layer resolution strategy:
1. In-memory cache (fastest)
2. Local file cache in knowledge/missions/*/metadata/ (instant, no network)
3. Master CDF skeleton file (network fallback)

Supports local file cache: if a dataset's info response is saved in
knowledge/missions/{mission}/metadata/{dataset_id}.json, it is loaded instantly
without a network request.
"""

import fnmatch
import json
import re
import requests
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

from ._utils import deep_merge
from agent.event_bus import get_event_bus, METADATA_FETCH, DEBUG

# Cache for metadata responses to avoid repeated API calls
_info_cache: dict[str, dict] = {}

# Cache for CDAWeb Notes HTML pages (keyed by base URL, e.g., NotesA.html)
_notes_cache: dict[str, str] = {}

# Directory containing per-mission folders with metadata cache files
_MISSIONS_DIR = Path(__file__).parent / "missions"
_SOURCE_DIRS = [_MISSIONS_DIR / "cdaweb", _MISSIONS_DIR / "ppi"]


def _find_local_cache(dataset_id: str) -> Optional[Path]:
    """Scan mission subfolders for a locally cached metadata file.

    Checks knowledge/missions/{source}/*/metadata/{dataset_id}.json across
    all source directories (cdaweb/, ppi/).

    Args:
        dataset_id: Dataset ID (CDAWeb or PDS URN).

    Returns:
        Path to local cache file, or None if not found.
    """
    # For URN IDs, use a filesystem-safe filename (colons not allowed on Windows)
    cache_filename = _dataset_id_to_cache_filename(dataset_id)

    for source_dir in _SOURCE_DIRS:
        if not source_dir.exists():
            continue
        for mission_dir in source_dir.iterdir():
            if not mission_dir.is_dir():
                continue
            cache_file = mission_dir / "metadata" / cache_filename
            if cache_file.exists():
                return cache_file
    return None


def _dataset_id_to_cache_filename(dataset_id: str) -> str:
    """Convert a dataset ID to a safe cache filename.

    URN IDs and PDS3 IDs contain colons and slashes which are invalid
    on some filesystems, so we replace them with underscores for the
    cache filename.

    Args:
        dataset_id: Dataset ID (CDAWeb, PDS URN, or pds3: prefixed).

    Returns:
        Safe filename like "urn_nasa_pds_cassini-mag-cal_data-1sec-krtp.json"
        or "pds3_JNO-J-3-FGM-CAL-V1.0_DATA.json".
    """
    safe_id = dataset_id.replace(":", "_").replace("/", "_")
    return f"{safe_id}.json"


def _load_dataset_override(
    dataset_id: str, mission_stem: str | None = None,
) -> dict | None:
    """Load a dataset-level override file.

    When *mission_stem* is provided, goes directly to
    ``{overrides_dir}/{mission_stem}/{safe_filename}.json`` (O(1) lookup).
    Otherwise falls back to scanning all stem subdirectories.

    Uses the same colon→underscore mapping as metadata cache filenames for
    filesystem safety.

    Returns:
        Parsed dict, or ``None`` if no override file exists.
    """
    from .mission_loader import _get_overrides_dir

    overrides_dir = _get_overrides_dir()
    if not overrides_dir.exists():
        return None

    safe_filename = _dataset_id_to_cache_filename(dataset_id)

    # Fast path: direct lookup when mission_stem is known
    if mission_stem is not None:
        path = overrides_dir / mission_stem / safe_filename
        if path.exists():
            return _read_override_json(path)
        return None

    # Slow path: scan stem subdirectories for a matching dataset override
    for stem_dir in overrides_dir.iterdir():
        if not stem_dir.is_dir():
            continue
        path = stem_dir / safe_filename
        if path.exists():
            return _read_override_json(path)
    return None


def _read_override_json(path: Path) -> dict | None:
    """Read and validate a single override JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            get_event_bus().emit(METADATA_FETCH, agent="MetadataClient", level="warning", msg=f"Dataset override {path} is not a JSON object; ignoring")
            return None
        return data
    except json.JSONDecodeError as exc:
        get_event_bus().emit(METADATA_FETCH, agent="MetadataClient", level="warning", msg=f"Malformed JSON in dataset override {path}: {exc}")
        return None
    except OSError as exc:
        get_event_bus().emit(METADATA_FETCH, agent="MetadataClient", level="warning", msg=f"Cannot read dataset override {path}: {exc}")
        return None


def get_dataset_quality_report(dataset_id: str) -> dict | None:
    """Summarise metadata discrepancies for a dataset from its override file.

    Returns a structured report that agents can relay to the user, or
    ``None`` if the dataset has never been validated (no override file).

    The report groups annotations into:

    * **metadata_only** — parameters listed in metadata but absent from actual
      data files (the user cannot fetch these).
    * **data_only** — parameters found in data files but missing from metadata
      (may be fetchable but undocumented).

    When ``_validations`` records are present (new format), discrepancies are
    unioned across all validation entries.  Falls back to top-level
    ``parameters_annotations`` for backward compatibility with old overrides.

    Args:
        dataset_id: CDAWeb or PDS URN dataset ID.

    Returns:
        Dict with ``validated``, ``metadata_only``, ``data_only``,
        ``validation_count``, and a human-readable ``summary`` string,
        or ``None``.
    """
    override = _load_dataset_override(dataset_id)
    if override is None:
        return None

    # Union annotations across all _validations records (new format)
    validations = override.get("_validations", [])
    if validations:
        annotations: dict = {}
        for v in validations:
            for param, ann in v.get("discrepancies", {}).items():
                if param not in annotations and isinstance(ann, dict):
                    annotations[param] = ann
    else:
        # Backward compat: use top-level parameters_annotations
        annotations = override.get("parameters_annotations", {})

    if not annotations and not override.get("_validated"):
        return None

    metadata_only: list[str] = []
    data_only: list[str] = []

    for param, ann in annotations.items():
        if not isinstance(ann, dict):
            continue
        category = ann.get("_category", "")
        note = ann.get("_note", "")
        # Use structured _category when available, substring fallback for old overrides
        if category == "phantom" or (
            not category and ("not found in data" in note or "not found in archive" in note)
        ):
            metadata_only.append(param)
        elif category == "undocumented" or (
            not category and ("found in data" in note or "found in archive" in note)
        ):
            data_only.append(param)

    # Build human-readable summary
    parts: list[str] = []
    if metadata_only:
        parts.append(
            f"{len(metadata_only)} parameter(s) listed in metadata but "
            f"absent from actual data files: {', '.join(sorted(metadata_only))}"
        )
    if data_only:
        parts.append(
            f"{len(data_only)} parameter(s) found in data files but "
            f"missing from metadata: {', '.join(sorted(data_only))}"
        )

    return {
        "validated": bool(override.get("_validated")),
        "validation_count": len(validations),
        "metadata_only": sorted(metadata_only),
        "data_only": sorted(data_only),
        "summary": "; ".join(parts) if parts else "No discrepancies detected.",
    }


def update_dataset_override(dataset_id: str, patch: dict,
                            mission_stem: str | None = None) -> dict:
    """Read-modify-write a dataset override file.

    Args:
        dataset_id: Dataset ID (CDAWeb or PDS URN, e.g. ``"AC_H2_MFI"``
            or ``"urn:nasa:pds:cassini-mag-cal:data-1sec-krtp"``).
        patch: Sparse dict to merge into the override.
        mission_stem: Mission stem directory name (e.g. ``"ace"``).
            If ``None``, auto-detected from local metadata cache.

    Returns:
        The full override dict after merging.

    Raises:
        ValueError: If *mission_stem* is not provided and cannot be
            auto-detected.
    """
    from .mission_loader import _get_overrides_dir

    if mission_stem is None:
        local = _find_local_cache(dataset_id)
        if local is not None:
            # .../missions/{stem}/metadata/{file}.json → stem = parent.parent.name
            mission_stem = local.parent.parent.name
        else:
            raise ValueError(
                f"Cannot auto-detect mission for dataset '{dataset_id}'. "
                f"Pass mission_stem explicitly."
            )

    overrides_dir = _get_overrides_dir()
    ds_dir = overrides_dir / mission_stem
    ds_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = _dataset_id_to_cache_filename(dataset_id)
    path = ds_dir / safe_filename

    # Load existing
    existing: dict = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    deep_merge(existing, patch)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
        f.write("\n")
    get_event_bus().emit(DEBUG, agent="MetadataClient", msg=f"Saved dataset override: {path}")

    # Invalidate in-memory cache for this dataset
    _info_cache.pop(dataset_id, None)

    return existing


def get_dataset_info(dataset_id: str, use_cache: bool = True) -> dict:
    """Fetch parameter metadata for a dataset.

    Checks three sources in order:
    1. In-memory cache (fastest)
    2. Local file cache in knowledge/missions/*/metadata/ (instant, no network)
    3. Master CDF skeleton file (network fallback)

    After loading from source 2 or 3, any dataset-level override from
    ``{overrides_dir}/{stem}/{dataset_id}.json`` is deep-merged on top.

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "PSP_FLD_L2_MAG_RTN_1MIN")
        use_cache: Whether to use cached results (in-memory and local file)

    Returns:
        Info dict with startDate, stopDate, parameters, etc.

    Raises:
        Exception: If all sources fail.
    """
    # 1. In-memory cache
    if use_cache and dataset_id in _info_cache:
        return _info_cache[dataset_id]

    # 2. Local file cache
    if use_cache:
        local_path = _find_local_cache(dataset_id)
        if local_path is not None:
            info = json.loads(local_path.read_text(encoding="utf-8"))
            # Apply dataset-level override (use stem from path for O(1) lookup)
            stem = local_path.parent.parent.name
            ds_override = _load_dataset_override(dataset_id, mission_stem=stem)
            if ds_override is not None:
                deep_merge(info, ds_override)
            _info_cache[dataset_id] = info
            return info

    # 3. Master CDF (network fallback — CDAWeb datasets only)
    if not dataset_id.startswith("urn:nasa:pds:"):
        from .master_cdf import fetch_dataset_metadata_from_master
        info = fetch_dataset_metadata_from_master(dataset_id)
        if info is not None:
            get_event_bus().emit(METADATA_FETCH, agent="MetadataClient", msg=f"Got metadata from Master CDF for {dataset_id}")
            # Apply dataset-level override
            ds_override = _load_dataset_override(dataset_id)
            if ds_override is not None:
                deep_merge(info, ds_override)
            if use_cache:
                _info_cache[dataset_id] = info
                _save_to_local_cache(dataset_id, info)
            return info

    # 4. PDS datasets (pds3: or urn:nasa:pds:) — return stub from mission JSON
    # Full parameter metadata gets populated on first data fetch from labels.
    if dataset_id.startswith("urn:nasa:pds:") or dataset_id.startswith("pds3:"):
        info = _build_pds_metadata_stub(dataset_id)
        if info is not None:
            get_event_bus().emit(METADATA_FETCH, agent="MetadataClient", msg=f"Built PDS metadata stub for {dataset_id}")
            ds_override = _load_dataset_override(dataset_id)
            if ds_override is not None:
                deep_merge(info, ds_override)
            if use_cache:
                _info_cache[dataset_id] = info
            return info

    raise ValueError(
        f"No metadata available for dataset '{dataset_id}'. "
        f"All metadata sources failed."
    )


def _build_pds_metadata_stub(dataset_id: str) -> dict | None:
    """Build a minimal metadata stub for a PDS dataset from mission JSONs.

    Searches PPI mission JSONs for the dataset and returns collection-level
    metadata (title, dates).  Full parameter metadata gets populated on
    first data fetch from the file labels.

    Args:
        dataset_id: PDS dataset ID (``urn:nasa:pds:...`` or ``pds3:...``).

    Returns:
        Minimal info dict, or ``None`` if dataset not found in any mission.
    """
    from .mission_prefixes import match_dataset_to_mission

    mission_stem, _ = match_dataset_to_mission(dataset_id)
    if not mission_stem:
        return None

    # Search PPI mission JSON for this dataset
    ppi_dir = _MISSIONS_DIR / "ppi"
    mission_json = ppi_dir / f"{mission_stem}.json"
    if not mission_json.exists():
        return None

    try:
        with open(mission_json, "r", encoding="utf-8") as f:
            mission_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    for inst in mission_data.get("instruments", {}).values():
        ds_entry = inst.get("datasets", {}).get(dataset_id)
        if ds_entry is not None:
            return {
                "description": ds_entry.get("description", ""),
                "startDate": ds_entry.get("start_date", ""),
                "stopDate": ds_entry.get("stop_date", ""),
                "parameters": [],  # populated on first data fetch
            }

    return None


def _save_to_local_cache(dataset_id: str, info: dict) -> None:
    """Persist metadata to the local file cache.

    Finds the appropriate mission directory by scanning existing dirs.
    For URN IDs, saves under ppi/{stem}/metadata/.
    For CDAWeb IDs, saves under cdaweb/{stem}/metadata/.
    If no matching mission dir is found, skips silently.
    """
    cache_filename = _dataset_id_to_cache_filename(dataset_id)

    # For PDS IDs (URN or pds3:), use mission prefix matching to find the PPI dir
    if dataset_id.startswith("urn:nasa:pds:") or dataset_id.startswith("pds3:"):
        from .mission_prefixes import match_dataset_to_mission
        mission_stem, _ = match_dataset_to_mission(dataset_id)
        if mission_stem:
            metadata_dir = _MISSIONS_DIR / "ppi" / mission_stem / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            cache_file = metadata_dir / cache_filename
            try:
                cache_file.write_text(
                    json.dumps(info, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                get_event_bus().emit(DEBUG, agent="MetadataClient", msg=f"Saved metadata to cache: {cache_file}")
            except OSError:
                pass
        return

    # For CDAWeb IDs, scan cdaweb/ subdirectory
    cdaweb_dir = _MISSIONS_DIR / "cdaweb"
    if not cdaweb_dir.exists():
        return
    for mission_dir in cdaweb_dir.iterdir():
        if not mission_dir.is_dir():
            continue
        metadata_dir = mission_dir / "metadata"
        if metadata_dir.exists():
            # Check if this mission has any datasets with matching prefix
            # by looking at existing cache files
            existing = list(metadata_dir.glob("*.json"))
            if not existing:
                continue
            # Check prefix match (e.g., AC_ for ACE datasets)
            sample_name = existing[0].stem
            if sample_name.startswith("_"):
                if len(existing) > 1:
                    sample_name = existing[1].stem
                else:
                    continue
            # Simple heuristic: same mission prefix
            ds_prefix = dataset_id.split("_")[0] if "_" in dataset_id else ""
            sample_prefix = sample_name.split("_")[0] if "_" in sample_name else ""
            if ds_prefix and ds_prefix == sample_prefix:
                cache_file = metadata_dir / cache_filename
                try:
                    cache_file.write_text(
                        json.dumps(info, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    get_event_bus().emit(DEBUG, agent="MetadataClient", msg=f"Saved metadata to cache: {cache_file}")
                except OSError:
                    pass
                return


def list_parameters(dataset_id: str) -> list[dict]:
    """List plottable 1D parameters for a dataset.

    Fetches metadata and filters to parameters that are:
    - Not the Time parameter
    - Numeric type (double or integer)
    - 1D with size <= 3 (scalars and small vectors)

    Args:
        dataset_id: CDAWeb dataset ID

    Returns:
        List of parameter dicts with name, description, units, size, dataset_id.
        Returns empty list if metadata fetch fails.
    """
    try:
        info = get_dataset_info(dataset_id)
    except (requests.RequestException, Exception) as e:
        get_event_bus().emit(METADATA_FETCH, agent="MetadataClient", level="warning", msg=f"Could not fetch info for {dataset_id}: {e}")
        return []

    # Cross-reference annotations from override (phantom/undocumented markers)
    annotations = info.get("parameters_annotations", {})

    params = []
    for p in info.get("parameters", []):
        name = p.get("name", "")

        # Skip Time parameter
        if name.lower() == "time":
            continue

        # Normalize size to a list
        size = p.get("size")
        if size is None:
            size = [1]
        elif isinstance(size, int):
            size = [size]

        # Filter: 1D with size <= 3
        if len(size) == 1 and size[0] <= 3:
            ptype = p.get("type", "")
            if ptype in ("double", "integer"):
                entry = {
                    "name": name,
                    "description": p.get("description", ""),
                    "units": p.get("units", ""),
                    "size": size,
                    "dataset_id": dataset_id,
                }
                # Attach annotation from parameters_annotations override
                ann = annotations.get(name)
                if isinstance(ann, dict):
                    note = ann.get("_note", "")
                    if note:
                        entry["note"] = note
                    category = ann.get("_category", "")
                    if category == "phantom" or (
                        not category
                        and ("not found in data" in note or "not found in archive" in note)
                    ):
                        entry["status"] = "NOT_IN_DATA_FILES"
                elif p.get("_note"):
                    entry["note"] = p["_note"]
                params.append(entry)

    return params


def get_dataset_time_range(dataset_id: str) -> Optional[dict]:
    """Get the available time range for a dataset.

    Args:
        dataset_id: CDAWeb dataset ID

    Returns:
        Dict with 'start' and 'stop' ISO date strings, or None if unavailable.
    """
    try:
        info = get_dataset_info(dataset_id)
        return {
            "start": info.get("startDate"),
            "stop": info.get("stopDate"),
        }
    except (requests.RequestException, ValueError):
        return None


def _source_dirs_for_mission(mission_id: str) -> list[Path]:
    """Return the source directory(ies) to use for a given mission ID.

    PPI missions (ID ending with ``_PPI``) only look in PPI source dirs.
    Other missions look in all source dirs (CDAWeb first, PPI as fallback
    for PPI-only missions like Galileo that don't have a ``_PPI`` suffix).
    Uses the module-level ``_SOURCE_DIRS`` list so tests can patch it.
    """
    mid = mission_id.upper()
    if mid.endswith("_PPI"):
        return [d for d in _SOURCE_DIRS if d.name == "ppi"]
    # For non-PPI IDs, return all source dirs (CDAWeb + PPI fallback
    # for PPI-only missions like GALILEO)
    return list(_SOURCE_DIRS)


def _stem_for_mission(mission_id: str) -> str:
    """Return the filesystem stem for a mission ID.

    For PPI missions (``VOYAGER1_PPI``), the stem is the part before
    ``_PPI`` (lowercased).  For others, lowercased as-is with hyphens
    replaced by underscores.
    """
    lower = mission_id.lower().replace("-", "_")
    if lower.endswith("_ppi"):
        return lower[:-4]  # strip _ppi suffix
    return lower


def list_cached_datasets(mission_id: str) -> Optional[dict]:
    """Load the _index.json summary for a mission's cached metadata.

    Routes to the correct source directory based on the mission ID:
    PPI missions (``_PPI`` suffix) look only in ``ppi/``, others in ``cdaweb/``.

    Args:
        mission_id: Mission identifier (e.g., "PSP", "VOYAGER1_PPI").
                    Case-insensitive.

    Returns:
        Parsed _index.json dict with mission_id, dataset_count, datasets list,
        or None if no index file exists.
    """
    stem = _stem_for_mission(mission_id)
    source_dirs = _source_dirs_for_mission(mission_id)
    for source_dir in source_dirs:
        index_path = source_dir / stem / "metadata" / "_index.json"
        if index_path.exists():
            return json.loads(index_path.read_text(encoding="utf-8"))
    return None


def _load_calibration_exclusions(mission_id: str) -> tuple[list[str], list[str]]:
    """Load calibration exclusion patterns and IDs for a mission.

    Routes to the correct source directory based on the mission ID.

    Args:
        mission_id: Mission identifier (case-insensitive).

    Returns:
        Tuple of (patterns, ids). Returns ([], []) if no exclusion file exists.
    """
    stem = _stem_for_mission(mission_id)
    source_dirs = _source_dirs_for_mission(mission_id)
    for source_dir in source_dirs:
        exclude_path = source_dir / stem / "metadata" / "_calibration_exclude.json"
        if exclude_path.exists():
            data = json.loads(exclude_path.read_text(encoding="utf-8"))
            return data.get("patterns", []), data.get("ids", [])
    return [], []


def _enrich_descriptions_from_mission(
    mission_id: str, datasets: list[dict],
) -> None:
    """Fill empty descriptions in dataset entries from mission JSON.

    The ``_index.json`` files for CDAWeb missions have empty ``description``
    fields.  The full descriptions live in the mission JSON under
    ``instruments.*.datasets.*.description``.  This helper builds a lookup
    and patches entries in-place.

    Args:
        mission_id: Mission identifier (e.g., ``"PSP"``, ``"ACE"``).
        datasets: List of dataset dicts (mutated in-place).
    """
    from .mission_loader import load_mission

    try:
        mission = load_mission(mission_id)
    except FileNotFoundError:
        return

    # Build {dataset_id: description} lookup from all instruments
    desc_lookup: dict[str, str] = {}
    for inst in mission.get("instruments", {}).values():
        for ds_id, ds_info in inst.get("datasets", {}).items():
            desc = ds_info.get("description", "")
            if desc:
                desc_lookup[ds_id] = desc

    if not desc_lookup:
        return

    for ds in datasets:
        if not ds.get("description") and ds.get("id") in desc_lookup:
            ds["description"] = desc_lookup[ds["id"]]


def browse_datasets(mission_id: str) -> Optional[list[dict]]:
    """Return non-calibration datasets from _index.json.

    Filters out datasets matching calibration exclusion patterns/IDs.
    Enriches entries with descriptions from mission JSON when the
    ``_index.json`` description is empty.
    Returns None if no _index.json exists for the mission.

    Args:
        mission_id: Mission identifier (e.g., 'PSP', 'ACE'). Case-insensitive.

    Returns:
        List of dataset summary dicts, or None if no index file exists.
    """
    index = list_cached_datasets(mission_id)
    if index is None:
        return None

    patterns, excluded_ids = _load_calibration_exclusions(mission_id)
    excluded_id_set = set(excluded_ids)

    result = []
    for ds in index.get("datasets", []):
        ds_id = ds.get("id", "")
        # Check exact ID exclusion
        if ds_id in excluded_id_set:
            continue
        # Check pattern exclusion
        if any(fnmatch.fnmatch(ds_id, pat) for pat in patterns):
            continue
        result.append(ds)

    # Enrich empty descriptions from mission JSON
    _enrich_descriptions_from_mission(mission_id, result)

    return result


def list_missions() -> list[dict]:
    """List all missions with descriptions and capabilities.

    Uses the routing table (derived from mission JSON files) as the single
    source of truth. Enriches each entry with description from the catalog
    and dataset count from the metadata index.

    Returns:
        Sorted list of dicts with mission_id, name, dataset_count,
        description, and capabilities.
    """
    from knowledge.catalog import MISSIONS
    from knowledge.mission_loader import get_routing_table

    results = []
    for entry in get_routing_table():
        mid = entry["id"]
        # Description from catalog profile
        cat_entry = MISSIONS.get(mid)
        desc = cat_entry.get("profile", {}).get("description", "") if cat_entry else ""
        # Dataset count from metadata index (if available)
        ds_count = 0
        index = list_cached_datasets(mid)
        if index is not None:
            ds_count = index.get("dataset_count", len(index.get("datasets", [])))
        results.append({
            "mission_id": mid,
            "name": entry["name"],
            "dataset_count": ds_count,
            "description": desc,
            "capabilities": entry["capabilities"],
        })
    return results


def validate_dataset_id(dataset_id: str) -> dict:
    """Check if a dataset ID exists in the local metadata cache.

    Uses _find_local_cache() to check all mission directories.
    No network call is made.

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "AC_H2_MFI").

    Returns:
        Dict with valid (bool), mission_id (str|None), message (str).
    """
    cache_path = _find_local_cache(dataset_id)
    if cache_path is not None:
        # Extract mission_id from the path: .../missions/{mission}/metadata/{file}.json
        mission_id = cache_path.parent.parent.name.upper()
        return {
            "valid": True,
            "mission_id": mission_id,
            "message": f"Dataset '{dataset_id}' found in {mission_id} cache.",
        }
    return {
        "valid": False,
        "mission_id": None,
        "message": (
            f"Dataset '{dataset_id}' not found in local metadata cache. "
            f"Use browse_datasets(mission_id) to see available datasets."
        ),
    }


def validate_parameter_id(dataset_id: str, parameter_id: str) -> dict:
    """Check if a parameter ID exists in a cached dataset's metadata.

    Reads the cached dataset JSON directly — no network call.

    Args:
        dataset_id: CDAWeb dataset ID.
        parameter_id: Parameter name to validate.

    Returns:
        Dict with valid (bool), available_parameters (list[str]), message (str).
    """
    cache_path = _find_local_cache(dataset_id)
    if cache_path is None:
        return {
            "valid": False,
            "available_parameters": [],
            "message": (
                f"Cannot validate parameter — dataset '{dataset_id}' "
                f"not found in local metadata cache."
            ),
        }

    try:
        info = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "valid": False,
            "available_parameters": [],
            "message": f"Cannot read cache file for dataset '{dataset_id}'.",
        }

    # Collect all non-Time parameter names
    available = [
        p["name"]
        for p in info.get("parameters", [])
        if p.get("name", "").lower() != "time"
    ]

    if parameter_id in available:
        return {
            "valid": True,
            "available_parameters": available,
            "message": f"Parameter '{parameter_id}' is valid for dataset '{dataset_id}'.",
        }

    return {
        "valid": False,
        "available_parameters": available,
        "message": (
            f"Parameter '{parameter_id}' not found in dataset '{dataset_id}'. "
            f"Available parameters: {', '.join(available)}. "
            f"Use list_parameters('{dataset_id}') for details."
        ),
    }


class _HTMLToText(HTMLParser):
    """Minimal HTML-to-text converter using stdlib HTMLParser.

    Strips tags while preserving block structure (newlines for <br>, <p>, <hr>,
    headings, list items). Skips <script> and <style> content.
    """

    _BLOCK_TAGS = {"p", "div", "br", "hr", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}
    _SKIP_TAGS = {"script", "style"}

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag in self._BLOCK_TAGS and not self._skip_depth:
            self._parts.append("\n")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str):
        if not self._skip_depth:
            self._parts.append(data)

    def get_text(self) -> str:
        """Return accumulated text, with collapsed whitespace."""
        raw = "".join(self._parts)
        # Collapse runs of blank lines to at most two newlines
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _extract_dataset_section(html: str, dataset_id: str) -> Optional[str]:
    """Extract the section for a specific dataset from a CDAWeb Notes HTML page.

    CDAWeb Notes pages use anchors like <a name="DATASET_ID"> or
    <strong>DATASET_ID</strong> to mark section starts, with <hr> tags
    or "Back to top" links as section boundaries.

    Args:
        html: Full HTML text of the Notes page.
        dataset_id: CDAWeb dataset ID to look for (e.g., "AC_H2_MFI").

    Returns:
        HTML string for the dataset section, or None if not found.
    """
    # Try anchor patterns: name="ID", id="ID"
    patterns = [
        rf'(?:name|id)\s*=\s*"{re.escape(dataset_id)}"',
        rf'<strong>\s*{re.escape(dataset_id)}\s*</strong>',
    ]

    start_pos = None
    for pat in patterns:
        match = re.search(pat, html, re.IGNORECASE)
        if match:
            start_pos = match.start()
            break

    if start_pos is None:
        return None

    # Find section end: next <hr> or "Back to top" after the anchor
    remaining = html[start_pos:]
    end_patterns = [
        r'<hr\b[^>]*>',
        r'Back to top',
    ]
    end_pos = len(remaining)
    for ep in end_patterns:
        m = re.search(ep, remaining[100:], re.IGNORECASE)  # skip first 100 chars to avoid self-match
        if m:
            end_pos = min(end_pos, m.start() + 100)

    return remaining[:end_pos]


def _fetch_notes_section(resource_url: str, dataset_id: str) -> Optional[str]:
    """Fetch a CDAWeb Notes page (cached) and extract the dataset section as text.

    Args:
        resource_url: URL like "https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H2_MFI"
        dataset_id: CDAWeb dataset ID.

    Returns:
        Plain text of the dataset documentation section, or None on failure.
    """
    # Strip fragment to get base URL (the page to fetch)
    base_url = resource_url.split("#")[0]

    # Check cache
    if base_url not in _notes_cache:
        try:
            from data_ops.http_utils import request_with_retry
            resp = request_with_retry(base_url)
            _notes_cache[base_url] = resp.text
        except Exception:
            return None

    html = _notes_cache[base_url]
    section_html = _extract_dataset_section(html, dataset_id)
    if section_html is None:
        return None

    parser = _HTMLToText()
    parser.feed(section_html)
    return parser.get_text()


def _fallback_resource_url(dataset_id: str) -> str:
    """Construct a plausible CDAWeb Notes URL from a dataset ID.

    CDAWeb Notes pages are organized by first letter: NotesA.html, NotesB.html, etc.

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "AC_H2_MFI").

    Returns:
        URL string like "https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H2_MFI"
    """
    first_letter = dataset_id[0].upper() if dataset_id else "A"
    return f"https://cdaweb.gsfc.nasa.gov/misc/Notes{first_letter}.html#{dataset_id}"


def get_dataset_docs(dataset_id: str, max_chars: int | None = None) -> dict:
    """Look up CDAWeb documentation for a dataset.

    Combines dataset metadata (contact, resourceURL) with the actual
    documentation text scraped from the CDAWeb Notes page.

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "AC_H2_MFI").
        max_chars: Maximum characters for the documentation text.
            Defaults to the ``context.dataset_docs`` truncation limit.

    Returns:
        Dict with dataset_id, contact, resource_url, and documentation fields.
        documentation may be None if the Notes page could not be fetched or
        the dataset section was not found.
    """
    if max_chars is None:
        from agent.truncation import get_limit
        max_chars = get_limit("context.dataset_docs")
    result = {"dataset_id": dataset_id, "contact": None, "resource_url": None, "documentation": None}

    # Try to get contact and resource URL from local cache or CDAWeb metadata
    try:
        info = get_dataset_info(dataset_id)
        result["contact"] = info.get("contact")
        result["resource_url"] = info.get("resourceURL")
    except Exception:
        pass

    # Enrich with CDAWeb REST API metadata (PI info, notes URL)
    if not result["contact"] or not result["resource_url"]:
        try:
            from .cdaweb_metadata import fetch_dataset_metadata
            cdaweb_meta = fetch_dataset_metadata()
            entry = cdaweb_meta.get(dataset_id)
            if entry:
                if not result["contact"] and entry.get("pi_name"):
                    result["contact"] = entry["pi_name"]
                    if entry.get("pi_affiliation"):
                        result["contact"] += f" @ {entry['pi_affiliation']}"
                if not result["resource_url"] and entry.get("notes_url"):
                    result["resource_url"] = entry["notes_url"]
        except Exception:
            pass

    # Determine the resource URL to fetch
    resource_url = result["resource_url"]
    if not resource_url:
        resource_url = _fallback_resource_url(dataset_id)
        result["resource_url"] = resource_url

    # Fetch and extract documentation
    doc_text = _fetch_notes_section(resource_url, dataset_id)
    if doc_text and max_chars > 0 and len(doc_text) > max_chars:
        doc_text = doc_text[:max_chars] + "\n[truncated]"
    result["documentation"] = doc_text

    return result


def clear_cache():
    """Clear the metadata info cache and Notes page cache."""
    global _info_cache, _notes_cache
    _info_cache = {}
    _notes_cache = {}
