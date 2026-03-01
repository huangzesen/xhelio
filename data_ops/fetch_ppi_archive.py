"""
PDS PPI direct file archive fetch backend.

Downloads data files directly from the PDS PPI file archive at
https://pds-ppi.igpp.ucla.edu/data/.

Data files are cached locally in ppi_data/ (mirrors cdaweb_data/ pattern).
Archive data is static/versioned so cached files never expire.

Supports PDS4 and PDS3 data file layouts:
- PDS4: Fixed-width ASCII tables (.TAB), delimited CSV (.csv) with XML labels
- PDS3: Fixed-width ASCII tables (.sts, .TAB, .tab) with .lbl labels

Column names and positions are parsed from companion label files.
Parameter metadata (units, fill values) is extracted from labels and cached
locally.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from agent.event_bus import get_event_bus, PPI_FETCH, DEBUG

PPI_ARCHIVE_BASE = "https://pds-ppi.igpp.ucla.edu/data"
CACHE_DIR = Path(__file__).resolve().parent.parent / "ppi_data"

# In-memory cache: URN → resolved collection URL
_collection_url_cache: dict[str, str] = {}

# PDS4 XML namespace
_PDS4_NS = "http://pds.nasa.gov/pds4/pds/v1"
_NS = {"pds": _PDS4_NS}

# Timeout for HTTP requests (seconds)
_TIMEOUT = 5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_ppi_archive_data(
    dataset_id: str,
    parameter_id: str,
    time_min: str,
    time_max: str,
) -> dict:
    """Fetch PPI data from the direct file archive.

    Args:
        dataset_id: PDS dataset ID — either a PDS4 URN
            (e.g., ``urn:nasa:pds:cassini-mag-cal:data-1sec-krtp``) or
            a PDS3 ID with ``pds3:`` prefix
            (e.g., ``pds3:JNO-J-3-FGM-CAL-V1.0:DATA``).
        parameter_id: Parameter name (e.g., ``BR``, ``BX PLANETOCENTRIC``).
        time_min: ISO start time (e.g., ``2005-01-01T00:00:00Z``).
        time_max: ISO end time (e.g., ``2005-01-02T00:00:00Z``).

    Returns:
        Dict with keys: data (DataFrame), units, description, fill_value.

    Raises:
        ValueError: If no data is available or parameter not found.
        requests.HTTPError: If a download fails.
    """
    # 1. Try to load parameter metadata from local cache
    units = ""
    description = ""
    fill_value = None
    info = _load_cached_metadata(dataset_id)
    if info is not None:
        param_meta = _find_param_meta_safe(info, parameter_id)
        if param_meta:
            units = param_meta.get("units", "")
            description = param_meta.get("description", "")
            fill_value = param_meta.get("fill", None)

    if fill_value is not None:
        try:
            fill_value = float(fill_value)
        except (ValueError, TypeError):
            fill_value = None

    # 2. Resolve collection URL
    collection_url = _resolve_collection_url(dataset_id)
    get_event_bus().emit(PPI_FETCH, agent="ppi", msg=f"[PPI] Collection URL: {collection_url}")

    # 3. Discover data files covering the time range
    file_pairs = _discover_data_files(collection_url, time_min, time_max)
    if not file_pairs:
        start = info.get("startDate", "unknown") if info else "unknown"
        stop = info.get("stopDate", "unknown") if info else "unknown"
        raise ValueError(
            f"No data files found for {dataset_id} "
            f"in range {time_min} to {time_max}. "
            f"Dataset covers {start} to {stop} — "
            f"try a time range within that window."
        )

    get_event_bus().emit(PPI_FETCH, agent="ppi", msg=f"[PPI] Found {len(file_pairs)} data file(s) for {dataset_id}")

    # 4. Download, parse, and concatenate
    frames = []
    first_label = None
    skipped = []
    for data_url, label_url in file_pairs:
        try:
            local_data = _download_file(data_url)
            local_label = _download_file(label_url)

            label = _parse_label(local_label)
            if first_label is None:
                first_label = label

                # Extract metadata from first label if not cached
                if info is None or not info.get("parameters"):
                    _populate_metadata_from_label(dataset_id, label)
                    # Reload to get the new metadata
                    info = _load_cached_metadata(dataset_id)
                    if info is not None:
                        param_meta = _find_param_meta_safe(info, parameter_id)
                        if param_meta:
                            units = units or param_meta.get("units", "")
                            description = description or param_meta.get("description", "")
                            if fill_value is None:
                                fv = param_meta.get("fill")
                                if fv is not None:
                                    try:
                                        fill_value = float(fv)
                                    except (ValueError, TypeError):
                                        pass

            df = _read_table(local_data, label, parameter_id)
            if df is not None and len(df) > 0:
                frames.append(df)
        except Exception as e:
            fname = data_url.rsplit("/", 1)[-1]
            get_event_bus().emit(PPI_FETCH, agent="ppi", level="warning", msg=f"[PPI] Skipping {fname}: {e}")
            skipped.append(fname)
            continue

    if not frames:
        msg = (
            f"No data rows for {dataset_id}/{parameter_id} "
            f"in range {time_min} to {time_max}"
        )
        if skipped:
            msg += f". {len(skipped)} file(s) skipped due to errors: {', '.join(skipped)}"
        raise ValueError(msg)

    df = pd.concat(frames)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    # 5. Trim to requested time range
    t_start = time_min.rstrip("Z")
    t_stop = time_max.rstrip("Z")
    df = df.loc[t_start:t_stop]

    if len(df) == 0:
        raise ValueError(
            f"No data rows for {dataset_id}/{parameter_id} "
            f"in range {time_min} to {time_max} after filtering"
        )

    # 6. Ensure float64 and replace fill values
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

    if fill_value is not None:
        for col in df.columns:
            mask = np.isclose(df[col].values, fill_value, rtol=1e-6,
                              equal_nan=False)
            df.loc[mask, col] = np.nan

    get_event_bus().emit(PPI_FETCH, agent="ppi", msg=f"[PPI] {dataset_id}/{parameter_id}: {len(df)} rows, {len(df.columns)} columns")

    return {
        "data": df,
        "units": units,
        "description": description,
        "fill_value": fill_value,
    }


# ---------------------------------------------------------------------------
# Metadata helpers — cached metadata loading and label-based population
# ---------------------------------------------------------------------------

def _load_cached_metadata(dataset_id: str) -> dict | None:
    """Try to load metadata from local cache.

    Returns info dict or None if no cache exists.
    """
    from knowledge.metadata_client import _find_local_cache

    cache_path = _find_local_cache(dataset_id)
    if cache_path is None:
        return None

    import json
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _find_param_meta_safe(info: dict, parameter_id: str) -> dict | None:
    """Find parameter metadata without raising on missing params."""
    for p in info.get("parameters", []):
        if p.get("name") == parameter_id:
            return p
    return None


def _parse_label(label_path: Path) -> dict:
    """Dispatch to PDS3 or PDS4 label parser based on file extension.

    Args:
        label_path: Local path to the label file (.lbl for PDS3,
                    .xml/.lblx for PDS4).

    Returns:
        Parsed label dict with fields, table_type, etc.
    """
    ext = label_path.suffix.lower()
    text = label_path.read_text(encoding="utf-8", errors="replace")

    if ext == ".lbl":
        from data_ops.pds3_label_parser import parse_pds3_label
        return parse_pds3_label(text)
    else:
        return _parse_xml_label(text)


def _build_metadata_from_label(label: dict) -> dict | None:
    """Build a metadata dict from a parsed label.

    Pure function — no I/O.  Returns a metadata dict in the standard
    cache format, or None if the label has no useful numeric parameters.
    """
    parameters = [{"name": "Time", "type": "isotime", "length": 24}]
    time_names = {"time", "epoch", "utc", "scet", "datetime", "date_time",
                  "timestamp", "sample utc"}

    for field in label.get("fields", []):
        fname = field.get("name", "")
        if fname.lower().strip() in time_names:
            continue

        ptype = field.get("type", "").upper()
        param_type = "double"
        if "INT" in ptype:
            param_type = "integer"
        elif "CHAR" in ptype or "TIME" in ptype or "DATE" in ptype:
            continue  # Skip non-numeric columns

        param = {
            "name": fname,
            "type": param_type,
            "units": field.get("unit", ""),
            "description": field.get("description", ""),
            "size": [1],
        }
        null_const = field.get("null_constant")
        if null_const:
            param["fill"] = null_const
        parameters.append(param)

    if len(parameters) <= 1:
        return None  # Only Time — nothing useful

    return {
        "parameters": parameters,
        "description": "",
        "startDate": "",
        "stopDate": "",
        "_meta": {"source": "label"},
    }


def _populate_metadata_from_label(dataset_id: str, label: dict) -> None:
    """Build and cache parameter metadata from a parsed label.

    Called on first data fetch when no metadata cache exists.  Extracts
    field names, units, and descriptions from the label and saves as a
    metadata cache file in the standard format.
    """
    import json
    from knowledge.metadata_client import _dataset_id_to_cache_filename
    from knowledge.mission_prefixes import match_dataset_to_mission

    info = _build_metadata_from_label(label)
    if info is None:
        return

    # Save to local cache
    mission_stem, _ = match_dataset_to_mission(dataset_id)
    if not mission_stem:
        return

    missions_dir = Path(__file__).resolve().parent.parent / "knowledge" / "missions"
    metadata_dir = missions_dir / "ppi" / mission_stem / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    cache_filename = _dataset_id_to_cache_filename(dataset_id)
    cache_file = metadata_dir / cache_filename
    try:
        cache_file.write_text(
            json.dumps(info, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        get_event_bus().emit(PPI_FETCH, agent="ppi", level="info", msg=f"[PPI] Saved label-derived metadata to {cache_file.name}")
    except OSError:
        pass


def _find_one_label(
    collection_url: str,
    _depth: int = 0,
    _max_depth: int = 4,
) -> Path | None:
    """Find and download exactly one label file from a collection directory.

    Uses breadth-first search: at each level, looks for a label file
    (.xml, .lblx, .lbl) that has a matching data file (.tab, .csv, .dat,
    .sts).  If none found, recurses into the first subdirectory.

    Returns:
        Local path to the downloaded label file, or None if no label found.
    """
    if _depth >= _max_depth:
        return None

    try:
        entries = _list_directory(collection_url)
    except Exception:
        return None

    data_exts = {".tab", ".csv", ".dat", ".sts"}
    label_exts = {".xml", ".lblx", ".lbl"}

    # Separate files and directories
    files = {}       # stem_lower -> {ext_lower: original_name}
    subdirs = []

    for e in sorted(entries, key=lambda x: x["name"]):
        name = e["name"]
        if e["is_dir"]:
            subdirs.append(name.rstrip("/"))
            continue

        stem_lower = Path(name).stem.lower()
        ext_lower = Path(name).suffix.lower()

        # Skip PDS inventory/collection files
        if stem_lower.startswith("collection"):
            continue

        files.setdefault(stem_lower, {})[ext_lower] = name

    # Look for a label file with a matching data file
    for stem_lower in sorted(files):
        exts = files[stem_lower]
        has_data = any(ext in exts for ext in data_exts)
        if not has_data:
            continue
        for label_ext in (".xml", ".lblx", ".lbl"):
            if label_ext in exts:
                label_url = f"{collection_url}{exts[label_ext]}"
                try:
                    return _download_file(label_url)
                except Exception:
                    continue

    # No match at this level — recurse into first subdirectory
    for d in subdirs:
        result = _find_one_label(
            f"{collection_url}{d}/",
            _depth=_depth + 1,
            _max_depth=_max_depth,
        )
        if result is not None:
            return result

    return None


def fetch_ppi_label_metadata(dataset_id: str, slot: str) -> dict | None:
    """Fetch parameter metadata for a PPI dataset from its label file.

    Downloads one label file from the dataset's archive directory,
    parses it, and returns the metadata dict.

    Args:
        dataset_id: PDS dataset ID (URN or pds3: prefixed).
        slot: PDS archive slot path (e.g., ``/data/JNO-J-3-FGM-CAL-V1.0/DATA``).

    Returns:
        Metadata dict with ``parameters`` list, or None if no label found
        or the label has no useful parameters.
    """
    collection_url = f"https://pds-ppi.igpp.ucla.edu{slot}/"
    label_path = _find_one_label(collection_url)
    if label_path is None:
        return None

    try:
        label = _parse_label(label_path)
    except Exception as e:
        get_event_bus().emit(DEBUG, agent="ppi", msg=f"[PPI] Failed to parse label for {dataset_id}: {e}")
        return None

    return _build_metadata_from_label(label)


# ---------------------------------------------------------------------------
# URN → Archive URL resolution
# ---------------------------------------------------------------------------

def _resolve_collection_url(dataset_id: str) -> str:
    """Convert a PDS dataset ID to an archive collection URL.

    PDS4: ``urn:nasa:pds:{bundle}:{collection}`` →
          ``{ARCHIVE_BASE}/{bundle_path}/{collection_dir}/``

    PDS3: ``pds3:{BUNDLE_ID}:{COLLECTION}`` →
          ``{ARCHIVE_BASE}/{bundle_id}/{COLLECTION}/`` (using slot if available)

    Bundle: underscores → hyphens for PDS4.
    Collection: discovered from directory listing (naming is inconsistent).
    """
    if dataset_id in _collection_url_cache:
        return _collection_url_cache[dataset_id]

    if dataset_id.startswith("pds3:"):
        url = _resolve_pds3_collection_url(dataset_id)
    else:
        url = _resolve_pds4_collection_url(dataset_id)

    _collection_url_cache[dataset_id] = url
    return url


def _resolve_pds4_collection_url(dataset_id: str) -> str:
    """Resolve a PDS4 URN to an archive collection URL."""
    parts = dataset_id.split(":")
    if len(parts) < 5:
        raise ValueError(f"Invalid PDS URN format: {dataset_id}")

    bundle_urn = parts[3]   # e.g., "cassini-mag-cal"
    collection_urn = parts[4]  # e.g., "data-1sec-krtp"

    # Bundle: underscores → hyphens
    bundle_path = bundle_urn.replace("_", "-")
    bundle_url = f"{PPI_ARCHIVE_BASE}/{bundle_path}/"

    # List bundle directory to find the actual collection name
    entries = _list_directory(bundle_url)
    dir_names = [e["name"].rstrip("/") for e in entries if e["is_dir"]]

    collection_dir = _match_collection(collection_urn, dir_names)
    if collection_dir is None:
        raise ValueError(
            f"Collection '{collection_urn}' not found in bundle "
            f"'{bundle_path}'. Available: {dir_names}"
        )

    return f"{bundle_url}{collection_dir}/"


def _resolve_pds3_collection_url(dataset_id: str) -> str:
    """Resolve a PDS3 dataset ID to an archive collection URL.

    PDS3 IDs look like ``pds3:JNO-J-3-FGM-CAL-V1.0:DATA``.
    The Metadex ``slot`` field gives the direct path, e.g.
    ``/data/JNO-J-3-FGM-CAL-V1.0/DATA``.

    Falls back to constructing the URL from the ID components.
    """
    raw_id = dataset_id[len("pds3:"):]  # Strip prefix

    # Try to get slot from mission JSON (stored at bootstrap time)
    slot = _get_pds3_slot(dataset_id)
    if slot:
        # slot is like "/data/JNO-J-3-FGM-CAL-V1.0/DATA"
        return f"https://pds-ppi.igpp.ucla.edu{slot}/"

    # Fallback: construct from ID
    # PDS3 ID format: BUNDLE_ID:COLLECTION (e.g., "JNO-J-3-FGM-CAL-V1.0:DATA")
    if ":" in raw_id:
        bundle_id, collection = raw_id.rsplit(":", 1)
    else:
        bundle_id = raw_id
        collection = "DATA"

    # PDS3 bundle IDs may contain "/" which is converted to "_" in URLs
    bundle_path = bundle_id.replace("/", "_")
    return f"{PPI_ARCHIVE_BASE}/{bundle_path}/{collection}/"


def _get_pds3_slot(dataset_id: str) -> str | None:
    """Look up the Metadex slot for a PDS3 dataset from mission JSONs."""
    import json
    from knowledge.mission_prefixes import match_dataset_to_mission

    mission_stem, _ = match_dataset_to_mission(dataset_id)
    if not mission_stem:
        return None

    missions_dir = Path(__file__).resolve().parent.parent / "knowledge" / "missions"
    mission_json = missions_dir / "ppi" / f"{mission_stem}.json"
    if not mission_json.exists():
        return None

    try:
        with open(mission_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    for inst in data.get("instruments", {}).values():
        ds_entry = inst.get("datasets", {}).get(dataset_id)
        if ds_entry is not None:
            return ds_entry.get("slot")

    return None


def _match_collection(urn_name: str, dir_names: list[str]) -> str | None:
    """Match a URN collection name to an actual directory name.

    Tries: exact → hyphen/underscore swap → normalized comparison.
    """
    # Exact match
    if urn_name in dir_names:
        return urn_name

    # Swap hyphens ↔ underscores
    swapped = urn_name.replace("-", "_")
    if swapped in dir_names:
        return swapped
    swapped = urn_name.replace("_", "-")
    if swapped in dir_names:
        return swapped

    # Normalize: strip hyphens/underscores and compare
    norm = urn_name.replace("-", "").replace("_", "").lower()
    for d in dir_names:
        if d.replace("-", "").replace("_", "").lower() == norm:
            return d

    return None


# ---------------------------------------------------------------------------
# Directory listing and file discovery
# ---------------------------------------------------------------------------

def _list_directory(url: str) -> list[dict]:
    """Fetch an Apache directory index and parse it into entries.

    Returns list of ``{name, is_dir, size}`` dicts.
    """
    from data_ops.http_utils import request_with_retry
    resp = request_with_retry(url)
    return _parse_html_listing(resp.text)


def _parse_html_listing(html: str) -> list[dict]:
    """Parse an Apache-style HTML directory listing.

    Handles both ``<a href="name">`` links and ``<pre>`` formatted listings.
    Returns list of ``{name, is_dir, size}`` dicts.
    """
    entries = []
    # Match hrefs — Apache uses <a href="name"> or <a href="name/">
    for m in re.finditer(r'<a\s+href="([^"?]+)"', html, re.IGNORECASE):
        name = m.group(1)
        # Skip parent dir and sorting links
        if name in ("../", "/", "?") or name.startswith("?") or name.startswith("/"):
            continue
        is_dir = name.endswith("/")
        entries.append({
            "name": name,
            "is_dir": is_dir,
            "size": 0,
        })

    return entries


def _discover_data_files(
    collection_url: str,
    time_min: str,
    time_max: str,
) -> list[tuple[str, str]]:
    """Find data+label file pairs covering a time range.

    Returns list of ``(data_url, label_url)`` tuples.
    """
    entries = _list_directory(collection_url)
    dir_names = [e["name"].rstrip("/") for e in entries if e["is_dir"]]
    file_names = [e["name"] for e in entries if not e["is_dir"]]

    # Detect organization pattern
    has_year_dirs = any(re.match(r"^\d{4}$", d) for d in dir_names)
    has_orbit_dirs = any(re.match(r"\d{7}_orbit_\d+", d) for d in dir_names)
    has_sol_dirs = any(re.match(r"^SOL\d+", d, re.IGNORECASE) for d in dir_names)
    has_freq_dirs = any(
        d.lower() in {"20hz", "2hz", "pt2hz", "1hz", "0.2hz"}
        for d in dir_names
    )

    t_min = pd.Timestamp(time_min.rstrip("Z"))
    t_max = pd.Timestamp(time_max.rstrip("Z"))

    if has_year_dirs:
        return _discover_year_organized(collection_url, dir_names, t_min, t_max)
    elif has_orbit_dirs:
        return _discover_orbit_organized(collection_url, dir_names, t_min, t_max)
    elif has_sol_dirs:
        return _discover_sol_organized(collection_url, dir_names, t_min, t_max)
    elif has_freq_dirs:
        return _discover_freq_organized(collection_url, dir_names, t_min, t_max)
    elif file_names:
        # Flat — scan files directly in collection dir
        return _discover_flat(collection_url, file_names)
    elif dir_names:
        # PDS3-style nested dirs (e.g., JUPITER/PC/PERI-01/) — recurse
        return _discover_recursive(collection_url, t_min, t_max, max_depth=4)
    else:
        return []


def _discover_year_organized(
    collection_url: str,
    dir_names: list[str],
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
) -> list[tuple[str, str]]:
    """Discover files in year-organized directories.

    Includes the year before t_min because a file at the end of that year
    may span into the target range (e.g., file ``04366_05032`` in the 2004
    directory covers Jan 2005).
    """
    pairs = []
    for d in sorted(dir_names):
        m = re.match(r"^(\d{4})$", d)
        if not m:
            continue
        year = int(m.group(1))
        # Include prior year — files there may span into our range
        if year < t_min.year - 1 or year > t_max.year:
            continue

        year_url = f"{collection_url}{d}/"
        entries = _list_directory(year_url)
        file_names = [e["name"] for e in entries if not e["is_dir"]]
        pairs.extend(_pair_data_and_labels(year_url, file_names))

    return pairs


def _discover_orbit_organized(
    collection_url: str,
    dir_names: list[str],
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
) -> list[tuple[str, str]]:
    """Discover files in orbit-organized directories.

    Orbit dirs typically look like ``2024017_orbit_58`` where the leading
    7 digits encode YYYYDDD (day-of-year).
    """
    pairs = []
    for d in sorted(dir_names):
        m = re.match(r"^(\d{4})(\d{3})_orbit_\d+", d)
        if not m:
            # Try to include non-matching dirs as a fallback
            continue
        year = int(m.group(1))
        doy = int(m.group(2))
        try:
            orbit_start = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=doy - 1)
        except Exception:
            continue

        # Include orbit if it could overlap with requested range.
        # Orbits can span weeks/months, so use generous window.
        orbit_end_est = orbit_start + pd.Timedelta(days=60)
        if orbit_end_est < t_min or orbit_start > t_max:
            continue

        orbit_url = f"{collection_url}{d}/"
        entries = _list_directory(orbit_url)
        file_names = [e["name"] for e in entries if not e["is_dir"]]
        pairs.extend(_pair_data_and_labels(orbit_url, file_names))

    return pairs


def _discover_sol_organized(
    collection_url: str,
    dir_names: list[str],
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
) -> list[tuple[str, str]]:
    """Discover files in sol-range organized directories.

    Handles directories like ``SOL0004_SOL0029_20181130_20181226`` where
    trailing ``YYYYMMDD_YYYYMMDD`` encode the Earth date range.
    """
    pairs = []
    for d in sorted(dir_names):
        if not re.match(r"^SOL\d+", d, re.IGNORECASE):
            continue

        dir_start, dir_end = _parse_sol_dir_dates(d)
        if dir_start is None:
            # Can't parse dates — include to be safe
            get_event_bus().emit(DEBUG, agent="ppi", msg=f"[PPI] Including sol dir {d} (could not parse dates)")
        else:
            # 1-day buffer on each side
            buf = pd.Timedelta(days=1)
            if dir_end + buf < t_min or dir_start - buf > t_max:
                continue

        dir_url = f"{collection_url}{d}/"
        entries = _list_directory(dir_url)
        file_names = [e["name"] for e in entries if not e["is_dir"]]
        dir_pairs = _pair_data_and_labels(dir_url, file_names)
        dir_pairs = _filter_pairs_by_filename_time(dir_pairs, t_min, t_max)
        pairs.extend(dir_pairs)

    return pairs


def _discover_freq_organized(
    collection_url: str,
    dir_names: list[str],
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
) -> list[tuple[str, str]]:
    """Discover files in frequency-organized directories.

    Handles layouts like ``20Hz/release02_SOL0120_SOL0209_20190329_20190629/``
    where the top level is a frequency band and the second level uses
    sol-range or release directories with embedded date ranges.
    """
    freq_names = {"20hz", "2hz", "pt2hz", "1hz", "0.2hz"}
    pairs = []

    for d in sorted(dir_names):
        if d.lower() not in freq_names:
            continue

        freq_url = f"{collection_url}{d}/"
        entries = _list_directory(freq_url)
        sub_dirs = [e["name"].rstrip("/") for e in entries if e["is_dir"]]
        sub_files = [e["name"] for e in entries if not e["is_dir"]]

        if sub_dirs:
            # Check for sol-range or release subdirs
            for sd in sorted(sub_dirs):
                sd_start, sd_end = _parse_sol_dir_dates(sd)
                if sd_start is not None:
                    buf = pd.Timedelta(days=1)
                    if sd_end + buf < t_min or sd_start - buf > t_max:
                        continue

                sd_url = f"{freq_url}{sd}/"
                sd_entries = _list_directory(sd_url)
                sd_files = [e["name"] for e in sd_entries if not e["is_dir"]]
                sd_pairs = _pair_data_and_labels(sd_url, sd_files)
                sd_pairs = _filter_pairs_by_filename_time(sd_pairs, t_min, t_max)
                pairs.extend(sd_pairs)
        else:
            # Files directly under frequency dir
            freq_pairs = _pair_data_and_labels(freq_url, sub_files)
            freq_pairs = _filter_pairs_by_filename_time(freq_pairs, t_min, t_max)
            pairs.extend(freq_pairs)

    return pairs


def _parse_sol_dir_dates(dir_name: str) -> tuple:
    """Extract Earth date range from a sol-range directory name.

    Handles patterns like:
    - ``SOL0004_SOL0029_20181130_20181226``
    - ``release02_SOL0120_SOL0209_20190329_20190629``

    Returns ``(start_timestamp, end_timestamp)`` or ``(None, None)``
    if dates cannot be extracted.
    """
    # Look for trailing YYYYMMDD_YYYYMMDD pattern
    m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})$", dir_name)
    if m:
        try:
            start = pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
            end = pd.Timestamp(f"{m.group(4)}-{m.group(5)}-{m.group(6)}")
            return start, end
        except Exception:
            pass

    # Try single trailing YYYYMMDD
    m = re.search(r"(\d{4})(\d{2})(\d{2})$", dir_name)
    if m:
        try:
            ts = pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
            return ts, ts
        except Exception:
            pass

    return None, None


# Regex for extracting timestamps from filenames
# Matches patterns like: _20181130T190519_20181130T192727_ or _20190101_20190201_
_FILENAME_TIME_RE = re.compile(
    r"(\d{4})(\d{2})(\d{2})(?:T(\d{2})(\d{2})(\d{2}))?"
    r"_"
    r"(\d{4})(\d{2})(\d{2})(?:T(\d{2})(\d{2})(\d{2}))?"
)

# PDS3 filename date pattern: YYYYDDD (7 digits, year + day of year)
# e.g., fgm_jno_l3_2016214pc_r1s_v01.sts → 2016214
_PDS3_FILENAME_DOY_RE = re.compile(r"(\d{4})(\d{3})[a-z]")


def _filter_pairs_by_filename_time(
    pairs: list[tuple[str, str]],
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
) -> list[tuple[str, str]]:
    """Filter file pairs by timestamps extracted from filenames.

    Looks for YYYYMMDD[THHMMSS]_YYYYMMDD[THHMMSS] or YYYYDDD patterns
    in data filenames.  Falls back to returning all pairs if no timestamp
    pattern is found in any filename.
    """
    if not pairs:
        return pairs

    filtered = []
    any_parsed = False
    buf = pd.Timedelta(days=1)

    for data_url, label_url in pairs:
        fname = data_url.rsplit("/", 1)[-1]

        # Try YYYYMMDD_YYYYMMDD pattern first
        m = _FILENAME_TIME_RE.search(fname)
        if m is not None:
            any_parsed = True
            try:
                g = m.groups()
                fstart = pd.Timestamp(
                    f"{g[0]}-{g[1]}-{g[2]}"
                    + (f"T{g[3]}:{g[4]}:{g[5]}" if g[3] else "")
                )
                fend = pd.Timestamp(
                    f"{g[6]}-{g[7]}-{g[8]}"
                    + (f"T{g[9]}:{g[10]}:{g[11]}" if g[9] else "")
                )
                if fend + buf >= t_min and fstart - buf <= t_max:
                    filtered.append((data_url, label_url))
            except Exception:
                filtered.append((data_url, label_url))
            continue

        # Try PDS3 YYYYDDD pattern (single date per file)
        m = _PDS3_FILENAME_DOY_RE.search(fname)
        if m is not None:
            any_parsed = True
            try:
                year = int(m.group(1))
                doy = int(m.group(2))
                fdate = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=doy - 1)
                # Single-day file — include if date overlaps range (with buffer)
                if fdate + buf >= t_min and fdate - buf <= t_max:
                    filtered.append((data_url, label_url))
            except Exception:
                filtered.append((data_url, label_url))
            continue

        # No timestamp pattern — include by default
        filtered.append((data_url, label_url))

    # If no filenames had timestamps, return all pairs unfiltered
    if not any_parsed:
        return pairs

    return filtered


def _discover_flat(
    collection_url: str,
    file_names: list[str],
) -> list[tuple[str, str]]:
    """Discover files in a flat (no subdirectory) collection."""
    return _pair_data_and_labels(collection_url, file_names)


def _discover_recursive(
    base_url: str,
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
    max_depth: int = 4,
    _depth: int = 0,
) -> list[tuple[str, str]]:
    """Recursively discover data files in PDS3-style nested directories.

    PDS3 collections can have arbitrary nesting (e.g.,
    ``DATA/JUPITER/PC/PERI-01/``).  We recurse up to *max_depth* levels,
    collecting file pairs from every leaf directory.
    Time filtering is applied to individual files by filename timestamps.

    Uses parallel HTTP requests at the deepest level to speed up traversal
    of many subdirectories (e.g., 75 PERI-XX directories).
    """
    if _depth >= max_depth:
        return []

    entries = _list_directory(base_url)
    dir_names = [e["name"].rstrip("/") for e in entries if e["is_dir"]]
    file_names = [e["name"] for e in entries if not e["is_dir"]]

    pairs = []

    # Collect files at this level
    if file_names:
        level_pairs = _pair_data_and_labels(base_url, file_names)
        level_pairs = _filter_pairs_by_filename_time(level_pairs, t_min, t_max)
        pairs.extend(level_pairs)

    if not dir_names:
        return pairs

    # At deeper levels with many subdirs, use parallel HTTP requests
    if _depth >= 2 and len(dir_names) > 5:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _scan_subdir(d):
            sub_url = f"{base_url}{d}/"
            return _discover_recursive(sub_url, t_min, t_max, max_depth, _depth + 1)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_scan_subdir, d): d for d in dir_names}
            for future in as_completed(futures):
                try:
                    pairs.extend(future.result())
                except Exception:
                    pass
    else:
        for d in sorted(dir_names):
            sub_url = f"{base_url}{d}/"
            pairs.extend(
                _discover_recursive(sub_url, t_min, t_max, max_depth, _depth + 1)
            )

    return pairs


def _pair_data_and_labels(
    base_url: str,
    file_names: list[str],
) -> list[tuple[str, str]]:
    """Pair data files with their labels.

    Data files: .TAB, .tab, .csv, .CSV, .dat, .sts
    Labels: .xml, .lblx, .XML, .lbl

    A data file ``foo.TAB`` pairs with ``foo.xml`` or ``foo.lbl``.
    """
    # Separate data and label files
    data_exts = {".tab", ".csv", ".dat", ".sts"}
    label_exts = {".xml", ".lblx", ".lbl"}

    data_files = {}
    label_files = {}

    for f in file_names:
        stem_lower = Path(f).stem.lower()
        ext_lower = Path(f).suffix.lower()
        # Skip PDS inventory/collection files (not science data)
        if stem_lower.startswith("collection"):
            continue
        if ext_lower in data_exts:
            data_files[stem_lower] = f
        elif ext_lower in label_exts:
            label_files[stem_lower] = f

    pairs = []
    for stem, data_name in sorted(data_files.items()):
        if stem in label_files:
            data_url = f"{base_url}{data_name}"
            label_url = f"{base_url}{label_files[stem]}"
            pairs.append((data_url, label_url))

    return pairs


# ---------------------------------------------------------------------------
# File download and caching
# ---------------------------------------------------------------------------

def _download_file(url: str) -> Path:
    """Download a file from the PPI archive with local caching.

    Cache structure preserves archive path:
    ``ppi_data/{bundle}/{collection}/.../{filename}``
    """
    # Extract relative path from URL
    marker = "/data/"
    idx = url.find(marker)
    if idx >= 0:
        rel_path = url[idx + len(marker):]
    else:
        rel_path = url.rsplit("/", 1)[-1]

    local_path = CACHE_DIR / rel_path

    # Cache hit
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    # Download
    get_event_bus().emit(PPI_FETCH, agent="ppi", level="info", msg=f"[PPI] Downloading: {local_path.name}")
    local_path.parent.mkdir(parents=True, exist_ok=True)

    from data_ops.http_utils import request_with_retry
    resp = request_with_retry(url, timeout=60)

    import os
    tmp_path = local_path.with_suffix(".tmp")
    tmp_path.write_bytes(resp.content)
    os.replace(tmp_path, local_path)
    size_kb = len(resp.content) / 1024
    get_event_bus().emit(PPI_FETCH, agent="ppi", msg=f"[PPI] Downloaded {size_kb:.1f} KB -> {local_path}")

    return local_path


# ---------------------------------------------------------------------------
# XML label parsing
# ---------------------------------------------------------------------------

def _parse_xml_label(xml_text: str) -> dict:
    """Parse a PDS4 XML label to extract table format information.

    Returns dict with:
        table_type: "fixed_width" or "delimited"
        fields: list of {name, offset, length, type, field_number} dicts
        delimiter: delimiter character (for delimited tables)
        records: expected number of records (if available)
    """
    root = ET.fromstring(xml_text)

    # Try Table_Delimited first, then Table_Character (fixed-width)
    table_delim = root.find(f".//{{{_PDS4_NS}}}Table_Delimited")
    if table_delim is not None:
        return _parse_delimited_label(table_delim)

    table_char = root.find(f".//{{{_PDS4_NS}}}Table_Character")
    if table_char is not None:
        return _parse_fixed_width_label(table_char)

    # Check for Table_Binary (not yet supported, but give a clear message)
    table_bin = root.find(f".//{{{_PDS4_NS}}}Table_Binary")
    if table_bin is not None:
        raise ValueError(
            "Table_Binary format not yet supported. "
            "This dataset uses binary tables which require a different parser."
        )

    # Fallback: try without namespace (some older labels)
    table_delim = root.find(".//Table_Delimited")
    if table_delim is not None:
        return _parse_delimited_label(table_delim)

    table_char = root.find(".//Table_Character")
    if table_char is not None:
        return _parse_fixed_width_label(table_char)

    # Collect table-like element names for debugging
    table_tags = [
        child.tag.split("}")[-1] if "}" in child.tag else child.tag
        for child in root.iter()
        if "table" in (child.tag.split("}")[-1] if "}" in child.tag else child.tag).lower()
    ]
    raise ValueError(
        f"No Table_Delimited or Table_Character found in XML label. "
        f"Table-like elements found: {table_tags or 'none'}"
    )


def _parse_delimited_label(table_elem) -> dict:
    """Parse a Table_Delimited element."""
    ns = _PDS4_NS

    # Get delimiter
    delimiter = ","
    delim_elem = table_elem.find(f"{{{ns}}}field_delimiter")
    if delim_elem is not None:
        delim_text = (delim_elem.text or "").strip().lower()
        if delim_text in ("semicolon", "semi colon"):
            delimiter = ";"
        elif delim_text in ("comma",):
            delimiter = ","
        elif delim_text in ("tab", "horizontal tab"):
            delimiter = "\t"

    # Get record delimiter (to detect if first line is header)
    records_elem = table_elem.find(f"{{{ns}}}records")
    records = int(records_elem.text) if records_elem is not None else None

    # Parse fields from Record_Delimited
    fields = []
    record = table_elem.find(f"{{{ns}}}Record_Delimited")
    if record is not None:
        for i, field in enumerate(record.findall(f"{{{ns}}}Field_Delimited")):
            name_elem = field.find(f"{{{ns}}}name")
            fn_elem = field.find(f"{{{ns}}}field_number")
            unit_elem = field.find(f"{{{ns}}}unit")
            desc_elem = field.find(f"{{{ns}}}description")

            name = name_elem.text.strip() if name_elem is not None else f"col_{i}"
            field_number = int(fn_elem.text) if fn_elem is not None else i + 1
            unit = unit_elem.text.strip() if unit_elem is not None and unit_elem.text else ""
            description = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""

            # Check for fill values in Special_Constants
            fill = _extract_special_constants(field, ns)

            entry = {
                "name": name,
                "field_number": field_number,
                "type": "delimited",
                "unit": unit,
                "description": description,
            }
            if fill is not None:
                entry["null_constant"] = fill
            fields.append(entry)

    return {
        "table_type": "delimited",
        "fields": fields,
        "delimiter": delimiter,
        "records": records,
    }


def _parse_fixed_width_label(table_elem) -> dict:
    """Parse a Table_Character (fixed-width) element."""
    ns = _PDS4_NS

    records_elem = table_elem.find(f"{{{ns}}}records")
    records = int(records_elem.text) if records_elem is not None else None

    fields = []
    record = table_elem.find(f"{{{ns}}}Record_Character")
    if record is not None:
        for i, field in enumerate(record.findall(f"{{{ns}}}Field_Character")):
            name_elem = field.find(f"{{{ns}}}name")
            loc_elem = field.find(f"{{{ns}}}field_location")
            len_elem = field.find(f"{{{ns}}}field_length")
            unit_elem = field.find(f"{{{ns}}}unit")
            desc_elem = field.find(f"{{{ns}}}description")

            name = name_elem.text.strip() if name_elem is not None else f"col_{i}"

            # field_location can be either a direct value or have a nested
            # <offset> child (PDS4 spec allows both forms)
            offset = 0
            if loc_elem is not None:
                offset_child = loc_elem.find(f"{{{ns}}}offset")
                if offset_child is not None:
                    offset = int(offset_child.text)
                elif loc_elem.text and loc_elem.text.strip():
                    offset = int(loc_elem.text.strip())

            length = 0
            if len_elem is not None:
                length_child = len_elem.find(f"{{{ns}}}length")
                if length_child is not None:
                    length = int(length_child.text)
                elif len_elem.text and len_elem.text.strip():
                    length = int(len_elem.text.strip())

            unit = unit_elem.text.strip() if unit_elem is not None and unit_elem.text else ""
            description = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""

            # Check for fill values in Special_Constants
            fill = _extract_special_constants(field, ns)

            entry = {
                "name": name,
                "offset": offset,
                "length": length,
                "type": "fixed_width",
                "unit": unit,
                "description": description,
            }
            if fill is not None:
                entry["null_constant"] = fill
            fields.append(entry)

    return {
        "table_type": "fixed_width",
        "fields": fields,
        "delimiter": None,
        "records": records,
    }


def _extract_special_constants(field_elem, ns: str) -> str | None:
    """Extract fill value from Special_Constants element."""
    sc = field_elem.find(f"{{{ns}}}Special_Constants")
    if sc is None:
        return None

    for tag in ("missing_constant", "missing_flag", "saturated_constant",
                "null_constant"):
        elem = sc.find(f"{{{ns}}}{tag}")
        if elem is not None and elem.text:
            return elem.text.strip()

    return None


# ---------------------------------------------------------------------------
# Table reading
# ---------------------------------------------------------------------------

def _read_table(
    file_path: Path,
    label: dict,
    parameter_id: str,
) -> pd.DataFrame | None:
    """Read a data file using its parsed XML label.

    Returns a DataFrame with DatetimeIndex and numeric columns for
    the requested parameter, or None if the parameter is not in this file.
    """
    if label["table_type"] == "fixed_width":
        return _read_fixed_width_table(file_path, label, parameter_id)
    else:
        return _read_delimited_table(file_path, label, parameter_id)


def _read_fixed_width_table(
    file_path: Path,
    label: dict,
    parameter_id: str,
) -> pd.DataFrame | None:
    """Read a fixed-width ASCII table (.TAB/.sts) using label-derived colspecs."""
    fields = label["fields"]
    if not fields:
        return None

    # Check if our parameter is in this file
    field_names = [f["name"] for f in fields]
    param_indices = _find_param_columns(field_names, parameter_id)
    time_idx = _find_time_column(field_names)

    if not param_indices:
        return None
    if time_idx is None:
        get_event_bus().emit(PPI_FETCH, agent="ppi", level="warning", msg=f"[PPI] No time column found in {file_path.name}")
        return None

    # Build colspecs: (start, end) for each field
    # PDS3/PDS4 field_location.offset is 1-based byte position
    colspecs = []
    col_names = []
    for f in fields:
        start = f["offset"] - 1  # 0-based
        end = start + f["length"]
        colspecs.append((start, end))
        col_names.append(f["name"])

    # PDS3 files with attached headers: skip header_bytes
    header_bytes = label.get("header_bytes", 0) or 0

    try:
        if header_bytes > 0:
            # Read file as bytes, skip header, then parse
            raw = file_path.read_bytes()[header_bytes:]
            import io
            df = pd.read_fwf(
                io.BytesIO(raw),
                colspecs=colspecs,
                names=col_names,
                header=None,
            )
        else:
            df = pd.read_fwf(
                file_path,
                colspecs=colspecs,
                names=col_names,
                header=None,
            )
    except Exception as e:
        get_event_bus().emit(PPI_FETCH, agent="ppi", level="warning", msg=f"[PPI] Failed to read {file_path.name}: {e}")
        return None

    return _extract_param_df(df, field_names, time_idx, param_indices)


def _read_delimited_table(
    file_path: Path,
    label: dict,
    parameter_id: str,
) -> pd.DataFrame | None:
    """Read a delimited table (CSV) using label-derived columns."""
    fields = label["fields"]
    delimiter = label["delimiter"] or ","

    if not fields:
        return None

    field_names = [f["name"] for f in fields]
    param_indices = _find_param_columns(field_names, parameter_id)
    time_idx = _find_time_column(field_names)

    if not param_indices:
        return None
    if time_idx is None:
        get_event_bus().emit(PPI_FETCH, agent="ppi", level="warning", msg=f"[PPI] No time column found in {file_path.name}")
        return None

    try:
        # Try reading without header first (label defines columns)
        df = pd.read_csv(
            file_path,
            sep=delimiter,
            header=None,
            names=field_names,
            skipinitialspace=True,
            on_bad_lines="skip",
        )
    except Exception:
        # Some files have a header row — skip it
        try:
            df = pd.read_csv(
                file_path,
                sep=delimiter,
                header=0,
                names=field_names,
                skipinitialspace=True,
                on_bad_lines="skip",
            )
        except Exception as e:
            get_event_bus().emit(PPI_FETCH, agent="ppi", level="warning", msg=f"[PPI] Failed to read {file_path.name}: {e}")
            return None

    # If the first row looks like a header (non-numeric in data columns),
    # detect and skip it
    if len(df) > 0:
        first_data_col = param_indices[0]
        first_val = df.iloc[0, first_data_col]
        if isinstance(first_val, str):
            try:
                float(first_val.strip().strip('"'))
            except (ValueError, AttributeError):
                # First row is a header — drop it
                df = df.iloc[1:].reset_index(drop=True)

    return _extract_param_df(df, field_names, time_idx, param_indices)


def _extract_param_df(
    df: pd.DataFrame,
    field_names: list[str],
    time_idx: int,
    param_indices: list[int],
) -> pd.DataFrame | None:
    """Extract time index + parameter columns from a raw DataFrame."""
    if len(df) == 0:
        return None

    time_col = field_names[time_idx]

    # Parse timestamps — filter out header/units rows first
    time_series = df.iloc[:, time_idx].astype(str).str.strip().str.strip('"')
    # Match ISO dates (2016-01-15) or PDS3 YYYY DOY format (2016 214)
    ts_mask = time_series.str.match(r"^\s*\d{4}[\s\-/]")
    if not ts_mask.any():
        get_event_bus().emit(PPI_FETCH, agent="ppi", level="warning", msg=f"[PPI] No timestamp-like values found in {field_names[time_idx]}")
        return None

    # Filter df to only rows with valid-looking timestamps
    df_clean = df[ts_mask].reset_index(drop=True)
    time_clean = time_series[ts_mask].reset_index(drop=True)

    times = _parse_pds_timestamps(time_clean)
    if times is None:
        get_event_bus().emit(PPI_FETCH, agent="ppi", level="warning", msg=f"[PPI] Failed to parse timestamps in {field_names[time_idx]}")
        return None

    # Build result DataFrame with integer column names (1, 2, ...)
    # to match the format used by fetch_cdf
    result = pd.DataFrame(index=times)
    result.index.name = "time"

    for i, col_idx in enumerate(param_indices):
        values = df_clean.iloc[:, col_idx]
        # Strip quotes if present
        if values.dtype == object:
            values = values.astype(str).str.strip().str.strip('"')
        # Use .values to avoid index alignment issues (source df has
        # integer index, result has DatetimeIndex)
        result[i + 1] = pd.to_numeric(values, errors="coerce").values

    return result


# ---------------------------------------------------------------------------
# Parameter and time column matching
# ---------------------------------------------------------------------------

def _parse_pds_timestamps(time_series: pd.Series) -> pd.DatetimeIndex | None:
    """Parse timestamps from PDS data files, handling various formats.

    Expects pre-cleaned series (header/non-timestamp rows already removed).

    Supports:
    - ISO 8601 (``2005-01-15T00:00:00``)
    - Year-DOY (``1979-185T23:44:47.500``)
    - PDS3 space-separated (``2016 214  0  1 24 505`` = YYYY DOY HR MIN SEC MSEC)
    - Various other formats via pandas mixed parsing
    """
    if len(time_series) == 0:
        return None

    # Check sample to pick the best parser upfront
    sample = str(time_series.iloc[0]).strip()

    # PDS3 space-separated format: "YYYY DOY HR MIN SEC MSEC"
    # e.g., "2016 214  0  1 24 505"
    if re.match(r"\d{4}\s+\d{1,3}\s+\d", sample):
        try:
            return _parse_pds3_space_timestamps(time_series)
        except Exception:
            pass

    # Year-DOY format: YYYY-DDDTHH:MM:SS.sss (common in PDS) — check first
    # because generic pd.to_datetime falls back to slow dateutil for this
    if re.match(r"\d{4}-\d{3}T", sample):
        try:
            return pd.to_datetime(time_series, format="%Y-%jT%H:%M:%S.%f")
        except Exception:
            pass
        try:
            return pd.to_datetime(time_series, format="%Y-%jT%H:%M:%S")
        except Exception:
            pass
        # Element-wise for mixed sub-second precision
        try:
            parsed = []
            for t in time_series:
                t_str = str(t).strip()
                if "." in t_str:
                    parsed.append(pd.Timestamp.strptime(t_str, "%Y-%jT%H:%M:%S.%f"))
                else:
                    parsed.append(pd.Timestamp.strptime(t_str, "%Y-%jT%H:%M:%S"))
            return pd.DatetimeIndex(parsed)
        except Exception:
            pass

    # Standard ISO 8601
    try:
        return pd.to_datetime(time_series, format="ISO8601")
    except Exception:
        pass

    # Generic pandas parsing (may use dateutil — slower)
    try:
        return pd.to_datetime(time_series, utc=False)
    except Exception:
        pass

    # Mixed format fallback
    try:
        return pd.to_datetime(time_series, format="mixed")
    except Exception:
        pass

    return None


def _parse_pds3_space_timestamps(time_series: pd.Series) -> pd.DatetimeIndex:
    """Parse PDS3 space-separated timestamps: ``YYYY DOY HR MIN SEC MSEC``.

    Uses vectorized operations for performance on large files (86,400+ rows).
    """
    # Split each row into components
    parts = time_series.str.split(expand=True)

    if parts.shape[1] >= 6:
        # YYYY DOY HR MIN SEC MSEC
        year = parts[0].astype(int)
        doy = parts[1].astype(int)
        hour = parts[2].astype(int)
        minute = parts[3].astype(int)
        sec = parts[4].astype(int)
        msec = parts[5].astype(int)

        # Build ISO strings: YYYY-DDDTHH:MM:SS.mmm
        iso = (
            year.astype(str) + "-" +
            doy.astype(str).str.zfill(3) + "T" +
            hour.astype(str).str.zfill(2) + ":" +
            minute.astype(str).str.zfill(2) + ":" +
            sec.astype(str).str.zfill(2) + "." +
            msec.astype(str).str.zfill(3)
        )
        return pd.to_datetime(iso, format="%Y-%jT%H:%M:%S.%f")

    elif parts.shape[1] >= 5:
        # YYYY DOY HR MIN SEC
        year = parts[0].astype(int)
        doy = parts[1].astype(int)
        hour = parts[2].astype(int)
        minute = parts[3].astype(int)
        sec = parts[4].astype(int)

        iso = (
            year.astype(str) + "-" +
            doy.astype(str).str.zfill(3) + "T" +
            hour.astype(str).str.zfill(2) + ":" +
            minute.astype(str).str.zfill(2) + ":" +
            sec.astype(str).str.zfill(2)
        )
        return pd.to_datetime(iso, format="%Y-%jT%H:%M:%S")

    raise ValueError(
        f"PDS3 time has {parts.shape[1]} components, expected >= 5"
    )


def _find_param_columns(
    field_names: list[str],
    parameter_id: str,
) -> list[int]:
    """Find column indices for a parameter in the field name list.

    Matches by exact name (case-insensitive). Returns list of indices
    for multi-component parameters.
    """
    # Exact match (case-insensitive)
    pid_lower = parameter_id.lower()
    matches = [
        i for i, name in enumerate(field_names)
        if name.lower() == pid_lower
    ]
    if matches:
        return matches

    # Try matching with common suffixes stripped
    # e.g., "BR" might appear as "BR" or "B_R"
    for i, name in enumerate(field_names):
        if name.lower().replace("_", "") == pid_lower.replace("_", ""):
            matches.append(i)
    if matches:
        return matches

    return []


_TIME_NAMES = {
    "time", "epoch", "utc", "scet", "datetime", "date_time",
    "timestamp", "t", "time_utc",
}

_TIME_PATTERNS = [
    re.compile(r"scet", re.IGNORECASE),
    re.compile(r"^time", re.IGNORECASE),
    re.compile(r"utc", re.IGNORECASE),
    re.compile(r"epoch", re.IGNORECASE),
    re.compile(r"date", re.IGNORECASE),
]


def _find_time_column(field_names: list[str]) -> int | None:
    """Find the time/epoch column index."""
    # Exact match on common names
    for i, name in enumerate(field_names):
        if name.lower().strip() in _TIME_NAMES:
            return i

    # Pattern match
    for pattern in _TIME_PATTERNS:
        for i, name in enumerate(field_names):
            if pattern.search(name):
                return i

    # Fall back to first column (many PDS tables have time first)
    if field_names:
        return 0

    return None


def _find_param_meta(info: dict, parameter_id: str) -> dict:
    """Find metadata for a specific parameter in cached info."""
    for p in info.get("parameters", []):
        if p.get("name") == parameter_id:
            return p

    available = [
        p["name"] for p in info.get("parameters", [])
        if p.get("name", "").lower() != "time"
    ]
    raise ValueError(
        f"Parameter '{parameter_id}' not found in dataset metadata. "
        f"Available: {available}"
    )
