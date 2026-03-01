"""
CDF file download backend — fetches data from CDAWeb via direct CDF file download.

Downloads CDF files from CDAWeb's REST API, caches them locally in cdaweb_data/,
and reads parameters using cdflib. Multi-day requests download CDF files in
parallel using a thread pool.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import cdflib
import numpy as np
import pandas as pd
import requests
import xarray as xr

from agent.event_bus import (
    get_event_bus,
    CDF_FILE_QUERY,
    CDF_CACHE_HIT,
    CDF_DOWNLOAD,
    CDF_DOWNLOAD_WARN,
    CDF_METADATA_SYNC,
    DEBUG,
    FETCH_ERROR,
)

from knowledge.metadata_client import get_dataset_info

CDAWEB_REST_BASE = "https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cdaweb_data"

_WARN_THRESHOLD_BYTES = 500 * 1024 * 1024   # 500 MB
_BLOCK_THRESHOLD_BYTES = 1024 * 1024 * 1024  # 1 GB


# CDF variable data types to skip (epoch/time and character/metadata types)
_EPOCH_TYPES = {"CDF_EPOCH", "CDF_EPOCH16", "CDF_TIME_TT2000"}
_SKIP_TYPES = _EPOCH_TYPES | {"CDF_CHAR", "CDF_UCHAR"}


def list_cdf_variables(dataset_id: str) -> list[dict]:
    """List data variables for a CDAWeb dataset.

    Uses the metadata resolution chain (local cache → Master CDF).
    Note: master CDF metadata may not perfectly match actual data CDF files
    (CDF versions can diverge). If a parameter listed here is missing from
    data files, fetch_cdf_data will return an error with the actual available
    variables so the agent can self-correct.

    Parameters that have been validated against actual data CDF files and found
    to be absent are marked with ``status: "NOT_IN_DATA_FILES"`` so the agent
    can prefer alternatives.

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "AC_H2_MFI").

    Returns:
        List of dicts with keys: name, description, units, size, and
        optionally status/note from validation annotations.
    """
    info = get_dataset_info(dataset_id)
    annotations = info.get("parameters_annotations", {})

    result = []
    for param in info.get("parameters", []):
        name = param.get("name", "")
        if name.lower() == "time":
            continue
        size = param.get("size", [1])
        entry = {
            "name": name,
            "description": param.get("description", ""),
            "units": param.get("units", ""),
            "size": size,
        }
        # Surface validation annotations so the agent can make informed choices
        ann = annotations.get(name)
        if isinstance(ann, dict):
            category = ann.get("_category", "")
            note = ann.get("_note", "")
            if category == "phantom" or (
                not category
                and ("not found in data" in note or "not found in archive" in note)
            ):
                entry["status"] = "NOT_IN_DATA_FILES"
            if note:
                entry["note"] = note
        result.append(entry)

    get_event_bus().emit(DEBUG, agent="cdf", msg=f"[CDF] Listed {len(result)} data variables for {dataset_id}")
    return result


def fetch_cdf_data(
    dataset_id: str,
    parameter_id: str,
    time_min: str,
    time_max: str,
    force: bool = False,
) -> dict:
    """Fetch timeseries data by downloading CDF files from CDAWeb.

    Same signature and return format as fetch_data().

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "AC_H2_MFI").
        parameter_id: Parameter name (e.g., "BGSEc").
        time_min: ISO start time (e.g., "2024-01-15T00:00:00Z").
        time_max: ISO end time (e.g., "2024-01-16T00:00:00Z").

    Returns:
        Dict with keys: data (DataFrame), units, description, fill_value.

    Raises:
        ValueError: If no data is available.
        requests.HTTPError: If a download fails.
    """
    # Get metadata from local cache if available
    cdf_native = False
    info = get_dataset_info(dataset_id)
    try:
        param_meta = _find_parameter_meta(info, parameter_id)
        units = param_meta.get("units", "")
        description = param_meta.get("description", "")
        fill_value = param_meta.get("fill", None)
    except ValueError:
        # Parameter not in metadata cache — it's a CDF-native variable name.
        # We'll extract metadata from the first CDF file below.
        cdf_native = True
        units = ""
        description = ""
        fill_value = None

    # Discover CDF files covering the time range
    file_list = _get_cdf_file_list(dataset_id, time_min, time_max)
    get_event_bus().emit(CDF_FILE_QUERY, agent="cdf", msg=f"[CDF] Found {len(file_list)} files for {dataset_id} ({time_min} to {time_max})")

    # Check download size before proceeding
    download_bytes, total_bytes, n_cached, n_to_download = _check_download_size(
        file_list, CACHE_DIR
    )

    if download_bytes > _BLOCK_THRESHOLD_BYTES and not force:
        size_mb = download_bytes / 1e6
        return {
            "status": "confirmation_required",
            "download_mb": round(size_mb),
            "n_files": n_to_download,
            "n_cached": n_cached,
            "dataset_id": dataset_id,
            "message": (
                f"This request requires downloading {size_mb:.0f} MB "
                f"({n_to_download} files) from CDAWeb. "
                f"Do you want to proceed?"
            ),
        }

    if download_bytes > _WARN_THRESHOLD_BYTES:
        get_event_bus().emit(CDF_DOWNLOAD_WARN, agent="cdf", level="warning", msg=f"[CDF] Large download: {download_bytes / 1e6:.0f} MB ({n_to_download} files) for {dataset_id}. Consider narrowing the time range.")

    # Download and read CDF files (parallel when enabled and multiple files)
    from config import PARALLEL_FETCH, PARALLEL_MAX_WORKERS
    use_parallel = PARALLEL_FETCH and len(file_list) > 1
    max_workers = min(len(file_list), PARALLEL_MAX_WORKERS, 6)
    frames = []
    validmin = None
    validmax = None

    if use_parallel:
        get_event_bus().emit(CDF_DOWNLOAD, agent="cdf", level="info", msg=f"[CDF] Downloading {len(file_list)} files in parallel (max_workers={max_workers})")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_download_and_read, fi["url"], parameter_id, CACHE_DIR): idx
                for idx, fi in enumerate(file_list)
            }
            results_by_idx: dict[int, tuple[Path, pd.DataFrame | xr.DataArray]] = {}
            for future in as_completed(futures):
                idx = futures[future]
                results_by_idx[idx] = future.result()
    else:
        results_by_idx = {}
        for idx, fi in enumerate(file_list):
            results_by_idx[idx] = _download_and_read(fi["url"], parameter_id, CACHE_DIR)

    for idx in range(len(file_list)):
        local_path, data = results_by_idx[idx]
        # Extract metadata from the first CDF file.
        # Always read FILLVAL/VALIDMIN/VALIDMAX from CDF (ground truth)
        # since cached fill values may have different precision (float32 vs float64).
        if not frames:
            # Sync cached metadata with actual data CDF variables.
            # Runs once per dataset — cheap since file is already downloaded.
            source_url = file_list[idx]["url"]
            _sync_metadata_with_data_cdf(dataset_id, local_path, source_url=source_url)

            try:
                cdf = cdflib.CDF(str(local_path))
                attrs = cdf.varattsget(parameter_id)
                if cdf_native:
                    units = attrs.get("UNITS", "") or ""
                    if isinstance(units, np.ndarray):
                        units = str(units)
                    description = (attrs.get("CATDESC", "")
                                   or attrs.get("FIELDNAM", "") or "")
                    if isinstance(description, np.ndarray):
                        description = str(description)
                fv = attrs.get("FILLVAL", None)
                if fv is not None:
                    try:
                        fill_value = float(fv)
                    except (ValueError, TypeError):
                        pass
                vmin = attrs.get("VALIDMIN", None)
                vmax = attrs.get("VALIDMAX", None)
                if vmin is not None:
                    try:
                        validmin = float(vmin)
                    except (ValueError, TypeError):
                        pass
                if vmax is not None:
                    try:
                        validmax = float(vmax)
                    except (ValueError, TypeError):
                        pass
            except Exception:
                pass
        frames.append(data)

    if not frames:
        raise ValueError(
            f"No data extracted for {dataset_id}/{parameter_id} "
            f"in range {time_min} to {time_max}"
        )

    # Branch: xarray DataArray (3D+) vs pandas DataFrame (1D/2D)
    is_xarray = isinstance(frames[0], xr.DataArray)

    # Check if this is non-time-varying data (RangeIndex / no "time" dim)
    is_non_time_varying = False
    if not is_xarray and hasattr(frames[0].index, 'name') and frames[0].index.name != "time":
        is_non_time_varying = True
    elif is_xarray and "time" not in frames[0].dims:
        is_non_time_varying = True

    if is_non_time_varying:
        # Non-time-varying: just use the first file's data (all files have same static values)
        data = frames[0]
        get_event_bus().emit(DEBUG, agent="cdf", msg=f"[CDF] {dataset_id}/{parameter_id}: non-time-varying, shape={data.shape}")
    elif is_xarray:
        data = _postprocess_xarray(frames, parameter_id, time_min, time_max,
                                   fill_value, validmin, validmax)
    else:
        data = _postprocess_dataframe(frames, time_min, time_max,
                                      fill_value, validmin, validmax)

    if is_non_time_varying:
        shape_info = f"{data.shape}"
    elif is_xarray:
        shape_info = f"{dict(data.sizes)}"
    else:
        shape_info = f"{len(data)} rows, {len(data.columns)} columns"
    get_event_bus().emit(DEBUG, agent="cdf", msg=f"[CDF] {dataset_id}/{parameter_id}: {shape_info}")

    return {
        "data": data,
        "units": units,
        "description": description,
        "fill_value": fill_value,
    }


def _postprocess_dataframe(
    frames: list[pd.DataFrame],
    time_min: str,
    time_max: str,
    fill_value: float | None,
    validmin: float | None,
    validmax: float | None,
) -> pd.DataFrame:
    """Concatenate, clean, and trim DataFrame results from CDF files."""
    df = pd.concat(frames)
    df.sort_index(inplace=True)

    # Remove duplicates (overlapping files)
    df = df[~df.index.duplicated(keep="first")]

    # Trim to requested time range (strip timezone info for naive index)
    t_start = _strip_utc_suffix(time_min)
    t_stop = _strip_utc_suffix(time_max)
    df = df.loc[t_start:t_stop]

    if len(df) == 0:
        raise ValueError(
            f"No data rows in range {time_min} to {time_max}"
        )

    # Ensure float64 dtype (CDF often stores float32)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

    # Replace fill values with NaN.
    if fill_value is not None:
        try:
            fill_f = float(fill_value)
            for col in df.columns:
                mask = np.isclose(df[col].values, fill_f, rtol=1e-6,
                                  equal_nan=False)
                df.loc[mask, col] = np.nan
        except (ValueError, TypeError):
            pass

    # Replace out-of-range values with NaN using CDF VALIDMIN/VALIDMAX.
    if validmin is not None or validmax is not None:
        for col in df.columns:
            if validmin is not None:
                df.loc[df[col] < validmin, col] = np.nan
            if validmax is not None:
                df.loc[df[col] > validmax, col] = np.nan

    return df


def _postprocess_xarray(
    frames: list[xr.DataArray],
    parameter_id: str,
    time_min: str,
    time_max: str,
    fill_value: float | None,
    validmin: float | None,
    validmax: float | None,
) -> xr.DataArray:
    """Concatenate, clean, and trim xarray DataArray results from CDF files."""
    da = xr.concat(frames, dim="time")
    da = da.sortby("time")

    # Remove duplicate times
    _, unique_idx = np.unique(da.coords["time"].values, return_index=True)
    da = da.isel(time=unique_idx)

    # Trim to requested time range
    t_start = np.datetime64(_strip_utc_suffix(time_min))
    t_stop = np.datetime64(_strip_utc_suffix(time_max))
    da = da.sel(time=slice(t_start, t_stop))

    if da.sizes["time"] == 0:
        raise ValueError(
            f"No data rows for {parameter_id} in range {time_min} to {time_max}"
        )

    # Ensure float64 dtype
    da = da.astype(np.float64)

    # Replace fill values with NaN
    if fill_value is not None:
        try:
            fill_f = float(fill_value)
            da = da.where(~np.isclose(da.values, fill_f, rtol=1e-6, equal_nan=False))
        except (ValueError, TypeError):
            pass

    # Replace out-of-range values with NaN
    if validmin is not None:
        da = da.where(da >= validmin)
    if validmax is not None:
        da = da.where(da <= validmax)

    da.name = parameter_id
    return da


def _download_and_read(
    url: str, parameter_id: str, cache_dir: Path
) -> tuple[Path, pd.DataFrame | xr.DataArray]:
    """Download a CDF file and read one parameter. Thread-safe."""
    local_path = _download_cdf_file(url, cache_dir)
    data = _read_cdf_parameter(local_path, parameter_id)
    return local_path, data


def _sync_metadata_with_data_cdf(
    dataset_id: str, cdf_path: Path, *, source_url: str = "",
) -> None:
    """Compare data CDF variables against cached metadata and update if needed.

    Called once per fetch on the first data CDF file.  If the data CDF contains
    variables not in the cached metadata (or vice versa), writes learned
    annotations to ``mission_overrides/`` (via ``update_dataset_override()``)
    so discrepancies are visible to the LLM and future fetches while keeping
    the metadata cache files as pure auto-generated data.

    Each validation is recorded as a structured entry in ``_validations`` with
    provenance (source URL, filename, timestamp).  If the same source URL has
    already been validated, the sync is skipped.  The legacy ``_validated: true``
    flag is still written for backward compatibility.

    This is cheap — the CDF is already downloaded, we just read its variable list.
    """
    from datetime import datetime, timezone
    from knowledge.metadata_client import (
        _find_local_cache, _load_dataset_override, update_dataset_override,
    )

    cache_path = _find_local_cache(dataset_id)
    if cache_path is None:
        get_event_bus().emit(CDF_METADATA_SYNC, agent="cdf", msg=f"[CDF] Metadata sync skipped for {dataset_id}: no local cache")
        return

    try:
        cached_info = json.loads(cache_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        get_event_bus().emit(CDF_METADATA_SYNC, agent="cdf", msg=f"[CDF] Metadata sync skipped for {dataset_id}: cache read failed: {exc}")
        return

    # Derive mission_stem from cache path for O(1) override lookup
    mission_stem = cache_path.parent.parent.name

    # Check existing override for prior validations
    existing_override = _load_dataset_override(dataset_id, mission_stem=mission_stem)
    if existing_override:
        existing_validations = existing_override.get("_validations", [])
        # If this exact source URL was already validated, skip
        if source_url and any(
            v.get("source_url") == source_url for v in existing_validations
        ):
            get_event_bus().emit(CDF_METADATA_SYNC, agent="cdf", msg=f"[CDF] Metadata sync skipped for {dataset_id}: source already validated ({cdf_path.name})")
            return
        # Backward compat: if old-format _validated exists but no _validations,
        # and no source_url to compare, skip (treat as "validated, provenance unknown")
        if (
            existing_override.get("_validated")
            and not existing_validations
            and not source_url
        ):
            get_event_bus().emit(CDF_METADATA_SYNC, agent="cdf", msg=f"[CDF] Metadata sync skipped for {dataset_id}: already validated (legacy)")
            return

    get_event_bus().emit(CDF_METADATA_SYNC, agent="cdf", msg=f"[CDF] Metadata sync: comparing {dataset_id} against {cdf_path.name}")

    try:
        data_cdf = cdflib.CDF(str(cdf_path))
        data_info = data_cdf.cdf_info()
        all_cdf_vars = set(data_info.zVariables) | set(data_info.rVariables)
    except Exception as exc:
        get_event_bus().emit(CDF_METADATA_SYNC, agent="cdf", level="warning", msg=f"[CDF] Metadata sync failed for {dataset_id}: could not read data CDF: {exc}")
        return

    # Use inspect_cdf_variables for filtered data variable names (skips
    # epoch, char, support_data, metadata types)
    data_var_names = {v["name"] for v in inspect_cdf_variables(cdf_path)}

    cached_names = {
        p.get("name") for p in cached_info.get("parameters", [])
        if p.get("name", "").lower() != "time"
    }

    # Compare against all CDF vars (unfiltered) for phantom detection
    master_only = cached_names - all_cdf_vars
    # Compare against filtered data vars for undocumented detection
    data_only = data_var_names - cached_names

    # Build this validation's discrepancies
    params_annotations: dict = {}

    if not data_only and not master_only:
        get_event_bus().emit(CDF_METADATA_SYNC, agent="cdf", msg=f"[CDF] Metadata sync for {dataset_id}: perfect match ({len(cached_names)} variables)")
    else:
        # Log discrepancies at WARNING so they're visible in normal mode
        if master_only:
            get_event_bus().emit(CDF_METADATA_SYNC, agent="cdf", level="warning", msg=f"[CDF] Metadata sync for {dataset_id}: {len(master_only)} vars in master CDF but NOT in data CDF: {', '.join(sorted(master_only))}")
        if data_only:
            get_event_bus().emit(CDF_METADATA_SYNC, agent="cdf", level="warning", msg=f"[CDF] Metadata sync for {dataset_id}: {len(data_only)} vars in data CDF but NOT in master CDF: {', '.join(sorted(data_only))}")

        for param in cached_info.get("parameters", []):
            name = param.get("name", "")
            if name.lower() == "time":
                continue
            if name in master_only:
                params_annotations[name] = {
                    "_category": "phantom",
                    "_note": "in master CDF but not found in data CDF",
                }

        # Undocumented data variables (already filtered by inspect_cdf_variables)
        for var_name in sorted(data_only):
            params_annotations[var_name] = {
                "_category": "undocumented",
                "_note": "found in data CDF but not in master CDF",
            }

    # Read existing validations for version numbering
    existing_validations = []
    if existing_override:
        existing_validations = existing_override.get("_validations", [])

    # Build validation record with provenance
    validation_record = {
        "version": len(existing_validations) + 1,
        "source_file": cdf_path.name,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "discrepancies": params_annotations,
    }
    if source_url:
        validation_record["source_url"] = source_url

    # Build override patch
    override_patch: dict = {
        "_validated": True,  # backward compat
    }

    # Merge params_annotations at top level for consumption by list_parameters
    # and get_dataset_quality_report (union across all validations)
    if params_annotations:
        override_patch["parameters_annotations"] = params_annotations

    # Append to _validations list (read-modify-write via update_dataset_override)
    override_patch["_validations"] = existing_validations + [validation_record]

    get_event_bus().emit(CDF_METADATA_SYNC, agent="cdf", level="info", msg=f"[CDF] Metadata sync for {dataset_id}: writing override — {len(master_only)} phantom, {len(params_annotations) - len(master_only)} undocumented, {len(cached_names & all_cdf_vars)} confirmed")

    update_dataset_override(
        dataset_id, override_patch, mission_stem=mission_stem,
    )


def inspect_cdf_variables(cdf_path: Path) -> list[dict]:
    """Read actual data variables from a CDF file on disk.

    Opens the CDF and returns metadata for each variable that is a plottable
    data variable (skips epoch/time types, character types, support_data, and
    metadata VAR_TYPE variables — same filtering as _sync_metadata_with_data_cdf).

    Args:
        cdf_path: Path to a local CDF file.

    Returns:
        List of dicts with keys: name, type, size, units, description, var_type.
    """
    data_cdf = cdflib.CDF(str(cdf_path))
    data_info = data_cdf.cdf_info()
    all_vars = data_info.zVariables + data_info.rVariables

    result = []
    for var_name in all_vars:
        try:
            var_inq = data_cdf.varinq(var_name)
            # Skip epoch/time and character types
            if var_inq.Data_Type_Description in _SKIP_TYPES:
                continue
            var_attrs = data_cdf.varattsget(var_name)
            var_type = var_attrs.get("VAR_TYPE", "")
            if isinstance(var_type, (bytes, np.bytes_)):
                var_type = var_type.decode()
            # Skip support/metadata variables (ISTP convention)
            if var_type in ("support_data", "metadata"):
                continue

            units = var_attrs.get("UNITS", "") or ""
            if isinstance(units, np.ndarray):
                units = str(units)
            description = (var_attrs.get("CATDESC", "")
                           or var_attrs.get("FIELDNAM", "") or "")
            if isinstance(description, np.ndarray):
                description = str(description)

            # Determine size from the variable's dimensions
            dim_sizes = list(var_inq.Dim_Sizes) if var_inq.Dim_Sizes else [1]

            result.append({
                "name": var_name,
                "type": var_inq.Data_Type_Description,
                "size": dim_sizes,
                "units": units,
                "description": description,
                "var_type": var_type,
            })
        except Exception:
            continue

    return result


def _find_parameter_meta(info: dict, parameter_id: str) -> dict:
    """Find metadata for a specific parameter in metadata info."""
    for p in info.get("parameters", []):
        if p.get("name") == parameter_id:
            return p
    available = [p.get("name") for p in info.get("parameters", [])]
    raise ValueError(
        f"Parameter '{parameter_id}' not found. Available: {available}"
    )


def _get_cdf_file_list(
    dataset_id: str, time_min: str, time_max: str
) -> list[dict]:
    """Query CDAWeb REST API for CDF file URLs covering a time range.

    Args:
        dataset_id: CDAWeb dataset ID.
        time_min: ISO start time.
        time_max: ISO end time.

    Returns:
        List of dicts with 'url', 'start_time', 'end_time', 'size' keys.
    """
    # Convert ISO times to CDAWeb format: YYYYMMDDTHHmmSSZ
    start_str = _iso_to_cdaweb_time(time_min)
    stop_str = _iso_to_cdaweb_time(time_max)

    url = (f"{CDAWEB_REST_BASE}/datasets/{dataset_id}"
           f"/orig_data/{start_str},{stop_str}")

    get_event_bus().emit(CDF_FILE_QUERY, agent="cdf", msg=f"[CDF] Querying file list: {url}")
    from data_ops.http_utils import request_with_retry
    resp = request_with_retry(url, headers={"Accept": "application/json"})

    data = resp.json()

    # Navigate the response structure
    file_descs = (data.get("FileDescription")
                  or data.get("FileDescriptionList", {}).get("FileDescription")
                  or [])

    if not file_descs:
        raise ValueError(
            f"No CDF files found for {dataset_id} "
            f"in range {time_min} to {time_max}"
        )

    result = []
    for fd in file_descs:
        file_url = fd.get("Name", "")
        if not file_url:
            continue
        result.append({
            "url": file_url,
            "start_time": fd.get("StartTime", ""),
            "end_time": fd.get("EndTime", ""),
            "size": fd.get("Length", 0),
        })

    return result


def _strip_utc_suffix(iso_time: str) -> str:
    """Strip timezone suffix from an ISO 8601 string, returning a naive form.

    '2024-01-15T00:00:00Z'       -> '2024-01-15T00:00:00'
    '2024-01-15T00:00:00+00:00'  -> '2024-01-15T00:00:00'
    '2024-01-15T00:00:00'        -> '2024-01-15T00:00:00'
    """
    for suffix in ("+00:00", "+0000", "Z"):
        if iso_time.endswith(suffix):
            return iso_time[: -len(suffix)]
    return iso_time


def _iso_to_cdaweb_time(iso_time: str) -> str:
    """Convert ISO 8601 time string to CDAWeb REST API format.

    '2024-01-15T00:00:00Z' -> '20240115T000000Z'
    '2024-01-15T00:00:00+00:00' -> '20240115T000000Z'
    '2024-01-15T00:00' -> '20240115T000000Z'
    """
    # Strip UTC offset variants before compacting
    # "+00:00" / "+0000" → "Z"
    t = iso_time
    for suffix in ("+00:00", "+0000"):
        if t.endswith(suffix):
            t = t[: -len(suffix)] + "Z"
            break
    # Strip common ISO separators
    t = t.replace("-", "").replace(":", "")
    # Ensure trailing Z
    if not t.endswith("Z"):
        t += "Z"
    # Ensure time portion is exactly HHMMSS (6 digits)
    if "T" in t:
        date_part, time_z = t.split("T", 1)
        time_part = time_z.rstrip("Z")
        # Strip sub-second precision
        if "." in time_part:
            time_part = time_part.split(".", 1)[0]
        time_part = time_part[:6].ljust(6, "0")
        t = f"{date_part}T{time_part}Z"
    return t


def _url_to_local_path(url: str, cache_base: Path) -> Path:
    """Resolve a CDAWeb CDF URL to its local cache path.

    Args:
        url: Full URL to the CDF file.
        cache_base: Local directory for cached files.

    Returns:
        Path where the file would be cached locally.
    """
    parsed = urlparse(url)
    path = parsed.path  # e.g., /sp_phys/data/ace/mag/.../file.cdf

    marker = "sp_phys/data/"
    idx = path.find(marker)
    if idx >= 0:
        rel_path = path[idx + len(marker):]
    else:
        rel_path = Path(parsed.path).name

    return cache_base / rel_path


def _check_download_size(
    file_list: list[dict], cache_dir: Path
) -> tuple[int, int, int, int]:
    """Calculate download size, excluding cached files.

    Args:
        file_list: List of dicts with 'url' and 'size' keys from _get_cdf_file_list().
        cache_dir: Local cache directory.

    Returns:
        Tuple of (download_bytes, total_bytes, n_cached, n_to_download).
    """
    download_bytes = 0
    total_bytes = 0
    n_cached = 0
    n_to_download = 0

    for fi in file_list:
        size = fi.get("size", 0)
        total_bytes += size
        local_path = _url_to_local_path(fi["url"], cache_dir)
        if local_path.exists() and local_path.stat().st_size > 0:
            n_cached += 1
        else:
            download_bytes += size
            n_to_download += 1

    return download_bytes, total_bytes, n_cached, n_to_download


def _download_cdf_file(url: str, cache_base: Path) -> Path:
    """Download a CDF file, using local cache if available.

    Preserves CDAWeb directory structure under cache_base.

    Args:
        url: Full URL to the CDF file.
        cache_base: Local directory for cached files.

    Returns:
        Path to the local CDF file.
    """
    local_path = _url_to_local_path(url, cache_base)

    # Skip download if cached
    if local_path.exists() and local_path.stat().st_size > 0:
        get_event_bus().emit(CDF_CACHE_HIT, agent="cdf", msg=f"[CDF] Cache hit: {local_path}")
        return local_path

    # Download
    get_event_bus().emit(CDF_DOWNLOAD, agent="cdf", level="info", msg=f"[CDF] Downloading: {local_path.name}")
    local_path.parent.mkdir(parents=True, exist_ok=True)

    from data_ops.http_utils import request_with_retry
    resp = request_with_retry(url)

    import os
    tmp_path = local_path.with_suffix(".tmp")
    tmp_path.write_bytes(resp.content)
    os.replace(tmp_path, local_path)
    size_mb = len(resp.content) / (1024 * 1024)
    get_event_bus().emit(CDF_DOWNLOAD, agent="cdf", msg=f"[CDF] Downloaded {size_mb:.1f} MB -> {local_path}")

    return local_path


def _read_cdf_parameter(
    cdf_path: Path, parameter_id: str
) -> pd.DataFrame | xr.DataArray:
    """Extract one parameter from a CDF file.

    Args:
        cdf_path: Path to local CDF file.
        parameter_id: CDF variable name to read.

    Returns:
        DataFrame with DatetimeIndex for 1D/2D data, or xarray DataArray
        for 3D+ data.
    """
    cdf = cdflib.CDF(str(cdf_path))
    info = cdf.cdf_info()

    # Read the parameter data
    try:
        param_data = cdf.varget(parameter_id)
    except Exception as e:
        all_vars = info.zVariables + info.rVariables
        raise ValueError(
            f"Variable '{parameter_id}' not found in {cdf_path.name}. "
            f"Available: {all_vars}"
        ) from e

    # Find the correct epoch for this parameter (using DEPEND_0)
    epoch_result = _get_parameter_epoch(cdf, info, parameter_id)

    if epoch_result is None:
        # Non-time-varying parameter (e.g., energy bins, static metadata).
        # Return as DataFrame with RangeIndex instead of DatetimeIndex.
        if param_data.ndim == 1:
            df = pd.DataFrame({1: param_data})
            df.index.name = "index"
            return df
        elif param_data.ndim == 2:
            ncols = param_data.shape[1]
            df = pd.DataFrame({i + 1: param_data[:, i] for i in range(ncols)})
            df.index.name = "index"
            return df
        else:
            # 3D+ without time coordinate
            dims = [f"dim{i}" for i in range(param_data.ndim)]
            da = xr.DataArray(param_data, dims=dims)
            da.name = parameter_id
            return da

    epoch_var, epoch_data, times = epoch_result

    # Shape validation: catch mismatches early with an actionable message
    if param_data.shape[0] != len(times):
        # Try to get the DEPEND_0 value for the error message
        depend_0_str = "unknown"
        try:
            attrs = cdf.varattsget(parameter_id)
            d0 = attrs.get("DEPEND_0", None)
            if d0 is not None:
                if isinstance(d0, np.ndarray):
                    d0 = str(d0.flat[0]) if d0.size > 0 else None
                if d0:
                    depend_0_str = str(d0).strip()
        except Exception:
            pass
        raise ValueError(
            f"Shape mismatch for '{parameter_id}': data has {param_data.shape[0]} "
            f"records but epoch '{epoch_var}' has {len(times)} records "
            f"(DEPEND_0='{depend_0_str}'). "
            f"This parameter may use a different time axis."
        )

    # Build DataFrame with integer column names (1D/2D) or DataArray (3D+)
    if param_data.ndim == 1:
        # Scalar parameter
        df = pd.DataFrame({1: param_data}, index=times)
        df.index.name = "time"
        return df
    elif param_data.ndim == 2:
        # Vector/multi-component parameter
        ncols = param_data.shape[1]
        columns = {i + 1: param_data[:, i] for i in range(ncols)}
        df = pd.DataFrame(columns, index=times)
        df.index.name = "time"
        return df
    else:
        # 3D+ variable — return xarray DataArray
        dims = ["time"] + [f"dim{i}" for i in range(1, param_data.ndim)]
        coords = {"time": times}
        da = xr.DataArray(param_data, dims=dims, coords=coords)
        da.name = parameter_id
        return da


def _try_virtual_epoch(cdf: cdflib.CDF, attrs: dict, parameter_id: str):
    """Try to construct timestamps from DEPEND_TIME + DEPEND_EPOCH0.

    THEMIS CDFs use a virtual epoch pattern where:
    - DEPEND_TIME names a variable with Unix seconds (CDF_DOUBLE)
    - DEPEND_EPOCH0 names a scalar CDF_EPOCH reference (base time)

    The actual times = epoch0 + time_array (both in seconds).

    Returns (time_var_name, None, times_array) or None if not applicable.
    """
    depend_time = attrs.get("DEPEND_TIME", None)
    depend_epoch0 = attrs.get("DEPEND_EPOCH0", None)

    if not depend_time or not depend_epoch0:
        return None

    # Normalize attribute values
    for attr_val in (depend_time, depend_epoch0):
        if isinstance(attr_val, np.ndarray):
            attr_val = str(attr_val.flat[0]) if attr_val.size > 0 else None

    if isinstance(depend_time, np.ndarray):
        depend_time = str(depend_time.flat[0]) if depend_time.size > 0 else None
    if isinstance(depend_epoch0, np.ndarray):
        depend_epoch0 = str(depend_epoch0.flat[0]) if depend_epoch0.size > 0 else None

    if not depend_time or not depend_epoch0:
        return None

    try:
        time_data = cdf.varget(str(depend_time))
        epoch0_val = cdf.varget(str(depend_epoch0))

        # epoch0_val is CDF_EPOCH milliseconds since 0000-01-01
        # Convert to a datetime reference, then add time_data (Unix seconds)
        epoch0_dt = cdflib.cdfepoch.to_datetime(epoch0_val)
        # epoch0_dt is a single datetime64 or scalar
        if hasattr(epoch0_dt, '__len__') and len(epoch0_dt) > 0:
            epoch0_dt = epoch0_dt[0]

        epoch0_ns = np.datetime64(str(epoch0_dt), 'ns')

        # time_data is in seconds (Unix-like offset from epoch0)
        # Convert to nanosecond offsets and add to epoch0
        time_offsets_ns = (time_data * 1e9).astype('int64')
        times = epoch0_ns + time_offsets_ns.astype('timedelta64[ns]')

        get_event_bus().emit(DEBUG, agent="cdf", msg=f"[CDF] Virtual epoch for '{parameter_id}': DEPEND_TIME='{depend_time}' ({len(time_data)} records) + DEPEND_EPOCH0='{depend_epoch0}'")
        return (str(depend_time), None, times)
    except Exception as exc:
        get_event_bus().emit(DEBUG, agent="cdf", msg=f"[CDF] Virtual epoch failed for '{parameter_id}': {exc}")
        return None


def _get_parameter_epoch(cdf: cdflib.CDF, info, parameter_id: str):
    """Get the epoch variable for a specific parameter using DEPEND_0.

    Reads the DEPEND_0 variable attribute from the parameter to determine
    which epoch/time variable it depends on. This is critical for CDF files
    with multiple epoch variables (e.g., MMS EDI, THEMIS FIT).

    Args:
        cdf: Open CDF file object.
        info: CDF info from cdf.cdf_info().
        parameter_id: CDF variable name to find the epoch for.

    Returns:
        Tuple of (epoch_var_name, epoch_data, times) if time-varying,
        or None if the parameter is not time-varying (e.g., energy bins).
    """
    all_vars = info.zVariables + info.rVariables

    # Read DEPEND_0 from the parameter's variable attributes
    depend_0 = None
    try:
        attrs = cdf.varattsget(parameter_id)
        depend_0 = attrs.get("DEPEND_0", None)
    except Exception:
        pass

    # Normalize: numpy arrays → str, strip whitespace
    if depend_0 is not None:
        if isinstance(depend_0, np.ndarray):
            depend_0 = str(depend_0.flat[0]) if depend_0.size > 0 else None
        elif isinstance(depend_0, bytes):
            depend_0 = depend_0.decode("utf-8", errors="replace")
        if isinstance(depend_0, str):
            depend_0 = depend_0.strip()

    # Empty / "NONE" / missing → not time-varying
    if not depend_0 or depend_0.upper() == "NONE":
        # Check if parameter itself is an epoch type — if so, it IS time data
        try:
            var_inq = cdf.varinq(parameter_id)
            if var_inq.Data_Type_Description in _EPOCH_TYPES:
                return None
        except Exception:
            pass
        # Check if the parameter has records matching any known epoch
        # (some older CDFs lack DEPEND_0 but are still time-varying)
        try:
            param_data = cdf.varget(parameter_id)
            if param_data is not None and param_data.ndim >= 1:
                n_records = param_data.shape[0]
                # Try to find an epoch with matching length
                epoch_var = _find_epoch_variable(cdf, info)
                epoch_data = cdf.varget(epoch_var)
                if epoch_data is not None and len(epoch_data) == n_records:
                    times = cdflib.cdfepoch.to_datetime(epoch_data)
                    return (epoch_var, epoch_data, times)
                # No matching epoch — non-time-varying
                return None
        except (ValueError, Exception):
            return None

    # DEPEND_0 names a valid variable — use it
    if depend_0 in all_vars:
        try:
            epoch_data = cdf.varget(depend_0)
            times = cdflib.cdfepoch.to_datetime(epoch_data)
            return (depend_0, epoch_data, times)
        except Exception as exc:
            # DEPEND_0 variable exists but is empty/unreadable.
            # Check for THEMIS-style virtual epoch: DEPEND_TIME + DEPEND_EPOCH0.
            # In this pattern, timestamps are stored as seconds-since-epoch0
            # in a separate variable, not as a CDF epoch array.
            virtual_result = _try_virtual_epoch(cdf, attrs, parameter_id)
            if virtual_result is not None:
                return virtual_result
            get_event_bus().emit(DEBUG, agent="cdf", level="warning", msg=f"[CDF] DEPEND_0='{depend_0}' for '{parameter_id}' failed to read: {exc}. Falling back to _find_epoch_variable().")

    # DEPEND_0 points to non-existent variable — fall back
    if depend_0 and depend_0 not in all_vars:
        get_event_bus().emit(DEBUG, agent="cdf", level="warning", msg=f"[CDF] DEPEND_0='{depend_0}' for '{parameter_id}' not found in CDF. Falling back to _find_epoch_variable().")

    # Fallback: use the global epoch finder
    epoch_var = _find_epoch_variable(cdf, info)
    epoch_data = cdf.varget(epoch_var)
    times = cdflib.cdfepoch.to_datetime(epoch_data)
    return (epoch_var, epoch_data, times)


def _find_epoch_variable(cdf: cdflib.CDF, info) -> str:
    """Find the epoch/time variable in a CDF file.

    Looks for common epoch variable names, then falls back to checking
    variable types.

    Args:
        cdf: Open CDF file object.
        info: CDF info from cdf.cdf_info().

    Returns:
        Name of the epoch variable.

    Raises:
        ValueError: If no epoch variable is found.
    """
    all_vars = info.zVariables + info.rVariables

    # Check common names first
    for name in ["Epoch", "EPOCH", "epoch", "Epoch1"]:
        if name in all_vars:
            return name

    # Fall back: look for CDF epoch data types
    for var_name in all_vars:
        try:
            var_info = cdf.varinq(var_name)
            # CDF epoch types: CDF_EPOCH (31), CDF_EPOCH16 (32), CDF_TIME_TT2000 (33)
            if var_info.Data_Type_Description in (
                "CDF_EPOCH", "CDF_EPOCH16", "CDF_TIME_TT2000"
            ):
                return var_name
        except Exception:
            continue

    raise ValueError(
        f"No epoch variable found in CDF file. Variables: {all_vars}"
    )
