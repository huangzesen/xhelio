"""
Master CDF skeleton file reader for CDAWeb parameter metadata.

Downloads Master CDF files from CDAWeb's static HTTP server and extracts
parameter metadata (names, types, units, fill values, sizes).

Master CDFs are lightweight skeleton files (~10-100 KB) that contain variable
definitions and attributes but no actual data. They serve as the authoritative
source for parameter metadata.

URL pattern: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0MASTERS/{id_lower}_00000000_v01.cdf
"""

from pathlib import Path

import numpy as np

from agent.event_bus import get_event_bus, CDF_METADATA_SYNC, DEBUG

try:
    import cdflib
except ImportError:
    cdflib = None

try:
    import requests
except ImportError:
    requests = None


from config import get_data_dir

MASTER_CDF_BASE = "https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0MASTERS"
MASTER_CDF_CACHE = get_data_dir() / "master_cdfs"

# CDF type string -> parameter type mapping
_CDF_TYPE_MAP = {
    "CDF_REAL4": "double",
    "CDF_REAL8": "double",
    "CDF_DOUBLE": "double",
    "CDF_FLOAT": "double",
    "CDF_INT1": "integer",
    "CDF_INT2": "integer",
    "CDF_INT4": "integer",
    "CDF_INT8": "integer",
    "CDF_UINT1": "integer",
    "CDF_UINT2": "integer",
    "CDF_UINT4": "integer",
    "CDF_BYTE": "integer",
}

# CDF types to skip (epoch/time and character/metadata)
_SKIP_TYPES = {
    "CDF_EPOCH", "CDF_EPOCH16", "CDF_TIME_TT2000",
    "CDF_CHAR", "CDF_UCHAR",
}


def get_master_cdf_url(dataset_id: str) -> str:
    """Construct the URL for a Master CDF skeleton file.

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "AC_H2_MFI").

    Returns:
        URL string for the Master CDF file.
    """
    return f"{MASTER_CDF_BASE}/{dataset_id.lower()}_00000000_v01.cdf"


def download_master_cdf(dataset_id: str, cache_dir: Path | None = None) -> Path:
    """Download a Master CDF file, using local cache if available.

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "AC_H2_MFI").
        cache_dir: Override cache directory (default: ~/.xhelio/master_cdfs/).

    Returns:
        Path to the local Master CDF file.

    Raises:
        RuntimeError: If requests or cdflib is not available.
        requests.HTTPError: If the download fails.
    """
    if requests is None:
        raise RuntimeError("'requests' package required for Master CDF download")

    if cache_dir is None:
        cache_dir = MASTER_CDF_CACHE

    filename = f"{dataset_id.lower()}_00000000_v01.cdf"
    local_path = cache_dir / filename

    # Return cached file if it exists and is non-empty
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    url = get_master_cdf_url(dataset_id)
    get_event_bus().emit(CDF_METADATA_SYNC, agent="MasterCDF", msg=f"Downloading Master CDF: {url}")

    from data_ops.http_utils import request_with_retry
    resp = request_with_retry(url)

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(resp.content)

    size_kb = len(resp.content) / 1024
    get_event_bus().emit(CDF_METADATA_SYNC, agent="MasterCDF", msg=f"Master CDF downloaded: {filename} ({size_kb:.1f} KB)")

    return local_path


def extract_metadata(cdf_path: Path) -> dict:
    """Extract parameter metadata from a Master CDF file.

    Reads all zVariables and rVariables from the CDF, filters to data
    variables (VAR_TYPE == "data"), and produces a structured dict.

    Args:
        cdf_path: Path to a local Master CDF file.

    Returns:
        Dict with "parameters" list, "startDate", "stopDate".
    """
    if cdflib is None:
        raise RuntimeError("'cdflib' package required for Master CDF reading")

    cdf = cdflib.CDF(str(cdf_path))
    cdf_info = cdf.cdf_info()

    # Start with synthetic Time parameter
    parameters = [
        {"name": "Time", "type": "isotime", "units": "UTC", "fill": None}
    ]

    all_vars = list(cdf_info.zVariables) + list(cdf_info.rVariables)

    for var_name in all_vars:
        try:
            var_inq = cdf.varinq(var_name)
        except Exception:
            continue

        # Skip epoch/time and character types
        dtype_desc = var_inq.Data_Type_Description
        if dtype_desc in _SKIP_TYPES:
            continue

        # Map CDF type to parameter type
        param_type = _CDF_TYPE_MAP.get(dtype_desc)
        if param_type is None:
            continue

        # Check VAR_TYPE — skip support/metadata variables
        # Accept "data" and "ignore_data" (the latter is used for real
        # multidimensional variables whose master CDF VAR_TYPE differs
        # from what older/newer data CDF versions use).
        try:
            attrs = cdf.varattsget(var_name)
            var_type = attrs.get("VAR_TYPE", "")
            if isinstance(var_type, np.ndarray):
                var_type = str(var_type)
            if var_type and var_type.lower() not in ("data", "ignore_data"):
                continue
        except Exception:
            pass  # If no attributes, include the variable

        # Extract metadata from variable attributes
        try:
            attrs = cdf.varattsget(var_name)
        except Exception:
            attrs = {}

        description = _get_str_attr(attrs, "CATDESC") or _get_str_attr(attrs, "FIELDNAM") or ""
        units = _get_str_attr(attrs, "UNITS") or ""

        # Fill value
        fill = None
        raw_fill = attrs.get("FILLVAL", None)
        if raw_fill is not None:
            try:
                fill = str(float(raw_fill))
            except (ValueError, TypeError):
                pass

        # Determine size from Dim_Sizes
        dim_sizes = var_inq.Dim_Sizes
        if isinstance(dim_sizes, (list, np.ndarray)) and len(dim_sizes) > 0:
            size = [int(d) for d in dim_sizes]
            # Squeeze leading 1-dimensions
            # e.g., [1, 25] -> [25], but keep [1] as scalar
            while len(size) > 1 and size[0] == 1:
                size = size[1:]
        else:
            size = [1]

        param = {
            "name": var_name,
            "type": param_type,
            "units": units,
            "description": description,
            "fill": fill,
        }

        # Only include size if not scalar
        if size != [1]:
            param["size"] = size

        parameters.append(param)

    return {
        "parameters": parameters,
        "startDate": "",
        "stopDate": "",
    }


def fetch_dataset_metadata_from_master(
    dataset_id: str,
    start_date: str = "",
    stop_date: str = "",
    cache_dir: Path | None = None,
) -> dict | None:
    """High-level: download Master CDF + extract parameter metadata.

    Args:
        dataset_id: CDAWeb dataset ID.
        start_date: Optional start date to inject into the result.
        stop_date: Optional stop date to inject into the result.
        cache_dir: Override cache directory for Master CDFs.

    Returns:
        Info dict, or None on failure.
    """
    try:
        cdf_path = download_master_cdf(dataset_id, cache_dir=cache_dir)
        info = extract_metadata(cdf_path)
        if start_date:
            info["startDate"] = start_date
        if stop_date:
            info["stopDate"] = stop_date
        return info
    except Exception as e:
        get_event_bus().emit(DEBUG, agent="MasterCDF", msg=f"Master CDF failed for {dataset_id}: {e}")
        return None


def _get_str_attr(attrs: dict, key: str) -> str:
    """Extract a string attribute from a CDF variable's attributes dict.

    Handles numpy arrays and bytes that cdflib sometimes returns.
    """
    val = attrs.get(key, "")
    if val is None:
        return ""
    if isinstance(val, np.ndarray):
        val = str(val.flat[0]) if val.size > 0 else ""
    if isinstance(val, bytes):
        val = val.decode("utf-8", errors="replace")
    if isinstance(val, (int, float)):
        return str(val)
    return str(val).strip() if val else ""
