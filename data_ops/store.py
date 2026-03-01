"""
Disk-backed data store.

DataEntry holds a single dataset as a pandas DataFrame or xarray DataArray.
Timeseries entries have DatetimeIndex; general-data entries may have numeric/string indices.
DataStore is a disk-backed container keyed by label strings, with an LRU in-memory cache.

Disk layout:
    {data_dir}/
        _labels.json          # label → hash folder name
        {hash}/
            data.pkl          # pickled DataFrame (or data.nc for xarray)
            meta.json         # label, units, description, stats, etc.
            .computing        # flag: present while async stats are being computed
"""

import collections
import hashlib
import json
import logging
import os
import shutil
import threading
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("xhelio")


@dataclass
class DataEntry:
    """A single dataset stored in memory.

    Attributes:
        label: Unique identifier (e.g., "AC_H2_MFI.BGSEc" or "Bmag").
        data: DataFrame with DatetimeIndex (1D/2D) or xarray DataArray (3D+).
        units: Physical units string (e.g., "nT").
        description: Human-readable description.
        source: Origin — "cdf" for fetched data, "computed" for derived data.
        is_timeseries: True if this entry has a DatetimeIndex (default).
            False for general-data entries (event catalogs, scatter data, etc.).
    """

    label: str
    data: pd.DataFrame | xr.DataArray
    units: str = ""
    description: str = ""
    source: str = "computed"
    metadata: dict | None = None
    is_timeseries: bool = True

    @property
    def is_xarray(self) -> bool:
        """True if this entry stores an xarray DataArray (3D+ data)."""
        return isinstance(self.data, xr.DataArray)

    @property
    def columns(self) -> list:
        """Column names (for DataFrames). Empty list for xarray."""
        if self.is_xarray:
            return []
        return list(self.data.columns)

    @property
    def time(self) -> np.ndarray:
        """Backward compat: numpy datetime64[ns] array."""
        if self.is_xarray:
            return self.data.coords["time"].values
        return self.data.index.values

    @property
    def values(self) -> np.ndarray:
        """Backward compat: numpy float64 array — (n,) for scalar, (n,k) for vector."""
        if self.is_xarray:
            return self.data.values
        v = self.data.values
        if v.shape[1] == 1:
            return v.squeeze(axis=1)
        return v

    def summary(self) -> dict:
        """Return a compact summary dict suitable for LLM responses."""
        if self.is_xarray:
            return self._summary_xarray()
        return self._summary_dataframe()

    def _summary_dataframe(self) -> dict:
        """Summary for DataFrame-backed entries."""
        n = len(self.data)
        ncols = len(self.data.columns)
        if self.metadata and self.metadata.get("type") == "spectrogram":
            shape_desc = f"spectrogram[{ncols} bins]"
        else:
            shape_desc = "scalar" if ncols == 1 else f"vector[{ncols}]"
        result = {
            "label": self.label,
            "columns": list(self.data.columns),
            "num_points": n,
            "shape": shape_desc,
            "units": self.units,
            "description": self.description,
            "source": self.source,
            "is_timeseries": self.is_timeseries,
        }
        if n > 0:
            if self.is_timeseries:
                result["time_min"] = str(self.data.index[0])
                result["time_max"] = str(self.data.index[-1])
            else:
                result["index_min"] = str(self.data.index[0])
                result["index_max"] = str(self.data.index[-1])
        else:
            if self.is_timeseries:
                result["time_min"] = None
                result["time_max"] = None
            else:
                result["index_min"] = None
                result["index_max"] = None
        # Cadence
        if n > 1 and self.is_timeseries:
            dt = self.data.index.to_series().diff().dropna()
            result["median_cadence"] = str(dt.median())

        # NaN stats
        nan_count = int(self.data.isna().sum().sum())
        result["nan_count"] = nan_count
        result["nan_percentage"] = round(nan_count / (n * ncols) * 100, 1) if n > 0 else 0

        # Per-column statistics (numeric only, keep it compact)
        if n > 0:
            stats = {}
            for col in self.data.columns:
                if self.data[col].dtype.kind in ("f", "i", "u"):
                    s = self.data[col].dropna()
                    if len(s) > 0:
                        stats[col] = {
                            "min": round(float(s.min()), 4),
                            "max": round(float(s.max()), 4),
                            "mean": round(float(s.mean()), 4),
                        }
            if stats:
                result["statistics"] = stats

        # Per-entry memory usage
        result["memory_bytes"] = int(self.data.memory_usage(deep=True).sum())

        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def _summary_xarray(self) -> dict:
        """Summary for xarray DataArray-backed entries."""
        da = self.data
        dims = dict(da.sizes)
        n_time = dims.get("time", 0)
        dim_desc = " x ".join(f"{k}={v}" for k, v in dims.items())
        result = {
            "label": self.label,
            "shape": f"ndarray[{dim_desc}]",
            "dims": dims,
            "num_points": n_time,
            "units": self.units,
            "time_min": str(da.coords["time"].values[0]) if n_time > 0 else None,
            "time_max": str(da.coords["time"].values[-1]) if n_time > 0 else None,
            "description": self.description,
            "source": self.source,
            "storage_type": "xarray",
        }

        # Cadence
        if n_time > 1:
            times = da.coords["time"].values
            dt = pd.Series(times).diff().dropna()
            result["median_cadence"] = str(dt.median())

        # NaN stats
        flat = da.values.flatten()
        finite = flat[np.isfinite(flat)]
        nan_count = int(flat.size - finite.size)
        result["nan_count"] = nan_count
        result["nan_percentage"] = round(nan_count / flat.size * 100, 1) if flat.size > 0 else 0

        # Global statistics
        if finite.size > 0:
            result["statistics"] = {
                "min": round(float(np.min(finite)), 4),
                "max": round(float(np.max(finite)), 4),
                "mean": round(float(np.mean(finite)), 4),
            }

        # Per-entry memory usage
        result["memory_bytes"] = int(da.nbytes)

        if self.metadata:
            result["metadata"] = self.metadata
        return result


# ---------------------------------------------------------------------------
# Disk-backed DataStore
# ---------------------------------------------------------------------------

_MAX_CACHE_ENTRIES = 50
_STATS_TIMEOUT = 10.0  # seconds to wait for .computing flag


def _label_hash(label: str) -> str:
    """Compute a short hash for a label (8 hex chars)."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()[:8]


class DataStore:
    """Disk-backed store mapping labels to DataEntry objects.

    All data lives on the filesystem under ``data_dir``. An LRU in-memory
    cache avoids redundant disk reads.  Thread-safe for concurrent access.

    Args:
        data_dir: Directory for persisting data.  Created if it doesn't exist.
    """

    def __init__(self, data_dir: Path):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        # label → hash folder name
        self._labels: dict[str, str] = {}

        # LRU cache: label → DataEntry (OrderedDict for move_to_end)
        self._cache: collections.OrderedDict[str, DataEntry] = collections.OrderedDict()

        # Load existing _labels.json
        labels_path = self._data_dir / "_labels.json"
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                self._labels = json.load(f)

    # ---- Public API ----

    def put(self, entry: DataEntry) -> None:
        """Store a DataEntry, writing data to disk immediately.

        Stats are computed asynchronously — a ``.computing`` flag file
        is present until stats are ready.
        """
        with self._lock:
            h = self._allocate_hash(entry.label)
            entry_dir = self._data_dir / h
            entry_dir.mkdir(parents=True, exist_ok=True)

            # Write .computing flag
            computing_flag = entry_dir / ".computing"
            computing_flag.touch()

            # Write data file (atomic via tmp + rename)
            if entry.is_xarray:
                fmt = "netcdf"
                data_file = entry_dir / "data.nc"
                tmp_file = entry_dir / "data.nc.tmp"
                encoding = {}
                if "time" in entry.data.dims:
                    encoding["time"] = {
                        "units": "seconds since 1970-01-01",
                        "dtype": "float64",
                    }
                entry.data.to_netcdf(str(tmp_file), encoding=encoding)
                os.replace(tmp_file, data_file)
            else:
                fmt = "pickle"
                data_file = entry_dir / "data.pkl"
                tmp_file = entry_dir / "data.pkl.tmp"
                entry.data.to_pickle(str(tmp_file))
                os.replace(tmp_file, data_file)

            # Write minimal meta.json
            meta = {
                "label": entry.label,
                "format": fmt,
                "units": entry.units,
                "description": entry.description,
                "source": entry.source,
                "is_timeseries": entry.is_timeseries,
            }
            if not entry.is_xarray:
                meta["columns"] = list(entry.data.columns)
            if entry.is_xarray:
                meta["memory_bytes"] = int(entry.data.nbytes)
            else:
                meta["memory_bytes"] = int(entry.data.memory_usage(deep=True).sum())
            if entry.metadata is not None:
                meta["metadata"] = entry.metadata
            self._write_meta(entry_dir, meta)

            # Update label index
            self._labels[entry.label] = h
            self._write_labels()

            # Update LRU cache
            self._cache[entry.label] = entry
            self._cache.move_to_end(entry.label)
            while len(self._cache) > _MAX_CACHE_ENTRIES:
                self._cache.popitem(last=False)

        # Compute full stats asynchronously
        t = threading.Thread(
            target=self._compute_stats_async,
            args=(entry.label, h),
            daemon=True,
        )
        t.start()

    def get(self, label: str) -> Optional[DataEntry]:
        """Retrieve a DataEntry by label, or None if not found."""
        with self._lock:
            # Cache hit
            if label in self._cache:
                self._cache.move_to_end(label)
                return self._cache[label]

            # Check label index
            h = self._labels.get(label)
            if h is None:
                return None

            # Load from disk
            entry = self._load_entry(label, h)
            if entry is None:
                return None

            # Add to cache
            self._cache[label] = entry
            self._cache.move_to_end(label)
            while len(self._cache) > _MAX_CACHE_ENTRIES:
                self._cache.popitem(last=False)

            return entry

    def has(self, label: str) -> bool:
        """Check if a label exists in the store."""
        with self._lock:
            return label in self._labels

    def remove(self, label: str) -> bool:
        """Remove an entry by label. Returns True if it existed."""
        with self._lock:
            h = self._labels.pop(label, None)
            if h is None:
                return False

            # Remove from cache
            self._cache.pop(label, None)

            # Delete the folder (ignore_errors for async stats thread race)
            entry_dir = self._data_dir / h
            if entry_dir.exists():
                shutil.rmtree(entry_dir, ignore_errors=True)

            # Rewrite labels index
            self._write_labels()
            return True

    def list_entries(self) -> list[dict]:
        """Return summary dicts for all stored entries.

        If a ``.computing`` flag is present for an entry, waits briefly
        for stats to finish, then reads the final ``meta.json``.
        """
        with self._lock:
            labels_snapshot = dict(self._labels)

        results = []
        for label, h in labels_snapshot.items():
            entry_dir = self._data_dir / h

            # Wait for .computing flag if present
            computing_flag = entry_dir / ".computing"
            if computing_flag.exists():
                deadline = _time.monotonic() + _STATS_TIMEOUT
                while computing_flag.exists() and _time.monotonic() < deadline:
                    _time.sleep(0.05)

            with self._lock:
                # Check cache first
                if label in self._cache:
                    self._cache.move_to_end(label)
                    results.append(self._cache[label].summary())
                    continue

            # Read meta.json for lightweight summary (no data load)
            meta = self._read_meta(entry_dir)
            if meta is None:
                continue
            results.append(self._meta_to_summary(meta))

        return results

    def clear(self) -> None:
        """Remove all entries and their disk files."""
        with self._lock:
            self._cache.clear()
            self._labels.clear()

            # Delete all hash folders (ignore_errors for async stats thread race)
            for item in self._data_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)

            # Remove _labels.json
            labels_path = self._data_dir / "_labels.json"
            if labels_path.exists():
                labels_path.unlink()

    def memory_usage_bytes(self) -> int:
        """Return approximate total memory usage of cached entries only."""
        with self._lock:
            total = 0
            for entry in self._cache.values():
                if entry.is_xarray:
                    total += entry.data.nbytes
                else:
                    total += int(entry.data.memory_usage(deep=True).sum())
            return total

    def __len__(self) -> int:
        with self._lock:
            return len(self._labels)

    # ---- Internal helpers ----

    def _allocate_hash(self, label: str) -> str:
        """Allocate a hash folder name for a label, handling collisions.

        If the label already has a hash, returns it.  If the candidate hash
        is taken by a different label, appends ``_2``, ``_3``, etc.
        """
        # Already allocated?
        existing = self._labels.get(label)
        if existing is not None:
            return existing

        candidate = _label_hash(label)
        # Check if candidate is used by a different label
        used_hashes = set(self._labels.values())
        if candidate not in used_hashes:
            return candidate

        # Collision — find a free slot
        n = 2
        while f"{candidate}_{n}" in used_hashes:
            n += 1
        return f"{candidate}_{n}"

    def _write_labels(self) -> None:
        """Atomically write _labels.json."""
        tmp = self._data_dir / "_labels.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._labels, f, indent=2)
        os.replace(tmp, self._data_dir / "_labels.json")

    def _write_meta(self, entry_dir: Path, meta: dict) -> None:
        """Atomically write meta.json."""
        tmp = entry_dir / "meta.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        os.replace(tmp, entry_dir / "meta.json")

    def _read_meta(self, entry_dir: Path) -> Optional[dict]:
        """Read meta.json, returning None if missing or corrupt."""
        meta_path = entry_dir / "meta.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _load_entry(self, label: str, h: str) -> Optional[DataEntry]:
        """Load a DataEntry from its hash folder."""
        entry_dir = self._data_dir / h
        meta = self._read_meta(entry_dir)
        if meta is None:
            return None

        fmt = meta.get("format", "pickle")
        if fmt == "netcdf":
            data_path = entry_dir / "data.nc"
            if not data_path.exists():
                return None
            data = xr.open_dataarray(data_path).load()
        else:
            data_path = entry_dir / "data.pkl"
            if not data_path.exists():
                return None
            data = pd.read_pickle(data_path)

        return DataEntry(
            label=label,
            data=data,
            units=meta.get("units", ""),
            description=meta.get("description", ""),
            source=meta.get("source", "computed"),
            metadata=meta.get("metadata"),
            is_timeseries=meta.get("is_timeseries", True),
        )

    def _compute_stats_async(self, label: str, h: str) -> None:
        """Compute full statistics and update meta.json, then remove .computing flag."""
        entry_dir = self._data_dir / h
        try:
            entry = self.get(label)
            if entry is None:
                return
            meta = self._read_meta(entry_dir)
            if meta is None:
                return
            meta.update(self._build_stats(entry))
            self._write_meta(entry_dir, meta)
        except Exception as exc:
            import logging
            logging.getLogger("xhelio").debug("Async stats computation failed: %s", exc)
        finally:
            flag = entry_dir / ".computing"
            if flag.exists():
                try:
                    flag.unlink()
                except OSError:
                    pass

    def _build_stats(self, entry: DataEntry) -> dict:
        """Extract statistics from a DataEntry for persisting in meta.json."""
        stats: dict = {}
        if entry.is_xarray:
            da = entry.data
            dims = dict(da.sizes)
            n_time = dims.get("time", 0)
            stats["dims"] = dims
            stats["num_points"] = n_time
            stats["shape"] = f"ndarray[{' x '.join(f'{k}={v}' for k, v in dims.items())}]"
            stats["storage_type"] = "xarray"
            if n_time > 0:
                stats["time_min"] = str(da.coords["time"].values[0])
                stats["time_max"] = str(da.coords["time"].values[-1])
            if n_time > 1:
                times = da.coords["time"].values
                dt = pd.Series(times).diff().dropna()
                stats["median_cadence"] = str(dt.median())
            flat = da.values.flatten()
            finite = flat[np.isfinite(flat)]
            nan_count = int(flat.size - finite.size)
            stats["nan_count"] = nan_count
            stats["nan_percentage"] = round(nan_count / flat.size * 100, 1) if flat.size > 0 else 0
            if finite.size > 0:
                stats["statistics"] = {
                    "min": round(float(np.min(finite)), 4),
                    "max": round(float(np.max(finite)), 4),
                    "mean": round(float(np.mean(finite)), 4),
                }
        else:
            df = entry.data
            n = len(df)
            ncols = len(df.columns)
            stats["num_points"] = n
            if entry.metadata and entry.metadata.get("type") == "spectrogram":
                stats["shape"] = f"spectrogram[{ncols} bins]"
            else:
                stats["shape"] = "scalar" if ncols == 1 else f"vector[{ncols}]"
            if n > 0:
                if entry.is_timeseries:
                    stats["time_min"] = str(df.index[0])
                    stats["time_max"] = str(df.index[-1])
                else:
                    stats["index_min"] = str(df.index[0])
                    stats["index_max"] = str(df.index[-1])
            if n > 1 and entry.is_timeseries:
                dt = df.index.to_series().diff().dropna()
                stats["median_cadence"] = str(dt.median())
            nan_count = int(df.isna().sum().sum())
            stats["nan_count"] = nan_count
            stats["nan_percentage"] = round(nan_count / (n * ncols) * 100, 1) if n > 0 else 0
            if n > 0:
                col_stats = {}
                for col in df.columns:
                    if df[col].dtype.kind in ("f", "i", "u"):
                        s = df[col].dropna()
                        if len(s) > 0:
                            col_stats[col] = {
                                "min": round(float(s.min()), 4),
                                "max": round(float(s.max()), 4),
                                "mean": round(float(s.mean()), 4),
                            }
                if col_stats:
                    stats["statistics"] = col_stats
        return stats

    def _meta_to_summary(self, meta: dict) -> dict:
        """Convert a meta.json dict to a summary dict matching DataEntry.summary() format."""
        result: dict = {
            "label": meta.get("label", ""),
            "units": meta.get("units", ""),
            "description": meta.get("description", ""),
            "source": meta.get("source", "computed"),
            "is_timeseries": meta.get("is_timeseries", True),
        }

        # Stats fields (present after async computation)
        for key in ("num_points", "shape", "nan_count", "nan_percentage",
                     "time_min", "time_max", "index_min", "index_max",
                     "median_cadence", "statistics", "dims", "storage_type",
                     "columns"):
            if key in meta:
                result[key] = meta[key]

        # Defaults if stats haven't been computed yet
        if "num_points" not in result:
            result["num_points"] = 0
        if "shape" not in result:
            cols = meta.get("columns")
            if cols is not None:
                ncols = len(cols)
                result["shape"] = "scalar" if ncols == 1 else f"vector[{ncols}]"
            else:
                result["shape"] = "unknown (computing)"

        # Estimate memory from disk file size (not loaded into RAM)
        result["memory_bytes"] = meta.get("memory_bytes", 0)

        if meta.get("metadata"):
            result["metadata"] = meta["metadata"]

        return result


# ---------------------------------------------------------------------------
# Module-level global store (replaces ContextVar)
# ---------------------------------------------------------------------------

_global_store: Optional[DataStore] = None
_store_lock = threading.Lock()


def get_store() -> DataStore:
    """Return the global DataStore. Raises RuntimeError if none has been set."""
    global _global_store
    if _global_store is None:
        raise RuntimeError("No DataStore has been set. Call set_store() first.")
    return _global_store


def set_store(store: DataStore) -> None:
    """Set the global DataStore."""
    global _global_store
    _global_store = store


def reset_store() -> None:
    """Reset the global DataStore (mainly for testing)."""
    global _global_store
    _global_store = None


# ---------------------------------------------------------------------------
# Utility functions (unchanged API)
# ---------------------------------------------------------------------------


def resolve_entry(
    store: DataStore, label: str
) -> tuple[DataEntry | None, str | None]:
    """Resolve a label to a DataEntry, supporting column sub-selection.

    Handles compound labels like 'PSP_B_DERIVATIVE_FINAL.B_mag' where
    'PSP_B_DERIVATIVE_FINAL' is the store key and 'B_mag' is a column.
    Also handles dedup suffixes (.1, .2) since CDF vectors use integer columns.

    Args:
        store: DataStore to look up entries.
        label: The label to resolve (may be compound).

    Returns:
        Tuple of (DataEntry, resolved_label) or (None, None) if not found.
    """
    # Exact match first
    entry = store.get(label)
    if entry is not None:
        return entry, label

    # Try column sub-selection: split from the right and check
    # progressively longer prefixes as parent labels.
    # E.g. "A.B.C" tries "A.B" with col "C", then "A" with col "B.C"
    parts = label.split(".")
    for i in range(len(parts) - 1, 0, -1):
        parent_label = ".".join(parts[:i])
        col_name = ".".join(parts[i:])
        parent = store.get(parent_label)
        if parent is not None:
            # Try string match first, then int (CDF vectors have integer column names)
            if col_name not in parent.columns:
                try:
                    col_name = int(col_name)
                except (ValueError, TypeError):
                    continue
                if col_name not in parent.columns:
                    continue
            sub_entry = DataEntry(
                label=label,
                data=parent.data[[col_name]],
                units=parent.units,
                description=f"{parent.description} [{col_name}]" if parent.description else str(col_name),
                source=parent.source,
                metadata=parent.metadata,
            )
            return sub_entry, label
    return None, None


def build_source_map(
    store: DataStore, labels: list[str]
) -> tuple[dict[str, pd.DataFrame | xr.DataArray] | None, str | None]:
    """Build a mapping of sandbox variable names to data from store labels.

    Each label becomes ``df_<SUFFIX>`` (DataFrame) or ``da_<SUFFIX>``
    (xarray DataArray) where SUFFIX is the part after the last '.' in
    the label.  If the label has no '.', the full label is used as suffix.

    Args:
        store: DataStore to look up entries.
        labels: List of store labels.

    Returns:
        Tuple of (source_map, error_string).  On success error_string is None.
        On failure source_map is None.
    """
    source_map: dict[str, pd.DataFrame | xr.DataArray] = {}
    for label in labels:
        entry = store.get(label)
        if entry is None:
            return None, f"Label '{label}' not found in store"
        suffix = label.rsplit(".", 1)[-1]
        prefix = "da" if entry.is_xarray else "df"
        var_name = f"{prefix}_{suffix}"
        if var_name in source_map:
            return None, (
                f"Duplicate sandbox variable '{var_name}' — labels "
                f"'{label}' and another share suffix '{suffix}'. "
                f"Use labels with distinct suffixes."
            )
        source_map[var_name] = entry.data
    return source_map, None


def describe_sources(store: DataStore, labels: list[str]) -> dict:
    """Return lightweight summaries for a list of store labels.

    Keyed by sandbox variable name (``df_SUFFIX`` or ``da_SUFFIX``).
    Reuses ``entry.summary()`` to avoid duplicating stat computation.

    Args:
        store: DataStore to look up entries.
        labels: List of store labels.

    Returns:
        Dict keyed by sandbox variable name (``df_SUFFIX`` or ``da_SUFFIX``),
        each containing label, shape info, points, cadence, nan_pct, and time_range.
    """
    result = {}
    for label in labels:
        entry = store.get(label)
        if entry is None:
            continue
        suffix = label.rsplit(".", 1)[-1]
        prefix = "da" if entry.is_xarray else "df"
        var_name = f"{prefix}_{suffix}"

        s = entry.summary()
        info = {
            "label": s["label"],
            "columns": s.get("columns") or (list(s["dims"].keys()) if "dims" in s else []),
            "points": s["num_points"],
            "nan_pct": s.get("nan_percentage", 0),
        }
        if entry.is_xarray:
            info["dims"] = s.get("dims", {})
            info["shape"] = list(entry.data.shape)
            info["storage_type"] = "xarray"
        if s.get("median_cadence"):
            info["cadence"] = s["median_cadence"]
        if s.get("time_min") and s.get("time_max"):
            info["time_range"] = [s["time_min"][:10], s["time_max"][:10]]
        elif s.get("index_min") and s.get("index_max"):
            info["index_range"] = [str(s["index_min"]), str(s["index_max"])]

        result[var_name] = info
    return result
