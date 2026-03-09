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
import os
import shutil
import threading
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from agent.logging import get_logger

logger = get_logger()


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
        id: Unique hash-based identifier (e.g., "a3f7c2e1_1").
        time_range: Tuple of (start_iso, end_iso) or None.
        physical_quantity: What the data measures (e.g., "magnetic_field").
        array_shape: Data structure description (e.g., "scalar", "vector[3]").
        comment: Agent-written note about the data.
    """

    label: str
    data: pd.DataFrame | xr.DataArray | dict | str | bytes | None = None
    units: str = ""
    description: str = ""
    source: str = "computed"
    metadata: dict | None = None
    is_timeseries: bool = True
    id: str = ""
    time_range: tuple | None = None
    physical_quantity: str = ""
    array_shape: str = ""
    comment: str = ""

    # Lazy-loading support (not part of public API)
    _data_path: Path | None = field(default=None, init=False, repr=False)
    _data_format: str = field(default="", init=False, repr=False)
    _data_loaded: bool = field(default=True, init=False, repr=False)
    _data_loader: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.data is not None:
            self._data_loaded = True

    def __getattribute__(self, name):
        if name == "data":
            loaded = object.__getattribute__(self, "_data_loaded")
            if not loaded:
                loader = object.__getattribute__(self, "_data_loader")
                path = object.__getattribute__(self, "_data_path")
                fmt = object.__getattribute__(self, "_data_format")
                if loader and path:
                    data = loader(path, fmt)
                    object.__setattr__(self, "data", data)
                    object.__setattr__(self, "_data_loaded", True)
        return object.__getattribute__(self, name)

    @property
    def is_xarray(self) -> bool:
        """True if this entry stores an xarray DataArray (3D+ data)."""
        return isinstance(self.data, xr.DataArray)

    @property
    def data_type(self) -> str:
        """Return a string tag for the type of data stored."""
        d = self.data
        if isinstance(d, pd.DataFrame):
            return "dataframe"
        if isinstance(d, xr.DataArray):
            return "xarray"
        if isinstance(d, dict):
            return "dict"
        if isinstance(d, str):
            return "text"
        if isinstance(d, bytes):
            return "bytes"
        if d is None:
            return "none"
        return "unknown"

    @property
    def columns(self) -> list:
        """Column names (for DataFrames). Empty list for xarray."""
        if self.is_xarray:
            return []
        d = self.data
        if isinstance(d, pd.DataFrame):
            return list(d.columns)
        return []

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
        if isinstance(self.data, pd.DataFrame):
            return self._summary_dataframe()
        return self._summary_generic()

    def _summary_dataframe(self) -> dict:
        """Summary for DataFrame-backed entries."""
        n = len(self.data)
        ncols = len(self.data.columns)
        if self.metadata and self.metadata.get("type") == "spectrogram":
            shape_desc = f"spectrogram[{ncols} bins]"
        else:
            shape_desc = "scalar" if ncols == 1 else f"vector[{ncols}]"
        result = {
            "id": self.id,
            "label": self.label,
            "columns": list(self.data.columns),
            "num_points": n,
            "shape": shape_desc,
            "units": self.units,
            "description": self.description,
            "source": self.source,
            "is_timeseries": self.is_timeseries,
            "time_range": self.time_range,
            "physical_quantity": self.physical_quantity,
            "array_shape": self.array_shape,
            "comment": self.comment,
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
        result["nan_percentage"] = (
            round(nan_count / (n * ncols) * 100, 1) if n > 0 else 0
        )

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
            "id": self.id,
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
            "time_range": self.time_range,
            "physical_quantity": self.physical_quantity,
            "array_shape": self.array_shape,
            "comment": self.comment,
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
        result["nan_percentage"] = (
            round(nan_count / flat.size * 100, 1) if flat.size > 0 else 0
        )

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

    def _summary_generic(self) -> dict:
        """Summary for non-DataFrame/non-xarray entries (dict, str, bytes, etc.)."""
        result = {
            "id": self.id,
            "label": self.label,
            "data_type": self.data_type,
            "units": self.units,
            "description": self.description,
            "source": self.source,
            "time_range": self.time_range,
            "physical_quantity": self.physical_quantity,
            "array_shape": self.array_shape,
            "comment": self.comment,
        }
        if isinstance(self.data, dict):
            result["num_keys"] = len(self.data)
            result["keys"] = list(self.data.keys())[:20]
        elif isinstance(self.data, str):
            result["length"] = len(self.data)
        elif isinstance(self.data, bytes):
            result["length"] = len(self.data)
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


def _compute_product_hash(
    *,
    dataset_id: str = "",
    parameter_id: str = "",
    source_ids: list[str] | None = None,
    code: str = "",
    output_label: str = "",
) -> str:
    """Compute an 8-char hash for a data product identity.

    The hash is deterministic based on what the data represents, NOT including
    time range. Same product = same hash = eligible for merging.
    """
    if dataset_id and parameter_id:
        key = f"fetch:{dataset_id}:{parameter_id}"
    elif source_ids and code:
        sorted_sources = sorted(source_ids)
        key = f"compute:{':'.join(sorted_sources)}:{code}"
    elif output_label:
        key = f"create:{output_label}"
    else:
        key = f"unknown:{output_label or 'anon'}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def generate_id(store: "DataStore", **kwargs) -> str:
    """Generate a unique hash-based ID for a new DataEntry.

    Args:
        store: DataStore instance to track sequential counters.
        kwargs: Passed to _compute_product_hash (dataset_id, parameter_id,
                source_ids, code, or output_label).

    Returns:
        A unique ID of the form hash8_N where hash8 is the 8-char product
        hash and N is a sequential counter starting from 1.
    """
    hash8 = _compute_product_hash(**kwargs)
    with store._lock:
        n = store._id_counter.get(hash8, 0) + 1
        store._id_counter[hash8] = n
    return f"{hash8}_{n}"


class DataStore:
    """Disk-backed store mapping IDs to DataEntry objects.

    All data lives on the filesystem under ``data_dir``. An LRU in-memory
    cache avoids redundant disk reads.  Thread-safe for concurrent access.

    Args:
        data_dir: Directory for persisting data.  Created if it doesn't exist.
    """

    def __init__(self, data_dir: Path):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        # id → hash folder name (primary index)
        self._ids: dict[str, str] = {}

        # label → [ids] (one label can map to multiple IDs)
        self._label_index: dict[str, list[str]] = {}

        # hash8 → max_N (for generating sequential IDs)
        self._id_counter: dict[str, int] = {}

        # LRU cache: id → DataEntry (OrderedDict for move_to_end)
        self._cache: collections.OrderedDict[str, DataEntry] = collections.OrderedDict()

        # Load existing _ids.json (new format) or migrate from _labels.json
        ids_path = self._data_dir / "_ids.json"
        labels_path = self._data_dir / "_labels.json"
        label_index_path = self._data_dir / "_label_index.json"

        if ids_path.exists():
            with open(ids_path, "r", encoding="utf-8") as f:
                self._ids = json.load(f)
            if label_index_path.exists():
                with open(label_index_path, "r", encoding="utf-8") as f:
                    self._label_index = json.load(f)
        elif labels_path.exists():
            # Migration from old format
            with open(labels_path, "r", encoding="utf-8") as f:
                old_labels = json.load(f)
            self._migrate_from_labels(old_labels)
        else:
            self._ids = {}
            self._label_index = {}

        # Rebuild _id_counter from existing IDs
        for entry_id in self._ids:
            hash8, n_str = entry_id.rsplit("_", 1)
            try:
                n = int(n_str)
                self._id_counter[hash8] = max(self._id_counter.get(hash8, 0), n)
            except ValueError:
                pass

    def _migrate_from_labels(self, old_labels: dict[str, str]) -> None:
        """Migrate from old _labels.json format to new ID-based format."""
        self._ids = {}
        self._label_index = {}
        self._id_counter = {}

        for label, h in old_labels.items():
            # Compute hash from label for migration
            hash8 = _compute_product_hash(output_label=label)
            n = self._id_counter.get(hash8, 0) + 1
            self._id_counter[hash8] = n
            entry_id = f"{hash8}_{n}"

            self._ids[entry_id] = h
            self._label_index.setdefault(label, []).append(entry_id)

            # Update meta.json with ID
            meta = self._read_meta(self._data_dir / h)
            if meta:
                meta["id"] = entry_id
                self._write_meta(self._data_dir / h, meta)

        self._write_ids()
        self._write_label_index()

    def _write_ids(self) -> None:
        """Atomically write _ids.json."""
        tmp = self._data_dir / "_ids.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._ids, f, indent=2)
        os.replace(tmp, self._data_dir / "_ids.json")

    def _write_label_index(self) -> None:
        """Atomically write _label_index.json."""
        tmp = self._data_dir / "_label_index.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._label_index, f, indent=2)
        os.replace(tmp, self._data_dir / "_label_index.json")

    # ---- Public API ----

    def put(self, entry: DataEntry) -> None:
        """Store a DataEntry, writing data to disk immediately.

        Stats are computed asynchronously — a ``.computing`` flag file
        is present until stats are ready.
        """
        with self._lock:
            # Generate ID if not set
            if not entry.id:
                entry.id = generate_id(self, output_label=entry.label)

            # Use ID as part of hash folder name
            h = f"{entry.id}_{_label_hash(entry.label)[:4]}"
            entry_dir = self._data_dir / h
            entry_dir.mkdir(parents=True, exist_ok=True)

            # Write .computing flag
            computing_flag = entry_dir / ".computing"
            computing_flag.touch()

            # Write data file (atomic via tmp + rename)
            fmt, data_file = self._save_data(entry, entry_dir)

            # Write minimal meta.json
            meta = {
                "id": entry.id,
                "label": entry.label,
                "format": fmt,
                "units": entry.units,
                "description": entry.description,
                "source": entry.source,
                "is_timeseries": entry.is_timeseries,
                "time_range": entry.time_range,
                "physical_quantity": entry.physical_quantity,
                "array_shape": entry.array_shape,
                "comment": entry.comment,
            }
            meta["data_type"] = entry.data_type
            if isinstance(entry.data, pd.DataFrame):
                meta["columns"] = list(entry.data.columns)
                meta["memory_bytes"] = int(entry.data.memory_usage(deep=True).sum())
            elif isinstance(entry.data, xr.DataArray):
                meta["memory_bytes"] = int(entry.data.nbytes)
            else:
                # dict, str, bytes, etc.
                meta["memory_bytes"] = 0
            if entry.metadata is not None:
                meta["metadata"] = entry.metadata
            self._write_meta(entry_dir, meta)

            # Update ID and label indices
            self._ids[entry.id] = h
            self._label_index.setdefault(entry.label, []).append(entry.id)
            self._write_ids()
            self._write_label_index()

            # Update LRU cache
            self._cache[entry.id] = entry
            self._cache.move_to_end(entry.id)
            while len(self._cache) > _MAX_CACHE_ENTRIES:
                self._cache.popitem(last=False)

        # Compute full stats asynchronously
        t = threading.Thread(
            target=self._compute_stats_async,
            args=(entry.id, h),
            daemon=True,
        )
        t.start()

    def get(self, key: str) -> Optional[DataEntry]:
        """Retrieve a DataEntry by ID or label, or None if not found.

        If a label is provided and multiple entries share that label,
        returns the most recent one (last in the list).
        """
        with self._lock:
            # Cache hit - cache is keyed by ID
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

            # Check if it's an exact ID match
            if key in self._ids:
                h = self._ids[key]
                entry = self._load_entry(key, h)
                if entry is not None:
                    self._cache[key] = entry
                    self._cache.move_to_end(key)
                    while len(self._cache) > _MAX_CACHE_ENTRIES:
                        self._cache.popitem(last=False)
                return entry

            # Try as label - return most recent (last in list)
            ids_for_label = self._label_index.get(key, [])
            if ids_for_label:
                most_recent_id = ids_for_label[-1]
                h = self._ids.get(most_recent_id)
                if h is not None:
                    entry = self._load_entry(most_recent_id, h)
                    if entry is not None:
                        self._cache[most_recent_id] = entry
                        self._cache.move_to_end(most_recent_id)
                        while len(self._cache) > _MAX_CACHE_ENTRIES:
                            self._cache.popitem(last=False)
                    return entry
            return None

    def get_by_label(self, label: str) -> list[DataEntry]:
        """Return all DataEntries with the given label."""
        with self._lock:
            ids_for_label = self._label_index.get(label, [])
            entries = []
            for entry_id in ids_for_label:
                h = self._ids.get(entry_id)
                if h is not None:
                    entry = self._load_entry(entry_id, h)
                    if entry is not None:
                        entries.append(entry)
            return entries

    def has(self, key: str) -> bool:
        """Check if an ID or label exists in the store."""
        with self._lock:
            # Check ID first
            if key in self._ids:
                return True
            # Check label
            return key in self._label_index

    def remove(self, key: str) -> bool:
        """Remove an entry by ID or label. Returns True if it existed."""
        with self._lock:
            # Determine the ID to remove
            entry_id = None
            if key in self._ids:
                entry_id = key
            elif key in self._label_index:
                ids = self._label_index[key]
                if ids:
                    entry_id = ids[-1]  # Remove most recent

            if entry_id is None:
                return False

            h = self._ids.pop(entry_id, None)
            if h is None:
                return False

            # Remove from label index
            label = None
            for lbl, ids in self._label_index.items():
                if entry_id in ids:
                    label = lbl
                    ids.remove(entry_id)
                    if not ids:
                        del self._label_index[lbl]
                    break

            # Remove from cache
            self._cache.pop(entry_id, None)

            # Delete the folder (ignore_errors for async stats thread race)
            entry_dir = self._data_dir / h
            if entry_dir.exists():
                shutil.rmtree(entry_dir, ignore_errors=True)

            # Rewrite indices
            self._write_ids()
            self._write_label_index()
            return True

    def list_entries(self) -> list[dict]:
        """Return summary dicts for all stored entries.

        If a ``.computing`` flag is present for an entry, waits briefly
        for stats to finish, then reads the final ``meta.json``.
        """
        with self._lock:
            ids_snapshot = dict(self._ids)

        results = []
        for entry_id, h in ids_snapshot.items():
            entry_dir = self._data_dir / h

            # Wait for .computing flag if present
            computing_flag = entry_dir / ".computing"
            if computing_flag.exists():
                deadline = _time.monotonic() + _STATS_TIMEOUT
                while computing_flag.exists() and _time.monotonic() < deadline:
                    _time.sleep(0.05)

            with self._lock:
                # Check cache first
                if entry_id in self._cache:
                    self._cache.move_to_end(entry_id)
                    results.append(self._cache[entry_id].summary())
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
            self._ids.clear()
            self._label_index.clear()
            self._id_counter.clear()

            # Delete all hash folders (ignore_errors for async stats thread race)
            for item in self._data_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)

            # Remove index files
            ids_path = self._data_dir / "_ids.json"
            if ids_path.exists():
                ids_path.unlink()
            label_index_path = self._data_dir / "_label_index.json"
            if label_index_path.exists():
                label_index_path.unlink()
            labels_path = self._data_dir / "_labels.json"
            if labels_path.exists():
                labels_path.unlink()

    def merge_entries(self, ids: list[str]) -> DataEntry:
        """Merge entries with the same hash prefix into one.

        Concatenates data, sorts by time, deduplicates.
        Removes the original entries and stores the merged result.

        Args:
            ids: List of entry IDs to merge. All must have the same hash prefix
                 (same data product).

        Returns:
            The merged DataEntry.

        Raises:
            ValueError: If IDs not found or have different hash prefixes.
        """
        with self._lock:
            entries = []
            hash_prefixes = set()
            for eid in ids:
                entry = self.get(eid)
                if entry is None:
                    raise ValueError(f"ID '{eid}' not found")
                entries.append(entry)
                hash_prefixes.add(eid.rsplit("_", 1)[0])

            if len(hash_prefixes) > 1:
                raise ValueError(
                    "All entries must be the same data product (same hash prefix)"
                )

            # Concatenate, sort, deduplicate
            ref = entries[0]
            if ref.is_xarray:
                merged_data = xr.concat([e.data for e in entries], dim="time")
                merged_data = merged_data.sortby("time")
                # Deduplicate time coords
                _, unique_idx = np.unique(
                    merged_data.coords["time"].values, return_index=True
                )
                merged_data = merged_data.isel(time=sorted(unique_idx))
            else:
                merged_data = pd.concat([e.data for e in entries])
                merged_data = merged_data.sort_index()
                merged_data = merged_data[~merged_data.index.duplicated(keep="last")]

            # Compute merged time range
            if ref.is_timeseries:
                if ref.is_xarray:
                    t_min = str(merged_data.coords["time"].values[0])
                    t_max = str(merged_data.coords["time"].values[-1])
                else:
                    t_min = str(merged_data.index[0])
                    t_max = str(merged_data.index[-1])
                time_range = (t_min, t_max)
            else:
                time_range = None

            # Build merged comment
            comments = [e.comment for e in entries if e.comment]
            merged_comment = f"Merged from {len(entries)} entries"
            if comments:
                merged_comment += f": {'; '.join(comments)}"

            # Create merged entry with new ID
            merged_entry = DataEntry(
                id=generate_id(self, output_label=ref.label),
                label=ref.label,
                data=merged_data,
                units=ref.units,
                description=ref.description,
                source=ref.source,
                is_timeseries=ref.is_timeseries,
                time_range=time_range,
                physical_quantity=ref.physical_quantity,
                array_shape=ref.array_shape,
                comment=merged_comment,
            )

            # Remove originals
            for eid in ids:
                self.remove(eid)

            # Store merged
            self.put(merged_entry)
            return merged_entry

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
            return len(self._ids)

    # ---- Internal helpers ----

    def _save_data(self, entry: DataEntry, entry_dir: Path) -> tuple[str, Path]:
        """Save entry data to disk, returning (format_string, data_file_path)."""
        data = entry.data
        if isinstance(data, xr.DataArray):
            fmt = "netcdf"
            data_file = entry_dir / "data.nc"
            tmp_file = entry_dir / "data.nc.tmp"
            encoding = {}
            if "time" in data.dims:
                encoding["time"] = {
                    "units": "seconds since 1970-01-01",
                    "dtype": "float64",
                }
            data.to_netcdf(str(tmp_file), encoding=encoding)
        elif isinstance(data, pd.DataFrame):
            fmt = "parquet"
            data_file = entry_dir / "data.parquet"
            tmp_file = entry_dir / "data.parquet.tmp"
            data.to_parquet(str(tmp_file))
        elif isinstance(data, dict):
            fmt = "json"
            data_file = entry_dir / "data.json"
            tmp_file = entry_dir / "data.json.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        elif isinstance(data, str):
            fmt = "text"
            data_file = entry_dir / "data.txt"
            tmp_file = entry_dir / "data.txt.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                f.write(data)
        elif isinstance(data, bytes):
            fmt = "bytes"
            data_file = entry_dir / "data.bin"
            tmp_file = entry_dir / "data.bin.tmp"
            with open(tmp_file, "wb") as f:
                f.write(data)
        else:
            import pickle
            fmt = "pickle"
            data_file = entry_dir / "data.pkl"
            tmp_file = entry_dir / "data.pkl.tmp"
            with open(tmp_file, "wb") as f:
                pickle.dump(data, f)
        os.replace(tmp_file, data_file)
        return fmt, data_file

    def _load_data(self, entry_dir: Path, fmt: str):
        """Load data from disk based on format string."""
        if fmt == "netcdf":
            data_path = entry_dir / "data.nc"
            if not data_path.exists():
                return None
            return xr.open_dataarray(data_path).load()
        elif fmt == "parquet":
            data_path = entry_dir / "data.parquet"
            if not data_path.exists():
                return None
            return pd.read_parquet(data_path)
        elif fmt == "json":
            data_path = entry_dir / "data.json"
            if not data_path.exists():
                return None
            with open(data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif fmt == "text":
            data_path = entry_dir / "data.txt"
            if not data_path.exists():
                return None
            with open(data_path, "r", encoding="utf-8") as f:
                return f.read()
        elif fmt == "bytes":
            data_path = entry_dir / "data.bin"
            if not data_path.exists():
                return None
            with open(data_path, "rb") as f:
                return f.read()
        else:
            # Legacy pickle format
            data_path = entry_dir / "data.pkl"
            if not data_path.exists():
                return None
            return pd.read_pickle(data_path)

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

    def _load_entry(self, entry_id: str, h: str) -> Optional[DataEntry]:
        """Load a DataEntry from its hash folder (lazy — data loads on first access)."""
        entry_dir = self._data_dir / h
        meta = self._read_meta(entry_dir)
        if meta is None:
            return None

        fmt = meta.get("format", "pickle")

        entry = DataEntry(
            id=entry_id,
            label=meta.get("label", ""),
            units=meta.get("units", ""),
            description=meta.get("description", ""),
            source=meta.get("source", "computed"),
            metadata=meta.get("metadata"),
            is_timeseries=meta.get("is_timeseries", True),
            time_range=meta.get("time_range"),
            physical_quantity=meta.get("physical_quantity", ""),
            array_shape=meta.get("array_shape", ""),
            comment=meta.get("comment", ""),
        )
        # Set up lazy loading — data will be loaded on first .data access
        entry._data_path = entry_dir
        entry._data_format = fmt
        entry._data_loaded = False
        entry._data_loader = self._load_data
        return entry

    def _compute_stats_async(self, entry_id: str, h: str) -> None:
        """Compute full statistics and update meta.json, then remove .computing flag."""
        entry_dir = self._data_dir / h
        try:
            entry = self.get(entry_id)
            if entry is None:
                return
            meta = self._read_meta(entry_dir)
            if meta is None:
                return
            meta.update(self._build_stats(entry))
            self._write_meta(entry_dir, meta)
        except Exception as exc:
            logger.debug("Async stats computation failed: %s", exc)
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
        # Non-DataFrame/xarray types have no statistical summary
        if entry.data_type not in ("dataframe", "xarray"):
            return stats
        if entry.is_xarray:
            da = entry.data
            dims = dict(da.sizes)
            n_time = dims.get("time", 0)
            stats["dims"] = dims
            stats["num_points"] = n_time
            stats["shape"] = (
                f"ndarray[{' x '.join(f'{k}={v}' for k, v in dims.items())}]"
            )
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
            stats["nan_percentage"] = (
                round(nan_count / flat.size * 100, 1) if flat.size > 0 else 0
            )
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
            stats["nan_percentage"] = (
                round(nan_count / (n * ncols) * 100, 1) if n > 0 else 0
            )
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
            "id": meta.get("id", ""),
            "label": meta.get("label", ""),
            "units": meta.get("units", ""),
            "description": meta.get("description", ""),
            "source": meta.get("source", "computed"),
            "is_timeseries": meta.get("is_timeseries", True),
            "time_range": meta.get("time_range"),
            "physical_quantity": meta.get("physical_quantity", ""),
            "array_shape": meta.get("array_shape", ""),
            "comment": meta.get("comment", ""),
        }

        # Stats fields (present after async computation)
        for key in (
            "num_points",
            "shape",
            "nan_count",
            "nan_percentage",
            "time_min",
            "time_max",
            "index_min",
            "index_max",
            "median_cadence",
            "statistics",
            "dims",
            "storage_type",
            "columns",
        ):
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
    store: DataStore, id_or_label: str
) -> tuple[DataEntry | None, str | None]:
    """Resolve an ID or label to a DataEntry, supporting column sub-selection.

    Handles compound labels like 'PSP_B_DERIVATIVE_FINAL.B_mag' where
    'PSP_B_DERIVATIVE_FINAL' is the store key and 'B_mag' is a column.
    Also handles dedup suffixes (.1, .2) since CDF vectors use integer columns.

    Args:
        store: DataStore to look up entries.
        id_or_label: The ID or label to resolve (may be compound with column).

    Returns:
        Tuple of (DataEntry, resolved_id) or (None, None) if not found.
    """
    # Exact match first (works for both ID and label)
    entry = store.get(id_or_label)
    if entry is not None:
        return entry, entry.id

    # Try column sub-selection: split from the right and check
    # progressively longer prefixes as parent labels.
    # E.g. "A.B.C" tries "A.B" with col "C", then "A" with col "B.C"
    parts = id_or_label.split(".")
    for i in range(len(parts) - 1, 0, -1):
        parent_key = ".".join(parts[:i])
        col_name = ".".join(parts[i:])
        parent = store.get(parent_key)
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
                id=parent.id,
                label=id_or_label,
                data=parent.data[[col_name]],
                units=parent.units,
                description=f"{parent.description} [{col_name}]"
                if parent.description
                else str(col_name),
                source=parent.source,
                metadata=parent.metadata,
                is_timeseries=parent.is_timeseries,
            )
            return sub_entry, parent.id
    return None, None


def build_source_map(
    store: DataStore, ids_or_labels: list[str]
) -> tuple[dict[str, pd.DataFrame | xr.DataArray] | None, str | None]:
    """Build a mapping of sandbox variable names to data from store IDs or labels.

    Each entry becomes ``df_<SUFFIX>`` (DataFrame) or ``da_<SUFFIX>``
    (xarray DataArray) where SUFFIX is the part after the last '.' in
    the label.  If the label has no '.', the full label is used as suffix.

    Args:
        store: DataStore to look up entries.
        ids_or_labels: List of store IDs or labels.

    Returns:
        Tuple of (source_map, error_string).  On success error_string is None.
        On failure source_map is None.
    """
    source_map: dict[str, pd.DataFrame | xr.DataArray] = {}
    for key in ids_or_labels:
        entry = store.get(key)
        if entry is None:
            return None, f"'{key}' not found in store"
        suffix = entry.label.rsplit(".", 1)[-1]
        prefix = "da" if entry.is_xarray else "df"
        var_name = f"{prefix}_{suffix}"
        if var_name in source_map:
            id_suffix = entry.id.rsplit("_", 1)[-1]
            var_name = f"{prefix}_{suffix}_{id_suffix}"
        source_map[var_name] = entry.data
    return source_map, None


def describe_sources(store: DataStore, ids_or_labels: list[str]) -> dict:
    """Return lightweight summaries for a list of store IDs or labels.

    Keyed by sandbox variable name (``df_SUFFIX`` or ``da_SUFFIX``).
    Reuses ``entry.summary()`` to avoid duplicating stat computation.

    Args:
        store: DataStore to look up entries.
        ids_or_labels: List of store IDs or labels.

    Returns:
        Dict keyed by sandbox variable name (``df_SUFFIX`` or ``da_SUFFIX``),
        each containing label, shape info, points, cadence, nan_pct, and time_range.
    """
    result = {}
    used_names = {}
    for key in ids_or_labels:
        entry = store.get(key)
        if entry is None:
            continue
        suffix = entry.label.rsplit(".", 1)[-1]
        prefix = "da" if entry.is_xarray else "df"
        var_name = f"{prefix}_{suffix}"
        if var_name in used_names:
            id_suffix = entry.id.rsplit("_", 1)[-1]
            var_name = f"{prefix}_{suffix}_{id_suffix}"
        used_names[var_name] = True

        s = entry.summary()
        info = {
            "id": s.get("id", ""),
            "label": s["label"],
            "columns": s.get("columns")
            or (list(s["dims"].keys()) if "dims" in s else []),
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
