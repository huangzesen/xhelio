"""
In-memory data store.

DataEntry holds a single dataset as a pandas DataFrame or xarray DataArray.
Timeseries entries have DatetimeIndex; general-data entries may have numeric/string indices.
DataStore is a singleton dict-like container keyed by label strings.
"""

import contextvars
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

# Characters unsafe for filenames on Windows
_UNSAFE_CHARS = re.compile(r'[/\\:*?"<>|]')


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
        if self.metadata:
            result["metadata"] = self.metadata
        return result


_UNLOADED = object()  # sentinel for LazyDataEntry


class LazyDataEntry(DataEntry):
    """DataEntry that defers loading its data file until first ``.data`` access.

    Constructed by ``DataStore.load_from_directory_lazy()`` — carries all
    metadata (label, units, columns, …) without touching the pickle/netcdf
    file until the data is actually needed.
    """

    def __init__(
        self,
        *,
        label: str,
        units: str,
        description: str,
        source: str,
        metadata: dict | None,
        is_timeseries: bool,
        file_path: Path,
        fmt: str,
        columns: list | None = None,
    ):
        # Bypass DataEntry.__init__ (which requires `data`)
        # by setting fields directly on the instance.
        object.__setattr__(self, '_file_path', file_path)
        object.__setattr__(self, '_format', fmt)
        object.__setattr__(self, '_loaded', False)
        object.__setattr__(self, '_load_lock', threading.Lock())
        object.__setattr__(self, '_columns', columns)
        # Set dataclass fields directly
        self.label = label
        self.units = units
        self.description = description
        self.source = source
        self.metadata = metadata
        self.is_timeseries = is_timeseries
        self._data = _UNLOADED

    # ---- data property: lazy-load on first access ----

    @property  # type: ignore[override]
    def data(self):
        if self._data is _UNLOADED:
            self._load_data()
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        object.__setattr__(self, '_loaded', True)

    # ---- Metadata-only overrides to avoid triggering a load ----

    @property
    def columns(self) -> list:
        """Column names from saved metadata (no disk load needed)."""
        if self._columns is not None:
            return list(self._columns)
        # Fallback: triggers load
        return list(self.data.columns)

    @property
    def is_xarray(self) -> bool:
        if self._data is not _UNLOADED:
            return isinstance(self._data, xr.DataArray)
        return self._format == "netcdf"

    def summary(self) -> dict:
        """Return a summary without loading data when possible."""
        if self._data is not _UNLOADED:
            return super().summary()
        # Build a lightweight summary from saved metadata
        result: dict = {
            "label": self.label,
            "units": self.units,
            "description": self.description,
            "source": self.source,
            "is_timeseries": self.is_timeseries,
        }
        if self._columns is not None:
            result["columns"] = list(self._columns)
            ncols = len(self._columns)
            result["shape"] = "scalar" if ncols == 1 else f"vector[{ncols}]"
        else:
            result["shape"] = "unknown (not loaded)"
        result["num_points"] = 0  # unknown without loading
        result["time_min"] = None
        result["time_max"] = None
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    # ---- Internal ----

    def _load_data(self):
        with self._load_lock:
            if self._data is not _UNLOADED:
                return
            if self._format == "netcdf":
                self._data = xr.open_dataarray(self._file_path).load()
            else:
                self._data = pd.read_pickle(self._file_path)
            object.__setattr__(self, '_loaded', True)


class DataStore:
    """Singleton in-memory store mapping labels to DataEntry objects."""

    def __init__(self):
        self._entries: dict[str, DataEntry] = {}
        self._lock = threading.RLock()

    def put(self, entry: DataEntry) -> None:
        """Store a DataEntry, overwriting any existing entry with the same label."""
        with self._lock:
            self._entries[entry.label] = entry

    def get(self, label: str) -> Optional[DataEntry]:
        """Retrieve a DataEntry by label, or None if not found."""
        with self._lock:
            return self._entries.get(label)

    def has(self, label: str) -> bool:
        """Check if a label exists in the store."""
        with self._lock:
            return label in self._entries

    def remove(self, label: str) -> bool:
        """Remove an entry by label. Returns True if it existed."""
        with self._lock:
            if label in self._entries:
                del self._entries[label]
                return True
            return False

    def list_entries(self) -> list[dict]:
        """Return summary dicts for all stored entries."""
        with self._lock:
            return [entry.summary() for entry in self._entries.values()]

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._entries.clear()

    def memory_usage_bytes(self) -> int:
        """Return approximate total memory usage of all stored data.

        Skips lazy entries that haven't been loaded yet.
        """
        with self._lock:
            total = 0
            for entry in self._entries.values():
                # Skip unloaded lazy entries
                if isinstance(entry, LazyDataEntry) and not entry._loaded:
                    continue
                if entry.is_xarray:
                    total += entry.data.nbytes
                else:
                    total += int(entry.data.memory_usage(deep=True).sum())
            return total

    def save_to_directory(self, dir_path: Path) -> None:
        """Persist all DataEntries to a directory.

        DataFrames are saved as pickle files, DataArrays as NetCDF files.
        An ``_index.json`` maps original labels to filenames and metadata.
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        with self._lock:
            index = {}
            for label, entry in self._entries.items():
                safe = _UNSAFE_CHARS.sub("_", label)
                if entry.is_xarray:
                    filename = f"{safe}.nc"
                    # Scipy NetCDF3 engine only supports int32; datetime64[ns]
                    # needs int64.  Encode time as float64 seconds-since-epoch
                    # so the round-trip works without netCDF4/h5netcdf.
                    encoding = {}
                    if "time" in entry.data.dims:
                        encoding["time"] = {
                            "units": "seconds since 1970-01-01",
                            "dtype": "float64",
                        }
                    entry.data.to_netcdf(dir_path / filename, encoding=encoding)
                    fmt = "netcdf"
                else:
                    filename = f"{safe}.pkl"
                    entry.data.to_pickle(dir_path / filename)
                    fmt = "pickle"
                entry_meta = {
                    "filename": filename,
                    "format": fmt,
                    "units": entry.units,
                    "description": entry.description,
                    "source": entry.source,
                    "is_timeseries": entry.is_timeseries,
                }
                if not entry.is_xarray:
                    entry_meta["columns"] = list(entry.data.columns)
                if entry.metadata is not None:
                    entry_meta["metadata"] = entry.metadata
                index[label] = entry_meta

        with open(dir_path / "_index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    def load_from_directory(self, dir_path: Path) -> int:
        """Restore DataEntries from a directory written by ``save_to_directory``.

        Returns:
            Number of entries loaded.
        """
        dir_path = Path(dir_path)
        index_path = dir_path / "_index.json"
        if not index_path.exists():
            return 0

        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        count = 0
        for label, info in index.items():
            file_path = dir_path / info["filename"]
            if not file_path.exists():
                continue
            fmt = info.get("format", "pickle")
            if fmt == "netcdf":
                data = xr.open_dataarray(file_path).load()
            else:
                data = pd.read_pickle(file_path)
            entry = DataEntry(
                label=label,
                data=data,
                units=info.get("units", ""),
                description=info.get("description", ""),
                source=info.get("source", "computed"),
                metadata=info.get("metadata"),
                is_timeseries=info.get("is_timeseries", True),
            )
            self.put(entry)
            count += 1

        return count

    def load_from_directory_lazy(self, dir_path: Path) -> int:
        """Like ``load_from_directory`` but creates lazy stubs (no pickle/netcdf I/O).

        Data files are only read when a ``LazyDataEntry.data`` property is
        first accessed.  This makes session resume near-instant for sessions
        with large datasets.

        Returns:
            Number of entries registered.
        """
        dir_path = Path(dir_path)
        index_path = dir_path / "_index.json"
        if not index_path.exists():
            return 0

        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        count = 0
        for label, info in index.items():
            file_path = dir_path / info["filename"]
            if not file_path.exists():
                continue
            entry = LazyDataEntry(
                label=label,
                units=info.get("units", ""),
                description=info.get("description", ""),
                source=info.get("source", "computed"),
                metadata=info.get("metadata"),
                is_timeseries=info.get("is_timeseries", True),
                file_path=file_path,
                fmt=info.get("format", "pickle"),
                columns=info.get("columns"),
            )
            self.put(entry)
            count += 1

        return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


# ContextVar-based store — each context (thread / asyncio task) gets its own.
# Default behavior (single implicit store) is unchanged for single-session use.
_store_var: contextvars.ContextVar[Optional[DataStore]] = contextvars.ContextVar(
    "_store_var", default=None
)


def get_store() -> DataStore:
    """Return the DataStore for the current context, creating one if needed."""
    store = _store_var.get()
    if store is None:
        store = DataStore()
        _store_var.set(store)
    return store


def set_store(store: DataStore) -> None:
    """Explicitly set the DataStore for the current context."""
    _store_var.set(store)


def reset_store() -> None:
    """Reset the DataStore for the current context (mainly for testing)."""
    _store_var.set(None)


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

    For each label, computes: columns/dims, point count, cadence, NaN%, and time range.
    Cheaper than full ``describe_data`` — just what the LLM needs for correct code.

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

        if entry.is_xarray:
            da = entry.data
            prefix = "da"
            var_name = f"{prefix}_{suffix}"
            times = da.coords["time"].values
            n_time = len(times)

            cadence_str = ""
            if n_time > 1:
                dt = pd.Series(times).diff().dropna().median()
                cadence_str = str(dt)

            total_cells = da.size
            nan_count = int(np.isnan(da.values).sum()) if total_cells > 0 else 0
            nan_pct = round(nan_count / total_cells * 100, 1) if total_cells > 0 else 0.0

            time_range = []
            if n_time > 0:
                time_range = [str(pd.Timestamp(times[0]).date()),
                              str(pd.Timestamp(times[-1]).date())]

            result[var_name] = {
                "label": label,
                "dims": dict(da.sizes),
                "shape": list(da.shape),
                "points": n_time,
                "cadence": cadence_str,
                "nan_pct": nan_pct,
                "time_range": time_range,
                "storage_type": "xarray",
            }
        else:
            df = entry.data
            var_name = f"df_{suffix}"

            total_cells = df.size
            nan_pct = round(df.isna().sum().sum() / total_cells * 100, 1) if total_cells > 0 else 0.0

            info = {
                "label": label,
                "columns": list(df.columns),
                "points": len(df),
                "nan_pct": nan_pct,
            }

            if entry.is_timeseries:
                cadence_str = ""
                if len(df) > 1:
                    dt = pd.Series(df.index).diff().dropna().median()
                    cadence_str = str(dt)
                info["cadence"] = cadence_str
                if len(df) > 0:
                    info["time_range"] = [str(df.index[0].date()), str(df.index[-1].date())]
                else:
                    info["time_range"] = []
            else:
                if len(df) > 0:
                    info["index_range"] = [str(df.index[0]), str(df.index[-1])]
                else:
                    info["index_range"] = []

            result[var_name] = info
    return result
