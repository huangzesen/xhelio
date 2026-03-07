"""Tool handler for loading local data files into the DataStore."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from config import get_data_dir
from data_ops.store import DataEntry, generate_id
from agent.event_bus import DATA_CREATED

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent

def _is_numeric_dtype(series: pd.Series) -> bool:
    """Check if a Series has a numeric dtype (int, float, etc.)."""
    return series.dtype.kind in ("i", "u", "f")


_DATETIME_NAMES = frozenset({
    "time", "datetime", "date", "timestamp", "epoch",
    "t", "time_utc", "datetime_utc", "date_utc",
})


def _get_allowed_dirs() -> list[Path]:
    """Return list of directories from which file loading is permitted."""
    return [
        Path.home(),
        get_data_dir(),
    ]


def _validate_file_path(file_path: str) -> str:
    """Validate that a file path exists and is within allowed directories.

    Returns the resolved absolute path string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is outside all allowed directories.
    """
    p = Path(file_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    allowed = _get_allowed_dirs()
    for allowed_dir in allowed:
        try:
            p.relative_to(allowed_dir)
            return str(p)
        except ValueError:
            continue

    raise ValueError(
        f"File path '{p}' is outside allowed directories: "
        f"{[str(d) for d in allowed]}"
    )


def _try_parse_datetime_index(
    df: pd.DataFrame, time_column: str | None = None
) -> pd.DataFrame:
    """Try to set a datetime index on the DataFrame.

    Strategy:
    1. If time_column is given explicitly, use it.
    2. Check if the current index name is a known datetime name.
    3. Auto-detect: scan columns for datetime-like names.
    4. Auto-detect: try first column if index is unnamed.
    5. Return df unchanged if no datetime found.
    """
    df = df.copy()

    # 1. Explicit time_column
    if time_column:
        if time_column in df.columns:
            df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
            df = df.set_index(time_column)
            return df
        elif df.index.name and df.index.name.lower() == time_column.lower():
            df.index = pd.to_datetime(df.index, errors="coerce")
            return df

    # 2. Current index name is a known datetime name
    if df.index.name and df.index.name.lower() in _DATETIME_NAMES:
        try:
            parsed = pd.to_datetime(df.index, errors="coerce")
            if parsed.notna().sum() > len(parsed) * 0.5:
                df.index = parsed
                return df
        except Exception:
            pass

    # 3. Auto-detect: scan columns for datetime-like names
    for col in df.columns:
        if col.lower() in _DATETIME_NAMES:
            if _is_numeric_dtype(df[col]):
                continue
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > len(parsed) * 0.5:
                    df[col] = parsed
                    df = df.set_index(col)
                    return df
            except Exception:
                continue

    # 4. Auto-detect: try first column if index is unnamed
    if df.index.name is None and len(df.columns) > 0:
        first_col = df.columns[0]
        if not _is_numeric_dtype(df[first_col]):
            try:
                parsed = pd.to_datetime(df[first_col], errors="coerce")
                if parsed.notna().sum() > len(parsed) * 0.5:
                    df[first_col] = parsed
                    df = df.set_index(first_col)
                    return df
            except Exception:
                pass

    return df


def _load_file_to_dataframe(
    file_path: str, time_column: str | None = None
) -> pd.DataFrame:
    """Load a data file into a pandas DataFrame based on its extension.

    Supported formats: CSV, TSV, JSON, Parquet, Excel.

    Args:
        file_path: Absolute path to the data file.
        time_column: Optional column name to use as datetime index.

    Returns:
        DataFrame with datetime index if detected/specified.

    Raises:
        ValueError: If the file format is unsupported.
    """
    p = Path(file_path)
    ext = p.suffix.lower()

    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(file_path, sep=sep)
    elif ext == ".json":
        try:
            df = pd.read_json(file_path, orient="table")
        except (ValueError, KeyError, TypeError):
            df = pd.read_json(file_path)
    elif ext == ".parquet":
        df = pd.read_parquet(file_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported formats: .csv, .tsv, .json, .parquet, .xlsx, .xls"
        )

    df = _try_parse_datetime_index(df, time_column=time_column)
    return df


def handle_load_file(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    """Handle the load_file tool: load a local data file into the DataStore."""
    file_path = tool_args.get("file_path", "")
    output_label = tool_args.get("output_label", "")

    if not file_path:
        return {"status": "error", "message": "file_path is required"}
    if not output_label:
        return {"status": "error", "message": "output_label is required"}

    try:
        validated_path = _validate_file_path(file_path)
    except FileNotFoundError as e:
        return {"status": "error", "message": str(e)}
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    try:
        df = _load_file_to_dataframe(
            validated_path,
            time_column=tool_args.get("time_column"),
        )
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"Failed to load file: {e}"}

    entry = DataEntry(
        label=output_label,
        data=df,
        units=tool_args.get("units", ""),
        description=tool_args.get("description", f"Loaded from {Path(validated_path).name}"),
        source="file",
        is_timeseries=isinstance(df.index, pd.DatetimeIndex),
    )

    store = orch._store
    entry.id = generate_id(store, output_label=output_label)
    store.put(entry)

    orch._event_bus.emit(
        DATA_CREATED,
        agent="orchestrator",
        msg=f"[DataIO] Loaded file -> '{output_label}' ({len(df)} rows, {len(df.columns)} cols)",
        data={
            "args": {
                "file_path": validated_path,
                "output_label": output_label,
            },
            "outputs": [entry.id],
        },
    )

    return {"status": "success", **entry.summary()}
