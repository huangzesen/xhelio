from __future__ import annotations
from typing import TYPE_CHECKING
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from config import DATA_BACKEND
from data_ops.store import DataEntry, generate_id
from data_ops.fetch import fetch_data
from knowledge.metadata_client import (
    validate_dataset_id,
    validate_parameter_id,
    get_dataset_quality_report,
)
from agent.event_bus import (
    DEBUG,
    DATA_FETCHED,
)
from agent.truncation import get_item_limit

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent


def handle_fetch_data(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    dataset_id = tool_args.get("dataset_id")
    parameter_id = tool_args.get("parameter_id")
    if not dataset_id or not parameter_id:
        missing = [k for k in ("dataset_id", "parameter_id") if not tool_args.get(k)]
        return {
            "status": "error",
            "message": f"Missing required parameter(s): {', '.join(missing)}",
        }

    ds_validation = validate_dataset_id(dataset_id)
    if not ds_validation["valid"]:
        return {"status": "error", "message": ds_validation["message"]}

    param_validation = validate_parameter_id(dataset_id, parameter_id)
    if not param_validation["valid"]:
        return {"status": "error", "message": param_validation["message"]}

    try:
        fetch_start = datetime.fromisoformat(tool_args.get("time_start", ""))
        fetch_end = datetime.fromisoformat(tool_args.get("time_end", ""))
    except (ValueError, TypeError) as e:
        return {
            "status": "error",
            "message": f"Invalid time_start/time_end: {e}",
        }
    if fetch_start >= fetch_end:
        return {
            "status": "error",
            "message": (
                f"time_start ({tool_args['time_start']}) must be before "
                f"time_end ({tool_args['time_end']})."
            ),
        }

    adjustment_note = None

    validation = orch._validate_time_range(dataset_id, fetch_start, fetch_end)
    if validation is not None:
        if validation.get("error"):
            return {"status": "error", "message": validation["note"]}
        fetch_start = validation["start"]
        fetch_end = validation["end"]
        adjustment_note = validation["note"]
        orch._event_bus.emit(
            DEBUG,
            level="debug",
            msg=f"[DataOps] Time range adjusted for {tool_args['dataset_id']}: "
            f"{adjustment_note}",
        )

    label = f"{dataset_id}.{parameter_id}"
    store = orch._store
    existing = store.get(label)
    if existing is not None and len(existing.time) > 0:
        if existing.is_xarray:
            existing_start = (
                pd.Timestamp(existing.time[0]).to_pydatetime().replace(tzinfo=None)
            )
            existing_end = (
                pd.Timestamp(existing.time[-1]).to_pydatetime().replace(tzinfo=None)
            )
        else:
            existing_start = existing.data.index[0].to_pydatetime().replace(tzinfo=None)
            existing_end = existing.data.index[-1].to_pydatetime().replace(tzinfo=None)
        if existing_start <= fetch_start and existing_end >= fetch_end:
            orch._event_bus.emit(
                DEBUG,
                level="debug",
                msg=f"[DataOps] Dedup: '{label}' already in memory "
                f"({existing_start} to {existing_end}), skipping fetch",
            )
            response = {
                "status": "success",
                "already_loaded": True,
                **existing.summary(),
            }
            if adjustment_note:
                response["time_range_note"] = adjustment_note
            orch._event_bus.emit(
                DATA_FETCHED,
                agent="orchestrator",
                msg=f"[Fetch] {label} (already loaded)",
                data={
                    "args": {
                        "dataset_id": tool_args["dataset_id"],
                        "parameter_id": tool_args["parameter_id"],
                        "time_start": tool_args.get("time_start", ""),
                        "time_end": tool_args.get("time_end", ""),
                        "time_range_resolved": [
                            fetch_start.isoformat(),
                            fetch_end.isoformat(),
                        ],
                        "already_loaded": True,
                    },
                    "outputs": [existing.id],
                },
            )
            return response

    try:
        result = fetch_data(
            dataset_id=tool_args["dataset_id"],
            parameter_id=tool_args["parameter_id"],
            time_min=fetch_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            time_max=fetch_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            force=tool_args.get("force_large_download", False),
        )
    except Exception as e:
        return {"status": "error", "message": str(e)}

    if isinstance(result, dict) and result.get("status") == "confirmation_required":
        return {
            "status": "clarification_needed",
            "question": result["message"],
            "options": [
                "Yes, proceed with the download",
                "No, try a shorter time range",
                "Other (please specify)",
            ],
            "context": (
                f"Dataset {result['dataset_id']}: {result['download_mb']} MB "
                f"across {result['n_files']} files to download "
                f"({result['n_cached']} already cached)."
            ),
        }

    fetched_data = result["data"]
    is_xarray = isinstance(fetched_data, xr.DataArray)

    if is_xarray:
        n_time = fetched_data.sizes["time"]

        if n_time > 0 and np.all(np.isnan(fetched_data.values)):
            return {
                "status": "error",
                "message": (
                    f"Parameter '{tool_args['parameter_id']}' in dataset "
                    f"'{tool_args['dataset_id']}' returned {n_time} time steps "
                    f"but ALL values are fill/NaN — no real data available "
                    f"for this parameter in the requested time range. "
                    f"Try a different parameter or dataset."
                ),
            }

        total_cells = fetched_data.size
        nan_total = int(np.isnan(fetched_data.values).sum())
        nan_pct = round(100 * nan_total / total_cells, 1) if total_cells > 0 else 0.0

        entry = DataEntry(
            label=label,
            data=fetched_data,
            units=result["units"],
            description=result["description"],
            source=DATA_BACKEND,
            time_range=(fetch_start.isoformat(), fetch_end.isoformat()),
            array_shape=f"ndarray[{dict(fetched_data.sizes)}]",
        )
        entry.id = generate_id(
            store,
            dataset_id=tool_args["dataset_id"],
            parameter_id=tool_args["parameter_id"],
        )
        store.put(entry)
        orch._event_bus.emit(
            DATA_FETCHED,
            agent="orchestrator",
            msg=f"[Fetch] {label} (xarray {dict(fetched_data.sizes)})",
            data={
                "args": {
                    "dataset_id": tool_args["dataset_id"],
                    "parameter_id": tool_args["parameter_id"],
                    "time_range": tool_args.get("time_range", ""),
                    "time_range_resolved": [
                        fetch_start.isoformat(),
                        fetch_end.isoformat(),
                    ],
                    "already_loaded": False,
                },
                "outputs": [entry.id],
                "status": "success",
                "nan_percentage": nan_pct,
            },
        )
        response = {"status": "success", **entry.summary()}
        response["note"] = (
            f"This is a {fetched_data.ndim}D variable with dims {dict(fetched_data.sizes)}. "
            f"Use run_code to read the data (xr.open_dataarray('{tool_args.get('label', 'data')}.nc')) "
            f"and slice/reduce it to a 2D DataFrame before plotting."
        )

        n_points = n_time
    else:
        df = fetched_data
        numeric_cols = df.select_dtypes(include="number")
        if (
            len(df) > 0
            and len(numeric_cols.columns) > 0
            and numeric_cols.isna().all(axis=None)
        ):
            return {
                "status": "error",
                "message": (
                    f"Parameter '{tool_args['parameter_id']}' in dataset "
                    f"'{tool_args['dataset_id']}' returned {len(df)} rows "
                    f"but ALL values are fill/NaN — no real data available "
                    f"for this parameter in the requested time range. "
                    f"Try a different parameter or dataset."
                ),
            }

        nan_total = numeric_cols.isna().sum().sum()
        nan_pct = (
            round(100 * nan_total / numeric_cols.size, 1)
            if numeric_cols.size > 0
            else 0.0
        )

        ncols = len(df.columns)
        array_shape = "scalar" if ncols == 1 else f"vector[{ncols}]"
        entry = DataEntry(
            label=label,
            data=df,
            units=result["units"],
            description=result["description"],
            source=DATA_BACKEND,
            time_range=(fetch_start.isoformat(), fetch_end.isoformat()),
            array_shape=array_shape,
        )
        entry.id = generate_id(
            store,
            dataset_id=tool_args["dataset_id"],
            parameter_id=tool_args["parameter_id"],
        )
        store.put(entry)
        orch._event_bus.emit(
            DATA_FETCHED,
            agent="orchestrator",
            msg=f"[Fetch] {label} ({len(entry.time)} points)",
            data={
                "args": {
                    "dataset_id": tool_args["dataset_id"],
                    "parameter_id": tool_args["parameter_id"],
                    "time_range": tool_args.get("time_range", ""),
                    "time_range_resolved": [
                        fetch_start.isoformat(),
                        fetch_end.isoformat(),
                    ],
                    "already_loaded": False,
                },
                "outputs": [entry.id],
                "status": "success",
                "nan_percentage": nan_pct,
            },
        )
        response = {"status": "success", **entry.summary()}

        n_points = len(df)

    if entry.is_timeseries and len(entry.time) > 1:
        actual_start = pd.Timestamp(entry.time[0])
        actual_end = pd.Timestamp(entry.time[-1])
        actual_span = (actual_end - actual_start).total_seconds()
        requested_span = (fetch_end - fetch_start).total_seconds()
        coverage_pct = (
            round(100 * actual_span / requested_span, 1) if requested_span > 0 else 0.0
        )
        response["time_coverage"] = {
            "requested_start": fetch_start.isoformat(),
            "requested_end": fetch_end.isoformat(),
            "actual_start": str(actual_start),
            "actual_end": str(actual_end),
            "coverage_pct": coverage_pct,
        }

    if n_points > 500_000:
        response["size_warning"] = (
            f"Very large dataset ({n_points:,} points). "
            f"Consider using a shorter time range or a lower-cadence dataset "
            f"to avoid slow downstream operations."
        )

    if adjustment_note:
        response["time_range_note"] = adjustment_note

    if nan_pct > 0:
        response["nan_percentage"] = nan_pct
        if nan_pct >= 25:
            response["quality_warning"] = (
                f"High NaN/fill ratio ({nan_pct}%). Data was stored but "
                f"quality is degraded. Consider trying a different "
                f"parameter or dataset if one with better coverage exists."
            )

    quality = get_dataset_quality_report(tool_args["dataset_id"])
    if quality and (quality["metadata_only"] or quality["data_only"]):
        response["metadata_discrepancies"] = quality

    return response


def handle_list_fetched_data(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    store = orch._store
    entries = store.list_entries()
    return {"status": "success", "entries": entries, "count": len(entries)}


def handle_list_assets(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    kind = tool_args.get("kind")
    assets = orch._asset_registry.list_assets(kind=kind)
    return {"status": "success", "assets": assets, "count": len(assets)}


def handle_describe_data(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    store = orch._store
    key = tool_args.get("data_id") or tool_args.get("label")
    entry = store.get(key)
    if entry is None:
        return {
            "status": "error",
            "message": f"Data '{key}' not found in memory",
        }

    time_start = tool_args.get("time_start")
    time_end = tool_args.get("time_end")

    if entry.is_xarray:
        da = entry.data
        if (time_start or time_end) and "time" in da.dims:
            sel_kw = {}
            if time_start and time_end:
                sel_kw["time"] = slice(time_start, time_end)
            elif time_start:
                sel_kw["time"] = slice(time_start, None)
            else:
                sel_kw["time"] = slice(None, time_end)
            da = da.sel(**sel_kw)
        dims = dict(da.sizes)
        n_time = dims.get("time", 0)

        coords_info = {}
        for cname, coord in da.coords.items():
            cvals = coord.values
            info = {"size": len(cvals), "dtype": str(cvals.dtype)}
            if np.issubdtype(cvals.dtype, np.number):
                info["min"] = float(np.nanmin(cvals))
                info["max"] = float(np.nanmax(cvals))
            elif np.issubdtype(cvals.dtype, np.datetime64):
                info["min"] = str(cvals[0])
                info["max"] = str(cvals[-1])
            coords_info[cname] = info

        flat = da.values.flatten()
        finite = flat[np.isfinite(flat)]
        nan_count = int(flat.size - finite.size)
        if finite.size > 0:
            pcts = np.percentile(finite, [25, 50, 75])
            statistics = {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "25%": float(pcts[0]),
                "50%": float(pcts[1]),
                "75%": float(pcts[2]),
            }
        else:
            statistics = {"min": None, "max": None, "mean": None, "std": None}

        time_start = time_end = time_span = median_cadence = None
        if n_time > 0:
            times = da.coords["time"].values
            time_start = str(times[0])
            time_end = str(times[-1])
            time_span = str(pd.Timestamp(times[-1]) - pd.Timestamp(times[0]))
            if n_time > 1:
                dt = pd.Series(times).diff().dropna()
                median_cadence = str(dt.median())

        return {
            "status": "success",
            "label": entry.label,
            "units": entry.units,
            "storage_type": "xarray",
            "dims": dims,
            "coordinates": coords_info,
            "num_points": n_time,
            "time_start": time_start,
            "time_end": time_end,
            "time_span": time_span,
            "median_cadence": median_cadence,
            "nan_count": nan_count,
            "nan_percentage": round(nan_count / flat.size * 100, 1)
            if flat.size > 0
            else 0,
            "statistics": statistics,
        }

    df = entry.data
    if (time_start or time_end) and entry.is_timeseries:
        try:
            if time_start and time_end:
                df = df.loc[pd.Timestamp(time_start) : pd.Timestamp(time_end)]
            elif time_start:
                df = df.loc[pd.Timestamp(time_start) :]
            else:
                df = df.loc[: pd.Timestamp(time_end)]
        except (ValueError, TypeError) as e:
            return {"status": "error", "message": f"Invalid time range: {e}"}
    stats = {}

    desc = df.describe(percentiles=[0.25, 0.5, 0.75], include="all")
    for col in df.columns:
        if df[col].dtype.kind in ("f", "i", "u"):
            col_stats = {
                "min": float(desc.loc["min", col]),
                "max": float(desc.loc["max", col]),
                "mean": float(desc.loc["mean", col]),
                "std": float(desc.loc["std", col]),
                "25%": float(desc.loc["25%", col]),
                "50%": float(desc.loc["50%", col]),
                "75%": float(desc.loc["75%", col]),
            }
        else:
            col_stats = {
                "type": str(df[col].dtype),
                "count": int(desc.loc["count", col]),
                "unique": int(desc.loc["unique", col])
                if "unique" in desc.index
                else None,
                "top": str(desc.loc["top", col]) if "top" in desc.index else None,
            }
        stats[col] = col_stats

    nan_count = int(df.isna().sum().sum())
    total_points = len(df)
    if total_points == 0:
        return {
            "status": "success",
            "label": entry.label,
            "units": entry.units,
            "num_points": 0,
            "message": "No data points in the requested range.",
        }
    time_span = str(df.index[-1] - df.index[0]) if total_points > 1 else "single point"

    if total_points > 1:
        dt = df.index.to_series().diff().dropna()
        median_cadence = str(dt.median())
    else:
        median_cadence = "N/A"

    return {
        "status": "success",
        "label": entry.label,
        "units": entry.units,
        "num_points": total_points,
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "time_start": str(df.index[0]),
        "time_end": str(df.index[-1]),
        "time_span": time_span,
        "median_cadence": median_cadence,
        "nan_count": nan_count,
        "nan_percentage": round(nan_count / (total_points * len(df.columns)) * 100, 1)
        if total_points > 0
        else 0,
        "statistics": stats,
    }


def handle_preview_data(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    store = orch._store
    key = tool_args.get("data_id") or tool_args.get("label")
    entry = store.get(key)
    if entry is None:
        return {
            "status": "error",
            "message": f"Data '{key}' not found in memory",
        }

    time_start = tool_args.get("time_start")
    time_end = tool_args.get("time_end")

    if entry.is_xarray:
        da = entry.data
        if (time_start or time_end) and "time" in da.dims:
            sel_kw = {}
            if time_start and time_end:
                sel_kw["time"] = slice(time_start, time_end)
            elif time_start:
                sel_kw["time"] = slice(time_start, None)
            else:
                sel_kw["time"] = slice(None, time_end)
            da = da.sel(**sel_kw)
        n_time = da.sizes.get("time", 0)
        n_rows = min(
            tool_args.get("n_rows", 3), get_item_limit("items.data_preview_xr")
        )
        position = tool_args.get("position", "both")

        def _xr_time_slice(indices):
            rows = []
            for i in indices:
                sl = da.isel(time=i)
                vals = sl.values.flatten()
                finite = vals[np.isfinite(vals)]
                rows.append(
                    {
                        "timestamp": str(da.coords["time"].values[i]),
                        "shape": list(sl.shape),
                        "min": float(np.min(finite)) if finite.size > 0 else None,
                        "max": float(np.max(finite)) if finite.size > 0 else None,
                        "mean": float(np.mean(finite)) if finite.size > 0 else None,
                        "nan_count": int(vals.size - finite.size),
                    }
                )
            return rows

        result = {
            "status": "success",
            "label": entry.label,
            "units": entry.units,
            "storage_type": "xarray",
            "dims": dict(da.sizes),
            "total_time_steps": n_time,
        }
        if n_time > 0:
            if position == "sampled":
                sample_limit = get_item_limit("items.data_sample_points")
                stride = tool_args.get("stride") or max(1, n_time // sample_limit)
                indices = list(range(0, n_time, stride))[:sample_limit]
                result["sampled"] = _xr_time_slice(indices)
                result["stride_used"] = stride
            else:
                head_idx = list(range(min(n_rows, n_time)))
                tail_idx = list(range(max(n_time - n_rows, 0), n_time))
                if position in ("head", "both"):
                    result["head"] = _xr_time_slice(head_idx)
                if position in ("tail", "both"):
                    result["tail"] = _xr_time_slice(tail_idx)

        return result

    df = entry.data
    if (time_start or time_end) and entry.is_timeseries:
        try:
            if time_start and time_end:
                df = df.loc[pd.Timestamp(time_start) : pd.Timestamp(time_end)]
            elif time_start:
                df = df.loc[pd.Timestamp(time_start) :]
            else:
                df = df.loc[: pd.Timestamp(time_end)]
        except (ValueError, TypeError) as e:
            return {"status": "error", "message": f"Invalid time range: {e}"}
    n_rows = min(tool_args.get("n_rows", 5), get_item_limit("items.data_preview_rows"))
    position = tool_args.get("position", "both")

    def _df_to_rows(sub_df):
        rows = []
        for ts, row in sub_df.iterrows():
            d = {"timestamp": str(ts)}
            for col in sub_df.columns:
                v = row[col]
                d[col] = float(v) if isinstance(v, (int, float)) else str(v)
            rows.append(d)
        return rows

    result = {
        "status": "success",
        "label": entry.label,
        "units": entry.units,
        "total_rows": len(df),
        "columns": list(df.columns),
    }

    if position == "sampled":
        sample_limit = get_item_limit("items.data_sample_points")
        stride = tool_args.get("stride") or max(1, len(df) // sample_limit)
        sampled = df.iloc[::stride]
        if len(sampled) > sample_limit:
            restride = max(1, len(df) // sample_limit)
            sampled = df.iloc[::restride].head(sample_limit)
        result["sampled"] = _df_to_rows(sampled)
        result["stride_used"] = stride
    else:
        if position in ("head", "both"):
            result["head"] = _df_to_rows(df.head(n_rows))
        if position in ("tail", "both"):
            result["tail"] = _df_to_rows(df.tail(n_rows))

    return result


def handle_manage_data(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    action = tool_args.get("action", "")
    if action == "merge":
        return _handle_merge_datasets(orch, tool_args)
    elif action == "save":
        return _handle_save_data(orch, tool_args)
    return {"status": "error", "message": f"Unknown manage_data action: {action!r}. Use 'merge' or 'save'."}


def _handle_save_data(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    store = orch._store
    key = tool_args.get("data_id") or tool_args.get("label")
    if not key:
        return {"status": "error", "message": "Missing required parameter: data_id"}
    entry = store.get(key)
    if entry is None:
        return {
            "status": "error",
            "message": f"Data '{key}' not found in memory",
        }

    from pathlib import Path

    if entry.is_xarray:
        da = entry.data
        filename = tool_args.get("filename", "")
        if not filename:
            safe_label = entry.label.replace(".", "_").replace("/", "_")
            filename = f"{safe_label}.nc"
        if not filename.endswith(".nc"):
            filename += ".nc"

        parent = Path(filename).parent
        if parent and str(parent) != "." and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        da.to_netcdf(
            filename,
            encoding={"time": {"units": "nanoseconds since 1970-01-01"}},
        )
        filepath = str(Path(filename).resolve())
        file_size = Path(filename).stat().st_size

        orch._event_bus.emit(
            DEBUG,
            level="debug",
            msg=f"[DataOps] Exported xarray '{entry.label}' to {filepath} ({file_size:,} bytes)",
        )

        return {
            "status": "success",
            "label": entry.label,
            "filepath": filepath,
            "format": "netcdf",
            "dims": dict(da.sizes),
            "file_size_bytes": file_size,
        }

    filename = tool_args.get("filename", "")
    if not filename:
        safe_label = entry.label.replace(".", "_").replace("/", "_")
        filename = f"{safe_label}.csv"
    if not filename.endswith(".csv"):
        filename += ".csv"

    parent = Path(filename).parent
    if parent and str(parent) != "." and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

    df = entry.data.copy()
    df.index.name = "timestamp"
    df.to_csv(filename, date_format="%Y-%m-%dT%H:%M:%S.%fZ")

    filepath = str(Path(filename).resolve())
    file_size = Path(filename).stat().st_size

    orch._event_bus.emit(
        DEBUG,
        level="debug",
        msg=f"[DataOps] Exported '{entry.label}' to {filepath} ({file_size:,} bytes)",
    )

    return {
        "status": "success",
        "label": entry.label,
        "filepath": filepath,
        "num_points": len(df),
        "num_columns": len(df.columns),
        "file_size_bytes": file_size,
    }


def _handle_merge_datasets(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    """Merge multiple datasets into one."""
    store = orch._store
    ids = tool_args.get("data_ids", [])
    if len(ids) < 2:
        return {"status": "error", "message": "Need at least 2 data IDs to merge"}

    try:
        merged = store.merge_entries(ids)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    return {"status": "success", **merged.summary()}
