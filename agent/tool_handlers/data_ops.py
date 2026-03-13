from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from data_ops.store import DataEntry, generate_id
from agent.truncation import get_item_limit

if TYPE_CHECKING:
    from agent.tool_caller import ToolCaller
    from agent.tool_context import ToolContext


def handle_assets(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    action = tool_args.get("action", "list")
    if action != "list":
        return {"status": "error", "message": f"Unknown assets action: {action!r}. Only 'list' is supported."}

    kind = tool_args.get("kind")
    asset_registry = ctx.asset_registry
    if asset_registry is None:
        return {"status": "error", "message": "Asset registry not available"}
    return {"status": "success", **asset_registry.list_assets_enriched(kind=kind)}


def handle_describe_data(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    store = ctx.store
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


def handle_preview_data(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    store = ctx.store
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


def handle_manage_data(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    action = tool_args.get("action", "")
    if action == "describe":
        return handle_describe_data(ctx, tool_args, caller)
    elif action == "preview":
        return handle_preview_data(ctx, tool_args, caller)
    elif action == "merge":
        return _handle_merge_datasets(ctx, tool_args)
    elif action == "save":
        return _handle_save_data(ctx, tool_args)
    elif action == "delete":
        return _handle_delete_data(ctx, tool_args)
    return {"status": "error", "message": f"Unknown manage_data action: {action!r}. Use 'describe', 'preview', 'merge', 'save', or 'delete'."}


def _handle_save_data(ctx: "ToolContext", tool_args: dict) -> dict:
    store = ctx.store
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

        from agent.event_bus import DEBUG
        if ctx.event_bus is not None:
            ctx.event_bus.emit(
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

    from agent.event_bus import DEBUG
    if ctx.event_bus is not None:
        ctx.event_bus.emit(
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


def _handle_merge_datasets(ctx: "ToolContext", tool_args: dict) -> dict:
    """Merge multiple datasets into one."""
    store = ctx.store
    ids = tool_args.get("data_ids", [])
    if len(ids) < 2:
        return {"status": "error", "message": "Need at least 2 data IDs to merge"}

    try:
        merged = store.merge_entries(ids)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    return {"status": "success", **merged.summary()}


def _handle_delete_data(ctx: "ToolContext", tool_args: dict) -> dict:
    """Remove a dataset from the in-memory store."""
    store = ctx.store
    key = tool_args.get("data_id") or tool_args.get("label")
    if not key:
        return {"status": "error", "message": "Missing required parameter: data_id"}
    entry = store.get(key)
    if entry is None:
        return {"status": "error", "message": f"Data '{key}' not found in memory"}
    label = entry.label
    removed = store.remove(key)
    if not removed:
        return {"status": "error", "message": f"Failed to remove '{key}'"}

    from agent.event_bus import DEBUG
    if ctx.event_bus is not None:
        ctx.event_bus.emit(
            DEBUG,
            level="debug",
            msg=f"[DataOps] Deleted '{label}' (id={key}) from memory",
        )
    return {"status": "success", "deleted": key, "label": label}
