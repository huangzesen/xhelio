"""Handler for the run_code tool — sandboxed Python execution."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from data_ops.sandbox import validate_code_blocklist, execute_sandboxed
from data_ops.store import DataEntry, generate_id
from agent.event_bus import DATA_COMPUTED

if TYPE_CHECKING:
    from agent.tool_caller import ToolCaller
    from agent.tool_context import ToolContext


def _stage_entry(entry: DataEntry, sandbox_dir: Path) -> None:
    """Write a store entry to the sandbox dir as a file the code can read."""
    data = entry.data
    if isinstance(data, pd.DataFrame):
        data.to_parquet(sandbox_dir / f"{entry.label}.parquet")
    elif isinstance(data, xr.DataArray):
        data.to_netcdf(str(sandbox_dir / f"{entry.label}.nc"))
    elif isinstance(data, dict):
        import json
        with open(sandbox_dir / f"{entry.label}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    elif isinstance(data, str):
        with open(sandbox_dir / f"{entry.label}.txt", "w", encoding="utf-8") as f:
            f.write(data)
    elif isinstance(data, bytes):
        with open(sandbox_dir / f"{entry.label}.bin", "wb") as f:
            f.write(data)


def _stage_meta(entry: DataEntry, sandbox_dir: Path) -> None:
    """Write entry metadata to the sandbox dir as a JSON file."""
    import json
    meta = {
        "label": entry.label,
        "units": entry.units,
        "description": entry.description,
        "source": entry.source,
        "time_range": entry.time_range,
        "physical_quantity": entry.physical_quantity,
        "array_shape": entry.array_shape,
    }
    if entry.metadata:
        meta.update(entry.metadata)
    with open(sandbox_dir / f"{entry.label}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def _read_output_file(sandbox_path: Path, var_name: str, fmt: str):
    """Read an output file written by the sandbox subprocess."""
    if fmt == "parquet":
        return pd.read_parquet(sandbox_path / f"{var_name}.parquet")
    elif fmt == "nc":
        return xr.open_dataarray(sandbox_path / f"{var_name}.nc")
    elif fmt == "json":
        import json
        with open(sandbox_path / f"{var_name}.json") as f:
            return json.load(f)
    return None


def handle_run_code(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    """Execute Python code in a sandboxed environment."""
    import shutil
    import tempfile

    code = tool_args["code"]
    inputs = tool_args.get("inputs", [])
    outputs = tool_args.get("outputs", {})  # {store_label: variable_name}
    description = tool_args.get("description", "")
    timeout = tool_args.get("timeout", 30)

    # 1. Validate code against blocklist
    violations = validate_code_blocklist(code)
    if violations:
        return {"status": "error", "message": "Blocked: " + "; ".join(violations)}

    # 2. Resolve sandbox directory — persistent if session exists, temp otherwise
    store = ctx.store
    if ctx.session_dir is not None:
        sandbox_path = ctx.session_dir / "sandbox"
        sandbox_path.mkdir(parents=True, exist_ok=True)
        use_temp = False
    else:
        sandbox_path = Path(tempfile.mkdtemp(prefix="xhelio_sandbox_"))
        use_temp = True

    try:
        # 3. Stage declared inputs
        staged_files = []
        staged_file_assets = {}
        for label in inputs:
            if label.startswith("file_") and ctx.asset_registry is not None:
                # File asset — stage from session files dir
                asset = ctx.asset_registry.get_asset(label)
                if asset is None:
                    return {"status": "error", "message": f"File asset '{label}' not found"}
                if not asset.session_path:
                    return {"status": "error", "message": f"File asset '{label}' has no session copy"}
                src = Path(asset.session_path)
                if not src.exists():
                    return {"status": "error", "message": f"File asset '{label}' missing: {asset.session_path}"}
                original_name = asset.metadata.get("original_filename", src.name)
                dest_in_sandbox = sandbox_path / original_name
                if dest_in_sandbox.exists():
                    return {"status": "error", "message": f"Filename collision: '{original_name}' already in sandbox"}
                shutil.copy2(str(src), str(dest_in_sandbox))
                staged_files.append(label)
                staged_file_assets[label] = original_name
            else:
                # DataStore entry — existing logic
                entry = store.get(label)
                if entry is None:
                    return {"status": "error", "message": f"Input '{label}' not found in store"}
                _stage_entry(entry, sandbox_path)
                _stage_meta(entry, sandbox_path)
                staged_files.append(label)

        # 4. Execute in sandbox subprocess
        output_vars = list(outputs.values())
        try:
            output, results = execute_sandboxed(
                code,
                work_dir=sandbox_path,
                timeout=timeout,
                output_vars=output_vars,
            )
        except (RuntimeError, TimeoutError) as e:
            return {"status": "error", "message": str(e)}

        # 5. Store each successful output
        stored = []
        errors = []
        for store_label, var_name in outputs.items():
            var_result = results.get(var_name, {"status": "missing"})
            if var_result["status"] != "ok":
                errors.append(f"Output '{var_name}' for label '{store_label}': missing from namespace")
                continue
            fmt = var_result["format"]
            data = _read_output_file(sandbox_path, var_name, fmt)
            if data is None:
                errors.append(f"Output '{var_name}': failed to read {fmt} file")
                continue
            entry = DataEntry(
                label=store_label,
                data=data,
                description=description,
                source="computed",
            )
            entry.id = generate_id(store, output_label=store_label)
            store.put(entry)
            stored.append({"label": store_label, "type": entry.data_type})

    finally:
        # Clean up only if using temp dir (non-persistent session)
        if use_temp:
            shutil.rmtree(sandbox_path, ignore_errors=True)

    # 6. Emit pipeline event for DAG tracking
    if ctx.event_bus is not None:
        outputs_map = {s["label"]: s.get("type", "") for s in stored} if stored else {}
        ctx.event_bus.emit(
            DATA_COMPUTED,
            agent=caller.agent_id if caller else "unknown",
            msg=f"[RunCode] {description}",
            data={
                "tool": "run_code",
                "args": {"code": code, "description": description},
                "inputs": inputs,
                "outputs": outputs_map,
                "status": "success" if not errors else "partial",
            },
        )

    # 7. Build response
    response = {"status": "success" if not errors else "partial", "output": output}
    if stored:
        response["stored"] = stored
    if errors:
        response["errors"] = errors
    if staged_files:
        response["staged_inputs"] = staged_files
    if staged_file_assets:
        response["staged_file_assets"] = staged_file_assets
    return response
