"""Handler for the run_code tool — sandboxed Python execution."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from data_ops.sandbox import validate_code_blocklist, execute_sandboxed
from data_ops.store import DataEntry, generate_id
from agent.event_bus import DATA_COMPUTED, DEBUG

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent


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


def handle_run_code(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    """Execute Python code in a sandboxed environment."""
    code = tool_args["code"]
    inputs = tool_args.get("inputs", [])
    store_as = tool_args.get("store_as")
    description = tool_args.get("description", "")

    sandbox_dir = Path(orch._session_dir) / "sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validate code against blocklist
    violations = validate_code_blocklist(code)
    if violations:
        return {"status": "error", "message": "Blocked: " + "; ".join(violations)}

    # 2. Stage inputs from the store as files
    store = orch._store
    staged_files = []
    for label in inputs:
        entry = store.get(label)
        if entry is None:
            return {"status": "error", "message": f"Input '{label}' not found in store"}
        _stage_entry(entry, sandbox_dir)
        staged_files.append(label)

    # 3. Execute in sandbox subprocess
    try:
        output, result_value = execute_sandboxed(
            code,
            work_dir=sandbox_dir,
        )
    except (RuntimeError, TimeoutError) as e:
        return {"status": "error", "message": str(e)}

    # 4. Optionally store result
    response = {"status": "success", "output": output}
    if store_as and result_value is not None:
        entry = DataEntry(
            label=store_as,
            data=result_value,
            description=description,
            source="computed",
        )
        entry.id = generate_id(store, output_label=store_as)
        store.put(entry)
        response["stored"] = {"label": store_as, "type": entry.data_type}

        orch._event_bus.emit(
            DATA_COMPUTED,
            agent="orchestrator",
            msg=f"[Sandbox] run_code -> '{store_as}'",
            data={
                "args": {
                    "inputs": inputs,
                    "code": code,
                    "store_as": store_as,
                    "description": description,
                },
                "inputs": inputs,
                "outputs": [entry.id],
            },
        )
    else:
        orch._event_bus.emit(
            DEBUG,
            level="debug",
            msg=f"[Sandbox] run_code executed: {description}",
        )

    if staged_files:
        response["staged_inputs"] = staged_files

    return response
