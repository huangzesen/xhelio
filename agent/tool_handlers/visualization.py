"""Visualization tool handlers."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

from data_ops.store import resolve_entry
from agent.event_bus import (
    RENDER_EXECUTED,
    MPL_RENDER_EXECUTED,
    JSX_RENDER_EXECUTED,
    PLOT_ACTION,
)
from agent.logging import get_logger

if TYPE_CHECKING:
    from agent.tool_caller import ToolCaller
    from agent.tool_context import ToolContext


def handle_render_plotly_json(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    """Handle render_plotly_json: fill data_label placeholders and render."""
    fig_json = tool_args.get("figure_json", {})
    data_traces = fig_json.get("data", [])
    if not data_traces:
        return {
            "status": "error",
            "message": (
                "figure_json.data is required and must be a non-empty array of traces. "
                "Each trace needs at least a 'data_label' key. Example: "
                'render_plotly_json(figure_json={"data": [{"type": "scatter", '
                '"data_label": "DATASET.Parameter"}], "layout": {}}). '
                "Call assets first to discover available labels."
            ),
        }

    # Guard: reject figures with too many layout objects (shapes + annotations).
    # LLMs struggle to generate large arrays of complex objects — the JSON
    # often arrives garbled (dicts collapsed to floats, arrays to integers).
    layout = fig_json.get("layout", {})
    n_shapes = len(layout.get("shapes", []))
    n_annotations = len(layout.get("annotations", []))
    _MAX_LAYOUT_OBJECTS = 30
    if n_shapes + n_annotations > _MAX_LAYOUT_OBJECTS:
        return {
            "status": "error",
            "message": (
                f"Too many layout objects: {n_shapes} shapes + {n_annotations} annotations "
                f"= {n_shapes + n_annotations} (limit: {_MAX_LAYOUT_OBJECTS}). "
                f"Reduce the number of shapes/annotations. For many similar markers, "
                f"consider: (1) showing only the most significant events, "
                f"(2) using a single legend entry instead of per-event labels, or "
                f"(3) omitting annotations and keeping only the shapes."
            ),
        }

    # Collect all data_label values and resolve entries
    store = ctx.store
    entry_map: dict = {}
    for trace in data_traces:
        label = trace.get("data_label")
        if label and label not in entry_map:
            entry, _ = resolve_entry(store, label)
            if entry is None:
                available = [e["label"] for e in store.list_entries()]
                return {
                    "status": "error",
                    "message": (
                        f"data_label '{label}' not found in memory. "
                        f"Available labels: {available}"
                    ),
                }
            entry_map[label] = entry

    # Validate non-empty data
    for label, entry in entry_map.items():
        if len(entry.data) == 0:
            return {
                "status": "error",
                "message": f"Entry '{label}' has no data points",
            }

    try:
        result = ctx.renderer.render_plotly_json(fig_json, entry_map)
    except Exception as e:
        return {"status": "error", "message": str(e)}

    if result.get("status") == "success":
        # Emit pipeline event for DAG tracking
        if ctx.event_bus is not None:
            ctx.event_bus.emit(
                RENDER_EXECUTED,
                agent=caller.agent_id if caller else "VizAgent[Plotly]",
                msg="[PlotlyViz] Rendered figure",
                data={
                    "tool": "render_plotly_json",
                    "args": {"description": result.get("description", "Plotly figure")},
                    "inputs": list(entry_map.keys()),
                    "outputs": {},
                    "status": "success",
                },
            )

        # Cache rendered PNG for auto-review via vision
        figure = ctx.renderer.get_figure()
        if figure is not None:
            import io

            try:
                buf = io.BytesIO()
                figure.write_image(
                    buf, format="png", width=1100, height=600, scale=2
                )
                if getattr(ctx, 'eureka_hooks', None) is not None:
                    ctx.eureka_hooks.latest_render_png = buf.getvalue()
            except Exception as e:
                get_logger().warning(
                    f"Failed to cache Plotly PNG for insight review: {e}"
                )

        # Save figure JSON + thumbnail PNG to session for output verification
        import json as _json
        from datetime import datetime, timezone

        session_dir = ctx.session_dir
        plotly_dir = session_dir / "plotly_outputs"
        plotly_dir.mkdir(parents=True, exist_ok=True)
        op_id = f"render_{datetime.now(timezone.utc).strftime('%H%M%S%f')}"

        json_path = plotly_dir / f"{op_id}.json"
        json_path.write_text(_json.dumps(fig_json, default=str))

        png_path = plotly_dir / f"{op_id}.png"
        if getattr(ctx, 'eureka_hooks', None) is not None and ctx.eureka_hooks.latest_render_png:
            png_path.write_bytes(ctx.eureka_hooks.latest_render_png)
        elif figure is not None:
            import io as _io

            try:
                buf = _io.BytesIO()
                figure.write_image(
                    buf, format="png", width=1100, height=600, scale=2
                )
                png_bytes = buf.getvalue()
                png_path.write_bytes(png_bytes)
                if getattr(ctx, 'eureka_hooks', None) is not None:
                    ctx.eureka_hooks.latest_render_png = png_bytes
            except Exception:
                pass

        result["output_files"] = [str(json_path)]
        if png_path.exists():
            result["output_files"].append(str(png_path))

        # Register figure in the asset registry
        if ctx.asset_registry is not None:
            thumbnail = str(png_path) if png_path.exists() else None
            fig_asset = ctx.asset_registry.register_figure(
                fig_json=fig_json,
                trace_labels=list(entry_map.keys()),
                panel_count=result.get("panels", 1),
                op_id=op_id,
                thumbnail_path=thumbnail,
            )
            fig_asset.figure_kind = "plotly"
            result["asset_id"] = fig_asset.asset_id

        if not json_path.is_file() or json_path.stat().st_size == 0:
            result["status"] = "error"
            result["message"] = (
                f"Render completed but output file is missing or empty: {json_path}. "
                "The figure JSON could not be saved to disk."
            )

        if getattr(ctx, 'eureka_hooks', None) is not None:
            review = ctx.eureka_hooks.sync_insight_review()
            if review is not None and not review.get("failed", False):
                passed = review.get("passed", True)
                verdict = "PASS" if passed else "NEEDS_IMPROVEMENT"
                result["insight_review"] = {
                    "verdict": verdict,
                    "feedback": review.get("text", ""),
                    "suggestions": review.get("suggestions", []),
                }
                if not passed:
                    result["insight_review"]["action_needed"] = (
                        "The automatic figure review found issues. "
                        "Review the feedback and suggestions above, then "
                        "re-render with improvements."
                    )

    return result


def handle_manage_plot(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    """Handle the manage_plot tool call."""
    action = tool_args.get("action")
    if not action:
        return {"status": "error", "message": "action is required"}

    if action == "reset":
        if ctx.event_bus is not None:
            ctx.event_bus.emit(
                PLOT_ACTION,
                agent="orchestrator",
                msg="[Plot] reset",
                data={"args": {"action": "reset"}, "outputs": []},
            )
        return ctx.renderer.reset()

    elif action == "get_state":
        return ctx.renderer.get_current_state()

    else:
        return {"status": "error", "message": f"Unknown action: {action}"}


# --- Matplotlib Visualization Tool Handlers ---


def handle_generate_mpl_script(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    """Handle the generate_mpl_script tool call."""
    code = tool_args.get("script")
    if not code:
        return {"status": "error", "message": "script parameter is required"}

    description = tool_args.get("description", "Matplotlib plot")

    # 1. Extract data labels from user script
    from rendering.mpl_sandbox import extract_data_labels

    data_labels = extract_data_labels(code)

    # 2. Stage data + metadata into sandbox dir
    from agent.tool_handlers.sandbox import _stage_entry, _stage_meta

    session_dir = ctx.session_dir
    sandbox_dir = session_dir / "sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    for label in data_labels:
        entry = ctx.store.get(label)
        if entry is not None:
            _stage_entry(entry, sandbox_dir)
            _stage_meta(entry, sandbox_dir)

    # available_labels() returns all store labels (without staging their data)
    all_labels = [e["label"] for e in ctx.store.list_entries()]

    # 3. Generate script_id and output path
    from datetime import datetime as _dt
    import secrets

    script_id = _dt.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
    output_dir = session_dir / "mpl_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{script_id}.png"

    # 4. Build preamble + epilogue
    import json as _json
    labels_json = _json.dumps(all_labels)

    preamble = f'''import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

_OUTPUT_PATH = {repr(str(output_path))}
_STAGED_LABELS = json.loads({repr(labels_json)})

def load_data(label):
    """Load a DataFrame from staged data by label."""
    from pathlib import Path
    p = Path(f"{{label}}.parquet")
    if not p.exists():
        raise KeyError(f"Label '{{label}}' not found. Available labels: {{available_labels()}}")
    return pd.read_parquet(p)

def load_meta(label):
    """Load metadata dict for a label."""
    from pathlib import Path
    p = Path(f"{{label}}.meta.json")
    if not p.exists():
        return {{}}
    with open(p) as f:
        return json.load(f)

def available_labels():
    """Return all available data labels."""
    return _STAGED_LABELS

# === User script starts below ===
'''

    epilogue = f'''

# === Auto-generated epilogue ===
plt.savefig(_OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close("all")
'''

    wrapped_code = preamble + code + epilogue

    # 5. Validate via blocklist
    from data_ops.sandbox import validate_code_blocklist

    violations = validate_code_blocklist(code)
    if violations:
        return {
            "status": "error",
            "message": "Script validation failed:\n" + "\n".join(f"  - {v}" for v in violations),
        }

    # 6. Save wrapped script to mpl_scripts/
    scripts_dir = session_dir / "mpl_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / f"{script_id}.py"
    script_path.write_text(wrapped_code, encoding="utf-8")

    # 7. Execute via sandbox
    from data_ops.sandbox import execute_sandboxed

    try:
        stdout_output, _ = execute_sandboxed(
            wrapped_code,
            work_dir=sandbox_dir,
            timeout=60,
        )
    except TimeoutError as e:
        return {
            "status": "error",
            "script_id": script_id,
            "message": f"Script execution timed out: {e}",
            "script_path": str(script_path),
        }

    # 8. Check if output PNG was created
    if output_path.exists() and output_path.stat().st_size > 0:
        # Success path
        _mpl_labels_used = data_labels
        if ctx.event_bus is not None:
            ctx.event_bus.emit(
                MPL_RENDER_EXECUTED,
                agent="VizAgent[Mpl]",
                msg=f"[MplViz] Script executed: {description}",
                data={
                    "script_id": script_id,
                    "description": description,
                    "output_path": str(output_path),
                    "script_path": str(script_path),
                    "args": {"script": code, "description": description},
                    "inputs": _mpl_labels_used,
                    "outputs": [],
                    "status": "success",
                },
            )
        # Cache rendered PNG for auto-review
        try:
            if getattr(ctx, 'eureka_hooks', None) is not None:
                ctx.eureka_hooks.latest_render_png = output_path.read_bytes()
        except Exception as e:
            get_logger().warning(
                f"Failed to cache matplotlib PNG for insight review: {e}"
            )

        response = {
            "status": "success",
            "script_id": script_id,
            "output_path": str(output_path),
            "output_files": [str(output_path)],
            "message": f"Matplotlib plot saved successfully. Script ID: {script_id}",
        }

        # Register in asset registry
        if ctx.asset_registry is not None:
            fig_asset = ctx.asset_registry.register_image(
                name=description,
                image_path=str(output_path),
            )
            fig_asset.figure_kind = "mpl"
            response["asset_id"] = fig_asset.asset_id

        if stdout_output and stdout_output.strip():
            response["stdout"] = stdout_output

        # Auto-review via vision
        if getattr(ctx, 'eureka_hooks', None) is not None:
            review = ctx.eureka_hooks.sync_insight_review()
            if review is not None and not review.get("failed", False):
                passed = review.get("passed", True)
                verdict = "PASS" if passed else "NEEDS_IMPROVEMENT"
                response["insight_review"] = {
                    "verdict": verdict,
                    "feedback": review.get("text", ""),
                    "suggestions": review.get("suggestions", []),
                }
                if not passed:
                    response["insight_review"]["action_needed"] = (
                        "The automatic figure review found issues. "
                        "Review the feedback and suggestions above, then "
                        "re-render with improvements."
                    )

        return response
    else:
        # Failure path
        response = {
            "status": "error",
            "script_id": script_id,
            "message": "Script execution failed. See stderr for details.",
            "stderr": stdout_output or "",
        }
        if script_path:
            response["script_path"] = str(script_path)
        return response


def handle_manage_mpl_output(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    """Handle the manage_mpl_output tool call."""
    action = tool_args.get("action")
    if not action:
        return {"status": "error", "message": "action is required"}

    session_dir = ctx.session_dir
    scripts_dir = session_dir / "mpl_scripts"
    outputs_dir = session_dir / "mpl_outputs"

    if action == "list":
        items = []
        if scripts_dir.exists():
            for script_file in sorted(scripts_dir.glob("*.py")):
                script_id = script_file.stem
                output_file = outputs_dir / f"{script_id}.png"
                items.append(
                    {
                        "script_id": script_id,
                        "has_output": output_file.exists(),
                        "script_path": str(script_file),
                        "output_path": str(output_file)
                        if output_file.exists()
                        else None,
                    }
                )
        return {"status": "success", "items": items, "count": len(items)}

    elif action == "get_script":
        script_id = tool_args.get("script_id")
        if not script_id:
            return {
                "status": "error",
                "message": "script_id is required for get_script",
            }
        script_file = scripts_dir / f"{script_id}.py"
        if not script_file.exists():
            return {"status": "error", "message": f"Script not found: {script_id}"}
        return {
            "status": "success",
            "script_id": script_id,
            "script": script_file.read_text(encoding="utf-8"),
        }

    elif action == "rerun":
        script_id = tool_args.get("script_id")
        if not script_id:
            return {"status": "error", "message": "script_id is required for rerun"}
        script_file = scripts_dir / f"{script_id}.py"
        if not script_file.exists():
            return {"status": "error", "message": f"Script not found: {script_id}"}

        # Read the saved script (already has preamble/epilogue baked in)
        saved_script = script_file.read_text(encoding="utf-8")

        # Re-stage all data into sandbox
        from agent.tool_handlers.sandbox import _stage_entry, _stage_meta
        from rendering.mpl_sandbox import extract_data_labels as _extract_mpl_labels

        sandbox_dir = ctx.session_dir / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        _rerun_labels = _extract_mpl_labels(saved_script)
        for label in _rerun_labels:
            entry = ctx.store.get(label)
            if entry is not None:
                _stage_entry(entry, sandbox_dir)
                _stage_meta(entry, sandbox_dir)

        # Execute via sandbox
        from data_ops.sandbox import execute_sandboxed

        output_path = outputs_dir / f"{script_id}.png"
        try:
            stdout_output, _ = execute_sandboxed(
                saved_script,
                work_dir=sandbox_dir,
                timeout=60,
            )
        except TimeoutError:
            return {"status": "error", "message": "Script timed out during rerun"}

        if output_path.exists() and output_path.stat().st_size > 0:
            if ctx.event_bus is not None:
                ctx.event_bus.emit(
                    MPL_RENDER_EXECUTED,
                    agent="VizAgent[Mpl]",
                    msg=f"[MplViz] Script re-executed: {script_id}",
                    data={
                        "script_id": script_id,
                        "description": f"Rerun of {script_id}",
                        "args": {
                            "script": saved_script,
                            "description": f"Rerun of {script_id}",
                        },
                        "inputs": _rerun_labels,
                        "outputs": [],
                        "status": "success",
                    },
                )
            # Cache rendered PNG for auto-review
            try:
                if getattr(ctx, 'eureka_hooks', None) is not None:
                    ctx.eureka_hooks.latest_render_png = output_path.read_bytes()
            except Exception as e:
                get_logger().warning(
                    f"Failed to cache matplotlib rerun PNG: {e}"
                )
            return {
                "status": "success",
                "script_id": script_id,
                "output_path": str(output_path),
                "output_files": [str(output_path)],
                "message": "Script re-executed successfully",
            }
        return {
            "status": "error",
            "stderr": stdout_output or "",
            "message": "Script re-execution failed",
        }

    elif action == "delete":
        script_id = tool_args.get("script_id")
        if not script_id:
            return {
                "status": "error",
                "message": "script_id is required for delete",
            }
        deleted = []
        script_file = scripts_dir / f"{script_id}.py"
        output_file = outputs_dir / f"{script_id}.png"
        if script_file.exists():
            script_file.unlink()
            deleted.append("script")
        if output_file.exists():
            output_file.unlink()
            deleted.append("output")
        if not deleted:
            return {
                "status": "error",
                "message": f"No files found for script_id: {script_id}",
            }
        return {"status": "success", "deleted": deleted, "script_id": script_id}

    else:
        return {"status": "error", "message": f"Unknown action: {action}"}


# --- JSX/Recharts Visualization ---


def handle_generate_jsx_component(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    """Handle the generate_jsx_component tool call."""
    code = tool_args.get("code")
    if not code:
        return {"status": "error", "message": "code parameter is required"}

    description = tool_args.get("description", "JSX component")

    # Generate script_id
    from datetime import datetime as _dt
    import secrets

    script_id = _dt.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)

    # Run the JSX pipeline
    from rendering.jsx_sandbox import run_jsx_pipeline
    import config as _config

    session_dir = ctx.session_dir
    output_dir = session_dir / "jsx_outputs"

    result = run_jsx_pipeline(
        code=code,
        store=ctx.store,
        output_dir=output_dir,
        script_id=script_id,
        timeout=30.0,
        max_points=_config.MAX_PLOT_POINTS,
    )

    if result.success:
        # Record in operations log
        from rendering.jsx_sandbox import extract_data_labels as _extract_jsx_labels

        _jsx_labels_used = _extract_jsx_labels(code)
        if ctx.event_bus is not None:
            ctx.event_bus.emit(
                JSX_RENDER_EXECUTED,
                agent="VizAgent[JSX]",
                msg=f"[JsxViz] Component compiled: {description}",
                data={
                    "script_id": script_id,
                    "description": description,
                    "output_path": result.output_path,
                    "data_path": result.data_path,
                    "script_path": result.script_path,
                    "args": {"code": code, "description": description},
                    "inputs": _jsx_labels_used,
                    "outputs": [],
                    "status": "success",
                },
            )
        response = {
            "status": "success",
            "script_id": script_id,
            "output_path": result.output_path,
            "data_path": result.data_path,
            "message": f"JSX component compiled successfully. Script ID: {script_id}",
        }

        # Register in asset registry
        if ctx.asset_registry is not None:
            fig_asset = ctx.asset_registry.register_image(
                name=description,
                image_path=result.output_path or "",
            )
            fig_asset.figure_kind = "jsx"
            response["asset_id"] = fig_asset.asset_id

        # Verify output bundle exists and is non-empty
        if result.output_path:
            output_path = Path(result.output_path)
            files = [str(output_path)]
            if result.data_path:
                files.append(result.data_path)
            response["output_files"] = files
            if not output_path.is_file() or output_path.stat().st_size == 0:
                response["status"] = "error"
                response["message"] = (
                    f"JSX compiled but output bundle is missing or empty: {output_path}."
                )

        return response
    else:
        response = {
            "status": "error",
            "script_id": script_id,
            "message": "JSX compilation failed. See stderr for details.",
            "stderr": result.stderr,
        }
        if result.script_path:
            response["script_path"] = result.script_path
        return response


def handle_manage_jsx_output(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    """Handle the manage_jsx_output tool call."""
    action = tool_args.get("action")
    if not action:
        return {"status": "error", "message": "action is required"}

    session_dir = ctx.session_dir
    scripts_dir = session_dir / "jsx_scripts"
    outputs_dir = session_dir / "jsx_outputs"

    if action == "list":
        items = []
        if scripts_dir.exists():
            for script_file in sorted(scripts_dir.glob("*.tsx")):
                script_id = script_file.stem
                bundle_file = outputs_dir / f"{script_id}.js"
                data_file = outputs_dir / f"{script_id}.data.json"
                items.append(
                    {
                        "script_id": script_id,
                        "has_bundle": bundle_file.exists(),
                        "has_data": data_file.exists(),
                        "script_path": str(script_file),
                    }
                )
        return {"status": "success", "items": items, "count": len(items)}

    elif action == "get_source":
        script_id = tool_args.get("script_id")
        if not script_id:
            return {
                "status": "error",
                "message": "script_id is required for get_source",
            }
        script_file = scripts_dir / f"{script_id}.tsx"
        if not script_file.exists():
            return {"status": "error", "message": f"Script not found: {script_id}"}
        return {
            "status": "success",
            "script_id": script_id,
            "source": script_file.read_text(encoding="utf-8"),
        }

    elif action == "recompile":
        script_id = tool_args.get("script_id")
        if not script_id:
            return {
                "status": "error",
                "message": "script_id is required for recompile",
            }
        script_file = scripts_dir / f"{script_id}.tsx"
        if not script_file.exists():
            return {"status": "error", "message": f"Script not found: {script_id}"}

        # Re-run the pipeline with the saved source
        code = script_file.read_text(encoding="utf-8")
        from rendering.jsx_sandbox import (
            run_jsx_pipeline,
            extract_data_labels as _extract_jsx_labels,
        )
        import config as _config

        result = run_jsx_pipeline(
            code=code,
            store=ctx.store,
            output_dir=outputs_dir,
            script_id=script_id,
            timeout=30.0,
            max_points=_config.MAX_PLOT_POINTS,
        )
        if result.success:
            _recompile_labels = _extract_jsx_labels(code)
            if ctx.event_bus is not None:
                ctx.event_bus.emit(
                    JSX_RENDER_EXECUTED,
                    agent="VizAgent[JSX]",
                    msg=f"[JsxViz] Component recompiled: {script_id}",
                    data={
                        "script_id": script_id,
                        "description": f"Recompile of {script_id}",
                        "args": {
                            "code": code,
                            "description": f"Recompile of {script_id}",
                        },
                        "inputs": _recompile_labels,
                        "outputs": [],
                        "status": "success",
                    },
                )
            return {
                "status": "success",
                "script_id": script_id,
                "message": "Component recompiled successfully",
            }
        return {"status": "error", "stderr": result.stderr}

    elif action == "delete":
        script_id = tool_args.get("script_id")
        if not script_id:
            return {
                "status": "error",
                "message": "script_id is required for delete",
            }
        deleted = []
        script_file = scripts_dir / f"{script_id}.tsx"
        bundle_file = outputs_dir / f"{script_id}.js"
        data_file = outputs_dir / f"{script_id}.data.json"
        for f, label in [
            (script_file, "source"),
            (bundle_file, "bundle"),
            (data_file, "data"),
        ]:
            if f.exists():
                f.unlink()
                deleted.append(label)
        if not deleted:
            return {
                "status": "error",
                "message": f"No files found for script_id: {script_id}",
            }
        return {"status": "success", "deleted": deleted, "script_id": script_id}

    else:
        return {"status": "error", "message": f"Unknown action: {action}"}
