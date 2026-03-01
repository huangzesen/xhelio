"""Pipeline tool handlers."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent


def handle_get_pipeline_info(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    if tool_args.get("list_library"):
        from data_ops.ops_library import get_ops_library

        entries = get_ops_library().get_top_entries()
        return {"status": "success", "library_entries": entries}

    pipeline = orch._get_or_build_pipeline()
    if len(pipeline) == 0:
        return {
            "status": "success",
            "message": "No pipeline operations recorded yet.",
        }

    node_id = tool_args.get("node_id")
    if node_id:
        detail = pipeline.node_detail(node_id)
        if detail is None:
            return {"status": "error", "message": f"Node '{node_id}' not found"}
        if detail.get("code"):
            from data_ops.ops_library import get_ops_library

            match = get_ops_library().find_matching_code(detail["code"])
            if match:
                detail["library_match"] = {
                    "id": match["id"],
                    "description": match["description"],
                    "use_count": match.get("use_count", 1),
                }
        return {"status": "success", **detail}

    return {"status": "success", **pipeline.to_summary()}


def handle_modify_pipeline_node(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    pipeline = orch._get_or_build_pipeline()
    action = tool_args.get("action", "")

    if action == "update_params":
        node_id = tool_args.get("node_id", "")
        params = tool_args.get("params", {})
        if not node_id:
            return {
                "status": "error",
                "message": "node_id is required for update_params",
            }
        if not params:
            return {
                "status": "error",
                "message": "params is required for update_params",
            }
        try:
            affected = pipeline.update_node_params(node_id, params)
            return {
                "status": "success",
                "action": "update_params",
                "node_id": node_id,
                "affected_nodes": sorted(affected),
                "stale_count": len(pipeline.get_stale_nodes()),
            }
        except KeyError as e:
            return {"status": "error", "message": str(e)}

    elif action == "remove":
        node_id = tool_args.get("node_id", "")
        if not node_id:
            return {
                "status": "error",
                "message": "node_id is required for remove",
            }
        try:
            result = pipeline.remove_node(node_id)
            return {"status": "success", "action": "remove", **result}
        except KeyError as e:
            return {"status": "error", "message": str(e)}

    elif action == "insert_after":
        after_id = tool_args.get("after_id", "")
        tool_type = tool_args.get("tool", "custom_operation")
        params = tool_args.get("params", {})
        output_label = tool_args.get("output_label", "")
        if not after_id:
            return {
                "status": "error",
                "message": "after_id is required for insert_after",
            }
        if not output_label:
            return {
                "status": "error",
                "message": "output_label is required for insert_after",
            }
        try:
            new_id = pipeline.insert_node(after_id, tool_type, params, output_label)
            return {
                "status": "success",
                "action": "insert_after",
                "new_node_id": new_id,
                "after_id": after_id,
                "stale_count": len(pipeline.get_stale_nodes()),
            }
        except KeyError as e:
            return {"status": "error", "message": str(e)}

    elif action == "apply_library_op":
        node_id = tool_args.get("node_id", "")
        library_entry_id = tool_args.get("library_entry_id", "")
        if not node_id:
            return {
                "status": "error",
                "message": "node_id is required for apply_library_op",
            }
        if not library_entry_id:
            return {
                "status": "error",
                "message": "library_entry_id is required for apply_library_op",
            }
        node = pipeline.get_node(node_id)
        if node is None:
            return {"status": "error", "message": f"Node '{node_id}' not found"}
        if node.tool != "custom_operation":
            return {
                "status": "error",
                "message": f"Node '{node_id}' is not a compute node (tool={node.tool})",
            }
        from data_ops.ops_library import get_ops_library

        lib = get_ops_library()
        entry = lib.get_entry_by_id(library_entry_id)
        if entry is None:
            return {
                "status": "error",
                "message": f"Library entry '{library_entry_id}' not found",
            }
        new_params = {
            "code": entry["code"],
            "description": entry["description"],
        }
        affected = pipeline.update_node_params(node_id, new_params)
        lib.record_reuse(library_entry_id)
        return {
            "status": "success",
            "action": "apply_library_op",
            "node_id": node_id,
            "library_entry_id": library_entry_id,
            "affected_nodes": sorted(affected),
            "stale_count": len(pipeline.get_stale_nodes()),
        }

    elif action == "save_to_library":
        node_id = tool_args.get("node_id", "")
        if not node_id:
            return {
                "status": "error",
                "message": "node_id is required for save_to_library",
            }
        node = pipeline.get_node(node_id)
        if node is None:
            return {"status": "error", "message": f"Node '{node_id}' not found"}
        if node.tool != "custom_operation":
            return {
                "status": "error",
                "message": f"Node '{node_id}' is not a compute node (tool={node.tool})",
            }
        code = node.params.get("code", "")
        if not code.strip():
            return {
                "status": "error",
                "message": f"Node '{node_id}' has no code to save",
            }
        description = node.params.get("description", "")
        source_labels = list(node.inputs)
        units = node.params.get("units", "")
        from data_ops.ops_library import get_ops_library

        entry = get_ops_library().add_or_update(
            description=description,
            code=code,
            source_labels=source_labels,
            units=units,
        )
        return {
            "status": "success",
            "action": "save_to_library",
            "node_id": node_id,
            "library_entry": entry,
        }

    else:
        return {
            "status": "error",
            "message": f"Unknown action: {action}. Use 'update_params', 'remove', 'insert_after', 'apply_library_op', or 'save_to_library'.",
        }


def handle_execute_pipeline(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    pipeline = orch._get_or_build_pipeline()
    stale = pipeline.get_stale_nodes()
    if not stale:
        return {
            "status": "success",
            "message": "No stale nodes to execute.",
            "executed": 0,
        }

    use_cache = tool_args.get("use_cache", True)
    cache_store = orch._store if use_cache else None
    store = orch._store

    result = pipeline.execute_stale(
        store,
        cache_store=cache_store,
        renderer=orch._renderer,
    )
    return {"status": "success", **result}


def handle_save_pipeline(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    session_id = orch._session_id
    if not session_id:
        return {
            "status": "error",
            "message": "No active session to save pipeline from",
        }

    import config

    ops_log = orch._ops_log
    session_dir = config.get_data_dir() / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    ops_log.save_to_file(session_dir / "operations.json")

    pipeline_actions = orch._run_memory_agent_for_pipelines()

    return {
        "status": "success",
        "pipelines_registered": pipeline_actions,
    }


def handle_run_pipeline(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from data_ops.pipeline import SavedPipeline, topological_sort_steps
    from knowledge.mission_prefixes import (
        match_dataset_to_mission,
        get_canonical_id,
    )
    from agent.llm.base import ToolCall
    import config

    if tool_args.get("list_pipelines"):
        pipelines = SavedPipeline.list_all()
        return {"status": "success", "pipelines": pipelines}

    pipeline_id = tool_args.get("pipeline_id")
    time_start = tool_args.get("time_start")
    time_end = tool_args.get("time_end")

    if not pipeline_id:
        return {
            "status": "error",
            "message": "pipeline_id is required (or set list_pipelines=true)",
        }
    if not time_start or not time_end:
        return {
            "status": "error",
            "message": "time_start and time_end are required",
        }

    try:
        pipeline = SavedPipeline.load(pipeline_id)
    except FileNotFoundError:
        return {
            "status": "error",
            "message": f"Pipeline '{pipeline_id}' not found",
        }

    issues = pipeline.validate()
    if issues:
        return {
            "status": "error",
            "message": f"Pipeline validation failed: {'; '.join(issues)}",
        }

    sorted_steps = topological_sort_steps(pipeline.steps)
    step_to_label = {
        s["step_id"]: s["output_label"] for s in sorted_steps if s.get("output_label")
    }

    batches: list[tuple[tuple[str, str | None], list[ToolCall]]] = []
    current_target: tuple[str, str | None] | None = None
    current_calls: list[ToolCall] = []

    for step in sorted_steps:
        tool = step["tool"]
        params = dict(step.get("params", {}))

        if tool == "fetch_data":
            step_args = dict(params)
            step_args["time_start"] = time_start
            step_args["time_end"] = time_end
            stem, _ = match_dataset_to_mission(params["dataset_id"])
            mission_id = get_canonical_id(stem) if stem else "UNKNOWN"
            target: tuple[str, str | None] = ("mission", mission_id)

        elif tool in ("custom_operation", "store_dataframe"):
            step_args = dict(params)
            if tool == "custom_operation":
                source_labels = [
                    step_to_label[sid]
                    for sid in step.get("inputs", [])
                    if sid in step_to_label
                ]
                step_args["source_labels"] = source_labels
            step_args["output_label"] = step.get("output_label", "")
            target = ("dataops", None)

        elif tool == "render_plotly_json":
            step_args = dict(params)
            target = ("viz_plotly", None)

        else:
            continue

        tc = ToolCall(name=tool, args=step_args)

        if target != current_target:
            if current_calls:
                batches.append((current_target, current_calls))
            current_target = target
            current_calls = [tc]
        else:
            current_calls.append(tc)

    if current_calls:
        batches.append((current_target, current_calls))

    steps_completed = 0
    errors = []

    from agent.event_bus import DELEGATION, DELEGATION_DONE

    for (delegation_type, mission_id), calls in batches:
        import json as _json

        calls_desc = _json.dumps(
            [
                {
                    "tool": tc.name,
                    "args": tc.args
                    if isinstance(tc.args, dict)
                    else dict(tc.args)
                    if tc.args
                    else {},
                }
                for tc in calls
            ],
            indent=2,
        )
        instruction = f"Execute these tool calls in sequence:\n{calls_desc}"

        if delegation_type == "mission":
            orch._event_bus.emit(
                DELEGATION,
                level="debug",
                msg=f"[Router] Delegating to {mission_id} specialist (pipeline replay)",
            )
            actor = orch._get_or_create_mission_agent(mission_id)
            sub_result = actor.send_and_wait(
                instruction, sender="pipeline_replay", timeout=300.0
            )
            orch._event_bus.emit(
                DELEGATION_DONE,
                level="debug",
                msg=f"[Router] {mission_id} specialist finished (pipeline replay)",
            )

        elif delegation_type == "dataops":
            orch._event_bus.emit(
                DELEGATION,
                level="debug",
                msg="[Router] Delegating to DataOps specialist (pipeline replay)",
            )
            actor = orch._get_available_dataops_actor()
            _eph = actor.agent_id != "DataOpsAgent"
            sub_result = actor.send_and_wait(
                instruction, sender="pipeline_replay", timeout=300.0
            )
            if _eph:
                orch._cleanup_ephemeral_actor(actor.agent_id)
            orch._event_bus.emit(
                DELEGATION_DONE,
                level="debug",
                msg="[Router] DataOps specialist finished (pipeline replay)",
            )

        elif delegation_type == "viz_plotly":
            orch._event_bus.emit(
                DELEGATION,
                level="debug",
                msg="[Router] Delegating to Visualization specialist (pipeline replay)",
            )
            if config.PREFER_VIZ_BACKEND == "matplotlib":
                actor = orch._get_or_create_viz_mpl_actor()
            else:
                actor = orch._get_or_create_viz_plotly_actor()
            sub_result = actor.send_and_wait(
                instruction, sender="pipeline_replay", timeout=300.0
            )
            orch._event_bus.emit(
                DELEGATION_DONE,
                level="debug",
                msg="[Router] Visualization specialist finished (pipeline replay)",
            )

        else:
            continue

        if sub_result.get("failed"):
            errors.extend(sub_result.get("errors", []))
            break

        steps_completed += len(calls)

    orch._invalidate_pipeline()

    data_labels = [s["output_label"] for s in sorted_steps if s.get("output_label")]
    figure_produced = any(s["tool"] == "render_plotly_json" for s in sorted_steps)

    return {
        "status": "success" if not errors else "partial",
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline.name,
        "steps_completed": steps_completed,
        "steps_total": len(sorted_steps),
        "data_labels": data_labels,
        **({"errors": errors} if errors else {}),
        **({"figure_produced": True} if figure_produced and not errors else {}),
    }


def handle_search_pipelines(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    query = tool_args.get("query", "")
    mission = tool_args.get("mission")
    dataset = tool_args.get("dataset")
    limit = tool_args.get("limit", 10)

    results = orch._pipeline_store.search(
        query=query,
        mission=mission,
        dataset=dataset,
        limit=limit,
    )
    pipelines = []
    for e in results:
        pipelines.append(
            {
                "id": e.id,
                "name": e.name,
                "description": e.description,
                "datasets": e.datasets,
                "missions": e.missions,
                "step_count": e.step_count,
                "tags": e.tags,
            }
        )
    return {
        "status": "success",
        "count": len(pipelines),
        "pipelines": pipelines,
    }
