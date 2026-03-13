"""Pipeline tool handlers — backed by PipelineDAG."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.tool_caller import ToolCaller
    from agent.tool_context import ToolContext


def handle_pipeline(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    action = tool_args.get("action", "")

    if action == "info":
        return handle_pipeline_info(ctx, tool_args, caller)
    elif action == "replay":
        return handle_pipeline_replay(ctx, tool_args, caller)
    else:
        return {"status": "error", "message": f"Unknown action: {action}"}


def handle_pipeline_info(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    """Return pipeline DAG summary."""
    if tool_args.get("list_library"):
        from data_ops.ops_library import get_ops_library

        entries = get_ops_library().get_top_entries()
        return {"status": "success", "library_entries": entries}

    dag = ctx.dag
    if dag is None or dag.node_count() == 0:
        return {
            "status": "success",
            "message": "No pipeline operations recorded yet.",
        }

    node_id = tool_args.get("node_id")
    if node_id:
        if node_id not in dag:
            return {"status": "error", "message": f"Node '{node_id}' not found"}
        return {"status": "success", **dag.node(node_id)}

    return {
        "status": "success",
        "node_count": dag.node_count(),
        "edge_count": dag.edge_count(),
        "sources": dag.nodes_by_kind("source"),
        "transforms": dag.nodes_by_kind("transform"),
        "sinks": dag.nodes_by_kind("sink"),
    }


def handle_pipeline_replay(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    """Replay a past session's pipeline into the current session."""
    from pathlib import Path
    from agent.session import SessionManager
    from agent.tool_context import ReplayContext
    from data_ops.dag import PipelineDAG
    from data_ops.replay import ReplayEngine

    session_id = tool_args.get("session_id")
    if not session_id:
        return {"status": "error", "message": "session_id is required for replay action"}

    op_id = tool_args.get("op_id")
    time_range = tool_args.get("time_range")

    # 1. Resolve session directory
    session_mgr = SessionManager()
    session_dir = session_mgr.base_dir / session_id
    pipeline_path = session_dir / "pipeline.json"
    if not pipeline_path.exists():
        return {"status": "error", "message": f"No pipeline found for session {session_id}"}

    # 2. Load source DAG
    source_dag = PipelineDAG.load(session_dir)
    if source_dag.node_count() == 0:
        return {"status": "error", "message": f"Pipeline for session {session_id} is empty"}

    # 3. Patch time range on fetch nodes if requested
    if time_range:
        start = time_range.get("start")
        end = time_range.get("end")
        if start or end:
            for node_id in source_dag.roots():
                node = source_dag.node(node_id)
                if node["tool"] in ("fetch", "fetch_data"):
                    with source_dag._lock:
                        args = source_dag._graph.nodes[node_id]["args"]
                        if start:
                            args["start_time"] = start
                        if end:
                            args["end_time"] = end

    # 4. Construct replay context with current session's store
    replay_ctx = ReplayContext(store=ctx.store, session_dir=ctx.session_dir)

    # 5. Run replay
    engine = ReplayEngine(source_dag, replay_ctx)
    all_results = []

    if op_id:
        # Replay specific subgraph
        if op_id not in source_dag:
            return {"status": "error", "message": f"Node '{op_id}' not found in source pipeline"}
        result = engine.replay(op_id)
        all_results.append(result)
    else:
        # Replay all leaf pipelines
        for leaf_id in source_dag.leaves():
            result = engine.replay(leaf_id)
            all_results.append(result)

    # 6. Aggregate results
    total_completed = sum(r.steps_completed for r in all_results)
    total_steps = sum(r.steps_total for r in all_results)
    all_errors = []
    for r in all_results:
        all_errors.extend(r.errors)

    # 7. Record summary node in current session's DAG
    if ctx.dag is not None:
        ctx.dag.add_node_auto(
            tool="pipeline_replay",
            agent="orchestrator",
            args={"source_session": session_id, "target_op_id": op_id},
            inputs=[],
            outputs={},
            status="success" if not all_errors else "partial",
        )
        ctx.dag.save()

    return {
        "status": "success" if not all_errors else "partial",
        "nodes_replayed": total_completed,
        "nodes_total": total_steps,
        "errors": all_errors,
    }
