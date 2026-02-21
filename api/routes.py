"""All REST + SSE endpoints for the FastAPI backend."""

import asyncio
import json
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from .models import (
    ChatRequest,
    CommandRequest,
    CommandResponse,
    CompletionRequest,
    ConfigUpdate,
    FetchDataRequest,
    GalleryItemInfo,
    GallerySaveRequest,
    InputHistoryEntry,
    MissionInfo,
    RenameSessionRequest,
    ReplayRequest,
    ResumeSessionRequest,
    SavedSessionInfo,
    SavedSessionWithOps,
    SessionInfo,
    SessionDetail,
    ServerStatus,
    ToggleGlobalMemoryRequest,
)
from .session_manager import APISessionManager
from .streaming import SSEBridge

router = APIRouter(prefix="/api")

# These are injected by app.py lifespan
session_manager: APISessionManager = None  # type: ignore[assignment]
_start_time: float = 0.0
_thread_pool: ThreadPoolExecutor = None  # type: ignore[assignment]


def _session_info(state) -> dict:
    return SessionInfo(
        session_id=state.session_id,
        model=state.model,
        created_at=state.created_at,
        last_active=state.last_active,
        busy=state.busy,
    ).model_dump(mode="json")


def _get_session_or_404(session_id: str):
    state = session_manager.get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return state


# Size threshold for inline figure JSON (20 MB)
_FIGURE_SIZE_LIMIT = 20 * 1024 * 1024


def _figure_response(fig, session_id: str) -> dict:
    """Serialize a Plotly figure, falling back to an HTML file if > 20 MB."""
    if fig is None:
        return {"figure": None}
    fig_json_str = fig.to_json()
    if len(fig_json_str) <= _FIGURE_SIZE_LIMIT:
        return {"figure": json.loads(fig_json_str)}
    # Large figure — write standalone HTML and return a URL
    from agent.session import SessionManager
    sm = SessionManager()
    session_dir = sm.base_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    html_path = session_dir / "figure.html"
    html_path.write_text(
        fig.to_html(include_plotlyjs="cdn", full_html=True),
        encoding="utf-8",
    )
    return {
        "figure": None,
        "figure_url": f"/api/sessions/{session_id}/figure.html",
    }


# ---- Saved sessions (on disk) ----
# These must be registered BEFORE /sessions/{session_id} to avoid
# "saved" and "resume" being captured as a session_id path param.

@router.get("/sessions/saved")
async def list_saved_sessions():
    """List sessions saved to disk (from previous runs)."""
    from agent.session import SessionManager
    sm = SessionManager()
    sessions = sm.list_sessions()
    return [
        SavedSessionInfo(
            id=s["id"],
            name=s.get("name"),
            model=s.get("model"),
            turn_count=s.get("turn_count", 0),
            last_message_preview=s.get("last_message_preview", ""),
            created_at=s.get("created_at"),
            updated_at=s.get("updated_at"),
            token_usage=s.get("token_usage", {}),
        ).model_dump()
        for s in sessions
    ]


@router.get("/sessions/saved-with-ops")
async def list_saved_sessions_with_ops():
    """List saved sessions that have operations.json."""
    from agent.session import SessionManager
    sm = SessionManager()
    sessions = sm.list_sessions()
    result = []
    for s in sessions:
        sid = s["id"]
        session_dir = sm.base_dir / sid
        ops_file = session_dir / "operations.json"
        if not ops_file.exists():
            continue
        try:
            ops = json.loads(ops_file.read_text(encoding="utf-8"))
            op_count = len(ops) if isinstance(ops, list) else 0
            has_renders = any(
                r.get("tool") == "render_plotly_json" for r in ops
            ) if isinstance(ops, list) else False
        except Exception:
            op_count = 0
            has_renders = False
        if op_count == 0 or not has_renders:
            continue
        result.append(SavedSessionWithOps(
            id=sid,
            name=s.get("name"),
            model=s.get("model"),
            turn_count=s.get("turn_count", 0),
            last_message_preview=s.get("last_message_preview", ""),
            created_at=s.get("created_at"),
            updated_at=s.get("updated_at"),
            op_count=op_count,
            has_renders=has_renders,
        ).model_dump())
    return result


@router.delete("/sessions/saved/{session_id}", status_code=204)
async def delete_saved_session(session_id: str):
    """Delete a saved session from disk and clean up live session if present."""
    # Also remove from live API sessions if present (no-op if not live)
    session_manager.delete_session(session_id)
    from agent.session import SessionManager
    sm = SessionManager()
    if not sm.delete_session(session_id):
        raise HTTPException(status_code=404, detail=f"Saved session '{session_id}' not found")


@router.post("/sessions/resume", status_code=201)
async def resume_session(req: ResumeSessionRequest):
    """Resume a saved session from disk into a new API session."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _thread_pool,
            lambda: session_manager.resume_session(req.session_id),
        )
        state, metadata, display_log, event_log = result
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    info = _session_info(state)
    info["resumed_from"] = req.session_id
    info["turn_count"] = metadata.get("turn_count", 0)
    info["last_message_preview"] = metadata.get("last_message_preview", "")
    info["display_log"] = display_log or []
    info["event_log"] = event_log or []
    return info


@router.patch("/sessions/saved/{session_id}/name")
async def rename_session(session_id: str, req: RenameSessionRequest):
    """Rename a saved session."""
    from agent.session import SessionManager
    from datetime import datetime, timezone
    sm = SessionManager()
    meta_path = sm.base_dir / session_id / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metadata: {e}")
    meta["name"] = req.name
    meta["updated_at"] = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"status": "ok", "name": req.name}


# ---- Sessions CRUD ----

@router.post("/sessions", status_code=201)
async def create_session():
    """Create a new agent session."""
    try:
        loop = asyncio.get_running_loop()
        state = await loop.run_in_executor(
            _thread_pool, session_manager.create_session
        )
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))
    return _session_info(state)


@router.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    return [_session_info(s) for s in session_manager.list_sessions()]


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session detail including token usage and data entries."""
    state = _get_session_or_404(session_id)
    usage = state.agent.get_token_usage()
    plan_status = None
    if state.agent._current_plan:
        plan_status = state.agent._current_plan.status.value
    return SessionDetail(
        session_id=state.session_id,
        model=state.model,
        created_at=state.created_at,
        last_active=state.last_active,
        busy=state.busy,
        token_usage=usage,
        data_entries=len(state.store),
        plan_status=plan_status,
    ).model_dump(mode="json")


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    """Delete a session."""
    if not session_manager.delete_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


# ---- Chat (SSE) ----

@router.post("/sessions/{session_id}/chat")
async def chat(session_id: str, req: ChatRequest):
    """Send a message and stream response events via SSE."""
    state = _get_session_or_404(session_id)

    # Check busy
    if state.busy:
        raise HTTPException(status_code=409, detail="Session is busy processing a request")

    state.busy = True
    state.touch()

    loop = asyncio.get_running_loop()
    bridge = SSEBridge(loop)

    def _run_agent():
        try:
            # Subscribe SSE bridge to EventBus display events
            state.agent.subscribe_sse(bridge.callback)

            # Run in session context so get_store()/get_operations_log() resolve correctly
            response_text = session_manager.run_in_session_context(
                state, state.agent.process_message, req.message
            )

            # Check if a plot was produced
            fig = state.agent._renderer.get_figure()
            if fig is not None:
                bridge.callback({"type": "plot", "available": True})

            # Send text response
            bridge.callback({"type": "text_delta", "text": response_text})

            # Emit session title if one was just generated
            if state.agent.get_session_id():
                try:
                    from agent.session import SessionManager
                    _sm = SessionManager()
                    _meta_path = _sm.base_dir / state.agent.get_session_id() / "metadata.json"
                    if _meta_path.exists():
                        _meta = json.loads(_meta_path.read_text(encoding="utf-8"))
                        if _meta.get("name"):
                            bridge.callback({"type": "session_title", "name": _meta["name"]})
                except Exception:
                    pass

            # Done
            usage = state.agent.get_token_usage()
            bridge.finish({"type": "done", "token_usage": usage})

        except Exception as e:
            bridge.error(str(e))
        finally:
            state.agent.unsubscribe_sse()
            state.busy = False
            state.touch()

    # Run the agent in a thread
    loop.run_in_executor(_thread_pool, _run_agent)

    async def event_generator():
        async for event in bridge.events():
            event_type = event.get("type", "message")
            yield {"event": event_type, "data": json.dumps(event)}

    return EventSourceResponse(event_generator())


# ---- Cancel ----

@router.post("/sessions/{session_id}/cancel", status_code=202)
async def cancel(session_id: str):
    """Cancel an in-flight request and wait for busy to clear."""
    state = _get_session_or_404(session_id)
    state.agent.request_cancel()
    # Wait up to 5s for the agent thread to finish
    for _ in range(50):
        if not state.busy:
            return {"status": "cancelled"}
        await asyncio.sleep(0.1)
    return {"status": "cancel_requested"}


# ---- Slash commands ----

_COMMAND_HELP_TABLE = """\
| Command | Description |
|---------|-------------|
| `/help` | Show this command list |
| `/status` | Session info: model, tokens, data count, plan status |
| `/data` | List data entries in memory |
| `/figure` | Figure availability status |
| `/branch` | Fork session into a new branch |
| `/reset` | Delete current session and create a new one |
| `/sessions` | List saved sessions |
| `/retry` | Retry the first failed plan task |
| `/cancel` | Cancel the current plan |
| `/errors` | Show recent error logs |
"""


@router.post("/sessions/{session_id}/command")
async def execute_command(session_id: str, req: CommandRequest):
    """Dispatch a slash command and return a structured response."""
    cmd = req.command.lower().strip()

    if cmd == "help":
        return CommandResponse(command=cmd, content=_COMMAND_HELP_TABLE).model_dump()

    # All other commands need a valid session
    state = _get_session_or_404(session_id)

    if cmd == "status":
        usage = state.agent.get_token_usage()
        plan_status = None
        if state.agent._current_plan:
            plan_status = state.agent._current_plan.status.value
        lines = [
            f"**Model:** `{state.model}`",
            f"**Session:** `{state.session_id}`",
            f"**Data entries:** {len(state.store)}",
            f"**Plan status:** {plan_status or 'none'}",
            f"**Token usage:**",
        ]
        if usage:
            for k, v in usage.items():
                lines.append(f"  - {k}: {v:,}")
        else:
            lines.append("  - (no tokens used yet)")
        return CommandResponse(command=cmd, content="\n".join(lines)).model_dump()

    if cmd == "data":
        entries = state.store.list_entries()
        if not entries:
            return CommandResponse(command=cmd, content="No data entries in memory.").model_dump()
        lines = ["| Label | Points | Units | Time Range |", "|-------|--------|-------|------------|"]
        for e in entries:
            t_min = e.get("time_min", "—") or "—"
            t_max = e.get("time_max", "—") or "—"
            lines.append(f"| `{e.get('label', '?')}` | {e.get('num_points', 0):,} | {e.get('units', '')} | {t_min} → {t_max} |")
        return CommandResponse(command=cmd, content="\n".join(lines)).model_dump()

    if cmd == "figure":
        fig = state.agent._renderer.get_figure()
        status = "A figure is available." if fig is not None else "No figure available."
        return CommandResponse(command=cmd, content=status).model_dump()

    if cmd == "branch":
        # Fork current session into a new branch
        loop = asyncio.get_running_loop()

        def _branch():
            from agent.session import SessionManager
            sm = SessionManager()

            # Force-save current session to disk
            session_manager.run_in_session_context(
                state, state.agent.save_session,
            )

            # Copy session directory to a new ID
            new_id = uuid.uuid4().hex[:12]
            src_dir = sm.base_dir / session_id
            dst_dir = sm.base_dir / new_id
            shutil.copytree(str(src_dir), str(dst_dir))

            # Resume the copy as a new API session
            new_state, _meta, _display_log, _event_log = session_manager.resume_session(new_id)
            return new_state.session_id

        new_id = await loop.run_in_executor(_thread_pool, _branch)
        return CommandResponse(
            command=cmd,
            content=f"Branched session. New branch: `{new_id}`",
            data={"session_id": new_id},
        ).model_dump()

    if cmd == "reset":
        # Delete current session and create a new one
        session_manager.delete_session(session_id)
        loop = asyncio.get_running_loop()
        new_state = await loop.run_in_executor(_thread_pool, session_manager.create_session)
        new_id = new_state.session_id
        return CommandResponse(
            command=cmd,
            content=f"Session reset. New session: `{new_id}`",
            data={"session_id": new_id},
        ).model_dump()

    if cmd == "sessions":
        from agent.session import SessionManager
        sm = SessionManager()
        sessions = sm.list_sessions()
        if not sessions:
            return CommandResponse(command=cmd, content="No saved sessions.").model_dump()
        lines = ["| Name | Turns | Last message | Updated |", "|------|-------|--------------|---------|"]
        for s in sessions:
            name = s.get("name") or s["id"][:8]
            turns = s.get("turn_count", 0)
            preview = (s.get("last_message_preview", "") or "")[:50]
            updated = s.get("updated_at", "—") or "—"
            lines.append(f"| {name} | {turns} | {preview} | {updated} |")
        return CommandResponse(command=cmd, content="\n".join(lines)).model_dump()

    if cmd == "retry":
        if state.busy:
            return CommandResponse(command=cmd, content="Session is busy.").model_dump()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _thread_pool,
            lambda: session_manager.run_in_session_context(
                state, state.agent.retry_failed_task
            ),
        )
        return CommandResponse(command=cmd, content=result or "No failed tasks to retry.").model_dump()

    if cmd == "cancel":
        result = state.agent.cancel_plan()
        return CommandResponse(command=cmd, content=result or "No active plan to cancel.").model_dump()

    if cmd == "errors":
        from agent.logging import get_recent_errors
        errors = get_recent_errors(days=7, limit=20)
        if not errors:
            return CommandResponse(command=cmd, content="No recent errors.").model_dump()
        lines = []
        for e in errors:
            lines.append(f"- **{e.get('timestamp', '?')}** `{e.get('level', '?')}` {e.get('message', '')}")
        return CommandResponse(command=cmd, content="\n".join(lines)).model_dump()

    raise HTTPException(status_code=400, detail=f"Unknown command: /{cmd}")


# ---- Data ----

@router.get("/sessions/{session_id}/data")
async def get_data(session_id: str):
    """List fetched data entries."""
    state = _get_session_or_404(session_id)
    return state.store.list_entries()


# ---- Figure ----

@router.get("/sessions/{session_id}/figure")
async def get_figure(session_id: str):
    """Get the current Plotly figure JSON (or URL if > 20 MB)."""
    state = _get_session_or_404(session_id)
    # Trigger deferred figure restore if needed (lazy load from session resume)
    if state.agent._deferred_figure_state is not None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _thread_pool,
            lambda: session_manager.run_in_session_context(
                state, state.agent._restore_deferred_figure,
            ),
        )
    fig = state.agent._renderer.get_figure()
    return _figure_response(fig, session_id)


@router.get("/sessions/{session_id}/figure-thumbnail")
async def get_figure_thumbnail(session_id: str):
    """Serve a pre-rendered PNG thumbnail of the session's plot.

    Used by the frontend on session resume — shows an instant preview
    while the full interactive Plotly figure loads lazily in the background.
    """
    from agent.session import SessionManager
    sm = SessionManager()
    thumb_path = sm.base_dir / session_id / "figure_thumbnail.png"
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="No figure thumbnail for this session")
    return FileResponse(str(thumb_path), media_type="image/png")


@router.get("/sessions/{session_id}/figure.html")
async def get_figure_html(session_id: str):
    """Serve a standalone Plotly HTML file for large figures."""
    from agent.session import SessionManager
    sm = SessionManager()
    html_path = sm.base_dir / session_id / "figure.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="No large figure saved for this session")
    return FileResponse(str(html_path), media_type="text/html")


# ---- Event log ----

@router.get("/sessions/{session_id}/events")
async def get_session_events(
    session_id: str,
    limit: int = Query(5000, ge=1, le=50000),
    offset: int = Query(0, ge=0),
):
    """Return the persisted event log for a session.

    Used by the frontend on page refresh to rebuild Activity/Console/chat
    state from the server-side event log without a full resume.
    Supports pagination via ``limit`` and ``offset`` to avoid returning
    very large event logs in a single response.
    """
    _get_session_or_404(session_id)
    from agent.session import SessionManager
    sm = SessionManager()
    events_path = sm.base_dir / session_id / "events.jsonl"
    if not events_path.exists():
        return {"events": [], "total": 0}
    events = sm._load_event_log(events_path) or []
    total = len(events)
    page = events[offset:offset + limit]
    return {"events": page, "total": total}


# ---- Follow-ups ----

@router.post("/sessions/{session_id}/follow-ups")
async def get_follow_ups(session_id: str):
    """Generate contextual follow-up suggestions."""
    state = _get_session_or_404(session_id)
    loop = asyncio.get_running_loop()
    try:
        suggestions = await loop.run_in_executor(
            _thread_pool,
            lambda: session_manager.run_in_session_context(
                state, state.agent.generate_follow_ups, 2
            ),
        )
    except Exception:
        suggestions = []
    return {"suggestions": suggestions}


# ---- Plan management ----

@router.get("/sessions/{session_id}/plan")
async def get_plan_status(session_id: str):
    """Get the current plan status text.

    Only returns the plan that belongs to this session's agent instance,
    not plans from the global TaskStore (which persist across sessions).
    """
    state = _get_session_or_404(session_id)
    plan = state.agent.get_current_plan()
    if plan is None:
        return {"plan_status": None}
    from agent.planner import format_plan_for_display
    return {"plan_status": format_plan_for_display(plan)}


@router.post("/sessions/{session_id}/plan/retry")
async def retry_plan_task(session_id: str):
    """Retry the first failed task in the current plan."""
    state = _get_session_or_404(session_id)
    if state.busy:
        raise HTTPException(status_code=409, detail="Session is busy")
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _thread_pool,
        lambda: session_manager.run_in_session_context(
            state, state.agent.retry_failed_task
        ),
    )
    return {"result": result}


@router.post("/sessions/{session_id}/plan/cancel")
async def cancel_plan(session_id: str):
    """Cancel the current plan."""
    state = _get_session_or_404(session_id)
    result = state.agent.cancel_plan()
    return {"result": result}


# ---- Memories ----

@router.post("/sessions/{session_id}/memories", status_code=200)
async def extract_memories(session_id: str):
    """No-op kept for backward compatibility with older frontend clients.

    Memory extraction now happens periodically during the session via
    _maybe_extract_memories() — no shutdown pass needed.
    """
    _get_session_or_404(session_id)  # validate session exists
    return {"status": "ok"}


# ---- Recent errors ----

@router.get("/errors")
async def recent_errors(days: int = 7, limit: int = 20):
    """Get recent errors from agent log files."""
    from agent.logging import get_recent_errors
    errors = get_recent_errors(days=days, limit=limit)
    return {"errors": errors, "count": len(errors)}


# ---- Validation overview (stateless) ----

@router.get("/validation/overview")
async def validation_overview():
    """Return validation status for all datasets with override files.

    Scans ~/.xhelio/mission_overrides/ and classifies discrepancies
    (phantom / undocumented) for each dataset, grouped by mission.
    """
    from knowledge.mission_loader import _get_overrides_dir
    from knowledge.catalog import list_spacecraft

    overrides_dir = _get_overrides_dir()
    if not overrides_dir.exists():
        return {"missions": []}

    # Build mission_stem -> display_name mapping
    stem_to_name: dict[str, str] = {}
    for sc in list_spacecraft():
        stem_to_name[sc["id"]] = sc["name"]

    missions: list[dict] = []

    for mission_dir in sorted(overrides_dir.iterdir()):
        if not mission_dir.is_dir():
            continue
        mission_stem = mission_dir.name
        display_name = stem_to_name.get(mission_stem, mission_stem)

        datasets: list[dict] = []
        total_phantom = 0
        total_undocumented = 0
        validated_count = 0
        issue_count = 0

        for json_file in sorted(mission_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, dict):
                continue

            dataset_id = json_file.stem
            validations_raw = data.get("_validations", [])
            is_validated = bool(data.get("_validated"))

            # Classify discrepancies (same logic as metadata_client.get_dataset_quality_report)
            if validations_raw:
                annotations: dict = {}
                for v in validations_raw:
                    for param, ann in v.get("discrepancies", {}).items():
                        if param not in annotations and isinstance(ann, dict):
                            annotations[param] = ann
            else:
                annotations = data.get("parameters_annotations", {})

            phantom_params: list[str] = []
            undocumented_params: list[str] = []
            for param, ann in annotations.items():
                if not isinstance(ann, dict):
                    continue
                category = ann.get("_category", "")
                note = ann.get("_note", "")
                if category == "phantom" or (
                    not category and ("not found in data" in note or "not found in archive" in note)
                ):
                    phantom_params.append(param)
                elif category == "undocumented" or (
                    not category and ("found in data" in note or "found in archive" in note)
                ):
                    undocumented_params.append(param)

            phantom_params.sort()
            undocumented_params.sort()

            # Build validation records
            validation_records: list[dict] = []
            for i, v in enumerate(validations_raw):
                disc_count = len(v.get("discrepancies", {}))
                validation_records.append({
                    "version": v.get("version", i + 1),
                    "source_file": v.get("source_file", ""),
                    "validated_at": v.get("validated_at", ""),
                    "source_url": v.get("source_url", ""),
                    "discrepancy_count": disc_count,
                })

            has_issues = bool(phantom_params or undocumented_params)
            if has_issues:
                issue_count += 1
            if is_validated:
                validated_count += 1
            total_phantom += len(phantom_params)
            total_undocumented += len(undocumented_params)

            datasets.append({
                "dataset_id": dataset_id,
                "validated": is_validated,
                "validation_count": len(validations_raw),
                "phantom_count": len(phantom_params),
                "undocumented_count": len(undocumented_params),
                "phantom_params": phantom_params,
                "undocumented_params": undocumented_params,
                "validations": validation_records,
            })

        if datasets:
            missions.append({
                "mission_stem": mission_stem,
                "display_name": display_name,
                "dataset_count": len(datasets),
                "validated_count": validated_count,
                "issue_count": issue_count,
                "total_phantom": total_phantom,
                "total_undocumented": total_undocumented,
                "datasets": datasets,
            })

    return {"missions": missions}


# ---- Catalog browsing (stateless) ----

@router.get("/catalog/missions")
async def list_missions():
    """List all supported spacecraft missions."""
    from knowledge.catalog import list_spacecraft
    return [MissionInfo(id=s["id"], name=s["name"]).model_dump() for s in list_spacecraft()]


@router.get("/catalog/missions/{mission_id}/datasets")
async def list_mission_datasets(mission_id: str):
    """List datasets for a mission."""
    from knowledge.metadata_client import browse_datasets
    datasets = browse_datasets(mission_id)
    if datasets is None:
        raise HTTPException(status_code=404, detail=f"Mission '{mission_id}' not found")
    return datasets


@router.get("/catalog/datasets/{dataset_id}/parameters")
async def list_dataset_parameters(dataset_id: str):
    """List plottable parameters for a dataset."""
    from knowledge.metadata_client import list_parameters
    loop = asyncio.get_running_loop()
    params = await loop.run_in_executor(_thread_pool, list_parameters, dataset_id)
    return params


@router.get("/catalog/datasets/{dataset_id}/time-range")
async def get_dataset_time_range(dataset_id: str):
    """Get the time range for a dataset."""
    from knowledge.metadata_client import get_dataset_time_range as _get_time_range
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_thread_pool, _get_time_range, dataset_id)
    if result is None:
        return {"start": None, "stop": None}
    return result


# ---- Mission data management ----

@router.get("/catalog/status")
async def mission_data_status():
    """Get mission data status (count, datasets, age)."""
    from knowledge.startup import get_mission_status
    status = get_mission_status()
    return {
        "mission_count": status["mission_count"],
        "mission_names": status["mission_names"],
        "total_datasets": status["total_datasets"],
        "oldest_date": status["oldest_date"].isoformat() if status["oldest_date"] else None,
    }


@router.post("/catalog/refresh")
async def refresh_mission_data():
    """Refresh dataset time ranges (SSE stream with progress)."""
    loop = asyncio.get_running_loop()
    bridge = SSEBridge(loop)

    def _refresh():
        try:
            from knowledge.startup import run_refresh
            bridge.callback({"type": "progress", "phase": "refresh", "step": "start",
                             "message": "Starting time-range refresh..."})
            result = run_refresh(progress_callback=bridge.callback)
            bridge.finish({"type": "done",
                           "message": f"Refresh complete: {result['datasets_updated']} datasets updated"})
        except Exception as e:
            bridge.error(str(e))

    loop.run_in_executor(_thread_pool, _refresh)

    async def event_generator():
        async for event in bridge.events():
            event_type = event.get("type", "message")
            yield {"event": event_type, "data": json.dumps(event)}

    return EventSourceResponse(event_generator())


@router.post("/catalog/rebuild-cdaweb")
async def rebuild_cdaweb():
    """Rebuild CDAWeb mission data (SSE stream with progress)."""
    loop = asyncio.get_running_loop()
    bridge = SSEBridge(loop)

    def _rebuild():
        try:
            from knowledge.startup import run_cdaweb_rebuild
            bridge.callback({"type": "progress", "phase": "cdaweb", "step": "start",
                             "message": "Starting CDAWeb rebuild..."})
            run_cdaweb_rebuild(progress_callback=bridge.callback)
            bridge.finish({"type": "done", "message": "CDAWeb rebuild complete"})
        except Exception as e:
            bridge.error(str(e))

    loop.run_in_executor(_thread_pool, _rebuild)

    async def event_generator():
        async for event in bridge.events():
            event_type = event.get("type", "message")
            yield {"event": event_type, "data": json.dumps(event)}

    return EventSourceResponse(event_generator())


@router.post("/catalog/rebuild-ppi")
async def rebuild_ppi():
    """Rebuild PPI mission data (SSE stream with progress)."""
    loop = asyncio.get_running_loop()
    bridge = SSEBridge(loop)

    def _rebuild():
        try:
            from knowledge.startup import run_ppi_rebuild
            bridge.callback({"type": "progress", "phase": "ppi", "step": "start",
                             "message": "Starting PPI rebuild..."})
            run_ppi_rebuild(progress_callback=bridge.callback)
            bridge.finish({"type": "done", "message": "PPI rebuild complete"})
        except Exception as e:
            bridge.error(str(e))

    loop.run_in_executor(_thread_pool, _rebuild)

    async def event_generator():
        async for event in bridge.events():
            event_type = event.get("type", "message")
            yield {"event": event_type, "data": json.dumps(event)}

    return EventSourceResponse(event_generator())


# ---- Data fetch + preview (session-scoped) ----

@router.post("/sessions/{session_id}/fetch-data", status_code=201)
async def fetch_data_endpoint(session_id: str, req: FetchDataRequest):
    """Fetch data from CDAWeb/PPI and store in session's DataStore."""
    state = _get_session_or_404(session_id)
    from data_ops.fetch import fetch_data
    from data_ops.store import DataEntry

    loop = asyncio.get_running_loop()

    def _do_fetch():
        result = fetch_data(req.dataset_id, req.parameter_id, req.time_min, req.time_max)
        label = f"{req.dataset_id}.{req.parameter_id}"
        entry = DataEntry(
            label=label,
            data=result["data"],
            units=result.get("units", ""),
            description=result.get("description", ""),
            source="cdf",
        )
        state.store.put(entry)
        return entry.summary()

    try:
        summary = await loop.run_in_executor(_thread_pool, _do_fetch)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return summary


@router.get("/sessions/{session_id}/data/{label}/preview")
async def get_data_preview(session_id: str, label: str):
    """Get head+tail preview of a data entry."""
    state = _get_session_or_404(session_id)
    entry = state.store.get(label)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Data entry '{label}' not found")

    import pandas as pd
    df = entry.data if isinstance(entry.data, pd.DataFrame) else entry.data.to_dataframe()
    n = len(df)
    if n <= 20:
        preview = df
    else:
        preview = pd.concat([df.head(10), df.tail(10)])

    # Convert to records with string index
    records = []
    for idx, row in preview.iterrows():
        rec = {"_index": str(idx)}
        for col in preview.columns:
            val = row[col]
            if pd.isna(val):
                rec[col] = None
            elif isinstance(val, float):
                rec[col] = round(val, 6)
            else:
                rec[col] = val
        records.append(rec)

    return {
        "label": label,
        "total_rows": n,
        "columns": list(preview.columns),
        "rows": records,
    }


# ---- Config ----

@router.get("/config")
async def get_config():
    """Get current config (no secrets)."""
    from config import _load_config, _PROVIDER_DEFAULTS
    # Built-in defaults (match config.template.json)
    defaults = {
        "llm_provider": "gemini",
        "providers": _PROVIDER_DEFAULTS,
        "catalog_search_method": "semantic",
        "parallel_fetch": True,
        "parallel_max_workers": 4,
        "max_plot_points": 10000,
        "memory_token_budget": 10000,
        "memory_extraction_interval": 2,
        "ops_library_max_entries": 50,
        "reasoning": {
            "observation_summaries": True,
            "self_reflection": True,
            "show_thinking": False,
        },
    }
    loaded = _load_config()
    # Deep-merge providers: start with defaults, overlay user's providers
    merged_providers = {}
    for prov, prov_defaults in _PROVIDER_DEFAULTS.items():
        merged_providers[prov] = {**prov_defaults}
    user_providers = loaded.pop("providers", {})
    for prov, prov_vals in user_providers.items():
        if prov in merged_providers:
            merged_providers[prov].update(prov_vals)
        else:
            merged_providers[prov] = prov_vals
    cfg = {**defaults, **loaded, "providers": merged_providers}
    # Remove any accidental secrets
    for key in ("api_key", "google_api_key", "llm_api_key"):
        cfg.pop(key, None)
    # Remove comment keys
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_comment")}
    # Attach setting descriptions for UI
    from config import CONFIG_DESCRIPTIONS
    cfg["_descriptions"] = CONFIG_DESCRIPTIONS
    return cfg


@router.get("/config/schema")
async def get_config_schema():
    """Return setting descriptions for the UI."""
    from config import CONFIG_DESCRIPTIONS
    return {"descriptions": CONFIG_DESCRIPTIONS}


@router.put("/config")
async def update_config(req: ConfigUpdate):
    """Merge partial config into config.json."""
    import config
    from config import CONFIG_PATH

    # Read current
    current = {}
    if CONFIG_PATH.exists():
        try:
            current = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # Deep merge
    def _merge(base, update):
        for k, v in update.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                _merge(base[k], v)
            else:
                base[k] = v

    # Strip secrets and read-only metadata from input
    _STRIP_KEYS = {"api_key", "google_api_key", "llm_api_key", "_descriptions"}
    sanitized = {k: v for k, v in req.config.items() if k not in _STRIP_KEYS}
    _merge(current, sanitized)

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")

    # Hot-reload module-level constants so new sessions pick up changes
    from config import reload_config
    old_provider = config.LLM_PROVIDER
    old_model = config.SMART_MODEL
    old_base_url = config.LLM_BASE_URL
    reload_config()

    # Signal that existing sessions should be recreated if LLM settings changed
    needs_new_session = (
        config.LLM_PROVIDER != old_provider
        or config.SMART_MODEL != old_model
        or config.LLM_BASE_URL != old_base_url
    )
    return {"status": "saved", "needs_new_session": needs_new_session}


@router.post("/config/avatar")
async def upload_avatar(file: bytes = None):
    """Upload avatar image (placeholder)."""
    return {"status": "not_implemented"}


@router.get("/models")
async def list_models(provider: str = "gemini"):
    """List available models for the given LLM provider.

    Returns model IDs suitable for the config dropdowns.
    Currently supports: gemini (live API call), openai, anthropic (static lists).
    """
    if provider == "gemini":
        try:
            import os
            from config import get_api_key
            api_key = get_api_key("gemini")
            if not api_key:
                return {"models": [], "error": "GOOGLE_API_KEY not set"}
            from google import genai
            client = genai.Client(api_key=api_key)
            raw = list(client.models.list())
            models = []
            for m in raw:
                actions = m.supported_actions or []
                if "generateContent" not in actions:
                    continue
                name = m.name.replace("models/", "")
                # Only include gemini-* models (skip gemma, robotics, deep-research, etc.)
                if not name.startswith("gemini-"):
                    continue
                # Skip TTS, image-generation, audio, computer-use, robotics variants
                lower = name.lower()
                if any(kw in lower for kw in ("tts", "image", "audio", "computer-use", "robotics")):
                    continue
                models.append({
                    "id": name,
                    "display_name": m.display_name or name,
                    "input_token_limit": m.input_token_limit,
                })
            # Sort: newest/highest version first, pro before flash
            models.sort(key=lambda m: m["id"], reverse=True)
            return {"models": models}
        except Exception as e:
            return {"models": [], "error": str(e)}

    elif provider == "openai":
        # Static list — users typically set custom base_url (OpenRouter, etc.)
        return {"models": [
            {"id": "minimax/minimax-m2.5", "display_name": "MiniMax M2.5"},
            {"id": "minimax/minimax-m2.1", "display_name": "MiniMax M2.1"},
        ]}

    elif provider == "anthropic":
        return {"models": [
            {"id": "claude-sonnet-4-5-20250514", "display_name": "Claude Sonnet 4.5"},
            {"id": "claude-haiku-4-5-20251001", "display_name": "Claude Haiku 4.5"},
            {"id": "claude-opus-4-6", "display_name": "Claude Opus 4.6"},
        ]}

    return {"models": []}


# ---- Memory CRUD ----

@router.get("/sessions/{session_id}/memories/list")
async def list_memories(session_id: str):
    """List all memories for a session with stats."""
    state = _get_session_or_404(session_id)
    if not hasattr(state.agent, '_memory_store') or state.agent._memory_store is None:
        return {"memories": [], "global_enabled": True, "stats": None}
    ms = state.agent._memory_store

    from agent.memory import estimate_memory_tokens, MEMORY_TOKEN_BUDGET
    import re

    memories = []
    type_counts: dict[str, int] = {}
    type_tokens: dict[str, int] = {}
    all_scopes: set[str] = set()

    _star_re = re.compile(r"^(\d)★")

    for m in ms.get_all():
        entry: dict = {
            "id": m.id,
            "type": m.type,
            "scopes": m.scopes,
            "content": m.content,
            "created_at": m.created_at,
            "enabled": m.enabled,
            "source": m.source,
            "tags": m.tags,
            "access_count": m.access_count,
            "last_accessed": m.last_accessed,
            "version": m.version,
            "supersedes": m.supersedes,
            "source_session": m.source_session,
            "review_of": m.review_of,
        }
        # For non-review memories, include historical review summary across all versions
        if m.type != "review":
            all_reviews = ms.get_all_reviews_for_lineage(m.id)
            if all_reviews:
                stars_list = []
                for r in all_reviews:
                    match = _star_re.match(r.content)
                    if match:
                        stars_list.append(int(match.group(1)))
                entry["review_summary"] = {
                    "total_count": len(all_reviews),
                    "avg_stars": sum(stars_list) / len(stars_list) if stars_list else 0,
                }
        memories.append(entry)
        type_counts[m.type] = type_counts.get(m.type, 0) + 1
        tokens = estimate_memory_tokens(m)
        type_tokens[m.type] = type_tokens.get(m.type, 0) + tokens
        all_scopes.update(m.scopes)

    stats = {
        "total_tokens": ms.total_tokens(),
        "token_budget": MEMORY_TOKEN_BUDGET,
        "type_counts": type_counts,
        "type_tokens": type_tokens,
        "all_scopes": sorted(all_scopes),
    }

    return {"memories": memories, "global_enabled": ms.is_global_enabled(), "stats": stats}


@router.delete("/sessions/{session_id}/memories/{memory_id}")
async def delete_memory(session_id: str, memory_id: str):
    """Delete a specific memory."""
    state = _get_session_or_404(session_id)
    if not hasattr(state.agent, '_memory_store') or state.agent._memory_store is None:
        raise HTTPException(status_code=404, detail="No memory store")
    if not state.agent._memory_store.remove(memory_id):
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found")
    return {"status": "deleted"}


@router.post("/sessions/{session_id}/memories/toggle-global")
async def toggle_global_memory(session_id: str, req: ToggleGlobalMemoryRequest):
    """Toggle global memory enabled/disabled."""
    state = _get_session_or_404(session_id)
    if not hasattr(state.agent, '_memory_store') or state.agent._memory_store is None:
        raise HTTPException(status_code=404, detail="No memory store")
    state.agent._memory_store.toggle_global(req.enabled)
    return {"global_enabled": req.enabled}


@router.delete("/sessions/{session_id}/memories")
async def clear_all_memories(session_id: str):
    """Clear all memories."""
    state = _get_session_or_404(session_id)
    if not hasattr(state.agent, '_memory_store') or state.agent._memory_store is None:
        return {"status": "ok"}
    state.agent._memory_store.replace_all([])
    return {"status": "cleared"}


@router.post("/sessions/{session_id}/memories/refresh")
async def refresh_memories(session_id: str):
    """Reload memories from disk."""
    state = _get_session_or_404(session_id)
    if not hasattr(state.agent, '_memory_store') or state.agent._memory_store is None:
        return {"status": "no_store"}
    state.agent._memory_store.load()
    return {"status": "refreshed"}


@router.get("/sessions/{session_id}/memories/search")
async def search_memories(session_id: str, q: str = "", type: str | None = None, scope: str | None = None, limit: int = 20):
    """Search memories by query, type, and/or scope."""
    state = _get_session_or_404(session_id)
    if not hasattr(state.agent, '_memory_store') or state.agent._memory_store is None:
        return {"results": []}
    ms = state.agent._memory_store
    results = ms.search(q, mem_type=type, scope=scope, limit=limit)
    from dataclasses import asdict
    return {"results": [asdict(m) for m in results]}


@router.get("/sessions/{session_id}/memories/archived")
async def get_archived_memories(session_id: str):
    """Get truly archived (dropped/deleted) memories, excluding versioned predecessors."""
    state = _get_session_or_404(session_id)
    if not hasattr(state.agent, '_memory_store') or state.agent._memory_store is None:
        return {"archived": []}
    ms = state.agent._memory_store
    from dataclasses import asdict
    return {"archived": [asdict(m) for m in ms.get_truly_archived()]}


@router.get("/sessions/{session_id}/memories/{memory_id}/history")
async def get_memory_version_history(session_id: str, memory_id: str):
    """Get version history chain for a specific memory."""
    state = _get_session_or_404(session_id)
    if not hasattr(state.agent, '_memory_store') or state.agent._memory_store is None:
        raise HTTPException(status_code=404, detail="No memory store")
    ms = state.agent._memory_store
    from dataclasses import asdict
    versions = ms.get_version_history(memory_id)
    return {"versions": [asdict(m) for m in versions]}


# ---- Pipeline & Replay ----

@router.get("/pipeline/{saved_id}/operations")
async def get_pipeline_operations(saved_id: str):
    """Get pipeline operations for a saved session."""
    from agent.session import SessionManager
    from data_ops.operations_log import OperationsLog

    sm = SessionManager()
    session_dir = sm.base_dir / saved_id
    ops_file = session_dir / "operations.json"
    if not ops_file.exists():
        raise HTTPException(status_code=404, detail=f"No operations for session '{saved_id}'")

    log = OperationsLog()
    log.load_from_file(ops_file)

    # Build pipeline — pass all successful output labels (same as plot_pipeline.py)
    records = log.get_records()
    all_labels = set()
    for r in records:
        if r.get("status") == "success":
            all_labels.update(r.get("outputs", []))
    pipeline = log.get_pipeline(all_labels) if all_labels else records
    return {"pipeline": pipeline, "all_records": records}


@router.get("/pipeline/{saved_id}/dag")
async def get_pipeline_dag(saved_id: str, render_op_id: str | None = None):
    """Get Plotly DAG figure for a saved session's pipeline."""
    from agent.session import SessionManager
    from data_ops.operations_log import OperationsLog

    sm = SessionManager()
    session_dir = sm.base_dir / saved_id
    ops_file = session_dir / "operations.json"
    if not ops_file.exists():
        raise HTTPException(status_code=404, detail=f"No operations for session '{saved_id}'")

    log = OperationsLog()
    log.load_from_file(ops_file)

    records = log.get_records()
    all_labels = set()
    for r in records:
        if r.get("status") == "success":
            all_labels.update(r.get("outputs", []))

    if render_op_id:
        pipeline = log.get_state_pipeline(render_op_id, all_labels)
    else:
        pipeline = log.get_pipeline(all_labels) if all_labels else records

    loop = asyncio.get_running_loop()

    def _build():
        from scripts.plot_pipeline import build_figure
        fig = build_figure(pipeline, saved_id)
        return _figure_response(fig, saved_id)

    return await loop.run_in_executor(_thread_pool, _build)


@router.post("/pipeline/{saved_id}/replay")
async def replay_pipeline_endpoint(saved_id: str, req: ReplayRequest = ReplayRequest()):
    """Replay a pipeline from a saved session."""
    loop = asyncio.get_running_loop()

    def _replay():
        from scripts.replay import replay_session, replay_state
        if req.render_op_id:
            result = replay_state(saved_id, req.render_op_id, use_cache=req.use_cache)
        else:
            result = replay_session(saved_id, use_cache=req.use_cache)

        fig_resp = _figure_response(result.figure, saved_id)
        resp = {
            "steps_completed": result.steps_completed,
            "steps_total": result.steps_total,
            "errors": result.errors,
            **fig_resp,
        }
        return resp

    try:
        result = await loop.run_in_executor(_thread_pool, _replay)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


# ---- Autocomplete + Input history + Token breakdown ----

@router.post("/sessions/{session_id}/completions")
async def get_completions(session_id: str, req: CompletionRequest):
    """Generate inline completions for partial input."""
    state = _get_session_or_404(session_id)
    loop = asyncio.get_running_loop()

    def _complete():
        if hasattr(state.agent, 'generate_inline_completions'):
            return session_manager.run_in_session_context(
                state, state.agent.generate_inline_completions, req.partial
            )
        return []

    try:
        completions = await loop.run_in_executor(_thread_pool, _complete)
    except Exception:
        completions = []
    return {"completions": completions}


@router.get("/input-history")
async def get_input_history():
    """Read input history."""
    from config import get_data_dir
    history_file = get_data_dir() / "input_history.txt"
    if not history_file.exists():
        return {"history": []}
    try:
        lines = history_file.read_text(encoding="utf-8").strip().splitlines()
        return {"history": lines[-200:]}  # Last 200 entries
    except OSError:
        return {"history": []}


@router.post("/input-history")
async def add_input_history(entry: InputHistoryEntry):
    """Append to input history."""
    from config import get_data_dir
    history_file = get_data_dir() / "input_history.txt"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(entry.text.replace("\n", " ").strip() + "\n")
    return {"status": "ok"}


@router.get("/sessions/{session_id}/token-breakdown")
async def get_token_breakdown(session_id: str):
    """Get detailed token usage breakdown by agent type."""
    state = _get_session_or_404(session_id)
    usage = state.agent.get_token_usage()

    breakdown = {}
    if hasattr(state.agent, 'get_token_usage_breakdown'):
        breakdown = state.agent.get_token_usage_breakdown()

    memory_bytes = state.store.memory_usage_bytes()
    data_entries = len(state.store)

    return {
        "total": usage,
        "breakdown": breakdown,
        "memory_bytes": memory_bytes,
        "data_entries": data_entries,
    }


# ---- Gallery ----

def _gallery_dir() -> Path:
    from config import get_data_dir
    d = get_data_dir() / "gallery"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_gallery_index() -> list[dict]:
    index_path = _gallery_dir() / "index.json"
    if not index_path.exists():
        return []
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _write_gallery_index(items: list[dict]) -> None:
    index_path = _gallery_dir() / "index.json"
    index_path.write_text(json.dumps(items, indent=2, default=str), encoding="utf-8")


@router.get("/gallery")
async def list_gallery_items():
    """List all gallery items."""
    return _read_gallery_index()


@router.post("/gallery", status_code=201)
async def save_to_gallery(req: GallerySaveRequest):
    """Save a data product to the gallery with a PNG thumbnail."""
    loop = asyncio.get_running_loop()

    def _save():
        from scripts.replay import replay_state

        item_id = uuid.uuid4().hex[:12]
        gallery = _gallery_dir()

        # Replay to get the figure
        result = replay_state(req.session_id, req.render_op_id, use_cache=True)
        if result.figure is None:
            raise ValueError("Replay produced no figure")

        # Export thumbnail PNG
        png_path = gallery / f"{item_id}.png"
        result.figure.write_image(str(png_path), scale=1)

        # Build gallery entry
        from datetime import datetime, timezone
        entry = {
            "id": item_id,
            "name": req.name,
            "session_id": req.session_id,
            "render_op_id": req.render_op_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "thumbnail": f"{item_id}.png",
        }

        # Append to index
        items = _read_gallery_index()
        items.append(entry)
        _write_gallery_index(items)

        return entry

    try:
        entry = await loop.run_in_executor(_thread_pool, _save)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return entry


@router.delete("/gallery/{item_id}", status_code=204)
async def delete_gallery_item(item_id: str):
    """Remove a gallery item and its thumbnail."""
    items = _read_gallery_index()
    updated = [it for it in items if it["id"] != item_id]
    if len(updated) == len(items):
        raise HTTPException(status_code=404, detail=f"Gallery item '{item_id}' not found")
    _write_gallery_index(updated)
    # Delete PNG
    png_path = _gallery_dir() / f"{item_id}.png"
    if png_path.exists():
        png_path.unlink()


@router.get("/gallery/{item_id}/thumbnail")
async def get_gallery_thumbnail(item_id: str):
    """Serve the PNG thumbnail for a gallery item."""
    png_path = _gallery_dir() / f"{item_id}.png"
    if not png_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(str(png_path), media_type="image/png")


@router.post("/gallery/{item_id}/replay")
async def replay_gallery_item(item_id: str):
    """Replay a gallery item's pipeline and return the figure."""
    items = _read_gallery_index()
    item = next((it for it in items if it["id"] == item_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Gallery item '{item_id}' not found")

    loop = asyncio.get_running_loop()

    def _replay():
        from scripts.replay import replay_state
        result = replay_state(item["session_id"], item["render_op_id"], use_cache=True)
        fig_resp = _figure_response(result.figure, item["session_id"])
        return {
            "steps_completed": result.steps_completed,
            "steps_total": result.steps_total,
            "errors": result.errors,
            **fig_resp,
        }

    try:
        result = await loop.run_in_executor(_thread_pool, _replay)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


# ---- Server status ----

@router.get("/status")
async def server_status():
    """Server status (active sessions, uptime)."""
    return ServerStatus(
        active_sessions=len(session_manager.list_sessions()),
        max_sessions=session_manager.max_sessions,
        uptime_seconds=time.time() - _start_time,
    ).model_dump()
