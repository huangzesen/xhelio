"""All REST + SSE endpoints for the FastAPI backend."""

import asyncio
import json
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from .models import (
    ApiKeyUpdate,
    ChatRequest,
    CommandRequest,
    CommandResponse,
    CompletionRequest,
    ConfigUpdate,
    ExecutePipelineRequest,
    FetchDataRequest,
    GalleryItemInfo,
    GallerySaveRequest,
    InputHistoryEntry,
    MissionInfo,
    PipelineFeedbackRequest,
    RenameSessionRequest,
    ReplayRequest,
    ResumeSessionRequest,
    SavedSessionInfo,
    SavedSessionWithOps,
    SavePipelineRequest,
    SessionInfo,
    SessionDetail,
    ServerStatus,
    ToggleGlobalMemoryRequest,
    UpdatePipelineRequest,
    AssetOverviewResponse,
    AssetCategoryResponse,
    CleanupRequest,
    CleanupResponse,
    DirStatsResponse,
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
        viz_backend=getattr(state.agent, "_viz_backend", config.PREFER_VIZ_BACKEND),
        created_at=state.created_at,
        last_active=state.last_active,
        busy=state.busy,
    ).model_dump(mode="json")


import re as _re

# session_id, gallery item_id: alphanumeric + underscore/dot/dash
_SAFE_PATH_RE = _re.compile(r"^[a-zA-Z0-9_.-]+$")
# op_id: also allows colons (format "session_id:op_NNN")
_SAFE_OP_ID_RE = _re.compile(r"^[a-zA-Z0-9_:.-]+$")


def _validate_path_component(value: str, name: str = "path component") -> None:
    """Reject path components containing traversal sequences."""
    pattern = _SAFE_OP_ID_RE if name == "op_id" else _SAFE_PATH_RE
    if not pattern.match(value):
        raise HTTPException(status_code=400, detail=f"Invalid {name}: {value!r}")


def _get_session_or_404(session_id: str):
    _validate_path_component(session_id, "session_id")
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
            round_count=s.get("round_count", 0),
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
            has_renders = (
                any(r.get("tool") == "render_plotly_json" for r in ops)
                if isinstance(ops, list)
                else False
            )
        except Exception:
            op_count = 0
            has_renders = False
        if op_count == 0 or not has_renders:
            continue
        result.append(
            SavedSessionWithOps(
                id=sid,
                name=s.get("name"),
                model=s.get("model"),
                turn_count=s.get("turn_count", 0),
                last_message_preview=s.get("last_message_preview", ""),
                created_at=s.get("created_at"),
                updated_at=s.get("updated_at"),
                op_count=op_count,
                has_renders=has_renders,
            ).model_dump()
        )
    return result


@router.delete("/sessions/saved/{session_id}", status_code=204)
async def delete_saved_session(session_id: str):
    """Delete a saved session from disk and clean up live session if present."""
    _validate_path_component(session_id, "session_id")
    # Also remove from live API sessions if present (no-op if not live)
    session_manager.delete_session(session_id)
    from agent.session import SessionManager

    sm = SessionManager()
    if not sm.delete_session(session_id):
        raise HTTPException(
            status_code=404, detail=f"Saved session '{session_id}' not found"
        )


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

    # Start persistent event loop on resume
    session_manager.start_loop(state, loop)

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
    _validate_path_component(session_id, "session_id")
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
    import os as _os

    meta["name"] = req.name
    meta["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _os.replace(tmp, meta_path)
    return {"status": "ok", "name": req.name}


# ---- Sessions CRUD ----


@router.post("/sessions", status_code=201)
async def create_session():
    """Create a new agent session."""
    try:
        loop = asyncio.get_running_loop()
        state = await loop.run_in_executor(_thread_pool, session_manager.create_session)
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
        viz_backend=getattr(state.agent, "_viz_backend", config.PREFER_VIZ_BACKEND),
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
    """Push a message into the agent's input queue.

    The persistent event loop (run_loop) processes it and emits events
    on the session's SSE bridge. The frontend subscribes via
    GET /sessions/{session_id}/events to receive those events.

    Always succeeds — no busy gate.
    """
    state = _get_session_or_404(session_id)
    state.touch()

    # Lazy-start the persistent event loop (no-ops if already running)
    if state._loop_thread is None or not state._loop_thread.is_alive():
        loop = asyncio.get_running_loop()
        session_manager.start_loop(state, loop)

    # Push message into the input queue — run_loop() will process it
    state.agent.push_input(req.message)

    return {"status": "queued"}


# ---- Cancel ----


@router.post("/sessions/{session_id}/cancel", status_code=202)
async def cancel(session_id: str):
    """Cancel an in-flight request and wait for busy to clear."""
    state = _get_session_or_404(session_id)
    state.agent.request_cancel()
    for _ in range(50):
        if not state.busy:
            return {"status": "cancelled"}
        await asyncio.sleep(0.1)
    return {"status": "cancel_requested"}


# ---- Session-lifetime SSE event stream (turnless mode) ----


@router.get("/sessions/{session_id}/events")
async def session_events(session_id: str):
    """Session-lifetime SSE stream. Client should reconnect on disconnect.

    Only available when the persistent event loop is running (turnless mode).
    Returns events as they are emitted by the agent's run_loop().
    """
    state = _get_session_or_404(session_id)
    if state.sse_bridge is None:
        raise HTTPException(
            status_code=400, detail="No active event stream (turnless mode not started)"
        )

    queue = state.sse_bridge.subscribe()

    async def event_generator():
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield {"event": event.get("type", "message"), "data": json.dumps(event)}
        finally:
            bridge = state.sse_bridge
            if bridge is not None:
                bridge.unsubscribe(queue)

    return EventSourceResponse(event_generator())


# ---- Control Center (direct frontend access) ----


@router.get("/sessions/{session_id}/work")
async def list_work(session_id: str):
    """List active work units in the Control Center."""
    state = _get_session_or_404(session_id)
    return {"work_units": state.agent._control_center.list_active()}


@router.post("/sessions/{session_id}/work/{unit_id}/cancel", status_code=202)
async def cancel_work_unit(session_id: str, unit_id: str):
    """Cancel a specific work unit via the Control Center."""
    state = _get_session_or_404(session_id)
    ok = state.agent._control_center.cancel(unit_id)
    return {"status": "ok" if ok else "not_found"}


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
            return CommandResponse(
                command=cmd, content="No data entries in memory."
            ).model_dump()
        lines = [
            "| Label | Points | Units | Time Range |",
            "|-------|--------|-------|------------|",
        ]
        for e in entries:
            t_min = e.get("time_min", "—") or "—"
            t_max = e.get("time_max", "—") or "—"
            lines.append(
                f"| `{e.get('label', '?')}` | {e.get('num_points', 0):,} | {e.get('units', '')} | {t_min} → {t_max} |"
            )
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
                state,
                state.agent.save_session,
            )

            # Copy session directory to a new ID
            new_id = uuid.uuid4().hex[:12]
            src_dir = sm.base_dir / session_id
            dst_dir = sm.base_dir / new_id
            shutil.copytree(str(src_dir), str(dst_dir))

            # Resume the copy as a new API session
            new_state, _meta, _display_log, _event_log = session_manager.resume_session(
                new_id
            )
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
        new_state = await loop.run_in_executor(
            _thread_pool, session_manager.create_session
        )
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
            return CommandResponse(
                command=cmd, content="No saved sessions."
            ).model_dump()
        lines = [
            "| Name | Turns | Last message | Updated |",
            "|------|-------|--------------|---------|",
        ]
        for s in sessions:
            name = s.get("name") or s["id"][:8]
            turns = s.get("turn_count", 0)
            from agent.truncation import trunc

            preview = trunc(
                s.get("last_message_preview", "") or "", "api.session_preview"
            )
            updated = s.get("updated_at", "—") or "—"
            lines.append(f"| {name} | {turns} | {preview} | {updated} |")
        return CommandResponse(command=cmd, content="\n".join(lines)).model_dump()

    if cmd == "errors":
        from agent.logging import get_recent_errors

        errors = get_recent_errors(days=7, limit=20)
        if not errors:
            return CommandResponse(
                command=cmd, content="No recent errors."
            ).model_dump()
        lines = []
        for e in errors:
            lines.append(
                f"- **{e.get('timestamp', '?')}** `{e.get('level', '?')}` {e.get('message', '')}"
            )
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
                state,
                state.agent._restore_deferred_figure,
            ),
        )
    fig = state.agent._renderer.get_figure()
    return _figure_response(fig, session_id)


def _lazy_generate_session_thumbnail(session_dir: Path) -> bool:
    """Try to generate a missing session-level figure thumbnail.

    Loads ``figure.json`` + data pickles (lazy), reconstructs the Plotly
    figure via ``fill_figure_data()``, and exports a PNG thumbnail.

    Returns True if the thumbnail was successfully generated.
    """
    figure_path = session_dir / "figure.json"
    data_dir = session_dir / "data"
    thumb_path = session_dir / "figure_thumbnail.png"

    if not figure_path.exists() or not data_dir.exists():
        return False

    try:
        fig_state = json.loads(figure_path.read_text(encoding="utf-8"))
        fig_json = fig_state.get("figure_json")
        input_labels = fig_state.get("input_labels", [])
        if not fig_json or not input_labels:
            return False

        from data_ops.store import DataStore, resolve_entry

        store = DataStore(data_dir)

        entry_map: dict = {}
        for label in input_labels:
            entry, _ = resolve_entry(store, label)
            if entry is None:
                return False
            entry_map[label] = entry

        from rendering.plotly_renderer import fill_figure_data

        result = fill_figure_data(fig_json, entry_map)
        if result.figure is None:
            return False

        result.figure.write_image(
            str(thumb_path),
            format="png",
            width=800,
            height=400,
            scale=1,
        )
        return True
    except Exception:
        return False


def _lazy_generate_render_thumbnail(session_dir: Path, op_id: str) -> bool:
    """Try to generate a missing per-render thumbnail for a specific operation.

    Loads ``operations.json`` to find the matching render op, loads data
    pickles (lazy), reconstructs the figure, and exports a PNG thumbnail.

    Returns True if the thumbnail was successfully generated.
    """
    ops_path = session_dir / "operations.json"
    data_dir = session_dir / "data"

    if not ops_path.exists() or not data_dir.exists():
        return False

    try:
        operations = json.loads(ops_path.read_text(encoding="utf-8"))
        if not isinstance(operations, list):
            return False

        # Find the matching render op
        op = None
        for record in operations:
            if (
                record.get("id") == op_id
                and record.get("tool") == "render_plotly_json"
                and record.get("status") == "success"
            ):
                op = record
                break
        if op is None:
            return False

        fig_json = op.get("args", {}).get("figure_json", {})
        input_labels = op.get("inputs", [])
        if not fig_json or not input_labels:
            return False

        from data_ops.store import DataStore, resolve_entry

        store = DataStore(data_dir)

        entry_map: dict = {}
        for label in input_labels:
            entry, _ = resolve_entry(store, label)
            if entry is None:
                return False
            entry_map[label] = entry

        from rendering.plotly_renderer import fill_figure_data

        result = fill_figure_data(fig_json, entry_map)
        if result.figure is None:
            return False

        thumb_dir = session_dir / "thumbnails"
        thumb_dir.mkdir(exist_ok=True)
        thumb_path = thumb_dir / f"{op_id}.png"
        result.figure.write_image(
            str(thumb_path),
            format="png",
            width=800,
            height=400,
            scale=1,
        )
        return True
    except Exception:
        return False


@router.head("/sessions/{session_id}/figure-thumbnail")
@router.get("/sessions/{session_id}/figure-thumbnail")
async def get_figure_thumbnail(session_id: str):
    """Serve a pre-rendered PNG thumbnail of the session's plot.

    Used by the frontend on session resume — shows an instant preview
    while the full interactive Plotly figure loads lazily in the background.
    HEAD is used by the frontend to probe existence before displaying.

    If the thumbnail doesn't exist on disk, attempts to regenerate it
    from figure.json + data pickles before returning 404.
    """
    _validate_path_component(session_id, "session_id")
    from agent.session import SessionManager

    sm = SessionManager()
    session_dir = sm.base_dir / session_id
    thumb_path = session_dir / "figure_thumbnail.png"
    if not thumb_path.exists():
        # Try lazy generation in the thread pool
        loop = asyncio.get_running_loop()
        generated = await loop.run_in_executor(
            _thread_pool,
            _lazy_generate_session_thumbnail,
            session_dir,
        )
        if not generated:
            raise HTTPException(
                status_code=404, detail="No figure thumbnail for this session"
            )
    return FileResponse(str(thumb_path), media_type="image/png")


@router.head("/sessions/{session_id}/thumbnails/{op_id}.png")
@router.get("/sessions/{session_id}/thumbnails/{op_id}.png")
async def get_render_thumbnail(session_id: str, op_id: str):
    """Serve a per-render PNG thumbnail for a specific operation.

    Used by the frontend on session resume to show inline plot previews
    for each render_plotly_json call in the session history.

    If the thumbnail doesn't exist on disk, attempts to regenerate it
    from operations.json + data pickles before returning 404.
    """
    _validate_path_component(session_id, "session_id")
    _validate_path_component(op_id, "op_id")
    from agent.session import SessionManager

    sm = SessionManager()
    session_dir = sm.base_dir / session_id
    thumb_path = session_dir / "thumbnails" / f"{op_id}.png"
    if not thumb_path.exists():
        # Try lazy generation in the thread pool
        loop = asyncio.get_running_loop()
        generated = await loop.run_in_executor(
            _thread_pool,
            _lazy_generate_render_thumbnail,
            session_dir,
            op_id,
        )
        if not generated:
            raise HTTPException(
                status_code=404, detail="No render thumbnail for this operation"
            )
    return FileResponse(str(thumb_path), media_type="image/png")


@router.get("/sessions/{session_id}/figure.html")
async def get_figure_html(session_id: str):
    """Serve a standalone Plotly HTML file for large figures."""
    _validate_path_component(session_id, "session_id")
    from agent.session import SessionManager

    sm = SessionManager()
    html_path = sm.base_dir / session_id / "figure.html"
    if not html_path.exists():
        raise HTTPException(
            status_code=404, detail="No large figure saved for this session"
        )
    return FileResponse(str(html_path), media_type="text/html")


# ---- Matplotlib outputs ----


@router.get("/sessions/{session_id}/mpl-outputs")
async def list_mpl_outputs(session_id: str):
    """List all matplotlib outputs for a session."""
    _validate_path_component(session_id, "session_id")
    from agent.session import SessionManager

    sm = SessionManager()
    session_dir = sm.base_dir / session_id
    outputs_dir = session_dir / "mpl_outputs"
    scripts_dir = session_dir / "mpl_scripts"

    items = []
    if scripts_dir.exists():
        for script_file in sorted(scripts_dir.glob("*.py")):
            script_id = script_file.stem
            output_file = outputs_dir / f"{script_id}.png"
            items.append(
                {
                    "script_id": script_id,
                    "has_output": output_file.exists(),
                }
            )
    return {"items": items, "count": len(items)}


@router.get("/sessions/{session_id}/mpl-outputs/{script_id}.png")
async def get_mpl_output(session_id: str, script_id: str):
    """Serve a matplotlib output image."""
    _validate_path_component(session_id, "session_id")
    _validate_path_component(script_id, "script_id")
    from agent.session import SessionManager

    sm = SessionManager()
    output_path = sm.base_dir / session_id / "mpl_outputs" / f"{script_id}.png"
    if not output_path.exists():
        raise HTTPException(
            status_code=404, detail=f"MPL output not found: {script_id}"
        )
    return FileResponse(str(output_path), media_type="image/png")


@router.get("/sessions/{session_id}/mpl-scripts/{script_id}.py")
async def get_mpl_script(session_id: str, script_id: str):
    """Serve a matplotlib script file."""
    _validate_path_component(session_id, "session_id")
    _validate_path_component(script_id, "script_id")
    from agent.session import SessionManager

    sm = SessionManager()
    script_path = sm.base_dir / session_id / "mpl_scripts" / f"{script_id}.py"
    if not script_path.exists():
        raise HTTPException(
            status_code=404, detail=f"MPL script not found: {script_id}"
        )
    return FileResponse(str(script_path), media_type="text/plain")


# ---- JSX/Recharts outputs ----


@router.get("/sessions/{session_id}/jsx-outputs")
async def list_jsx_outputs(session_id: str):
    """List all JSX component outputs for a session."""
    _validate_path_component(session_id, "session_id")
    from agent.session import SessionManager

    sm = SessionManager()
    session_dir = sm.base_dir / session_id
    outputs_dir = session_dir / "jsx_outputs"
    scripts_dir = session_dir / "jsx_scripts"

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
                }
            )
    return {"items": items, "count": len(items)}


@router.get("/sessions/{session_id}/jsx-outputs/{script_id}.js")
async def get_jsx_bundle(session_id: str, script_id: str):
    """Serve a compiled JSX bundle."""
    _validate_path_component(session_id, "session_id")
    _validate_path_component(script_id, "script_id")
    from agent.session import SessionManager

    sm = SessionManager()
    bundle_path = sm.base_dir / session_id / "jsx_outputs" / f"{script_id}.js"
    if not bundle_path.exists():
        raise HTTPException(
            status_code=404, detail=f"JSX bundle not found: {script_id}"
        )
    return FileResponse(str(bundle_path), media_type="application/javascript")


@router.get("/sessions/{session_id}/jsx-outputs/{script_id}.data.json")
async def get_jsx_data(session_id: str, script_id: str):
    """Serve JSX component data JSON."""
    _validate_path_component(session_id, "session_id")
    _validate_path_component(script_id, "script_id")
    from agent.session import SessionManager

    sm = SessionManager()
    data_path = sm.base_dir / session_id / "jsx_outputs" / f"{script_id}.data.json"
    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"JSX data not found: {script_id}")
    return FileResponse(str(data_path), media_type="application/json")


@router.get("/sessions/{session_id}/jsx-scripts/{script_id}.tsx")
async def get_jsx_source(session_id: str, script_id: str):
    """Serve an original JSX/TSX source file."""
    _validate_path_component(session_id, "session_id")
    _validate_path_component(script_id, "script_id")
    from agent.session import SessionManager

    sm = SessionManager()
    script_path = sm.base_dir / session_id / "jsx_scripts" / f"{script_id}.tsx"
    if not script_path.exists():
        raise HTTPException(
            status_code=404, detail=f"JSX source not found: {script_id}"
        )
    return FileResponse(str(script_path), media_type="text/plain")


# ---- Event log ----


@router.get("/sessions/{session_id}/event-log")
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
    page = events[offset : offset + limit]
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
    from knowledge.catalog import list_missions_catalog

    overrides_dir = _get_overrides_dir()
    if not overrides_dir.exists():
        return {"missions": []}

    # Build mission_stem -> display_name mapping
    stem_to_name: dict[str, str] = {}
    for sc in list_missions_catalog():
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
                    not category
                    and ("not found in data" in note or "not found in archive" in note)
                ):
                    phantom_params.append(param)
                elif category == "undocumented" or (
                    not category
                    and ("found in data" in note or "found in archive" in note)
                ):
                    undocumented_params.append(param)

            phantom_params.sort()
            undocumented_params.sort()

            # Build validation records
            validation_records: list[dict] = []
            for i, v in enumerate(validations_raw):
                disc_count = len(v.get("discrepancies", {}))
                validation_records.append(
                    {
                        "version": v.get("version", i + 1),
                        "source_file": v.get("source_file", ""),
                        "validated_at": v.get("validated_at", ""),
                        "source_url": v.get("source_url", ""),
                        "discrepancy_count": disc_count,
                    }
                )

            has_issues = bool(phantom_params or undocumented_params)
            if has_issues:
                issue_count += 1
            if is_validated:
                validated_count += 1
            total_phantom += len(phantom_params)
            total_undocumented += len(undocumented_params)

            datasets.append(
                {
                    "dataset_id": dataset_id,
                    "validated": is_validated,
                    "validation_count": len(validations_raw),
                    "phantom_count": len(phantom_params),
                    "undocumented_count": len(undocumented_params),
                    "phantom_params": phantom_params,
                    "undocumented_params": undocumented_params,
                    "validations": validation_records,
                }
            )

        if datasets:
            missions.append(
                {
                    "mission_stem": mission_stem,
                    "display_name": display_name,
                    "dataset_count": len(datasets),
                    "validated_count": validated_count,
                    "issue_count": issue_count,
                    "total_phantom": total_phantom,
                    "total_undocumented": total_undocumented,
                    "datasets": datasets,
                }
            )

    return {"missions": missions}


# ---- Catalog browsing (stateless) ----


@router.get("/catalog/missions")
async def list_missions():
    """List all supported missions."""
    from knowledge.catalog import list_missions_catalog

    return [
        MissionInfo(id=s["id"], name=s["name"]).model_dump()
        for s in list_missions_catalog()
    ]


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
    """Get mission data status (count, datasets, age) + loading state."""
    from knowledge.startup import get_mission_status
    from knowledge.loading_state import get_loading_state

    status = get_mission_status()
    loading = get_loading_state()
    return {
        "mission_count": status["mission_count"],
        "mission_names": status["mission_names"],
        "total_datasets": status["total_datasets"],
        "oldest_date": status["oldest_date"].isoformat()
        if status["oldest_date"]
        else None,
        "loading": loading.to_dict(),
    }


@router.get("/catalog/loading-progress")
async def catalog_loading_progress():
    """SSE stream of mission data loading progress.

    Sends current state immediately, then streams updates as they occur.
    Closes when loading reaches COMPLETE or FAILED.
    """
    from knowledge.loading_state import get_loading_state

    loading = get_loading_state()
    loop = asyncio.get_running_loop()
    bridge = SSEBridge(loop)

    # Subscribe to state updates before checking current state to avoid
    # race where loading completes between check and subscribe.
    loading.subscribe(bridge.callback)

    async def event_generator():
        try:
            # Send current state immediately
            current = loading.to_dict()
            phase = current.get("phase", "")

            if phase in ("complete", "failed"):
                yield {"event": "done", "data": json.dumps(current)}
                return

            yield {"event": "progress", "data": json.dumps(current)}

            async for event in bridge.events():
                event_type = "progress"
                if event.get("is_ready") or event.get("phase") == "failed":
                    event_type = "done"
                yield {"event": event_type, "data": json.dumps(event)}
                if event_type == "done":
                    break
        finally:
            loading.unsubscribe(bridge.callback)

    return EventSourceResponse(event_generator())


@router.post("/catalog/refresh")
async def refresh_mission_data():
    """Refresh dataset time ranges (SSE stream with progress)."""
    loop = asyncio.get_running_loop()
    bridge = SSEBridge(loop)

    def _refresh():
        try:
            from knowledge.startup import run_refresh

            bridge.callback(
                {
                    "type": "progress",
                    "phase": "refresh",
                    "step": "start",
                    "message": "Starting time-range refresh...",
                }
            )
            result = run_refresh(progress_callback=bridge.callback)
            bridge.finish(
                {
                    "type": "done",
                    "message": f"Refresh complete: {result['datasets_updated']} datasets updated",
                }
            )
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

            bridge.callback(
                {
                    "type": "progress",
                    "phase": "cdaweb",
                    "step": "start",
                    "message": "Starting CDAWeb rebuild...",
                }
            )
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

            bridge.callback(
                {
                    "type": "progress",
                    "phase": "ppi",
                    "step": "start",
                    "message": "Starting PPI rebuild...",
                }
            )
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
        result = fetch_data(
            req.dataset_id, req.parameter_id, req.time_min, req.time_max
        )
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
    from agent.truncation import get_item_limit

    preview_n = get_item_limit("items.api_data_preview")
    df = (
        entry.data
        if isinstance(entry.data, pd.DataFrame)
        else entry.data.to_dataframe()
    )
    n = len(df)
    if n <= preview_n * 2:
        preview = df
    else:
        preview = pd.concat([df.head(preview_n), df.tail(preview_n)])

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
        "memory_token_budget": 100000,
        "memory_extraction_interval": 2,
        "ops_library_max_entries": 50,
        "reasoning": {
            "observation_summaries": True,
            "self_reflection": True,
            "show_thinking": False,
            "insight_feedback": False,
            "insight_feedback_max_iterations": 2,
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

    import os

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=2), encoding="utf-8")
    os.replace(tmp, CONFIG_PATH)

    # Hot-reload module-level constants so config.* reflects new values
    from config import reload_config

    reload_config()

    # Hot-reload all active sessions — each agent compares its own cached
    # state against the new config values and rebuilds only if needed.
    result = session_manager.reload_config_for_sessions()

    # Only require a new session if some sessions were busy and couldn't
    # be hot-reloaded (user needs to retry after they finish processing)
    needs_new_session = result["skipped"] > 0

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
                return {
                    "models": [],
                    "error": "API key not set — add GOOGLE_API_KEY to .env",
                }
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
                if any(
                    kw in lower
                    for kw in ("tts", "image", "audio", "computer-use", "robotics")
                ):
                    continue
                models.append(
                    {
                        "id": name,
                        "display_name": m.display_name or name,
                        "input_token_limit": m.input_token_limit,
                    }
                )
            # Sort: newest/highest version first, pro before flash
            models.sort(key=lambda m: m["id"], reverse=True)
            return {"models": models}
        except Exception as e:
            return {"models": [], "error": str(e)}

    # OpenAI-compatible and Anthropic: return empty list so the frontend
    # renders free-text inputs.  Users can type any model ID they want —
    # the model namespace depends on the base_url they configure.
    return {"models": []}


# ---- API Key Management ----


@router.get("/api-key-status")
async def api_key_status(provider: str = "gemini"):
    """Check whether the API key for the given provider is configured. Never returns the full key."""
    from config import get_api_key

    key = get_api_key(provider)
    if key and len(key) >= 4:
        return {"configured": True, "masked": f"...{key[-4:]}"}
    elif key:
        return {"configured": True, "masked": "...****"}
    return {"configured": False, "masked": None}


@router.put("/api-key")
async def update_api_key(req: ApiKeyUpdate):
    """Write/update the API key for the specified provider in .env and validate it."""
    from dotenv import set_key as dotenv_set_key
    from config import reload_config, _PROVIDER_ENV_KEYS

    env_path = Path(__file__).resolve().parent.parent / ".env"

    # Get the env var name for the specified provider
    env_key = _PROVIDER_ENV_KEYS.get(req.provider, "GOOGLE_API_KEY")

    # Write key to .env
    dotenv_set_key(str(env_path), env_key, req.key)

    # Reload so os.getenv picks up the new value
    reload_config()

    # Validate with a quick API call - only for Gemini
    valid = False
    error_msg = None
    if req.provider == "gemini":
        try:
            from google import genai

            client = genai.Client(api_key=req.key)
            # Lightweight check — list models (fetches first page)
            list(client.models.list())
            valid = True
        except Exception as e:
            error_msg = str(e)
    else:
        # For OpenAI/Anthropic, just save and return valid
        valid = True

    masked = f"...{req.key[-4:]}" if len(req.key) >= 4 else "...****"
    return {
        "status": "saved",
        "valid": valid,
        "error": error_msg,
        "masked": masked,
    }


# ---- Memory CRUD ----


@router.get("/sessions/{session_id}/memories/list")
async def list_memories(session_id: str):
    """List all memories for a session with stats."""
    state = _get_session_or_404(session_id)
    if not hasattr(state.agent, "_memory_store") or state.agent._memory_store is None:
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
        # For non-review memories, include two-tier historical review summary and accumulated access count
        if m.type != "review":
            # Accumulate access_count across the entire version lineage
            lineage_access = m.access_count
            current_id: str | None = m.supersedes or None
            while current_id:
                prev = ms.get_by_id(current_id)
                if prev is None:
                    break
                lineage_access += prev.access_count
                current_id = prev.supersedes or None
            entry["lineage_access_count"] = lineage_access

            all_reviews = [
                r for r in ms.get_all_reviews_for_lineage(m.id) if not r.archived
            ]
            if all_reviews:
                recent_reviews = [
                    r
                    for r in ms.get_recent_reviews_for_lineage(m.id, n=10)
                    if not r.archived
                ]

                def _tier_stats(reviews):
                    stars = [
                        int(match.group(1))
                        for r in reviews
                        if (match := _star_re.match(r.content))
                    ]
                    return {
                        "total_count": len(reviews),
                        "avg_stars": sum(stars) / len(stars) if stars else 0,
                    }

                entry["review_summary"] = {
                    "all_time": _tier_stats(all_reviews),
                    "recent": _tier_stats(recent_reviews),
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

    return {
        "memories": memories,
        "global_enabled": ms.is_global_enabled(),
        "stats": stats,
    }


@router.delete("/sessions/{session_id}/memories/{memory_id}")
async def delete_memory(session_id: str, memory_id: str):
    """Delete a specific memory."""
    state = _get_session_or_404(session_id)
    if not hasattr(state.agent, "_memory_store") or state.agent._memory_store is None:
        raise HTTPException(status_code=404, detail="No memory store")
    if not state.agent._memory_store.remove(memory_id):
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found")
    return {"status": "deleted"}


@router.post("/sessions/{session_id}/memories/toggle-global")
async def toggle_global_memory(session_id: str, req: ToggleGlobalMemoryRequest):
    """Toggle global memory enabled/disabled."""
    state = _get_session_or_404(session_id)
    if not hasattr(state.agent, "_memory_store") or state.agent._memory_store is None:
        raise HTTPException(status_code=404, detail="No memory store")
    state.agent._memory_store.toggle_global(req.enabled)
    return {"global_enabled": req.enabled}


@router.delete("/sessions/{session_id}/memories")
async def clear_all_memories(session_id: str):
    """Clear all memories."""
    state = _get_session_or_404(session_id)
    if not hasattr(state.agent, "_memory_store") or state.agent._memory_store is None:
        return {"status": "ok"}
    state.agent._memory_store.replace_all([])
    return {"status": "cleared"}


@router.post("/sessions/{session_id}/memories/refresh")
async def refresh_memories(session_id: str):
    """Reload memories from disk."""
    state = _get_session_or_404(session_id)
    if not hasattr(state.agent, "_memory_store") or state.agent._memory_store is None:
        return {"status": "no_store"}
    state.agent._memory_store.load()
    return {"status": "refreshed"}


@router.get("/sessions/{session_id}/memories/search")
async def search_memories(
    session_id: str,
    q: str = "",
    type: str | None = None,
    scope: str | None = None,
    limit: int = 20,
):
    """Search memories by query, type, and/or scope."""
    state = _get_session_or_404(session_id)
    if not hasattr(state.agent, "_memory_store") or state.agent._memory_store is None:
        return {"results": []}
    ms = state.agent._memory_store
    results = ms.search(q, mem_type=type, scope=scope, limit=limit)
    from dataclasses import asdict

    return {"results": [asdict(m) for m in results]}


@router.get("/sessions/{session_id}/memories/archived")
async def get_archived_memories(session_id: str):
    """Get truly archived (dropped/deleted) memories, excluding versioned predecessors."""
    state = _get_session_or_404(session_id)
    if not hasattr(state.agent, "_memory_store") or state.agent._memory_store is None:
        return {"archived": []}
    ms = state.agent._memory_store
    from dataclasses import asdict

    return {"archived": [asdict(m) for m in ms.get_truly_archived()]}


@router.get("/sessions/{session_id}/memories/{memory_id}/history")
async def get_memory_version_history(session_id: str, memory_id: str):
    """Get version history chain for a specific memory."""
    state = _get_session_or_404(session_id)
    if not hasattr(state.agent, "_memory_store") or state.agent._memory_store is None:
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
        raise HTTPException(
            status_code=404, detail=f"No operations for session '{saved_id}'"
        )

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
        raise HTTPException(
            status_code=404, detail=f"No operations for session '{saved_id}'"
        )

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
        if hasattr(state.agent, "generate_inline_completions"):
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
        from agent.truncation import get_item_limit

        lines = history_file.read_text(encoding="utf-8").strip().splitlines()
        return {"history": lines[-get_item_limit("items.api_input_history") :]}
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
    from agent.llm_utils import get_context_limit
    import config as app_config

    state = _get_session_or_404(session_id)
    usage = state.agent.get_token_usage()

    breakdown = {}
    if hasattr(state.agent, "get_token_usage_breakdown"):
        breakdown = state.agent.get_token_usage_breakdown()

    memory_bytes = state.store.memory_usage_bytes()
    data_entries = len(state.store)

    context_limits = {
        "smart": get_context_limit(app_config.SMART_MODEL),
        "sub_agent": get_context_limit(app_config.SUB_AGENT_MODEL),
        "inline": get_context_limit(app_config.INLINE_MODEL),
        "planner": get_context_limit(app_config.PLANNER_MODEL),
        "insight": get_context_limit(app_config.INSIGHT_MODEL),
    }

    return {
        "total": usage,
        "breakdown": breakdown,
        "memory_bytes": memory_bytes,
        "data_entries": data_entries,
        "context_limits": context_limits,
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
    _validate_path_component(item_id, "item_id")
    items = _read_gallery_index()
    updated = [it for it in items if it["id"] != item_id]
    if len(updated) == len(items):
        raise HTTPException(
            status_code=404, detail=f"Gallery item '{item_id}' not found"
        )
    _write_gallery_index(updated)
    # Delete PNG
    png_path = _gallery_dir() / f"{item_id}.png"
    if png_path.exists():
        png_path.unlink()


@router.get("/gallery/{item_id}/thumbnail")
async def get_gallery_thumbnail(item_id: str):
    """Serve the PNG thumbnail for a gallery item."""
    _validate_path_component(item_id, "item_id")
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
        raise HTTPException(
            status_code=404, detail=f"Gallery item '{item_id}' not found"
        )

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


# ---- Saved Pipelines ----


@router.get("/pipelines")
async def list_pipelines():
    """List all saved pipelines."""
    from data_ops.pipeline import SavedPipeline

    return SavedPipeline.list_all()


@router.get("/pipelines/archived")
async def list_archived_pipelines():
    """List all archived pipelines."""
    from data_ops.pipeline import SavedPipeline

    return SavedPipeline.list_archived()


@router.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Get full pipeline detail."""
    from data_ops.pipeline import SavedPipeline

    try:
        pipeline = SavedPipeline.load(pipeline_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Pipeline '{pipeline_id}' not found"
        )
    return pipeline.to_dict()


@router.post("/pipelines", status_code=201)
async def create_pipeline(req: SavePipelineRequest):
    """Create a saved pipeline from a session's pipeline."""
    from data_ops.pipeline import SavedPipeline

    loop = asyncio.get_running_loop()

    def _create():
        pipeline = SavedPipeline.from_session(
            req.session_id,
            render_op_id=req.render_op_id,
            name=req.name,
            description=req.description,
            tags=req.tags,
        )
        issues = pipeline.validate()
        pipeline.save()
        result = pipeline.to_dict()
        if issues:
            result["validation_warnings"] = issues
        return result

    try:
        result = await loop.run_in_executor(_thread_pool, _create)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@router.put("/pipelines/{pipeline_id}")
async def update_pipeline(pipeline_id: str, req: UpdatePipelineRequest):
    """Update pipeline metadata or steps."""
    from data_ops.pipeline import SavedPipeline

    try:
        pipeline = SavedPipeline.load(pipeline_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Pipeline '{pipeline_id}' not found"
        )

    if req.name is not None:
        pipeline.name = req.name
    if req.description is not None:
        pipeline.description = req.description
    if req.tags is not None:
        pipeline.tags = req.tags
    if req.steps is not None:
        pipeline._data["steps"] = req.steps

    pipeline.save()
    return pipeline.to_dict()


@router.delete("/pipelines/{pipeline_id}", status_code=204)
async def delete_pipeline(pipeline_id: str):
    """Delete (archive) a saved pipeline."""
    from data_ops.pipeline import SavedPipeline

    if not SavedPipeline.delete(pipeline_id):
        raise HTTPException(
            status_code=404, detail=f"Pipeline '{pipeline_id}' not found"
        )


@router.post("/pipelines/{pipeline_id}/restore")
async def restore_pipeline(pipeline_id: str):
    """Restore an archived pipeline back to active status."""
    from data_ops.pipeline import SavedPipeline

    try:
        pipeline = SavedPipeline.restore(pipeline_id)
        return pipeline.to_dict()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Pipeline '{pipeline_id}' not found"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/pipelines/{pipeline_id}/feedback")
async def add_pipeline_feedback(pipeline_id: str, req: PipelineFeedbackRequest):
    """Add user feedback to a saved pipeline."""
    from data_ops.pipeline import SavedPipeline

    try:
        pipeline = SavedPipeline.load(pipeline_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Pipeline '{pipeline_id}' not found"
        )
    entry = pipeline.add_feedback(req.comment)
    pipeline.save()
    return entry


@router.post("/pipelines/{pipeline_id}/validate")
async def validate_pipeline(pipeline_id: str):
    """Validate a saved pipeline and return issues."""
    from data_ops.pipeline import SavedPipeline

    try:
        pipeline = SavedPipeline.load(pipeline_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Pipeline '{pipeline_id}' not found"
        )
    issues = pipeline.validate()
    return {"valid": len(issues) == 0, "issues": issues}


@router.post("/pipelines/{pipeline_id}/execute")
async def execute_pipeline_saved(pipeline_id: str, req: ExecutePipelineRequest):
    """Execute a saved pipeline with a new time range."""
    from data_ops.pipeline import SavedPipeline

    loop = asyncio.get_running_loop()

    def _execute():
        pipeline = SavedPipeline.load(pipeline_id)
        issues = pipeline.validate()
        if issues:
            raise ValueError(f"Pipeline validation failed: {'; '.join(issues)}")
        result = pipeline.execute(req.time_start, req.time_end)
        fig_resp = _figure_response(result.figure, pipeline_id)
        return {
            "steps_completed": result.steps_completed,
            "steps_total": result.steps_total,
            "errors": result.errors,
            "data_labels": [e["label"] for e in result.store.list_entries()],
            **fig_resp,
        }

    try:
        result = await loop.run_in_executor(_thread_pool, _execute)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@router.get("/pipelines/{pipeline_id}/dag")
async def get_pipeline_dag_saved(pipeline_id: str):
    """Get a DAG visualization of a saved pipeline's steps."""
    from data_ops.pipeline import SavedPipeline

    try:
        pipeline = SavedPipeline.load(pipeline_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Pipeline '{pipeline_id}' not found"
        )

    loop = asyncio.get_running_loop()

    def _build_dag():
        # Convert pipeline steps to records for plot_pipeline
        records = []
        for step in pipeline.steps:
            records.append(
                {
                    "id": step["step_id"],
                    "tool": step["tool"],
                    "status": "success",
                    "inputs": [],  # labels, not step_ids
                    "outputs": [step["output_label"]]
                    if step.get("output_label")
                    else [],
                    "args": step.get("params", {}),
                }
            )
        # Build input_producers for edges
        step_to_label = {
            s["step_id"]: s["output_label"]
            for s in pipeline.steps
            if s.get("output_label")
        }
        for step, rec in zip(pipeline.steps, records):
            for inp_step_id in step.get("inputs", []):
                label = step_to_label.get(inp_step_id)
                if label:
                    rec["inputs"].append(label)
                    rec.setdefault("input_producers", {})[label] = inp_step_id

        # Compute contributes_to via backward BFS from presentation steps
        rec_by_id = {r["id"]: r for r in records}
        label_to_step = {
            s["output_label"]: s["step_id"]
            for s in pipeline.steps
            if s.get("output_label")
        }
        presentation_ids = [
            s["step_id"] for s in pipeline.steps if s.get("phase") == "presentation"
        ]
        for product_id in presentation_ids:
            rec_by_id[product_id].setdefault("contributes_to", []).append(product_id)
            queue = list(rec_by_id[product_id]["inputs"])
            visited: set[str] = set()
            while queue:
                label = queue.pop()
                if label in visited:
                    continue
                visited.add(label)
                producer_sid = label_to_step.get(label)
                if producer_sid and producer_sid in rec_by_id:
                    rec_by_id[producer_sid].setdefault("contributes_to", []).append(
                        product_id
                    )
                    for inp in rec_by_id[producer_sid]["inputs"]:
                        if inp not in visited:
                            queue.append(inp)

        from scripts.plot_pipeline import build_figure

        fig = build_figure(records, f"pipeline:{pipeline_id}")
        return _figure_response(fig, pipeline_id)

    return await loop.run_in_executor(_thread_pool, _build_dag)


# ---- Asset Management ----


def _epoch_to_iso(epoch: float | None) -> str | None:
    """Convert epoch seconds to ISO 8601 string."""
    if epoch is None:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _dir_stats_to_response(ds) -> dict:
    """Convert a DirStats dataclass to DirStatsResponse-compatible dict."""
    return {
        "name": ds.name,
        "path": ds.path,
        "total_bytes": ds.total_bytes,
        "file_count": ds.file_count,
        "oldest_mtime": _epoch_to_iso(ds.oldest_mtime),
        "newest_mtime": _epoch_to_iso(ds.newest_mtime),
        "turn_count": ds.turn_count,
        "session_name": ds.session_name,
    }


def _category_to_response(cat) -> dict:
    """Convert an AssetCategory dataclass to AssetCategoryResponse-compatible dict."""
    return {
        "name": cat.name,
        "path": cat.path,
        "total_bytes": cat.total_bytes,
        "file_count": cat.file_count,
        "subcategories": [_dir_stats_to_response(s) for s in cat.subcategories],
    }


@router.get("/assets")
async def get_assets():
    """Scan all 4 asset categories and return overview."""
    from data_ops.asset_manager import get_asset_overview

    loop = asyncio.get_running_loop()
    overview = await loop.run_in_executor(_thread_pool, get_asset_overview)
    return {
        "categories": [_category_to_response(c) for c in overview.categories],
        "total_bytes": overview.total_bytes,
        "scan_time_ms": overview.scan_time_ms,
    }


@router.get("/assets/cdf-cache")
async def get_cdf_cache():
    """Per-mission breakdown of CDF cache."""
    from data_ops.asset_manager import get_cdf_cache_detail

    loop = asyncio.get_running_loop()
    cat = await loop.run_in_executor(_thread_pool, get_cdf_cache_detail)
    return _category_to_response(cat)


@router.get("/assets/ppi-cache")
async def get_ppi_cache():
    """Per-collection breakdown of PPI cache."""
    from data_ops.asset_manager import get_ppi_cache_detail

    loop = asyncio.get_running_loop()
    cat = await loop.run_in_executor(_thread_pool, get_ppi_cache_detail)
    return _category_to_response(cat)


@router.get("/assets/sessions")
async def get_sessions_assets():
    """Per-session breakdown with metadata."""
    from data_ops.asset_manager import get_sessions_detail

    loop = asyncio.get_running_loop()
    cat = await loop.run_in_executor(_thread_pool, get_sessions_detail)
    return _category_to_response(cat)


@router.get("/assets/spice-kernels")
async def get_spice_kernels():
    """Per-mission breakdown of SPICE kernels."""
    from data_ops.asset_manager import get_spice_kernels_detail

    loop = asyncio.get_running_loop()
    cat = await loop.run_in_executor(_thread_pool, get_spice_kernels_detail)
    return _category_to_response(cat)


@router.post("/assets/cdf-cache/clean")
async def clean_cdf_cache(req: CleanupRequest):
    """Clean CDF cache files."""
    from data_ops.asset_manager import clean_cdf_cache as _clean

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _thread_pool,
        lambda: _clean(
            missions=req.targets or None,
            older_than_days=req.older_than_days,
            dry_run=req.dry_run,
        ),
    )
    return result


@router.post("/assets/ppi-cache/clean")
async def clean_ppi_cache(req: CleanupRequest):
    """Clean PPI cache files."""
    from data_ops.asset_manager import clean_ppi_cache as _clean

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _thread_pool,
        lambda: _clean(
            collections=req.targets or None,
            older_than_days=req.older_than_days,
            dry_run=req.dry_run,
        ),
    )
    return result


@router.post("/assets/sessions/clean")
async def clean_sessions(req: CleanupRequest):
    """Clean session data. Auto-excludes active API sessions."""
    from data_ops.asset_manager import clean_sessions as _clean

    # Get active session IDs to protect
    active_ids = {s["session_id"] for s in session_manager.list_sessions()}
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _thread_pool,
        lambda: _clean(
            session_ids=req.targets or None,
            older_than_days=req.older_than_days,
            empty_only=req.empty_only,
            exclude_ids=active_ids,
            dry_run=req.dry_run,
        ),
    )
    return result


@router.post("/assets/spice-kernels/clean")
async def clean_spice_kernels(req: CleanupRequest):
    """Clean SPICE kernel files."""
    from data_ops.asset_manager import clean_spice_kernels as _clean

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _thread_pool,
        lambda: _clean(
            missions=req.targets or None,
            dry_run=req.dry_run,
        ),
    )
    return result


# ---- Server status ----


@router.get("/eureka")
async def list_eurekas(session_id: str | None = None, status: str | None = None):
    """List all eurekas with optional filters."""
    from agent.eureka_store import EurekaStore
    from dataclasses import asdict

    store = EurekaStore()
    entries = store.list(session_id=session_id, status=status)
    return {"eurekas": [asdict(e) for e in entries]}


@router.get("/eureka/{eureka_id}")
async def get_eureka(eureka_id: str):
    """Get a single eureka by ID."""
    from agent.eureka_store import EurekaStore
    from dataclasses import asdict

    store = EurekaStore()
    entry = store.get(eureka_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Eureka not found")
    return asdict(entry)


@router.patch("/eureka/{eureka_id}")
async def update_eureka(eureka_id: str, body: dict):
    """Update eureka status."""
    from agent.eureka_store import EurekaStore

    status = body.get("status")
    if status not in ("proposed", "reviewed", "confirmed", "rejected"):
        raise HTTPException(status_code=400, detail="Invalid status")
    store = EurekaStore()
    ok = store.update_status(eureka_id, status)
    if not ok:
        raise HTTPException(status_code=404, detail="Eureka not found")
    return {"ok": True}


@router.get("/status")
async def server_status():
    """Server status (active sessions, uptime)."""
    from config import get_api_key

    return ServerStatus(
        active_sessions=len(session_manager.list_sessions()),
        max_sessions=session_manager.max_sessions,
        uptime_seconds=time.time() - _start_time,
        api_key_configured=bool(get_api_key("gemini")),
    ).model_dump()
