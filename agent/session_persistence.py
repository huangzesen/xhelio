"""Session persistence and plan management.

Lifecycle functions (start_session, save_session, load_session) take an
OrchestratorAgent but only access public API methods (get_chat_state,
restore_chat, get_token_usage, restore_token_state, cycle_number) and
session_ctx fields. No private attribute access.

Helper functions (save_plan, load_plan, get_plot_status, etc.) accept
a SessionContext directly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from data_ops.store import DataStore, set_store, resolve_entry
from data_ops.asset_registry import AssetRegistry
from agent.event_bus import (
    EventLogWriter,
    TokenLogListener,
    DEBUG,
    USER_MESSAGE,
    SESSION_TITLE,
)
from agent.logging import set_session_id, attach_log_file, LOG_DIR, get_logger
from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY
from agent.truncation import trunc

from agent.tool_caller import OrchestratorState

if TYPE_CHECKING:
    from agent.orchestrator_agent import OrchestratorAgent
    from agent.session_context import SessionContext

logger = get_logger()

_EMPTY_ORCH = OrchestratorState()


def _orch(sctx: "SessionContext") -> OrchestratorState:
    """Get orchestrator state from a SessionContext, or empty sentinel."""
    return sctx.agent_state.get("orchestrator", _EMPTY_ORCH)


def _ensure_orch(sctx: "SessionContext") -> OrchestratorState:
    """Get or create orchestrator state section."""
    if "orchestrator" not in sctx.agent_state:
        sctx.agent_state["orchestrator"] = OrchestratorState()
    return sctx.agent_state["orchestrator"]


# ---------------------------------------------------------------------------
# Internal listeners
# ---------------------------------------------------------------------------


def start_event_log_writer(ctx: "SessionContext") -> None:
    """Create and subscribe an EventLogWriter for the current session directory."""
    orch_state = _ensure_orch(ctx)
    event_bus = ctx.event_bus
    # Close any existing writer first
    if orch_state.event_log_writer is not None:
        event_bus.unsubscribe(orch_state.event_log_writer)
        orch_state.event_log_writer.close()
    session_dir = ctx.session_manager.base_dir / ctx.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    orch_state.event_log_writer = EventLogWriter(session_dir / "events.jsonl")
    event_bus.subscribe(orch_state.event_log_writer)


def start_token_log_listener(ctx: "SessionContext") -> None:
    """Create and subscribe a TokenLogListener for the current session."""
    orch_state = _ensure_orch(ctx)
    event_bus = ctx.event_bus
    if orch_state.token_log_listener is not None:
        event_bus.unsubscribe(orch_state.token_log_listener)
        orch_state.token_log_listener.close()
    token_log_path = LOG_DIR / f"token_{ctx.session_id}.jsonl"
    orch_state.token_log_listener = TokenLogListener(token_log_path)
    event_bus.subscribe(orch_state.token_log_listener)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def start_session(orch: "OrchestratorAgent") -> str:
    """Create a new session on disk, attach the log file, and enable auto-save.

    The session directory and metadata are created immediately so the
    session appears in the sidebar right away (with a default name).
    Empty sessions from previous runs are cleaned up first.

    Returns:
        The new session_id.
    """
    sctx = orch.session_ctx
    orch_state = _ensure_orch(sctx)
    sctx.session_manager.cleanup_empty_sessions()
    sctx.session_id = sctx.session_manager.create_session(orch.model_name)
    orch_state.auto_save = True
    orch_state.session_title_generated = False
    set_session_id(sctx.session_id)
    attach_log_file(sctx.session_id)
    start_token_log_listener(sctx)
    sctx.event_bus.session_id = sctx.session_id

    # Create disk-backed DataStore for this session
    session_dir = sctx.session_manager.base_dir / sctx.session_id
    sctx.session_dir = session_dir
    (session_dir / "sandbox").mkdir(exist_ok=True)
    sctx.store = DataStore(session_dir / "data")
    set_store(sctx.store)
    sctx.asset_registry = AssetRegistry(session_dir, sctx.store)

    # Create session-scoped PipelineDAG
    from data_ops.dag import PipelineDAG
    sctx.dag = PipelineDAG(session_dir=session_dir)

    # Start writing structured event log to disk
    start_event_log_writer(sctx)

    # Push kind handlers into global TOOL_REGISTRY for dispatch
    ENVOY_KIND_REGISTRY.register_handlers_globally()

    sctx.event_bus.emit(
        DEBUG, level="debug", msg=f"[Session] Started: {sctx.session_id}"
    )
    return sctx.session_id


def save_session(orch: "OrchestratorAgent") -> None:
    """Persist the current chat history and DataStore to disk."""
    sctx = orch.session_ctx
    if not sctx.session_id:
        return
    event_bus = sctx.event_bus

    # Use interface as single source of truth
    chat_state = orch.get_chat_state()
    interface_dict = chat_state.get("messages", [])
    if not interface_dict and chat_state == {}:
        # get_chat_state returns {} when no chat or serialization fails
        pass

    store = sctx.store
    usage = orch.get_token_usage()

    # EventBus user messages — used for turn count and preview extraction
    # (original text, not augmented with injected context headers).
    bus_user_msgs = event_bus.get_events(types={USER_MESSAGE})

    # Turn count: prefer EventBus user messages (always available); fall
    # back to counting "user" roles in interface for sessions.
    # Interactions API sessions don't expose full history client-side, so
    # the EventBus count is the primary source.
    turn_count = (
        len(bus_user_msgs)
        if bus_user_msgs
        else sum(1 for e in interface_dict if e.get("role") == "user")
    )

    # Round count: track orchestrator cycles (round_start → round_end pairs).
    round_count = orch.cycle_number

    # Don't persist empty sessions (no user messages, no data, no events)
    if turn_count == 0 and len(store) == 0:
        return

    # Preview from last user message (original, not augmented)
    last_preview = ""
    if bus_user_msgs:
        last_event = bus_user_msgs[-1]
        last_text = (
            (last_event.data or {}).get("text", "")
            if hasattr(last_event, "data")
            else ""
        )
        if last_text:
            last_preview = trunc(last_text, "history.error.brief")
    # Fallback to interface history if EventBus had no text
    if not last_preview:
        for h in reversed(interface_dict):
            if h.get("role") == "user":
                content = h.get("content", [])
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            last_preview = trunc(text, "history.error.brief")
                            break
                if last_preview:
                    break

    # Display log: build from EventBus events (display_log_builder was removed)
    display_log = []

    metadata_updates = {
        "turn_count": turn_count,
        "round_count": round_count,
        "last_message_preview": last_preview,
        "token_usage": usage,
        "model": orch.model_name,
    }

    # Auto-generate a session title after the first round
    orch_state = _orch(sctx)
    if round_count >= 1 and not orch_state.session_title_generated:
        inline = orch_state.inline
        session_name = None
        if inline is not None:
            try:
                session_name = inline.generate_session_title()
            except Exception:
                pass
        if session_name:
            metadata_updates["name"] = session_name
            orch_state.session_title_generated = True
            event_bus.emit(
                SESSION_TITLE,
                level="info",
                msg=f"Session: {session_name}",
                data={"name": session_name},
            )

    if sctx.asset_registry is not None:
        sctx.asset_registry.save()

    renderer = sctx.renderer
    sctx.session_manager.save_session(
        session_id=sctx.session_id,
        chat_history=interface_dict,
        data_store=store,
        metadata_updates=metadata_updates,
        figure_state=renderer.save_state(),
        figure_obj=renderer.get_figure(),
        operations=[],  # Pipeline tracking now uses pipeline.json via PipelineDAG
        display_log=display_log,
    )
    event_bus.emit(
        DEBUG,
        level="debug",
        msg=f"[Session] Saved ({turn_count} turns, {len(store)} data entries)",
    )


def load_session(
    orch: "OrchestratorAgent",
    session_id: str,
    *,
    skip_interaction_resume: bool = False,
) -> tuple[dict, list[dict] | None, list[dict] | None]:
    """Restore chat history and DataStore from a saved session.

    Args:
        orch: The orchestrator agent.
        session_id: The session to load.
        skip_interaction_resume: Deprecated — kept for API compatibility.
            Sessions always create fresh LLM chats seeded with history.

    Returns:
        Tuple of (metadata dict, display_log list or None, event_log list or None).
    """
    sctx = orch.session_ctx
    (
        history_dicts,
        data_dir,
        metadata,
        figure_state,
        operations,
        display_log,
        event_log,
    ) = sctx.session_manager.load_session(session_id)

    event_bus = sctx.event_bus

    # Build state dict and restore chat via public API
    if history_dicts:
        saved_state = {
            "session_id": session_id,
            "messages": history_dicts,
            "metadata": metadata,
        }
        orch.restore_chat(saved_state)
        event_bus.emit(
            DEBUG, level="debug", msg=f"[Session] Chat restored for {session_id}"
        )
    else:
        orch.restore_chat({})

    # Restore DataStore — constructor auto-loads _labels.json (or migrates _index.json)
    sctx.session_dir = data_dir.parent
    (sctx.session_dir / "sandbox").mkdir(exist_ok=True)
    load_plan(sctx)
    sctx.store = DataStore(data_dir)
    set_store(sctx.store)
    orch_state = _ensure_orch(sctx)
    sctx.asset_registry = AssetRegistry(sctx.session_dir, sctx.store)
    event_bus.emit(
        DEBUG,
        level="debug",
        msg=f"[Session] DataStore opened at {data_dir} ({len(sctx.store)} entries)",
    )

    # Load PipelineDAG from session directory
    from data_ops.dag import PipelineDAG
    sctx.dag = PipelineDAG.load(sctx.session_dir)

    # Clear sub-agent caches (they'll be recreated on next use)
    if sctx.delegation is not None:
        sctx.delegation.stop_all()
    ENVOY_KIND_REGISTRY.clear_active()
    sctx.renderer.reset()

    # Defer figure restore — the full Plotly figure is built lazily when
    # the frontend first requests GET /figure (via restore_deferred_figure).
    # This avoids loading all data pickle files during session resume.
    renderer = sctx.renderer
    if figure_state:
        sctx.deferred_figure_state = figure_state
        # Restore lightweight metadata on the renderer (no data loading)
        renderer._panel_count = figure_state.get("panel_count", 0)
        renderer._trace_labels = figure_state.get("trace_labels", [])
        renderer._last_fig_json = figure_state.get("last_fig_json")
        event_bus.emit(
            DEBUG,
            level="debug",
            msg="[Session] Figure restore deferred until first access",
        )
    else:
        sctx.deferred_figure_state = None

    sctx.session_id = session_id
    orch_state.auto_save = True
    orch_state.session_title_generated = bool(metadata.get("name"))
    set_session_id(session_id)
    attach_log_file(session_id)
    start_token_log_listener(sctx)
    event_bus.session_id = session_id

    # Restore cumulative token usage from previous session runs.
    saved_usage = metadata.get("token_usage", {})
    if saved_usage:
        orch.restore_token_state(saved_usage)

    # Restore cycle counter from previous session runs
    orch.cycle_number = metadata.get("round_count", 0)

    # Start writing structured event log (append mode — resumes keep adding)
    start_event_log_writer(sctx)

    # Push kind handlers into global TOOL_REGISTRY for dispatch
    ENVOY_KIND_REGISTRY.register_handlers_globally()

    event_bus.emit(
        DEBUG, level="debug", msg=f"[Session] Loaded: {session_id}"
    )
    return metadata, display_log, event_log


# ---------------------------------------------------------------------------
# Figure management
# ---------------------------------------------------------------------------


def restore_deferred_figure(ctx) -> None:
    """Restore the Plotly figure from deferred state (lazy).

    Called when the frontend first requests the figure via
    GET /figure, or when the user sends their first message.

    Accepts either a SessionContext or an OrchestratorAgent (for
    backward compatibility with start_session/load_session callers).
    """
    sctx = ctx if hasattr(ctx, "deferred_figure_state") else ctx.session_ctx
    figure_state = sctx.deferred_figure_state
    if figure_state is None:
        return
    sctx.deferred_figure_state = None  # clear so we only do this once

    event_bus = sctx.event_bus

    try:
        entries = None
        last_fig_json = figure_state.get("last_fig_json")
        if last_fig_json:
            store = sctx.store
            entries = {}
            for trace in last_fig_json.get("data", []):
                label = trace.get("data_label")
                if label and label not in entries:
                    entry, _ = resolve_entry(store, label)
                    if entry is not None:
                        entries[label] = entry
            # Only pass entries if we resolved all of them
            all_labels = {
                t.get("data_label")
                for t in last_fig_json.get("data", [])
                if t.get("data_label")
            }
            if not all_labels.issubset(entries.keys()):
                entries = None  # fall back to legacy path

        sctx.renderer.restore_state(figure_state, entries=entries)
        if event_bus is not None:
            event_bus.emit(
                DEBUG, level="debug", msg="[Session] Deferred figure restore complete"
            )
    except Exception as e:
        if event_bus is not None:
            event_bus.emit(
                DEBUG,
                level="warning",
                msg=f"[Session] Could not restore deferred figure: {e}",
            )


def get_latest_figure_png(ctx) -> bytes | None:
    """Return PNG bytes for the most recent figure, checking memory then disk.

    Lookup order:
    1. In-memory cache (_latest_render_png) -- fast path, works during active session
    2. Plotly renderer export -- works if figure is in memory (including deferred restore)
    3. Disk files (mpl_outputs/, plotly_outputs/) -- always works after reload

    Accepts either a SessionContext or an OrchestratorAgent.
    """
    sctx = ctx if hasattr(ctx, "deferred_figure_state") else ctx.session_ctx
    eureka = sctx.eureka_hooks
    if eureka is not None and eureka.latest_render_png is not None:
        return eureka.latest_render_png

    restore_deferred_figure(ctx)
    figure = sctx.renderer.get_figure()
    if figure is not None:
        import io

        try:
            buf = io.BytesIO()
            figure.write_image(buf, format="png", width=1100, height=600, scale=2)
            png_bytes = buf.getvalue()
            if eureka is not None:
                eureka.latest_render_png = png_bytes
            return png_bytes
        except Exception:
            pass

    session_dir = sctx.session_dir
    if session_dir is not None:
        latest_png = find_latest_png(session_dir)
        if latest_png is not None:
            try:
                png_bytes = latest_png.read_bytes()
                if eureka is not None:
                    eureka.latest_render_png = png_bytes
                return png_bytes
            except Exception:
                pass

    return None


def find_latest_png(session_dir: Path) -> Path | None:
    """Find the most recently modified PNG file in the session output dirs."""
    png_dirs = ["mpl_outputs", "plotly_outputs"]
    latest: Path | None = None
    latest_mtime: float = 0
    for dirname in png_dirs:
        d = session_dir / dirname
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix == ".png" and f.stat().st_size > 0:
                mtime = f.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest = f
    return latest


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


def get_plot_status(ctx) -> dict:
    """Return plot status dict: state, panel_count, traces.

    Accepts either a SessionContext or an OrchestratorAgent.
    """
    sctx = ctx if hasattr(ctx, "deferred_figure_state") else ctx.session_ctx
    renderer = sctx.renderer
    figure = renderer.get_figure()
    if figure is not None:
        return {
            "state": "active",
            "panel_count": renderer._panel_count,
            "traces": list(renderer._trace_labels),
        }
    if sctx.deferred_figure_state is not None:
        return {
            "state": "restorable",
            "panel_count": sctx.deferred_figure_state.get("panel_count", 0),
            "traces": sctx.deferred_figure_state.get("trace_labels", []),
        }
    return {"state": "none", "panel_count": 0, "traces": []}


def get_data_status(ctx) -> dict:
    """Return data store status: total and cached counts.

    Accepts either a SessionContext or an OrchestratorAgent.
    """
    sctx = ctx if hasattr(ctx, "deferred_figure_state") else ctx.session_ctx
    store = sctx.store
    if store is None:
        return {"total_entries": 0, "loaded": 0, "deferred": 0}
    with store._lock:
        total = len(store._ids)
        cached = len(store._cache)
    return {
        "total_entries": total,
        "loaded": cached,
        "deferred": total - cached,
    }


def handle_get_session_assets(ctx) -> dict:
    """Handler for the get_session_assets tool.

    Accepts either a SessionContext or an OrchestratorAgent.
    """
    sctx = ctx if hasattr(ctx, "deferred_figure_state") else ctx.session_ctx
    plot_status = get_plot_status(ctx)
    data_status = get_data_status(ctx)
    return {
        "status": "success",
        "plot": plot_status,
        "data": data_status,
        "operations_count": sctx.dag.node_count() if sctx.dag is not None else 0,
    }


def handle_restore_plot(ctx) -> dict:
    """Handler for the restore_plot tool.

    Accepts either a SessionContext or an OrchestratorAgent.
    """
    sctx = ctx if hasattr(ctx, "deferred_figure_state") else ctx.session_ctx
    renderer = sctx.renderer
    # Already active — no-op
    if renderer.get_figure() is not None:
        return {
            "status": "success",
            "message": "Plot is already active.",
            "panel_count": renderer._panel_count,
            "traces": list(renderer._trace_labels),
        }
    # Attempt deferred restore
    if sctx.deferred_figure_state is not None:
        restore_deferred_figure(ctx)
        if renderer.get_figure() is not None:
            return {
                "status": "success",
                "message": "Plot restored from deferred session state.",
                "panel_count": renderer._panel_count,
                "traces": list(renderer._trace_labels),
            }
        return {
            "status": "error",
            "message": "Deferred figure state existed but restoration failed.",
        }
    return {
        "status": "error",
        "message": "No plot to restore — no active or deferred figure state.",
    }


# ---------------------------------------------------------------------------
# Plan persistence
# ---------------------------------------------------------------------------


def plan_path(ctx) -> Path:
    """Return the path to the plan.json file.

    Accepts either a SessionContext or an OrchestratorAgent.
    """
    sctx = ctx if hasattr(ctx, "deferred_figure_state") else ctx.session_ctx
    return Path(sctx.session_dir) / "plan.json"


def save_plan(ctx) -> None:
    """Persist current_plan to disk (or delete the file if plan is None).

    Accepts either a SessionContext or an OrchestratorAgent.
    """
    sctx = ctx if hasattr(ctx, "deferred_figure_state") else ctx.session_ctx
    if sctx.session_dir is None:
        return
    orch_state = _orch(sctx)
    path = Path(sctx.session_dir) / "plan.json"
    if orch_state.current_plan is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(json.dumps(orch_state.current_plan, ensure_ascii=False))


def load_plan(ctx) -> None:
    """Restore current_plan from disk if the file exists.

    Accepts either a SessionContext or an OrchestratorAgent.
    """
    sctx = ctx if hasattr(ctx, "deferred_figure_state") else ctx.session_ctx
    orch_state = _ensure_orch(sctx)
    if sctx.session_dir is None:
        orch_state.current_plan = None
        return
    path = Path(sctx.session_dir) / "plan.json"
    if path.exists():
        try:
            orch_state.current_plan = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            orch_state.current_plan = None
    else:
        orch_state.current_plan = None
