"""Delegation tool handlers."""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent.tool_caller import ToolCaller
    from agent.tool_context import ToolContext

from agent.base_agent import AgentState
from agent.tool_caller import OrchestratorState

_EMPTY_ORCH = OrchestratorState()
from agent.delegation import AGENT_ID_DATAOPS, AGENT_ID_DATA_IO
from agent.event_bus import DELEGATION, DELEGATION_DONE


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_done_callback(ctx: "ToolContext", label: str, *, ephemeral_id: str | None = None):
    """Create a post-delegation callback that emits DELEGATION_DONE.

    Args:
        ctx: The orchestrator context.
        label: Human-readable label for the log message.
        ephemeral_id: If provided, clean up this ephemeral agent after completion.
    """
    def _post(result):
        if ctx.event_bus is not None:
            ctx.event_bus.emit(
                DELEGATION_DONE,
                level="debug",
                msg=f"[Router] {label} specialist finished",
                data={
                    "status": result.get("status"),
                    "text_preview": result.get("result", "")[:200],
                },
            )
        if ephemeral_id is not None:
            ctx.delegation.cleanup_ephemeral(ephemeral_id)
        return result
    return _post


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_delegate_to_envoy(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    mission_id = tool_args.get("envoy") if tool_args.get("envoy") is not None else tool_args.get("mission_id")
    request = tool_args["request"]
    wait = tool_args.get("wait", True)
    mode_text = " (run in background)" if not wait else ""
    if ctx.event_bus is not None:
        ctx.event_bus.emit(
            DELEGATION,
            level="debug",
            msg=f"[Router] Delegating to {mission_id} specialist{mode_text}",
        )

    try:
        primary = ctx.delegation.get_or_create_envoy_agent(mission_id)
        if primary.state != AgentState.SLEEPING or primary.inbox.qsize() > 0:
            agent = ctx.delegation.create_ephemeral_envoy_agent(mission_id)
            is_ephemeral = True
        else:
            agent = primary
            is_ephemeral = False
        full_request = ctx.delegation.build_envoy_request(mission_id, request, agent=agent)
        tool_call_id = caller.tool_call_id if caller else None

        base_post = _make_done_callback(
            ctx, mission_id,
            ephemeral_id=agent.agent_id if is_ephemeral else None,
        )

        def _envoy_post(result, _mid=mission_id):
            result["mission"] = _mid
            return base_post(result)

        return ctx.delegation.delegate_to_sub_agent(
            agent,
            full_request,
            store_snapshot=ctx.store.list_entries(),
            tool_call_id=tool_call_id,
            agent_type=f"envoy:{mission_id}",
            agent_name=agent.agent_id,
            task_summary=request[:200],
            post_process=_envoy_post,
            wait=wait,
        )
    except (KeyError, FileNotFoundError, NotImplementedError):
        return {
            "status": "error",
            "message": f"Unknown mission '{mission_id}'. Check the supported missions table.",
        }


def handle_delegate_to_viz(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    import config

    from agent.delegation import DelegationBus
    from agent.viz_backends import VIZ_BACKENDS

    request = tool_args["request"]
    context = tool_args.get("context", "")
    backend = tool_args.get("backend") or config.PREFER_VIZ_BACKEND
    wait = tool_args.get("wait", True)

    if backend not in VIZ_BACKENDS:
        return {"status": "error", "message": f"Unknown viz backend: {backend!r}. Available: {', '.join(VIZ_BACKENDS)}"}

    cfg = VIZ_BACKENDS[backend]
    agent_id = cfg["agent_id"]

    if ctx.event_bus is not None:
        ctx.event_bus.emit(
            DELEGATION,
            level="debug",
            msg=f"[Router] Delegating to {backend} visualization specialist",
        )

    full_request = f"{request}\n\nContext: {context}" if context else request

    store = ctx.store
    entries = store.list_entries()
    if entries:
        _orch_state = ctx.agent_state.get("orchestrator", _EMPTY_ORCH)
        if _orch_state.ctx_tracker is not None:
            delta_text = DelegationBus.format_store_delta(
                _orch_state.ctx_tracker, agent_id, entries, fmt="json"
            )
            if delta_text:
                full_request += delta_text

    agent = ctx.delegation.get_or_create_viz_agent(backend)
    if agent.state != AgentState.SLEEPING or agent.inbox.qsize() > 0:
        return {
            "status": "error",
            "message": (
                f"The {backend} visualization agent is already processing a delegation. "
                "Wait for it to finish before sending another request."
            ),
        }
    tool_call_id = caller.tool_call_id if caller else None

    return ctx.delegation.delegate_to_sub_agent(
        agent,
        full_request,
        store_snapshot=ctx.store.list_entries(),
        tool_call_id=tool_call_id,
        agent_type=cfg["agent_type"],
        agent_name=agent_id,
        task_summary=request[:200],
        post_process=_make_done_callback(ctx, f"{backend} visualization"),
        wait=wait,
    )


def handle_delegate_to_data_ops(ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None) -> dict:
    request = tool_args["request"]
    context = tool_args.get("context", "")
    wait = tool_args.get("wait", True)
    mode_text = " (run in background)" if not wait else ""
    if ctx.event_bus is not None:
        ctx.event_bus.emit(
            DELEGATION,
            level="debug",
            msg=f"[Router] Delegating to DataOps specialist{mode_text}",
        )

    agent = ctx.delegation.get_available_dataops_agent()
    is_ephemeral = agent.agent_id != AGENT_ID_DATAOPS
    full_request = ctx.delegation.build_dataops_request(request, context, agent=agent)
    tool_call_id = caller.tool_call_id if caller else None

    return ctx.delegation.delegate_to_sub_agent(
        agent,
        full_request,
        store_snapshot=ctx.store.list_entries(),
        tool_call_id=tool_call_id,
        agent_type="dataops",
        agent_name=agent.agent_id,
        task_summary=request[:200],
        post_process=_make_done_callback(
            ctx, "DataOps",
            ephemeral_id=agent.agent_id if is_ephemeral else None,
        ),
        wait=wait,
    )


def handle_delegate_to_data_io(
    ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None,
) -> dict:
    request = tool_args["request"]
    context = tool_args.get("context", "")
    wait = tool_args.get("wait", True)
    mode_text = " (run in background)" if not wait else ""
    if ctx.event_bus is not None:
        ctx.event_bus.emit(
            DELEGATION,
            level="debug",
            msg=f"[Router] Delegating to DataIO specialist{mode_text}",
        )

    agent = ctx.delegation.get_or_create_data_io_agent()
    if agent.state != AgentState.SLEEPING or agent.inbox.qsize() > 0:
        return {
            "status": "error",
            "message": (
                "The data I/O agent is already processing a delegation. "
                "Wait for it to finish before sending another request."
            ),
        }
    full_request = ctx.delegation.build_data_io_request(request, context)
    tool_call_id = caller.tool_call_id if caller else None

    return ctx.delegation.delegate_to_sub_agent(
        agent,
        full_request,
        store_snapshot=ctx.store.list_entries(),
        tool_call_id=tool_call_id,
        agent_type="data_io",
        agent_name=AGENT_ID_DATA_IO,
        task_summary=request[:200],
        post_process=_make_done_callback(ctx, "DataIO"),
        wait=wait,
    )
