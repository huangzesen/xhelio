"""Delegation tool handlers."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent

from agent.sub_agent import AgentState
from agent.event_bus import DELEGATION, DELEGATION_DONE


def handle_delegate_to_mission(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    mission_id = tool_args["mission_id"]
    request = tool_args["request"]
    orch._event_bus.emit(
        DELEGATION,
        level="debug",
        msg=f"[Router] Delegating to {mission_id} specialist",
    )

    try:
        primary = orch._get_or_create_mission_agent(mission_id)
        if primary.state != AgentState.SLEEPING or primary.inbox.qsize() > 0:
            actor = orch._create_ephemeral_mission_actor(mission_id)
            is_ephemeral = True
        else:
            actor = primary
            is_ephemeral = False
        full_request = orch._build_mission_request(mission_id, request, agent=actor)
        tool_call_id = getattr(orch._tls, "current_tool_call_id", None)

        def _mission_post(
            result, _mid=mission_id, _actor_id=actor.agent_id, _eph=is_ephemeral
        ):
            result["mission"] = _mid
            orch._event_bus.emit(
                DELEGATION_DONE,
                level="debug",
                msg=f"[Router] {_mid} specialist finished",
            )
            if _eph:
                orch._cleanup_ephemeral_actor(_actor_id)
            return result

        return orch._delegate_to_sub_agent(
            actor,
            full_request,
            store_snapshot=orch._store.list_entries(),
            tool_call_id=tool_call_id,
            agent_type="mission",
            agent_name=actor.agent_id,
            task_summary=request[:200],
            post_process=_mission_post,
        )
    except (KeyError, FileNotFoundError):
        return {
            "status": "error",
            "message": f"Unknown mission '{mission_id}'. Check the supported missions table.",
        }


def handle_delegate_to_viz_plotly(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    request = tool_args["request"]
    context = tool_args.get("context", "")
    orch._event_bus.emit(
        DELEGATION,
        level="debug",
        msg="[Router] Delegating to Visualization specialist",
    )

    req_lower = request.lower()
    if "export" in req_lower or ".png" in req_lower or ".pdf" in req_lower:
        import re as _re

        fn_match = _re.search(r"[\w.-]+\.(?:png|pdf|svg)", request, _re.IGNORECASE)
        filename = fn_match.group(0) if fn_match else "output.png"
        fmt = "pdf" if filename.endswith(".pdf") else "png"
        result = orch._renderer.export(filename, format=fmt)
        if (
            result.get("status") == "success"
            and not orch.gui_mode
            and not orch.web_mode
        ):
            try:
                import os, platform, subprocess

                fp = result["filepath"]
                if platform.system() == "Darwin":
                    subprocess.Popen(["open", fp])
                elif platform.system() == "Windows":
                    os.startfile(fp)
                else:
                    subprocess.Popen(["xdg-open", fp])
            except Exception:
                pass
        return {
            "status": "success",
            "result": f"Exported plot to {result.get('filepath', filename)}",
        }

    full_request = f"{request}\n\nContext: {context}" if context else request

    _viz_agent_id = "VizAgent[Plotly]"
    store = orch._store
    entries = store.list_entries()
    if entries:
        import json

        new_entries, removed_labels, store_hash = orch._ctx_tracker.get_store_delta(
            _viz_agent_id, entries
        )
        if new_entries or removed_labels:
            store_text = json.dumps(new_entries, indent=2, default=str)
            if new_entries and not removed_labels:
                full_request += (
                    "\n\nNew data added to memory:\n```json\n" + store_text + "\n```"
                )
            elif removed_labels and not new_entries:
                full_request += "\n\nData removed from memory: " + ", ".join(
                    removed_labels
                )
            else:
                full_request += (
                    "\n\nData store updated:\n```json\n" + store_text + "\n```"
                )
                if removed_labels:
                    full_request += "\nRemoved: " + ", ".join(removed_labels)
            orch._ctx_tracker.record(
                _viz_agent_id, store_entries=entries, store_hash=store_hash
            )

    state = orch._renderer.get_current_state()
    if state["has_plot"]:
        if state.get("figure_json"):
            import json

            fig_json_str = json.dumps(state["figure_json"], indent=2)
            plot_repr = fig_json_str
        else:
            plot_repr = str(state.get("traces", ""))
        if orch._ctx_tracker.is_changed(_viz_agent_id, "plot", plot_repr):
            if state.get("figure_json"):
                full_request += (
                    f"\n\nCurrently displayed: {state['traces']}"
                    f"\n\nCurrent figure_json (modify this, don't rebuild from scratch):\n{fig_json_str}"
                )
            else:
                full_request += f"\n\nCurrently displayed: {state['traces']}"
            orch._ctx_tracker.record(_viz_agent_id, plot=plot_repr)
        else:
            full_request += (
                f"\n\nPlot unchanged, currently displayed: {state['traces']}"
            )
    else:
        full_request += "\n\nNo plot currently displayed."

    actor = orch._get_or_create_viz_plotly_actor()
    if actor.state != AgentState.SLEEPING or actor.inbox.qsize() > 0:
        return {
            "status": "error",
            "message": (
                "The visualization agent is already processing a delegation. "
                "Wait for it to finish before sending another request."
            ),
        }
    tool_call_id = getattr(orch._tls, "current_tool_call_id", None)

    def _viz_post(result):
        orch._event_bus.emit(
            DELEGATION_DONE,
            level="debug",
            msg="[Router] Plotly Visualization specialist finished",
        )
        return result

    return orch._delegate_to_sub_agent(
        actor,
        full_request,
        store_snapshot=orch._store.list_entries(),
        tool_call_id=tool_call_id,
        agent_type="viz_plotly",
        agent_name="VizAgent[Plotly]",
        task_summary=request[:200],
        post_process=_viz_post,
    )


def handle_delegate_to_viz_mpl(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    request = tool_args["request"]
    context = tool_args.get("context", "")
    orch._event_bus.emit(
        DELEGATION,
        level="debug",
        msg="[Router] Delegating to MPL Visualization specialist",
    )

    full_request = f"{request}\n\nContext: {context}" if context else request

    _mpl_agent_id = "VizAgent[Mpl]"
    store = orch._store
    entries = store.list_entries()
    if entries:
        import json

        new_entries, removed_labels, store_hash = orch._ctx_tracker.get_store_delta(
            _mpl_agent_id, entries
        )
        if new_entries or removed_labels:
            store_text = json.dumps(new_entries, indent=2, default=str)
            if new_entries and not removed_labels:
                full_request += (
                    "\n\nNew data added to memory:\n```json\n" + store_text + "\n```"
                )
            elif removed_labels and not new_entries:
                full_request += "\n\nData removed from memory: " + ", ".join(
                    removed_labels
                )
            else:
                full_request += (
                    "\n\nData store updated:\n```json\n" + store_text + "\n```"
                )
                if removed_labels:
                    full_request += "\nRemoved: " + ", ".join(removed_labels)
            orch._ctx_tracker.record(
                _mpl_agent_id, store_entries=entries, store_hash=store_hash
            )

    actor = orch._get_or_create_viz_mpl_actor()
    if actor.state != AgentState.SLEEPING or actor.inbox.qsize() > 0:
        return {
            "status": "error",
            "message": (
                "The MPL visualization agent is already processing a delegation. "
                "Wait for it to finish before sending another request."
            ),
        }
    tool_call_id = getattr(orch._tls, "current_tool_call_id", None)

    def _mpl_viz_post(result):
        orch._event_bus.emit(
            DELEGATION_DONE,
            level="debug",
            msg="[Router] MPL Visualization specialist finished",
        )
        return result

    return orch._delegate_to_sub_agent(
        actor,
        full_request,
        store_snapshot=orch._store.list_entries(),
        tool_call_id=tool_call_id,
        agent_type="viz_mpl",
        agent_name="VizAgent[Mpl]",
        task_summary=request[:200],
        post_process=_mpl_viz_post,
    )


def handle_delegate_to_viz_jsx(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    request = tool_args["request"]
    context = tool_args.get("context", "")
    orch._event_bus.emit(
        DELEGATION,
        level="debug",
        msg="[Router] Delegating to JSX Visualization specialist",
    )

    full_request = f"{request}\n\nContext: {context}" if context else request

    _jsx_agent_id = "VizAgent[JSX]"
    store = orch._store
    entries = store.list_entries()
    if entries:
        import json

        new_entries, removed_labels, store_hash = orch._ctx_tracker.get_store_delta(
            _jsx_agent_id, entries
        )
        if new_entries or removed_labels:
            store_text = json.dumps(new_entries, indent=2, default=str)
            if new_entries and not removed_labels:
                full_request += (
                    "\n\nNew data added to memory:\n```json\n" + store_text + "\n```"
                )
            elif removed_labels and not new_entries:
                full_request += "\n\nData removed from memory: " + ", ".join(
                    removed_labels
                )
            else:
                full_request += (
                    "\n\nData store updated:\n```json\n" + store_text + "\n```"
                )
                if removed_labels:
                    full_request += "\nRemoved: " + ", ".join(removed_labels)
            orch._ctx_tracker.record(
                _jsx_agent_id, store_entries=entries, store_hash=store_hash
            )

    actor = orch._get_or_create_viz_jsx_actor()
    if actor.state != AgentState.SLEEPING or actor.inbox.qsize() > 0:
        return {
            "status": "error",
            "message": (
                "The JSX visualization agent is already processing a delegation. "
                "Wait for it to finish before sending another request."
            ),
        }
    tool_call_id = getattr(orch._tls, "current_tool_call_id", None)

    def _jsx_viz_post(result):
        orch._event_bus.emit(
            DELEGATION_DONE,
            level="debug",
            msg="[Router] JSX Visualization specialist finished",
        )
        return result

    return orch._delegate_to_sub_agent(
        actor,
        full_request,
        store_snapshot=orch._store.list_entries(),
        tool_call_id=tool_call_id,
        agent_type="viz_jsx",
        agent_name="VizAgent[JSX]",
        task_summary=request[:200],
        post_process=_jsx_viz_post,
    )


def handle_delegate_to_data_ops(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    request = tool_args["request"]
    context = tool_args.get("context", "")
    orch._event_bus.emit(
        DELEGATION,
        level="debug",
        msg="[Router] Delegating to DataOps specialist",
    )

    actor = orch._get_available_dataops_actor()
    is_ephemeral = actor.agent_id != "DataOpsAgent"
    full_request = orch._build_dataops_request(request, context, agent=actor)
    tool_call_id = getattr(orch._tls, "current_tool_call_id", None)

    def _dataops_post(result, _actor_id=actor.agent_id, _eph=is_ephemeral):
        orch._event_bus.emit(
            DELEGATION_DONE,
            level="debug",
            msg="[Router] DataOps specialist finished",
        )
        if _eph:
            orch._cleanup_ephemeral_actor(_actor_id)
        return result

    return orch._delegate_to_sub_agent(
        actor,
        full_request,
        store_snapshot=orch._store.list_entries(),
        tool_call_id=tool_call_id,
        agent_type="data_ops",
        agent_name=actor.agent_id,
        task_summary=request[:200],
        post_process=_dataops_post,
    )


def handle_delegate_to_data_extraction(
    orch: "OrchestratorAgent", tool_args: dict
) -> dict:
    request = tool_args["request"]
    context = tool_args.get("context", "")
    orch._event_bus.emit(
        DELEGATION,
        level="debug",
        msg="[Router] Delegating to DataExtraction specialist",
    )

    actor = orch._get_or_create_extraction_actor()
    if actor.state != AgentState.SLEEPING or actor.inbox.qsize() > 0:
        return {
            "status": "error",
            "message": (
                "The data extraction agent is already processing a delegation. "
                "Wait for it to finish before sending another request."
            ),
        }
    full_request = orch._build_extraction_request(request, context)
    tool_call_id = getattr(orch._tls, "current_tool_call_id", None)

    def _extraction_post(result):
        orch._event_bus.emit(
            DELEGATION_DONE,
            level="debug",
            msg="[Router] DataExtraction specialist finished",
        )
        return result

    return orch._delegate_to_sub_agent(
        actor,
        full_request,
        store_snapshot=orch._store.list_entries(),
        tool_call_id=tool_call_id,
        agent_type="data_extraction",
        agent_name="DataExtractionAgent",
        task_summary=request[:200],
        post_process=_extraction_post,
    )


def handle_delegate_to_insight(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    user_request = tool_args["request"]
    extra_context = tool_args.get("context", "")
    orch._event_bus.emit(
        DELEGATION,
        level="debug",
        msg="[Router] Delegating to Insight specialist",
    )

    orch._restore_deferred_figure()

    figure = orch._renderer.get_figure()
    if figure is None:
        return {
            "status": "error",
            "message": "No plot currently displayed. Create a plot first before requesting analysis.",
        }

    import io

    try:
        buf = io.BytesIO()
        figure.write_image(buf, format="png", width=1100, height=600, scale=2)
        image_bytes = buf.getvalue()
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to export plot as PNG for analysis: {e}. Is kaleido installed?",
        }

    _insight_agent_id = "InsightAgent"
    data_context = orch._build_insight_context()
    if extra_context:
        data_context += f"\n\nAdditional context: {extra_context}"
    if not orch._ctx_tracker.is_changed(_insight_agent_id, "session", data_context):
        data_context = "[Data context unchanged from previous analysis.]"
    else:
        orch._ctx_tracker.record(_insight_agent_id, session=data_context)

    actor = orch._get_or_create_insight_actor()
    if actor.state != AgentState.SLEEPING or actor.inbox.qsize() > 0:
        return {
            "status": "error",
            "message": (
                "The insight agent is already processing a delegation. "
                "Wait for it to finish before sending another request."
            ),
        }
    tool_call_id = getattr(orch._tls, "current_tool_call_id", None)

    def _insight_post(result):
        orch._event_bus.emit(
            DELEGATION_DONE,
            level="debug",
            msg="[Router] Insight specialist finished",
        )
        return result

    return orch._delegate_to_sub_agent(
        actor,
        {
            "action": "analyze",
            "image_bytes": image_bytes,
            "data_context": data_context,
            "user_request": user_request,
        },
        tool_call_id=tool_call_id,
        agent_type="insight",
        agent_name="InsightAgent",
        task_summary=user_request[:200],
        post_process=_insight_post,
    )
