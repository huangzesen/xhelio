"""
Matplotlib visualization agent with optional think phase.

Owns matplotlib-based visualization through:
- generate_mpl_script — create plots via matplotlib Python scripts in a subprocess sandbox
- manage_mpl_output — list, view scripts, rerun, delete MPL outputs
- list_fetched_data — discover available data in memory

For plot-creation requests, runs a think→execute pattern: inspect data
shapes, types, units, and NaN counts before generating scripts.
Style/manage requests skip the think phase to avoid wasting tokens.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .llm import LLMAdapter
from .sub_agent import SubAgent
from .tools import get_function_schemas
from .tool_loop import run_tool_loop, extract_text_from_response
from .event_bus import EventBus, DEBUG, PROGRESS
from .loop_guard import LoopGuard
from .turn_limits import get_limit
from .model_fallback import get_active_model
from .agent_registry import VIZ_MPL_TOOLS, VIZ_THINK_TOOLS
from .llm_utils import send_with_timeout
from .logging import get_logger
from .viz_plotly_agent import _needs_think_phase, _check_think_rejection
from knowledge.prompt_builder import (
    build_viz_mpl_prompt,
    build_viz_think_prompt,
)

if TYPE_CHECKING:
    from .memory import MemoryStore


class VizMplAgent(SubAgent):
    """A SubAgent specialized for matplotlib visualization with think→execute pattern.

    Persistent — stays alive across delegations. The think phase runs
    as an ephemeral LLM chat (separate from the main SubAgent session) to
    inspect data before the execute phase.

    Unlike VizPlotlyAgent, this agent generates standalone Python scripts
    that execute in a subprocess sandbox. The session_dir is needed to locate
    the data directory for script execution.
    """

    _has_deferred_reviews = True

    def __init__(
        self,
        adapter: LLMAdapter,
        model_name: str,
        tool_executor,
        *,
        gui_mode: bool = False,
        event_bus: EventBus | None = None,
        memory_store: MemoryStore | None = None,
        memory_scope: str = "",
        session_dir: Path | None = None,
    ):
        self.gui_mode = gui_mode
        self.session_dir = session_dir

        # Build tool schemas
        tool_schemas = get_function_schemas(names=VIZ_MPL_TOOLS)

        # Think-phase tool schemas (data inspection only — shared with Plotly agent)
        self._think_tool_schemas = get_function_schemas(names=VIZ_THINK_TOOLS)

        super().__init__(
            agent_id="VizAgent[Mpl]",
            adapter=adapter,
            model_name=model_name,
            tool_executor=tool_executor,
            system_prompt=build_viz_mpl_prompt(gui_mode=gui_mode),
            tool_schemas=tool_schemas,
            event_bus=event_bus,
            memory_store=memory_store,
            memory_scope=memory_scope or "visualization",
        )

    def _handle_request(self, msg):
        """Override to inject think→execute pattern for plot-creation requests."""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Prepend incremental memory context if available
        memory_prefix = self._build_memory_prefix()
        if memory_prefix:
            content = f"{memory_prefix}\n\n{content}"

        if _needs_think_phase(content):
            # Think phase: run ephemeral data inspection
            think_context = self._run_think_phase(content)
            rejection = _check_think_rejection(think_context)
            if rejection:
                self._deliver_result(msg, {
                    "text": rejection, "failed": True, "errors": [rejection]
                })
                return

            # Enrich the request with think findings
            if think_context:
                content = (
                    f"{content}\n\n"
                    f"## Data Inspection Findings\n{think_context}\n\n"
                    f"Now create the visualization using generate_mpl_script.\n"
                    f"Use the data labels and column info from the findings above."
                )
        else:
            self._event_bus.emit(
                DEBUG, agent=self.agent_id,
                msg="[MplViz] Skipping think phase (style/manage request)",
            )

        # Execute phase via standard SubAgent._handle_request
        # Fresh loop guard for the execute phase
        self._guard = LoopGuard(
            max_total_calls=get_limit("sub_agent.max_total_calls"),
            max_iterations=get_limit("sub_agent.max_iterations"),
        )
        response = self._llm_send(content)
        result = self._process_response(response, msg)
        self._deliver_result(msg, result)
        self._run_deferred_reviews(msg)

    def _run_think_phase(self, user_request: str) -> str:
        """Inspect data before visualization (ephemeral chat, not the main session)."""
        self._event_bus.emit(
            DEBUG, agent=self.agent_id,
            msg="[MplViz] Think phase: inspecting data...",
        )

        think_prompt = build_viz_think_prompt()
        chat = self.adapter.create_chat(
            model=get_active_model(self.model_name),
            system_prompt=think_prompt,
            tools=self._think_tool_schemas,
            thinking="high",
        )

        response = send_with_timeout(
            chat=chat,
            message=user_request,
            timeout_pool=self._timeout_pool,
            cancel_event=self._cancel_event,
            retry_timeout=180,
            agent_name=f"{self.agent_id}/Think",
            logger=get_logger(),
        )
        self._track_usage(response)

        response = run_tool_loop(
            chat=chat,
            response=response,
            tool_executor=self.tool_executor,
            agent_name=f"{self.agent_id}/Think",
            max_total_calls=get_limit("think.max_total_calls"),
            max_iterations=get_limit("think.max_iterations"),
            track_usage=self._track_usage,
            cancel_event=self._cancel_event,
            send_fn=lambda msg: send_with_timeout(
                chat=chat, message=msg,
                timeout_pool=self._timeout_pool,
                cancel_event=self._cancel_event,
                retry_timeout=180,
                agent_name=f"{self.agent_id}/Think",
                logger=get_logger(),
            ),
            adapter=self.adapter,
        )

        text = extract_text_from_response(response)
        self._event_bus.emit(
            PROGRESS, agent=self.agent_id,
            msg="[MplViz] Think phase complete",
        )
        return text or ""
