"""
Visualization sub-agent with optional think phase.

Both entry points (process_request and execute_task) run a think→execute
pattern for plot-creation requests: inspect data shapes, types, units, and
NaN counts before constructing render_plotly_json calls.  Style/manage
requests skip the think phase to avoid wasting tokens.

Owns all visualization through three tools:
- render_plotly_json — create/update plots via Plotly figure JSON with data_label placeholders
- manage_plot — export, reset, zoom, get state, add/remove traces
- list_fetched_data — discover available data in memory
"""

import re

from .llm import LLMAdapter, FunctionSchema
from .base_agent import BaseSubAgent
from .tasks import Task, TaskStatus
from .tools import get_tool_schemas
from .tool_loop import run_tool_loop, extract_text_from_response
from .event_bus import DEBUG, PROGRESS
from .model_fallback import get_active_model
from knowledge.prompt_builder import (
    build_visualization_prompt,
    build_visualization_think_prompt,
)

# Visualization agent gets its own tool category + list_fetched_data from data_ops
# render_plotly_json and manage_plot are exposed; plot_data and style_plot are
# excluded (legacy primitives superseded by render_plotly_json).
VIZ_TOOL_CATEGORIES = ["visualization"]
VIZ_EXTRA_TOOLS = ["list_fetched_data", "manage_plot", "review_memory"]

# Think phase: data inspection (custom_operation available for spectrograms etc.)
VIZ_THINK_EXTRA_TOOLS = ["list_fetched_data", "describe_data", "preview_data", "custom_operation"]

# Keywords for the skip heuristic
_STYLE_MANAGE_KEYWORDS = {
    "title", "zoom", "log scale", "linear scale", "color", "colour",
    "font", "legend", "annotation", "vline", "vrect",
    "bigger", "smaller", "resize", "canvas", "width", "height",
    "theme", "dark", "export", "reset", "get_state",
    "remove trace", "set time", "time range", "dash", "line style",
}

_PLOT_KEYWORDS = {
    "plot", "show", "display", "visualize", "create", "draw",
    "compare", "panel", "spectrogram", "overlay", "side by side",
}


def _extract_labels_from_instruction(instruction: str) -> list[str]:
    """Extract data labels from a task instruction that has store contents appended.

    The orchestrator appends lines like "  - AC_H0_MFI.Magnitude (37800 pts)"
    to the instruction. This extracts the label portion.
    """
    labels = []
    for match in re.finditer(r"^\s+-\s+(\S+)\s+\(", instruction, re.MULTILINE):
        labels.append(match.group(1))
    return labels


class VisualizationAgent(BaseSubAgent):
    """An LLM session specialized for visualization.

    Uses three tools: render_plotly_json (create/update plots via Plotly
    figure JSON with data_label placeholders), manage_plot (export, reset,
    zoom, add/remove traces), and list_fetched_data (discover available data).

    For plot-creation requests (both process_request and execute_task),
    runs a think phase to inspect data before the execute phase.
    """

    # Disable forced tool calling: render_plotly_json requires complex nested
    # JSON that the LLM emits as empty {} under forced-calling mode (mode="ANY").
    # The task prompt already provides explicit instructions and examples.
    _force_tool_call_in_tasks = False

    def __init__(
        self,
        adapter: LLMAdapter,
        model_name: str,
        tool_executor,
        verbose: bool = False,
        gui_mode: bool = False,
        cancel_event=None,
        event_bus=None,
    ):
        self.gui_mode = gui_mode
        super().__init__(
            adapter=adapter,
            model_name=model_name,
            tool_executor=tool_executor,
            verbose=verbose,
            agent_name="Visualization Agent",
            system_prompt=build_visualization_prompt(gui_mode=gui_mode),
            tool_categories=VIZ_TOOL_CATEGORIES,
            extra_tool_names=VIZ_EXTRA_TOOLS,
            cancel_event=cancel_event,
            event_bus=event_bus,
            llm_retry_timeout=180,
        )

        # Build think-phase tool schemas (data inspection only)
        self._think_tool_schemas: list[FunctionSchema] = []
        for tool_schema in get_tool_schemas(
            categories=[],
            extra_names=VIZ_THINK_EXTRA_TOOLS,
        ):
            self._think_tool_schemas.append(FunctionSchema(
                name=tool_schema["name"],
                description=tool_schema["description"],
                parameters=tool_schema["parameters"],
            ))

    def _needs_think_phase(self, request: str) -> bool:
        """Decide whether a request needs data inspection before plotting.

        Plot-creation requests benefit from inspecting data shapes and types.
        Style/manage requests (title changes, zoom, export) do not.

        Returns True if ambiguous (conservative — better to over-inspect).
        """
        req_lower = request.lower()
        has_plot_signal = any(kw in req_lower for kw in _PLOT_KEYWORDS)
        has_style_signal = any(kw in req_lower for kw in _STYLE_MANAGE_KEYWORDS)
        # If only style signals, skip. If any plot signal (or ambiguous), run think.
        if has_style_signal and not has_plot_signal:
            return False
        return True

    def _run_think_phase(self, user_request: str) -> str:
        """Inspect data in memory before creating a visualization.

        Creates an ephemeral chat session with data inspection tools.
        Runs a tool-calling loop to explore data shapes, types, and values,
        then returns a text summary of findings.

        Args:
            user_request: The user's visualization request.

        Returns:
            Text summary of data inspection findings (shapes, panel layout
            recommendations, sizing hints).
        """
        think_prompt = build_visualization_think_prompt()

        self._event_bus.emit(DEBUG, agent=self.agent_name, msg="[Viz] Think phase: inspecting data...")

        chat = self.adapter.create_chat(
            model=get_active_model(self.model_name),
            system_prompt=think_prompt,
            tools=self._think_tool_schemas,
            thinking="high",
        )

        self._last_tool_context = "viz_think_initial"
        response = self._send_with_timeout(chat, user_request)
        self._track_usage(response)

        response = run_tool_loop(
            chat=chat,
            response=response,
            tool_executor=self.tool_executor,
            agent_name="Viz/Think",
            max_total_calls=10,
            max_iterations=4,
            track_usage=self._track_usage,
            cancel_event=self._cancel_event,
            send_fn=lambda msg: self._send_with_timeout(chat, msg),
            adapter=self.adapter,
        )

        text = extract_text_from_response(response)
        if self.verbose and text:
            self._event_bus.emit(DEBUG, agent=self.agent_name, msg=f"[Viz] Think result: {text[:500]}")

        self._event_bus.emit(PROGRESS, agent=self.agent_name, msg="[Viz] Think phase complete")
        return text or ""

    def _check_think_rejection(self, think_context: str) -> str | None:
        """Check if the think phase output contains a REJECT signal.

        Returns the full rejection text if found, None otherwise.
        """
        if not think_context:
            return None
        for line in think_context.split("\n"):
            if line.strip().startswith("**REJECT"):
                idx = think_context.index(line.strip())
                return think_context[idx:].strip()
        return None

    def _build_enriched_message(
        self, user_message: str, think_context: str
    ) -> str:
        """Combine the user request with think-phase findings."""
        if not think_context:
            return user_message
        return (
            f"{user_message}\n\n"
            f"## Data Inspection Findings\n{think_context}\n\n"
            f"Now create the visualization using render_plotly_json.\n"
            f"Follow the recommended time range from the findings above for x-axis range."
        )

    def process_request(self, user_message: str) -> dict:
        """Conditionally run think→execute for plot-creation requests.

        Style/manage requests skip the think phase and go straight to
        the execute phase (standard BaseSubAgent.process_request).

        If the execute phase fails with render_plotly_json errors, retries
        once: re-runs the think phase with error context so it can produce
        corrected recommendations, then executes again.

        Args:
            user_message: The user's visualization request.

        Returns:
            Dict with text, failed, errors (same as BaseSubAgent.process_request).
        """
        if not self._needs_think_phase(user_message):
            self._event_bus.emit(DEBUG, agent=self.agent_name, msg="[Viz] Skipping think phase (style/manage request)")
            return super().process_request(user_message)

        # First attempt: think → execute
        think_context = self._run_think_phase(user_message)
        rejection = self._check_think_rejection(think_context)
        if rejection:
            self._event_bus.emit(DEBUG, agent=self.agent_name, msg="[Viz] Think phase rejected request")
            return {"text": rejection, "failed": True, "errors": [rejection]}
        enriched = self._build_enriched_message(user_message, think_context)
        result = super().process_request(enriched)

        # Check if retry is warranted (render errors only) — but skip if
        # the overall result succeeded (LLM self-corrected within the attempt)
        render_errors = [
            e for e in result.get("errors", []) if "render_plotly_json" in e
        ]
        if not render_errors or not result.get("failed", False):
            return result

        # Retry: re-run think phase with error context, then execute again
        self._event_bus.emit(DEBUG, agent=self.agent_name, msg="[Viz] Render failed, retrying with error feedback")
        error_feedback = "\n".join(f"- {e}" for e in render_errors)
        retry_request = (
            f"{user_message}\n\n"
            f"## Previous Attempt Failed\n"
            f"Errors:\n{error_feedback}\n\n"
            f"Re-inspect the data and provide corrected recommendations."
        )
        think_context = self._run_think_phase(retry_request)
        enriched = self._build_enriched_message(user_message, think_context)
        return super().process_request(enriched)

    def execute_task(self, task: Task) -> str:
        """Execute a visualization task with think→execute pattern.

        For plot-creation tasks, runs the think phase to inspect data
        before delegating to the base execute_task loop.  Style/manage
        tasks skip the think phase.

        On render_plotly_json errors, retries once with error feedback
        (matching process_request behavior).
        """
        if not self._needs_think_phase(task.instruction):
            self._event_bus.emit(DEBUG, agent=self.agent_name, msg="[Viz] Skipping think phase for task (style/manage)")
            return super().execute_task(task)

        # Think phase: inspect data
        think_context = self._run_think_phase(task.instruction)
        rejection = self._check_think_rejection(think_context)
        if rejection:
            self._event_bus.emit(DEBUG, agent=self.agent_name, msg="[Viz] Think phase rejected task")
            task.status = TaskStatus.FAILED
            task.error = rejection
            task.result = rejection
            return rejection
        task.instruction = self._build_enriched_message(
            task.instruction, think_context
        )

        # Execute phase (via base class)
        result = super().execute_task(task)

        # Check for render errors that warrant a retry — but skip if the LLM
        # already self-corrected (i.e. a later render succeeded)
        render_errors = [
            tr.get("message", str(tr))
            for tr in task.tool_results
            if tr.get("tool") == "render_plotly_json"
            and tr.get("status") == "error"
        ]
        render_ok = any(
            tr.get("tool") == "render_plotly_json" and tr.get("status") == "success"
            for tr in task.tool_results
        )
        if not render_errors or render_ok:
            return result

        # Retry: re-run think phase with error context, then execute again
        self._event_bus.emit(DEBUG, agent=self.agent_name, msg="[Viz] Task render failed, retrying with error feedback")
        error_feedback = "\n".join(f"- {e}" for e in render_errors)
        retry_request = (
            f"{task.instruction}\n\n"
            f"## Previous Attempt Failed\n"
            f"Errors:\n{error_feedback}\n\n"
            f"Re-inspect the data and provide corrected recommendations."
        )
        think_context = self._run_think_phase(retry_request)
        task.instruction = self._build_enriched_message(
            task.instruction, think_context
        )

        # super().execute_task() resets tool_calls/tool_results/status at top
        return super().execute_task(task)

    def _get_task_prompt(self, task: Task) -> str:
        """Build an explicit task prompt with concrete label values.

        Extracts actual data labels from the instruction (injected by
        _execute_plan_task) and constructs the exact render_plotly_json call
        so Gemini Flash sees the precise command to execute.

        When a current figure_json is embedded in the instruction (canvas
        exists), instructs the LLM to modify it instead of starting fresh.

        Note: Export tasks are handled directly by the orchestrator and
        never reach this method.
        """
        labels = _extract_labels_from_instruction(task.instruction)
        has_canvas = "Current figure_json" in task.instruction

        no_labels = not labels

        if has_canvas:
            # Canvas exists — instruct LLM to modify the provided figure_json
            task_prompt = (
                f"Execute this task: {task.instruction}\n\n"
                "A figure_json is provided above. Modify it to fulfil the request "
                "and pass the full modified JSON to render_plotly_json.\n"
                "If modification is too complex, call manage_plot(action=\"reset\") "
                "first, then create a new figure_json from scratch.\n\n"
                "RULES:\n"
                "- Prefer modifying the existing figure_json over rebuilding.\n"
                "- Use manage_plot for export, reset, or other structural operations if needed."
            )
        else:
            # No canvas — build from scratch
            if labels:
                traces_example = ", ".join(
                    f'{{"type": "scatter", "data_label": "{lbl}"}}'
                    for lbl in labels
                )
                first_call = (
                    f'render_plotly_json(figure_json={{"data": [{traces_example}], '
                    f'"layout": {{}}}})'
                )
            else:
                first_call = "render_plotly_json with the appropriate labels"

            task_prompt = (
                f"Execute this task: {task.instruction}\n\n"
                f"Your FIRST call must be: {first_call}\n\n"
                "RULES:\n"
                "- Call render_plotly_json with the labels shown above.\n"
                "- Use manage_plot for export, reset, zoom, or trace operations if needed."
            )

            if no_labels:
                task_prompt += "\n\nNote: Labels were not pre-extracted. Call list_fetched_data first to discover available labels.\n"

        return task_prompt
