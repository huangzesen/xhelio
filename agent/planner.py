"""
Planning logic for multi-step task handling.

This module provides:
- The orchestrator LLM decides when to invoke planning via the request_planning tool
- PlannerAgent: Chat-based planner with plan-execute-replan loop
- format_plan_for_display(): Human-readable plan rendering
"""

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .llm import LLMAdapter, LLMResponse, FunctionSchema
from .event_bus import get_event_bus, DEBUG, PROGRESS, PLAN_CREATED, LLM_CALL
from .llm_utils import _LLM_RETRY_TIMEOUT, send_with_timeout, track_llm_usage
from .token_counter import count_tokens, count_tool_tokens
from .tasks import Task, TaskPlan, create_task, create_plan
from .tools import get_function_schemas
from .tool_loop import run_tool_loop, extract_text_from_response
from .agent_registry import PLANNER_TOOLS
from .turn_limits import get_limit
from knowledge.prompt_builder import build_planner_agent_prompt

from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context_tracker import ContextTracker


# Schema for the produce_plan tool — forces the LLM to return a structured
# plan via tool calling instead of raw JSON text.
PRODUCE_PLAN_SCHEMA = FunctionSchema(
    name="produce_plan",
    description=(
        "Submit the complete task plan. Call this tool when you have "
        "finished researching and are ready to emit your plan. "
        "Include ALL tasks needed to fulfill the request — fetch, "
        "compute, AND visualization — in a single call."
    ),
    parameters={
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of your planning decision.",
            },
            "tasks": {
                "type": "array",
                "description": "All tasks needed to fulfill the request.",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Brief human-readable summary.",
                        },
                        "instruction": {
                            "type": "string",
                            "description": "Detailed instruction for executing the task.",
                        },
                        "mission": {
                            "type": "string",
                            "description": "Mission ID, '__visualization__', '__data_ops__', '__data_extraction__', or null.",
                        },
                        "candidate_datasets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Recommended dataset IDs from browse_datasets research.",
                        },
                    },
                    "required": ["description", "instruction", "mission"],
                },
            },
            "summary": {
                "type": "string",
                "description": "Brief user-facing summary of what the plan will accomplish.",
            },
            "time_range_validated": {
                "type": "boolean",
                "description": (
                    "Set to true ONLY if you verified that every candidate dataset's "
                    "stop_date is AFTER the requested time range start date. "
                    "If false, you MUST adjust the time range before submitting."
                ),
            },
        },
        "required": ["reasoning", "tasks", "summary", "time_range_validated"],
    },
)


class PlannerAgent:
    """Chat-based planner that decomposes complex requests into task batches.

    Uses a single persistent session with tools. The LLM researches context
    (list_missions, web_search, list_fetched_data, etc.) and then responds
    with a JSON plan. If JSON parsing fails, the error is fed back to the
    same session for retry (up to 2 attempts).
    """

    def __init__(
        self,
        adapter: LLMAdapter,
        model_name: str,
        tool_executor=None,
        verbose: bool = False,
        cancel_event=None,
        event_bus=None,
        ctx_tracker: "ContextTracker | None" = None,
        session_id: str | None = None,
    ):
        self.adapter = adapter
        self.model_name = model_name
        self.tool_executor = tool_executor
        self.verbose = verbose
        self._cancel_event = cancel_event
        self._ctx_tracker = ctx_tracker
        self._session_id = session_id
        self._orchestrator_inbox = None  # Set by orchestrator when creating planner
        self._chat = None
        self._current_system_prompt = ""
        self._token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cached_tokens": 0,
        }
        self._api_calls = 0
        self._last_tool_context = "send_message"

        # Token decomposition
        self._current_system_tokens = 0
        self._current_tools_tokens = 0
        self._latest_input_tokens = 0
        self._timeout_pool = ThreadPoolExecutor(max_workers=1)
        self._event_bus = event_bus or get_event_bus()

        # Build FunctionSchema list when tools are available
        self._tool_schemas: list[FunctionSchema] = []
        if self.tool_executor is not None:
            self._tool_schemas = get_function_schemas(names=PLANNER_TOOLS)
        # Always include produce_plan — it's a planner-internal tool
        self._tool_schemas.append(PRODUCE_PLAN_SCHEMA)

    def _make_plan_executor(self):
        """Wrap the external tool executor to handle produce_plan internally."""

        def executor(tool_name, tool_args):
            if tool_name == "produce_plan":
                return {"status": "success", "message": "Plan submitted."}
            return self.tool_executor(tool_name, tool_args)

        return executor

    def save_plan_to_file(self, plan: dict, session_id: str) -> str:
        """Save plan to filesystem as JSON.

        Args:
            plan: The plan dict from produce_plan.
            session_id: Current session ID for filename.

        Returns:
            Path to the saved plan file.
        """
        import os
        from pathlib import Path

        # Save to session-specific file
        plan_dir = Path(os.environ.get("XHELIO_DATA_DIR", "/tmp/xhelio")) / "plans"
        plan_dir.mkdir(parents=True, exist_ok=True)

        plan_file = plan_dir / f"{session_id}_plan.json"

        with open(plan_file, "w") as f:
            json.dump(plan, f, indent=2)

        return str(plan_file)

    def send_to_orchestrator(self, content: str, priority: int = 1):
        """Send message to orchestrator's inbox.

        Args:
            content: Message text explaining what was done and recommendations.
            priority: Message priority (0=user, 1=subagent).
        """
        if self._orchestrator_inbox:
            from .sub_agent import _make_message
            msg = _make_message(
                "subagent_result",
                sender=self.__class__.__name__,
                content=content,
            )
            self._orchestrator_inbox.put((priority, msg.timestamp, msg))

    @staticmethod
    def _extract_plan_from_collected(collected: dict) -> Optional[dict]:
        """Extract plan dict from collected produce_plan tool results, if any."""
        entries = collected.get("produce_plan")
        if not entries:
            return None
        # Use the last produce_plan call's args
        plan = dict(entries[-1]["args"]) if entries[-1].get("args") else {}
        # Normalize mission "null"/"none" strings to None
        for task in plan.get("tasks", []):
            mission = task.get("mission")
            if isinstance(mission, str) and mission.lower() in ("null", "none", ""):
                task["mission"] = None
        # Warn if planner did not validate time ranges
        if not plan.get("time_range_validated"):
            get_event_bus().emit(
                DEBUG,
                agent="PlannerAgent",
                level="warning",
                msg="[PlannerAgent] Plan submitted with time_range_validated=false — "
                "time coverage may not have been checked",
            )
        return plan

    def _on_reset(self, chat, failed_message):
        """Rollback: new chat with last assistant turn dropped."""
        history = chat.get_history()
        while history and history[-1].get("role") == "assistant":
            history.pop()

        get_event_bus().emit(
            LLM_CALL,
            agent="PlannerAgent",
            level="warning",
            msg=f"[PlannerAgent] Session rollback — new chat ({len(history)} msgs kept)",
        )

        new_chat = self.adapter.create_chat(
            model=self.model_name,
            system_prompt=self._current_system_prompt,
            tools=self._tool_schemas,
            thinking="high",
            history=history,
        )
        self._chat = new_chat
        return (
            new_chat,
            "The previous response was lost due to a server error. Please try again.",
        )

    def _send_with_timeout(self, chat, message) -> LLMResponse:
        """Send a message to the LLM with periodic warnings and retry on timeout."""
        return send_with_timeout(
            chat=chat,
            message=message,
            timeout_pool=self._timeout_pool,
            cancel_event=self._cancel_event,
            retry_timeout=_LLM_RETRY_TIMEOUT,
            agent_name="PlannerAgent",
            logger=None,
            on_reset=self._on_reset,
        )

    def _track_usage(self, response: LLMResponse):
        """Accumulate token usage from an LLMResponse."""
        token_state = {
            "input": self._token_usage["input_tokens"],
            "output": self._token_usage["output_tokens"],
            "thinking": self._token_usage["thinking_tokens"],
            "cached": self._token_usage["cached_tokens"],
            "api_calls": self._api_calls,
        }
        track_llm_usage(
            response=response,
            token_state=token_state,
            agent_name="PlannerAgent",
            last_tool_context=self._last_tool_context,
            system_tokens=self._current_system_tokens,
            tools_tokens=self._current_tools_tokens,
        )
        self._token_usage["input_tokens"] = token_state["input"]
        self._token_usage["output_tokens"] = token_state["output"]
        self._token_usage["thinking_tokens"] = token_state["thinking"]
        self._token_usage["cached_tokens"] = token_state["cached"]
        self._api_calls = token_state["api_calls"]
        self._latest_input_tokens = response.usage.input_tokens

    def _parse_or_retry(
        self, chat, response: LLMResponse, max_retries: int = 2
    ) -> Optional[dict]:
        """Parse JSON from response text with specific error feedback, retrying on failure.

        Validates the response is raw JSON (no fences, no prose). If parsing
        fails, sends a targeted error message back to the same session so the
        LLM knows exactly what to fix. Caps retries at max_retries attempts.
        """
        for attempt in range(max_retries + 1):
            text = response.text
            if not text:
                error_reason = "Your response was empty. Respond with the plan JSON."
            else:
                stripped = text.strip()
                # Try direct JSON parse
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError as e:
                    # Give a specific error depending on the shape of the response
                    if not stripped.startswith("{"):
                        error_reason = (
                            "Your response must be raw JSON starting with `{`. "
                            "Do not include prose, markdown fences, or any text "
                            "before or after the JSON object."
                        )
                    else:
                        error_reason = (
                            f"Your response starts with `{{` but is not valid JSON: {e}"
                        )
                    data = None
                else:
                    # JSON parsed — validate required fields
                    missing = [
                        f
                        for f in (
                            "reasoning",
                            "tasks",
                            "summary",
                            "time_range_validated",
                        )
                        if f not in data
                    ]
                    if missing:
                        error_reason = f"JSON is valid but missing required field(s): {', '.join(repr(f) for f in missing)}"
                        data = None
                    else:
                        # Normalize mission "null"/"none" strings to None
                        for task_data in data.get("tasks", []):
                            mission = task_data.get("mission")
                            if isinstance(mission, str) and mission.lower() in (
                                "null",
                                "none",
                                "",
                            ):
                                task_data["mission"] = None
                        return data

            # Parse/validation failed — retry or give up
            if attempt < max_retries:
                from .truncation import trunc

                self._event_bus.emit(
                    DEBUG,
                    agent="PlannerAgent",
                    level="warning",
                    msg=f"[PlannerAgent] JSON parse failed (attempt {attempt + 1}/{max_retries + 1}): {error_reason}",
                )
                if text:
                    self._event_bus.emit(
                        DEBUG,
                        agent="PlannerAgent",
                        msg=f"[PlannerAgent] Raw response: {trunc(text.strip(), 'inline.debug')}",
                    )
                error_msg = (
                    f"{error_reason}\n\n"
                    f"Respond with ONLY the plan JSON object. Required fields: "
                    f"reasoning, tasks, summary, time_range_validated. No markdown fences, no prose."
                )
                self._last_tool_context = f"json_retry_{attempt + 1}"
                response = self._send_with_timeout(chat, error_msg)
                self._track_usage(response)
            else:
                self._event_bus.emit(
                    DEBUG,
                    agent="PlannerAgent",
                    level="warning",
                    msg=f"[PlannerAgent] JSON parse failed after {max_retries + 1} attempts: {error_reason}",
                )

        return None

    def start_planning(self, user_request: str) -> Optional[dict]:
        """Begin planning by sending the user request to a fresh chat.

        Creates a single session with tools. The LLM can call research tools
        (list_missions, web_search, list_fetched_data, etc.) during its
        tool loop, then responds with a JSON plan.

        Args:
            user_request: The user's original request.

        Returns:
            Dict with {status, reasoning, tasks, summary} or None on failure.
        """
        try:
            system_prompt = build_planner_agent_prompt()
            self._current_system_prompt = system_prompt

            self._current_system_tokens = count_tokens(system_prompt)
            self._current_tools_tokens = count_tool_tokens(self._tool_schemas)

            self._chat = self.adapter.create_chat(
                model=self.model_name,
                system_prompt=system_prompt,
                tools=self._tool_schemas if self._tool_schemas else None,
                thinking="high",
            )

            self._event_bus.emit(
                PROGRESS,
                agent="PlannerAgent",
                msg="[Planning] Researching and decomposing request...",
            )

            # Prepend current time
            current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            user_request = f"[Current time: {current_time}]\n\n{user_request}"

            self._last_tool_context = "planning_initial"
            response = self._send_with_timeout(self._chat, user_request)
            self._track_usage(response)

            # Run tool loop — LLM may call research tools before producing plan
            collected = {}
            if self._tool_schemas:
                executor = (
                    self._make_plan_executor()
                    if self.tool_executor
                    else lambda n, a: (
                        {"status": "success", "message": "Plan submitted."}
                        if n == "produce_plan"
                        else {"status": "error", "message": f"Unknown tool: {n}"}
                    )
                )
                response = run_tool_loop(
                    chat=self._chat,
                    response=response,
                    tool_executor=executor,
                    adapter=self.adapter,
                    agent_name="PlannerAgent",
                    max_total_calls=get_limit("think.max_total_calls"),
                    max_iterations=get_limit("think.max_iterations"),
                    track_usage=self._track_usage,
                    collect_tool_results=collected,
                    cancel_event=self._cancel_event,
                    send_fn=lambda msg: self._send_with_timeout(self._chat, msg),
                    terminal_tools={"produce_plan"},
                )

            # Check if produce_plan was called during the tool loop
            result = self._extract_plan_from_collected(collected)
            if result is None:
                # Fallback: try parsing text as JSON (backward compat)
                result = self._parse_or_retry(self._chat, response)
            if result and self.verbose:
                self._event_bus.emit(
                    DEBUG,
                    agent="PlannerAgent",
                    msg=f"[PlannerAgent] Round 1: {len(result.get('tasks', []))} tasks",
                )
                self._log_plan_details(result, round_num=1)

            # Save plan to filesystem and notify orchestrator
            if result:
                plan_file = self.save_plan_to_file(result, self._session_id or "default")
                self.send_to_orchestrator(
                    f"Planning complete. Plan saved to {plan_file}. "
                    f"Use the plan_check tool to review and execute tasks."
                )

            # Return None - orchestrator will check inbox for message
            return None

        except Exception as e:
            self._event_bus.emit(
                DEBUG,
                agent="PlannerAgent",
                level="warning",
                msg=f"[PlannerAgent] Error in start_planning: {e}",
            )
            # Notify orchestrator of failure
            self.send_to_orchestrator(f"Planning failed: {e}")
            return None

    def _log_plan_details(self, result: dict, round_num: int) -> None:
        """Log full task details (instructions, candidates, reasoning) for debugging."""
        if not result:
            return
        lines = [f"[PlannerAgent] === Round {round_num} Plan Details ==="]
        if result.get("reasoning"):
            lines.append(f"  Reasoning: {result['reasoning']}")
        for i, task in enumerate(result.get("tasks", []), 1):
            lines.append(f"  Task {i}:")
            lines.append(f"    description: {task.get('description', '?')}")
            lines.append(f"    mission: {task.get('mission', 'null')}")
            lines.append(f"    instruction: {task.get('instruction', '?')}")
            candidates = task.get("candidate_datasets")
            if candidates:
                lines.append(f"    candidate_datasets: {candidates}")
        if result.get("summary"):
            lines.append(f"  Summary: {result['summary']}")
        self._event_bus.emit(DEBUG, agent="PlannerAgent", msg="\n".join(lines))

    def get_token_usage(self) -> dict:
        """Return accumulated token usage."""
        return {
            "input_tokens": self._token_usage["input_tokens"],
            "output_tokens": self._token_usage["output_tokens"],
            "thinking_tokens": self._token_usage["thinking_tokens"],
            "cached_tokens": self._token_usage["cached_tokens"],
            "api_calls": self._api_calls,
            "ctx_system_tokens": self._current_system_tokens,
            "ctx_tools_tokens": self._current_tools_tokens,
            "ctx_history_tokens": max(
                0,
                self._latest_input_tokens
                - self._current_system_tokens
                - self._current_tools_tokens,
            ),
            "ctx_total_tokens": self._latest_input_tokens,
        }

    def reset(self):
        """Reset the chat session."""
        self._chat = None


def format_plan_for_display(plan: "TaskPlan | dict") -> str:
    """Format a plan for display to the user.

    Accepts either a ``TaskPlan`` object or a plain ``dict`` (from the
    research-only planner).  Groups tasks by round when round > 0.

    Args:
        plan: The plan to format

    Returns:
        Human-readable string representation
    """
    # Plain dict from the research-only planner
    if isinstance(plan, dict):
        raw_tasks = plan.get("tasks", [])
        lines = [f"Plan: {len(raw_tasks)} steps"]
        lines.append("-" * 40)
        summary = plan.get("summary", "")
        if summary:
            lines.append(f"Summary: {summary}")
        for i, task in enumerate(raw_tasks, 1):
            mission = task.get("mission", "")
            mission_tag = f" [{mission}]" if mission else ""
            lines.append(f"  {i}. [o]{mission_tag} {task.get('description', '')}")
        lines.append("-" * 40)
        lines.append(f"Progress: 0/{len(raw_tasks)} completed")
        return "\n".join(lines)

    lines = [f"Plan: {len(plan.tasks)} steps"]
    lines.append("-" * 40)

    # Check if any task has a non-zero round
    has_rounds = any(t.round > 0 for t in plan.tasks)

    if has_rounds:
        # Group by round
        rounds: dict[int, list[tuple[int, Task]]] = {}
        for i, task in enumerate(plan.tasks):
            rounds.setdefault(task.round, []).append((i, task))

        for round_num in sorted(rounds.keys()):
            if round_num > 0:
                lines.append(f"  Round {round_num}:")
            for i, task in rounds[round_num]:
                status_icon = {
                    "pending": "o",
                    "in_progress": "*",
                    "completed": "+",
                    "failed": "x",
                    "skipped": "-",
                }.get(task.status.value, "~")

                mission_tag = f" [{task.mission}]" if task.mission else ""
                lines.append(
                    f"  {i + 1}. [{status_icon}]{mission_tag} {task.description}"
                )

                if task.status.value == "failed" and task.error:
                    lines.append(f"       Error: {task.error}")
    else:
        for i, task in enumerate(plan.tasks):
            status_icon = {
                "pending": "o",
                "in_progress": "*",
                "completed": "+",
                "failed": "x",
                "skipped": "-",
            }.get(task.status.value, "~")

            mission_tag = f" [{task.mission}]" if task.mission else ""
            lines.append(f"  {i + 1}. [{status_icon}]{mission_tag} {task.description}")

            if task.status.value == "failed" and task.error:
                lines.append(f"       Error: {task.error}")

    lines.append("-" * 40)
    lines.append(f"Progress: {plan.progress_summary()}")

    return "\n".join(lines)


def format_plan_structured(plan: "TaskPlan | dict") -> dict:
    """Format a plan as structured JSON for the frontend.

    Accepts either a ``TaskPlan`` object (from the old plan-execute loop) or a
    plain ``dict`` (from the research-only planner).

    Returns a dict with total_steps, progress, and steps (grouped by round).
    Each step has title, details, status, mission, round, error, and
    candidate_datasets.
    """
    # Plain dict from the research-only planner
    if isinstance(plan, dict):
        raw_tasks = plan.get("tasks", [])
        steps = []
        for task in raw_tasks:
            step: dict = {
                "title": task.get("description", ""),
                "details": task.get("instruction", ""),
                "status": "pending",
                "mission": task.get("mission"),
                "round": 0,
            }
            ds = task.get("candidate_datasets")
            if ds:
                step["candidate_datasets"] = ds
            steps.append(step)
        return {
            "total_steps": len(raw_tasks),
            "progress": f"0/{len(raw_tasks)} completed",
            "summary": plan.get("summary", ""),
            "reasoning": plan.get("reasoning", ""),
            "steps": steps,
        }

    # TaskPlan object (legacy path)
    steps = []
    for task in plan.tasks:
        step = {
            "title": task.description,
            "details": task.instruction,
            "status": task.status.value,
            "mission": task.mission,
            "round": task.round,
        }
        if task.error:
            step["error"] = task.error
        if task.candidate_datasets:
            step["candidate_datasets"] = task.candidate_datasets
        steps.append(step)

    return {
        "total_steps": len(plan.tasks),
        "progress": plan.progress_summary(),
        "steps": steps,
    }
