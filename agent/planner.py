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
from .event_bus import get_event_bus, DEBUG, PROGRESS, PLAN_CREATED
from .model_fallback import get_active_model
from .llm_utils import _LLM_RETRY_TIMEOUT, send_with_timeout, track_llm_usage
from .token_counter import count_tokens, count_tool_tokens
from .tasks import Task, TaskPlan, create_task, create_plan
from .tools import get_function_schemas
from .tool_loop import run_tool_loop, extract_text_from_response
from .agent_registry import PLANNER_TOOLS
from .turn_limits import get_limit
from knowledge.prompt_builder import build_planner_agent_prompt, build_planner_think_prompt

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .context_tracker import ContextTracker



# JSON schema for PlannerAgent's structured output
PLANNER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["continue", "done"],
        },
        "reasoning": {
            "type": "string",
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "instruction": {"type": "string"},
                    "mission": {"type": "string"},
                    "candidate_datasets": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["description", "instruction"],
            },
        },
        "summary": {
            "type": "string",
        },
    },
    "required": ["status", "reasoning", "tasks"],
}


class PlannerAgent:
    """Chat-based planner that decomposes complex requests into task batches.

    Uses a two-phase approach when tools are available:
    1. **Think phase**: A tool-calling session verifies dataset IDs and
       parameter names via discovery tools (search_datasets, list_parameters, etc.).
    2. **Planning phase**: A JSON-schema-enforced session produces the task plan,
       enriched with the think phase context from phase 1.

    Without a tool_executor, skips the think phase.
    The planning phase always uses JSON schema enforcement for guaranteed output.
    """

    def __init__(self, adapter: LLMAdapter, model_name: str,
                 tool_executor=None, verbose: bool = False,
                 cancel_event=None,
                 event_bus=None,
                 ctx_tracker: 'ContextTracker | None' = None):
        self.adapter = adapter
        self.model_name = model_name
        self.tool_executor = tool_executor
        self.verbose = verbose
        self._cancel_event = cancel_event
        self._ctx_tracker = ctx_tracker
        self._chat = None
        self._token_usage = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "cached_tokens": 0}
        self._api_calls = 0
        self._last_tool_context = "send_message"

        # Token decomposition — updated per-phase
        self._current_system_tokens = 0
        self._current_tools_tokens = 0
        self._latest_input_tokens = 0
        self._timeout_pool = ThreadPoolExecutor(max_workers=1)
        self._event_bus = event_bus or get_event_bus()

        # Build FunctionSchema list when tools are available
        self._tool_schemas: list[FunctionSchema] = []
        if self.tool_executor is not None:
            self._tool_schemas = get_function_schemas(names=PLANNER_TOOLS)

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

    def _parse_response(self, response: LLMResponse) -> Optional[dict]:
        """Parse JSON response from the LLM, normalizing mission fields.

        The planning phase uses JSON schema enforcement, but some models
        (e.g. via OpenRouter) may wrap JSON in markdown code fences or
        return non-JSON text.  We strip common wrappers before parsing.
        """
        try:
            text = response.text
            if not text:
                return None

            # Strip markdown code fences (```json ... ``` or ``` ... ```)
            stripped = text.strip()
            if stripped.startswith("```"):
                # Remove opening fence (with optional language tag)
                first_newline = stripped.index("\n")
                stripped = stripped[first_newline + 1:]
                # Remove closing fence
                if stripped.rstrip().endswith("```"):
                    stripped = stripped.rstrip()[:-3].rstrip()
                text = stripped

            data = json.loads(text)

            # Normalize mission "null"/"none" strings to None
            for task_data in data.get("tasks", []):
                mission = task_data.get("mission")
                if isinstance(mission, str) and mission.lower() in ("null", "none", ""):
                    task_data["mission"] = None

            return data
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            self._event_bus.emit(DEBUG, agent="PlannerAgent", level="warning", msg=f"[PlannerAgent] Failed to parse response: {e}")
            if text:
                from .truncation import trunc
                self._event_bus.emit(DEBUG, agent="PlannerAgent", msg=f"[PlannerAgent] Raw response text (first 500 chars): {trunc(text, 'inline.debug')}")
            return None

    def _run_think_phase(self, user_request: str) -> str:
        """Phase 1: Gather high-level research context.

        Creates a one-shot tool-calling chat that researches the user's request
        and returns a text summary of findings (resolved time ranges, mission
        identification, web context, data already in memory). Dataset/parameter
        discovery is left to mission agents.

        Args:
            user_request: The user's original request.

        Returns:
            Text summary of research findings.
        """
        think_prompt = build_planner_think_prompt()

        # Update token decomposition for think phase
        self._current_system_tokens = count_tokens(think_prompt)
        self._current_tools_tokens = count_tool_tokens(self._tool_schemas)

        chat = self.adapter.create_chat(
            model=get_active_model(self.model_name),
            system_prompt=think_prompt,
            tools=self._tool_schemas,
            thinking="high",
        )

        self._event_bus.emit(PROGRESS, agent="PlannerAgent", msg="[Planner] Think phase: researching datasets...")

        self._last_tool_context = "think_initial"
        response = self._send_with_timeout(chat, user_request)
        self._track_usage(response)

        # Collect raw tool results so we can extract list_parameters data
        tool_results = {}
        response = run_tool_loop(
            chat=chat,
            response=response,
            tool_executor=self.tool_executor,
            adapter=self.adapter,
            agent_name="PlannerAgent/Think",
            max_total_calls=get_limit("think.max_total_calls"),
            max_iterations=get_limit("think.max_iterations"),
            track_usage=self._track_usage,
            collect_tool_results=tool_results,
            cancel_event=self._cancel_event,
            send_fn=lambda msg: self._send_with_timeout(chat, msg),
        )

        text = extract_text_from_response(response)
        if self.verbose and text:
            self._event_bus.emit(DEBUG, agent="PlannerAgent", msg=f"[PlannerAgent] Think result: {text}")

        self._event_bus.emit(PROGRESS, agent="PlannerAgent", msg="[Planner] Think phase complete")

        # Show findings in web UI live-log
        if text:
            self._event_bus.emit(
                PROGRESS, agent="PlannerAgent",
                msg=f"[Planner] Think phase findings:\n{text}",
            )

        return text

    def start_planning(self, user_request: str) -> Optional[dict]:
        """Begin planning by sending the user request to a fresh chat.

        When tools are available, runs a two-phase process:
        1. Think phase — calls tools to verify datasets/parameters.
        2. Planning phase — JSON-schema-enforced chat produces the task plan.

        Without tools, goes straight to the planning phase.

        Args:
            user_request: The user's original request.

        Returns:
            Dict with {status, reasoning, tasks, summary} or None on failure.
        """
        try:
            # Phase 1: Think (only when tools are available)
            think_context = ""
            if self._tool_schemas and self.tool_executor:
                think_context = self._run_think_phase(user_request)

            # Phase 2: Planning (always JSON-schema-enforced)
            system_prompt = build_planner_agent_prompt()

            # Update token decomposition for planning phase (no tools)
            self._current_system_tokens = count_tokens(system_prompt)
            self._current_tools_tokens = 0

            self._chat = self.adapter.create_chat(
                model=get_active_model(self.model_name),
                system_prompt=system_prompt,
                json_schema=PLANNER_RESPONSE_SCHEMA,
                thinking="high",
            )

            # Build the planning message with think phase context
            if think_context:
                planning_message = (
                    f"{user_request}\n\n"
                    f"## Research Context\n\n"
                    f"{think_context}"
                )
            else:
                planning_message = user_request

            self._event_bus.emit(PROGRESS, agent="PlannerAgent", msg="[Planning] Decomposing request into tasks...")

            self._last_tool_context = "planning_initial"
            response = self._send_with_timeout(self._chat, planning_message)
            self._track_usage(response)

            result = self._parse_response(response)
            if result and self.verbose:
                self._event_bus.emit(
                    DEBUG, agent="PlannerAgent",
                    msg=f"[PlannerAgent] Round 1: status={result['status']}, "
                    f"{len(result.get('tasks', []))} tasks",
                )
                self._log_plan_details(result, round_num=1)

            return result

        except Exception as e:
            self._event_bus.emit(DEBUG, agent="PlannerAgent", level="warning", msg=f"[PlannerAgent] Error in start_planning: {e}")
            return None

    def continue_planning(self, round_results: list[dict],
                           round_num: int = 0, max_rounds: int | None = None) -> Optional[dict]:
        """Send execution results back to the planner for the next round.

        Args:
            round_results: List of dicts with {description, status, result_summary, error}
            round_num: Current round number (1-based).
            max_rounds: Maximum number of rounds allowed.

        Returns:
            Dict with {status, reasoning, tasks, summary} or None on failure.
        """
        if max_rounds is None:
            max_rounds = get_limit("planner.max_rounds")

        if self._chat is None:
            self._event_bus.emit(DEBUG, agent="PlannerAgent", level="warning", msg="[PlannerAgent] No active chat session for continue_planning")
            return None

        try:
            # Format results as structured text
            lines = ["Execution results:"]
            for r in round_results:
                status = r.get("status", "unknown")
                desc = r.get("description", "")
                line = f"- Task: {desc} | Status: {status}"
                if r.get("result_summary"):
                    line += f" | Result: {r['result_summary']}"
                if r.get("error"):
                    line += f" | Error: {r['error']}"
                lines.append(line)

            # Include current data-store state so planner can make informed decisions
            # Use context tracker to only inject new/changed store entries
            data_details = None
            for r in round_results:
                if r.get("data_in_memory"):
                    data_details = r["data_in_memory"]
                    break
            if data_details:
                _planner_id = "PlannerAgent"
                _inject_store = True
                if self._ctx_tracker:
                    # Convert to list of dicts for delta computation
                    _entries = [d if isinstance(d, dict) else {"label": d} for d in data_details]
                    new_entries, removed_labels, store_hash = self._ctx_tracker.get_store_delta(_planner_id, _entries)
                    if not new_entries and not removed_labels:
                        _inject_store = False  # Store unchanged — skip
                    else:
                        self._ctx_tracker.record(_planner_id, store_entries=_entries, store_hash=store_hash)
                        # Only inject the new/changed entries
                        if new_entries:
                            data_details = new_entries
                            if removed_labels:
                                lines.append(f"\nData removed from memory: {', '.join(removed_labels)}")

                if _inject_store:
                    lines.append("\nData currently in memory:")
                    for d in data_details:
                        if isinstance(d, dict):
                            cols = d.get("columns", [])
                            cols_str = f", columns={cols}" if cols else ""
                            lines.append(
                                f"  - {d['label']} ({d.get('shape', '?')}, "
                                f"{d.get('num_points', '?')} pts, "
                                f"units={d.get('units', '?')}{cols_str})"
                            )
                        else:
                            # Backward compat: flat label string
                            lines.append(f"  - {d}")
                    label_names = [d["label"] if isinstance(d, dict) else d for d in data_details]
                    self._event_bus.emit(DEBUG, agent="PlannerAgent", msg=f"[Planner] Data in memory: {', '.join(label_names)}")

            # Collect ALL failed task descriptions (current + previous rounds)
            failed_descs = []
            for r in round_results:
                if r.get("status") == "failed":
                    failed_descs.append(r)
            if failed_descs:
                lines.append("\n## FAILED TASKS — REFLECTION REQUIRED:")
                for r in failed_descs:
                    desc = r.get("description", "unknown")
                    error = r.get("error", "unknown error")
                    lines.append(f"  - Task: {desc}")
                    lines.append(f"    Error: {error}")
                    lines.append(
                        f"    What to try instead: Use different dataset IDs, "
                        f"adjust time ranges, or skip this data and proceed "
                        f"with what's available."
                    )
                lines.append(
                    "Do NOT retry the same failed tasks. "
                    "Either find alternatives or set status='done'."
                )

            # Round budget awareness
            remaining = max_rounds - round_num
            if remaining <= 2 and remaining > 0:
                lines.append(f"\n## BUDGET WARNING: Only {remaining} round(s) remaining.")
                lines.append("Prioritize essential tasks. Consider setting status='done' with partial results.")
            if remaining <= 0:
                lines.append("\n## FINAL ROUND: This is the last round. Set status='done' unless critical work remains.")

            message = "\n".join(lines)

            if self.verbose:
                self._event_bus.emit(DEBUG, agent="PlannerAgent", msg=f"[PlannerAgent] Sending results:\n{message}")

            self._last_tool_context = f"continue_planning_round{round_num}"
            response = self._send_with_timeout(self._chat, message)
            self._track_usage(response)

            result = self._parse_response(response)
            if result and self.verbose:
                self._event_bus.emit(
                    DEBUG, agent="PlannerAgent",
                    msg=f"[PlannerAgent] Next round: status={result['status']}, "
                    f"{len(result.get('tasks', []))} tasks",
                )
                self._log_plan_details(result, round_num=round_num)

            return result

        except Exception as e:
            self._event_bus.emit(DEBUG, agent="PlannerAgent", level="warning", msg=f"[PlannerAgent] Error in continue_planning: {e}")
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
            "ctx_history_tokens": max(0, self._latest_input_tokens - self._current_system_tokens - self._current_tools_tokens),
            "ctx_total_tokens": self._latest_input_tokens,
        }

    def reset(self):
        """Reset the chat session."""
        self._chat = None


def format_plan_for_display(plan: TaskPlan) -> str:
    """Format a plan for display to the user.

    Groups tasks by round when round > 0.

    Args:
        plan: The plan to format

    Returns:
        Human-readable string representation
    """
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
                lines.append(f"  {i+1}. [{status_icon}]{mission_tag} {task.description}")

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
            lines.append(f"  {i+1}. [{status_icon}]{mission_tag} {task.description}")

            if task.status.value == "failed" and task.error:
                lines.append(f"       Error: {task.error}")

    lines.append("-" * 40)
    lines.append(f"Progress: {plan.progress_summary()}")

    return "\n".join(lines)
