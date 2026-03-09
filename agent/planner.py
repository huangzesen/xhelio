"""
Planning logic for multi-step task handling.

This module provides:
- The orchestrator LLM decides when to invoke planning via the delegate_to_planner tool
- PlannerAgent: SubAgent-based planner with research tools and produce_plan
- format_plan_for_display(): Human-readable plan rendering
"""

import json
from typing import TYPE_CHECKING

from .sub_agent import SubAgent
from .llm import FunctionSchema
from .event_bus import get_event_bus, DEBUG
from .agent_registry import PLANNER_TOOLS
from .tools import get_function_schemas
from .tasks import Task, TaskPlan

from knowledge.prompt_builder import build_planner_agent_prompt


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


class PlannerAgent(SubAgent):
    """Planning agent that decomposes complex requests into task batches.

    Uses research tools (list_missions, search_datasets, browse_datasets,
    web_search, etc.) to investigate data availability, then calls
    produce_plan to save a structured plan to disk. The natural language
    summary is delivered to the orchestrator via the standard SubAgent
    result flow.
    """

    _PARALLEL_SAFE_TOOLS: set[str] = {
        "envoy_query",
        "get_dataset_docs",
        "list_fetched_data",
        "web_search",
    }

    def __init__(
        self,
        service,
        tool_executor=None,
        *,
        event_bus=None,
        cancel_event=None,
        memory_store=None,
        memory_scope: str = "planner",
        session_id: str | None = None,
    ):
        self._session_id_str = session_id or "default"

        # Build tool schemas before super().__init__ so _tool_schemas is set
        schemas = get_function_schemas(names=PLANNER_TOOLS)
        schemas.append(PRODUCE_PLAN_SCHEMA)

        super().__init__(
            agent_id="PlannerAgent",
            service=service,
            agent_type="planner",
            tool_executor=self._wrap_tool_executor(tool_executor),
            tool_schemas=schemas,
            event_bus=event_bus,
            cancel_event=cancel_event,
            memory_store=memory_store,
            memory_scope=memory_scope,
        )

    def _wrap_tool_executor(self, external_executor):
        """Wrap the external executor to handle produce_plan internally."""
        def executor(tool_name, tool_args, tc_id=None):
            if tool_name == "produce_plan":
                return self._handle_produce_plan(tool_args)
            if external_executor:
                return external_executor(tool_name, tool_args, tc_id)
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}
        return executor

    def _build_system_prompt(self) -> str:
        return build_planner_agent_prompt()

    def _handle_produce_plan(self, tool_args: dict) -> dict:
        """Handle the produce_plan tool call -- normalize and save plan to disk."""
        plan = dict(tool_args)
        # Normalize mission "null"/"none" strings to None
        for task in plan.get("tasks", []):
            mission = task.get("mission")
            if isinstance(mission, str) and mission.lower() in ("null", "none", ""):
                task["mission"] = None

        if not plan.get("time_range_validated"):
            self._event_bus.emit(
                DEBUG,
                agent="PlannerAgent",
                level="warning",
                msg="[PlannerAgent] Plan submitted with time_range_validated=false",
            )

        plan_file = self._save_plan_to_file(plan)
        return {
            "status": "success",
            "message": f"Plan saved to {plan_file}. Summarize the plan for the orchestrator.",
            "plan_file": plan_file,
            "task_count": len(plan.get("tasks", [])),
        }

    def _save_plan_to_file(self, plan: dict) -> str:
        """Save plan to filesystem as JSON."""
        import os
        from pathlib import Path

        plan_dir = Path(os.environ.get("XHELIO_DATA_DIR", "/tmp/xhelio")) / "plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_file = plan_dir / f"{self._session_id_str}_plan.json"

        with open(plan_file, "w") as f:
            json.dump(plan, f, indent=2)

        return str(plan_file)


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
