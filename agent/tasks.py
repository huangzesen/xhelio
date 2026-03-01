"""
Task data structures and persistence for multi-step task handling.

This module provides:
- TaskStatus: Enum for task lifecycle states
- Task: Individual task with status, result, and tool tracking
- TaskPlan: A collection of tasks for a complex user request
- TaskStore: JSON persistence to ~/.xhelio/tasks/
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
import json
import uuid


class TaskStatus(Enum):
    """Lifecycle states for a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """A single task in a multi-step plan.

    Attributes:
        id: Unique identifier for the task
        description: Human-readable summary of what this task does
        instruction: The instruction to send to Gemini for this task
        mission: Spacecraft ID (e.g., "PSP", "ACE") or None for cross-mission tasks
        depends_on: List of task IDs that must complete before this task
        status: Current lifecycle state
        result: Result text from Gemini after completion
        error: Error message if the task failed
        tool_calls: List of tool names called during execution
    """
    id: str
    description: str
    instruction: str
    mission: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    tool_calls: list[str] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    round: int = 0
    candidate_datasets: Optional[list[str]] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "instruction": self.instruction,
            "mission": self.mission,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "round": self.round,
            "candidate_datasets": self.candidate_datasets,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """Create a Task from a dictionary."""
        return cls(
            id=data["id"],
            description=data["description"],
            instruction=data["instruction"],
            mission=data.get("mission"),
            depends_on=data.get("depends_on", []),
            status=TaskStatus(data["status"]),
            result=data.get("result"),
            error=data.get("error"),
            tool_calls=data.get("tool_calls", []),
            tool_results=data.get("tool_results", []),
            round=data.get("round", 0),
            candidate_datasets=data.get("candidate_datasets") or (
                [data["dataset_id"]] if data.get("dataset_id") else None
            ),
        )


class PlanStatus(Enum):
    """Lifecycle states for a plan."""
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskPlan:
    """A plan containing multiple tasks for a complex user request.

    Attributes:
        id: Unique identifier for the plan
        user_request: The original user request that triggered planning
        tasks: List of tasks to execute
        created_at: When the plan was created
        status: Current lifecycle state of the plan
        current_task_index: Index of the task being executed
    """
    id: str
    user_request: str
    tasks: list[Task]
    created_at: datetime
    status: PlanStatus = PlanStatus.PLANNING
    current_task_index: int = 0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "id": self.id,
            "user_request": self.user_request,
            "tasks": [t.to_dict() for t in self.tasks],
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "current_task_index": self.current_task_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskPlan":
        """Create a TaskPlan from a dictionary."""
        return cls(
            id=data["id"],
            user_request=data["user_request"],
            tasks=[Task.from_dict(t) for t in data["tasks"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            status=PlanStatus(data["status"]),
            current_task_index=data.get("current_task_index", 0),
        )

    def add_tasks(self, new_tasks: list[Task]) -> None:
        """Append new tasks to the plan (used by replan loop)."""
        self.tasks.extend(new_tasks)

    def get_current_task(self) -> Optional[Task]:
        """Get the current task being executed."""
        if 0 <= self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index]
        return None

    def get_pending_tasks(self) -> list[Task]:
        """Get all tasks that haven't started yet."""
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def get_completed_tasks(self) -> list[Task]:
        """Get all successfully completed tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]

    def get_failed_tasks(self) -> list[Task]:
        """Get all failed tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.FAILED]

    def is_complete(self) -> bool:
        """Check if all tasks have been processed (completed, failed, or skipped)."""
        terminal_states = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED}
        return all(t.status in terminal_states for t in self.tasks)

    def progress_summary(self) -> str:
        """Return a brief progress summary."""
        completed = len(self.get_completed_tasks())
        failed = len(self.get_failed_tasks())
        total = len(self.tasks)
        return f"{completed}/{total} completed" + (f", {failed} failed" if failed else "")


class TaskStore:
    """Persistent storage for task plans.

    Plans are stored as JSON files in ~/.xhelio/tasks/
    with filenames like: 20260205_143022_abc123.json
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the task store.

        Args:
            base_dir: Override the default storage directory (for testing)
        """
        if base_dir is None:
            from config import get_data_dir
            base_dir = get_data_dir() / "tasks"
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _plan_path(self, plan: TaskPlan) -> Path:
        """Get the file path for a plan (stable across saves)."""
        ts = plan.created_at.strftime("%Y%m%d_%H%M%S")
        short_id = plan.id[:8]
        return self.base_dir / f"{ts}_{short_id}.json"

    def save(self, plan: TaskPlan) -> None:
        """Persist a plan to disk as JSON (overwrites on each call)."""
        path = self._plan_path(plan)
        with open(path, "w") as fp:
            json.dump(plan.to_dict(), fp, indent=2)

    def get_incomplete_plans(self) -> list[TaskPlan]:
        """Get all plans that are not in a terminal state.

        Terminal states: COMPLETED, FAILED, CANCELLED
        """
        incomplete = []
        terminal = {PlanStatus.COMPLETED.value, PlanStatus.FAILED.value, PlanStatus.CANCELLED.value}
        for f in self.base_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    if data.get("status") not in terminal:
                        incomplete.append(TaskPlan.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue
        return incomplete

    def clear_all(self) -> int:
        """Delete all saved plans.

        Returns:
            Number of files deleted
        """
        count = 0
        for f in self.base_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count


def create_task(
    description: str,
    instruction: str,
    mission: Optional[str] = None,
    depends_on: Optional[list[str]] = None,
) -> Task:
    """Factory function to create a new task with a unique ID."""
    return Task(
        id=str(uuid.uuid4()),
        description=description,
        instruction=instruction,
        mission=mission,
        depends_on=depends_on or [],
    )


def create_plan(user_request: str, tasks: list[Task]) -> TaskPlan:
    """Factory function to create a new plan with a unique ID."""
    return TaskPlan(
        id=str(uuid.uuid4()),
        user_request=user_request,
        tasks=tasks,
        created_at=datetime.now(),
    )


# Global store instance (singleton pattern like data_ops.store)
_store: Optional[TaskStore] = None


def get_task_store() -> TaskStore:
    """Get the global TaskStore instance (creates on first call)."""
    global _store
    if _store is None:
        _store = TaskStore()
    return _store


def reset_task_store():
    """Reset the global TaskStore instance (for testing)."""
    global _store
    _store = None
