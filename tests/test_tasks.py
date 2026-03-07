"""
Tests for agent.tasks — Task, TaskPlan, and TaskStore.

Run with: python -m pytest tests/test_tasks.py
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from agent.tasks import (
    Task,
    TaskStatus,
    TaskPlan,
    PlanStatus,
    TaskStore,
    create_task,
    create_plan,
    get_task_store,
    reset_task_store,
)


@pytest.fixture
def temp_store():
    """Create a TaskStore with a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield TaskStore(base_dir=Path(tmpdir))


@pytest.fixture(autouse=True)
def reset_global_store():
    """Reset the global store before each test."""
    reset_task_store()
    yield
    reset_task_store()


class TestTask:
    def test_create_with_defaults(self):
        task = Task(
            id="task-1",
            description="Fetch PSP data",
            instruction="Use fetch_data to get PSP magnetic field"
        )
        assert task.id == "task-1"
        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.error is None
        assert task.tool_calls == []
        assert task.mission is None
        assert task.depends_on == []

    def test_create_with_mission(self):
        task = Task(
            id="task-m",
            description="Fetch PSP mag",
            instruction="fetch_data PSP_FLD_L2_MAG_RTN_1MIN",
            mission="PSP",
            depends_on=["task-0"],
        )
        assert task.mission == "PSP"
        assert task.depends_on == ["task-0"]

    def test_to_dict(self):
        task = Task(
            id="task-1",
            description="Test task",
            instruction="Do something",
            mission="ACE",
            depends_on=["task-0"],
            status=TaskStatus.COMPLETED,
            result="Done",
            tool_calls=["fetch_data", "compute_magnitude"],
            candidate_datasets=["AC_H2_MFI", "AC_H0_MFI"],
        )
        d = task.to_dict()
        assert d["id"] == "task-1"
        assert d["status"] == "completed"
        assert d["result"] == "Done"
        assert d["tool_calls"] == ["fetch_data", "compute_magnitude"]
        assert d["mission"] == "ACE"
        assert d["depends_on"] == ["task-0"]
        assert d["candidate_datasets"] == ["AC_H2_MFI", "AC_H0_MFI"]
        assert "dataset_id" not in d
        assert "parameter_id" not in d

    def test_to_dict_null_mission(self):
        task = Task(id="t", description="D", instruction="I")
        d = task.to_dict()
        assert d["mission"] is None
        assert d["depends_on"] == []

    def test_from_dict(self):
        data = {
            "id": "task-2",
            "description": "Another task",
            "instruction": "Do another thing",
            "mission": "PSP",
            "depends_on": ["task-1"],
            "status": "failed",
            "result": None,
            "error": "Something went wrong",
            "tool_calls": ["plot_data"],
            "candidate_datasets": ["AC_H2_MFI", "AC_H0_MFI"],
        }
        task = Task.from_dict(data)
        assert task.id == "task-2"
        assert task.status == TaskStatus.FAILED
        assert task.error == "Something went wrong"
        assert task.tool_calls == ["plot_data"]
        assert task.mission == "PSP"
        assert task.depends_on == ["task-1"]
        assert task.candidate_datasets == ["AC_H2_MFI", "AC_H0_MFI"]

    def test_from_dict_missing_new_fields(self):
        """Backward compat: old task dicts without mission/depends_on/candidate_datasets still load."""
        data = {
            "id": "task-old",
            "description": "Old task",
            "instruction": "Do old thing",
            "status": "completed",
        }
        task = Task.from_dict(data)
        assert task.mission is None
        assert task.depends_on == []
        assert task.candidate_datasets is None

    def test_from_dict_backward_compat_dataset_id(self):
        """Backward compat: old dicts with dataset_id convert to candidate_datasets."""
        data = {
            "id": "task-old2",
            "description": "Old fetch task",
            "instruction": "Fetch data from AC_H2_MFI",
            "status": "completed",
            "dataset_id": "AC_H2_MFI",
            "parameter_id": "BGSEc",
        }
        task = Task.from_dict(data)
        assert task.candidate_datasets == ["AC_H2_MFI"]

    def test_roundtrip(self):
        original = Task(
            id="task-rt",
            description="Roundtrip test",
            instruction="Test serialization",
            mission="OMNI",
            depends_on=["task-a", "task-b"],
            status=TaskStatus.IN_PROGRESS,
            tool_calls=["search_datasets"],
            candidate_datasets=["OMNI_HRO_1MIN"],
        )
        restored = Task.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.status == original.status
        assert restored.tool_calls == original.tool_calls
        assert restored.mission == original.mission
        assert restored.depends_on == original.depends_on
        assert restored.candidate_datasets == original.candidate_datasets


class TestTaskPlan:
    def test_create_with_defaults(self):
        tasks = [
            create_task("Step 1", "Do step 1"),
            create_task("Step 2", "Do step 2"),
        ]
        plan = TaskPlan(
            id="plan-1",
            user_request="Do a complex thing",
            tasks=tasks,
            created_at=datetime(2026, 2, 5, 10, 0, 0),
        )
        assert plan.id == "plan-1"
        assert plan.status == PlanStatus.PLANNING
        assert plan.current_task_index == 0
        assert len(plan.tasks) == 2

    def test_get_current_task(self):
        tasks = [
            Task(id="t1", description="First", instruction="Do first"),
            Task(id="t2", description="Second", instruction="Do second"),
        ]
        plan = TaskPlan(
            id="plan",
            user_request="Test",
            tasks=tasks,
            created_at=datetime.now(),
        )
        assert plan.get_current_task().id == "t1"
        plan.current_task_index = 1
        assert plan.get_current_task().id == "t2"
        plan.current_task_index = 5  # Out of range
        assert plan.get_current_task() is None

    def test_get_pending_completed_failed(self):
        tasks = [
            Task(id="t1", description="D1", instruction="I1", status=TaskStatus.COMPLETED),
            Task(id="t2", description="D2", instruction="I2", status=TaskStatus.FAILED),
            Task(id="t3", description="D3", instruction="I3", status=TaskStatus.PENDING),
        ]
        plan = TaskPlan(
            id="plan",
            user_request="Test",
            tasks=tasks,
            created_at=datetime.now(),
        )
        assert len(plan.get_pending_tasks()) == 1
        assert len(plan.get_completed_tasks()) == 1
        assert len(plan.get_failed_tasks()) == 1

    def test_is_complete(self):
        tasks = [
            Task(id="t1", description="D1", instruction="I1", status=TaskStatus.COMPLETED),
            Task(id="t2", description="D2", instruction="I2", status=TaskStatus.PENDING),
        ]
        plan = TaskPlan(
            id="plan",
            user_request="Test",
            tasks=tasks,
            created_at=datetime.now(),
        )
        assert not plan.is_complete()
        tasks[1].status = TaskStatus.COMPLETED
        assert plan.is_complete()
        # Failed and skipped also count as terminal
        tasks[1].status = TaskStatus.FAILED
        assert plan.is_complete()
        tasks[1].status = TaskStatus.SKIPPED
        assert plan.is_complete()

    def test_progress_summary(self):
        tasks = [
            Task(id="t1", description="D1", instruction="I1", status=TaskStatus.COMPLETED),
            Task(id="t2", description="D2", instruction="I2", status=TaskStatus.FAILED),
            Task(id="t3", description="D3", instruction="I3", status=TaskStatus.PENDING),
        ]
        plan = TaskPlan(
            id="plan",
            user_request="Test",
            tasks=tasks,
            created_at=datetime.now(),
        )
        summary = plan.progress_summary()
        assert "1/3 completed" in summary
        assert "1 failed" in summary

    def test_to_dict_and_from_dict(self):
        tasks = [create_task("Step 1", "Do step 1")]
        plan = TaskPlan(
            id="plan-serial",
            user_request="Serialize me",
            tasks=tasks,
            created_at=datetime(2026, 2, 5, 12, 30, 45),
            status=PlanStatus.EXECUTING,
            current_task_index=0,
        )
        d = plan.to_dict()
        assert d["id"] == "plan-serial"
        assert d["status"] == "executing"
        assert d["created_at"] == "2026-02-05T12:30:45"

        restored = TaskPlan.from_dict(d)
        assert restored.id == plan.id
        assert restored.status == PlanStatus.EXECUTING
        assert restored.created_at == plan.created_at
        assert len(restored.tasks) == 1


class TestTaskStore:
    def test_get_incomplete_plans(self, temp_store):
        # Save plans as JSON files directly (save/load methods removed)
        import json
        from datetime import datetime as _dt

        def _save_plan(store, plan):
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            path = store.base_dir / f"{ts}_{plan.id[:8]}.json"
            with open(path, "w") as f:
                json.dump(plan.to_dict(), f)

        # Completed plan
        p1 = create_plan("Completed", [create_task("T1", "I1")])
        p1.status = PlanStatus.COMPLETED
        _save_plan(temp_store, p1)

        # Executing plan
        p2 = create_plan("Executing", [create_task("T2", "I2")])
        p2.status = PlanStatus.EXECUTING
        _save_plan(temp_store, p2)

        # Cancelled plan
        p3 = create_plan("Cancelled", [create_task("T3", "I3")])
        p3.status = PlanStatus.CANCELLED
        _save_plan(temp_store, p3)

        incomplete = temp_store.get_incomplete_plans()
        assert len(incomplete) == 1
        assert incomplete[0].id == p2.id

    def test_clear_all(self, temp_store):
        import json
        from datetime import datetime as _dt

        for i in range(2):
            plan = create_plan(f"P{i}", [create_task("T", "I")])
            path = temp_store.base_dir / f"plan_{i}.json"
            with open(path, "w") as f:
                json.dump(plan.to_dict(), f)

        count = temp_store.clear_all()
        assert count == 2
        assert list(temp_store.base_dir.glob("*.json")) == []


class TestFactoryFunctions:
    def test_create_task(self):
        task = create_task("Description", "Instruction")
        assert task.id  # Has a UUID
        assert len(task.id) == 36  # UUID format
        assert task.description == "Description"
        assert task.instruction == "Instruction"
        assert task.status == TaskStatus.PENDING
        assert task.mission is None
        assert task.depends_on == []

    def test_create_task_with_mission(self):
        task = create_task("Fetch PSP data", "fetch_data ...", mission="PSP")
        assert task.mission == "PSP"
        assert task.depends_on == []

    def test_create_task_with_depends_on(self):
        task = create_task("Plot comparison", "plot ...", depends_on=["id-1", "id-2"])
        assert task.depends_on == ["id-1", "id-2"]
        assert task.mission is None

    def test_create_task_with_all_new_fields(self):
        task = create_task("Fetch ACE", "fetch ...", mission="ACE", depends_on=["id-0"])
        assert task.mission == "ACE"
        assert task.depends_on == ["id-0"]

    def test_create_plan(self):
        tasks = [create_task("T1", "I1"), create_task("T2", "I2")]
        plan = create_plan("User request", tasks)
        assert plan.id
        assert len(plan.id) == 36
        assert plan.user_request == "User request"
        assert len(plan.tasks) == 2
        assert plan.status == PlanStatus.PLANNING
        assert isinstance(plan.created_at, datetime)


class TestGlobalStore:
    def test_singleton(self):
        s1 = get_task_store()
        s2 = get_task_store()
        assert s1 is s2

    def test_reset(self):
        s1 = get_task_store()
        reset_task_store()
        s2 = get_task_store()
        assert s1 is not s2
