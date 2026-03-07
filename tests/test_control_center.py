"""Tests for agent.control_center module."""

import threading
import time

import pytest

from agent.control_center import (
    ControlCenter,
    WorkStatus,
    WorkUnit,
    _make_unit_id,
)


class TestControlCenter:
    """Tests for ControlCenter thread-safe coordination."""

    def test_register(self):
        cc = ControlCenter()
        unit = cc.register(
            "delegation", "envoy", "EnvoyAgent[ACE]", "Fetch ACE mag data"
        )
        assert unit.id.startswith("wu_")
        assert unit.kind == "delegation"
        assert unit.agent_type == "envoy"
        assert unit.agent_name == "EnvoyAgent[ACE]"
        assert unit.status == WorkStatus.RUNNING
        assert unit.result is None
        assert isinstance(unit.cancel_event, threading.Event)
        assert not unit.cancel_event.is_set()

    def test_mark_completed(self):
        cc = ControlCenter()
        unit = cc.register("delegation", "envoy", "EnvoyAgent[ACE]", "Fetch ACE")
        cc.mark_completed(unit.id, {"status": "success"}, [{"tool": "fetch_data"}])
        assert unit.status == WorkStatus.COMPLETED
        assert unit.result == {"status": "success"}
        assert len(unit.operation_log) == 1
        assert unit.completed_at is not None

    def test_mark_failed(self):
        cc = ControlCenter()
        unit = cc.register("delegation", "envoy", "EnvoyAgent[ACE]", "Fetch ACE")
        cc.mark_failed(unit.id, "something broke")
        assert unit.status == WorkStatus.FAILED
        assert unit.error == "something broke"
        assert unit.result == {"status": "error", "message": "something broke"}

    def test_has_pending(self):
        cc = ControlCenter()
        assert not cc.has_pending()
        unit = cc.register("delegation", "envoy", "EnvoyAgent[ACE]", "Fetch")
        assert cc.has_pending()
        cc.mark_completed(unit.id, {})
        assert not cc.has_pending()

    def test_get(self):
        cc = ControlCenter()
        unit = cc.register("delegation", "envoy", "EnvoyAgent[ACE]", "Fetch")
        assert cc.get(unit.id) is unit
        assert cc.get("nonexistent") is None

    # ── List / Status Line ────────────────────────────────────────

    def test_list_active(self):
        cc = ControlCenter()
        u1 = cc.register("delegation", "envoy", "EnvoyAgent[ACE]", "Fetch ACE")
        u2 = cc.register("delegation", "data_ops", "DataOpsAgent", "Compute magnitude")
        cc.mark_completed(u2.id, {})

        active = cc.list_active()
        assert len(active) == 1
        assert active[0]["id"] == u1.id
        assert active[0]["agent_name"] == "EnvoyAgent[ACE]"
        assert "elapsed_s" in active[0]

    def test_status_line_empty(self):
        cc = ControlCenter()
        assert cc.status_line() == ""

    def test_status_line_with_active(self):
        cc = ControlCenter()
        cc.register("delegation", "envoy", "EnvoyAgent[ACE]", "Fetch ACE")
        line = cc.status_line()
        assert line.startswith("[Active work:")
        assert "EnvoyAgent[ACE]" in line

    # ── Selective Cancellation ────────────────────────────────────

    def test_cancel_specific(self):
        cc = ControlCenter()
        u1 = cc.register("delegation", "envoy", "EnvoyAgent[ACE]", "Fetch ACE")
        u2 = cc.register("delegation", "envoy", "EnvoyAgent[PSP]", "Fetch PSP")

        assert cc.cancel(u1.id)
        assert u1.status == WorkStatus.CANCELLED
        assert u1.cancel_event.is_set()
        assert u1.completed_at is not None
        # u2 still running
        assert u2.status == WorkStatus.RUNNING
        assert not u2.cancel_event.is_set()

    def test_cancel_not_found(self):
        cc = ControlCenter()
        assert not cc.cancel("nonexistent")

    def test_cancel_already_completed(self):
        cc = ControlCenter()
        unit = cc.register("delegation", "envoy", "EnvoyAgent[ACE]", "Fetch")
        cc.mark_completed(unit.id, {})
        assert not cc.cancel(unit.id)  # Can't cancel completed work

    def test_cancel_by_type(self):
        cc = ControlCenter()
        u1 = cc.register("delegation", "envoy", "EnvoyAgent[ACE]", "Fetch ACE")
        u2 = cc.register("delegation", "envoy", "EnvoyAgent[PSP]", "Fetch PSP")
        u3 = cc.register("delegation", "data_ops", "DataOpsAgent", "Compute")

        count = cc.cancel_by_type("envoy")
        assert count == 2
        assert u1.status == WorkStatus.CANCELLED
        assert u2.status == WorkStatus.CANCELLED
        assert u3.status == WorkStatus.RUNNING  # data_ops unaffected

    def test_cancel_all(self):
        cc = ControlCenter()
        u1 = cc.register("delegation", "envoy", "EnvoyAgent[ACE]", "Fetch")
        u2 = cc.register("delegation", "data_ops", "DataOpsAgent", "Compute")
        u3 = cc.register("planner", "planner", "PlannerAgent", "Build plan")

        count = cc.cancel_all()
        assert count == 3
        assert all(
            u.status == WorkStatus.CANCELLED and u.cancel_event.is_set()
            for u in [u1, u2, u3]
        )

    # ── Clear ─────────────────────────────────────────────────────

    def test_clear(self):
        cc = ControlCenter()
        cc.register("delegation", "envoy", "EnvoyAgent[ACE]", "Fetch")
        cc.register("delegation", "data_ops", "DataOpsAgent", "Compute")
        assert cc.has_pending()
        cc.clear()
        assert not cc.has_pending()
        assert cc.list_active() == []

    def test_clear_sets_cancel_events(self):
        """clear() should set cancel_event on all RUNNING units."""
        cc = ControlCenter()
        u1 = cc.register("delegation", "envoy", "EnvoyAgent[ACE]", "Fetch")
        u2 = cc.register("delegation", "data_ops", "DataOpsAgent", "Compute")
        cc.mark_completed(u2.id, {"ok": True})  # u2 already done
        cc.clear()
        assert u1.cancel_event.is_set()
        # u2 was not RUNNING at clear() time — cancel_event should NOT be set
        assert not u2.cancel_event.is_set()

    # ── Thread Safety ─────────────────────────────────────────────

    def test_concurrent_register_and_complete(self):
        cc = ControlCenter()
        n_threads = 10
        created_ids: list[str] = []
        lock = threading.Lock()

        def worker(idx):
            unit = cc.register("delegation", "envoy", f"Agent[{idx}]", f"Task {idx}")
            with lock:
                created_ids.append(unit.id)
            time.sleep(0.05)
            cc.mark_completed(unit.id, {"idx": idx})

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(created_ids) == n_threads
        assert not cc.has_pending()

    def test_mark_completed_on_nonrunning_is_noop(self):
        cc = ControlCenter()
        unit = cc.register("delegation", "envoy", "EnvoyAgent[ACE]", "Fetch")
        cc.cancel(unit.id)
        # Trying to complete a cancelled unit should be a no-op
        cc.mark_completed(unit.id, {"late": "result"})
        assert unit.status == WorkStatus.CANCELLED
        assert unit.result is None

    def test_mark_completed_unknown_id_is_noop(self):
        cc = ControlCenter()
        cc.mark_completed("nonexistent", {})
        cc.mark_failed("nonexistent", "err")


class TestWorkUnit:
    """Tests for WorkUnit dataclass."""

    def test_defaults(self):
        unit = WorkUnit(
            id="wu_test",
            kind="delegation",
            agent_type="envoy",
            agent_name="EnvoyAgent[ACE]",
            task_summary="Fetch ACE data",
            status=WorkStatus.RUNNING,
            cancel_event=threading.Event(),
        )
        assert unit.result is None
        assert unit.operation_log == []
        assert unit.error is None
        assert unit._collected is False
        assert unit.thread is None
        assert unit.completed_at is None

    def test_make_unit_id_unique(self):
        ids = {_make_unit_id() for _ in range(100)}
        assert len(ids) == 100
