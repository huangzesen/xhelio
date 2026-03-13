# tests/test_work_tracker.py
"""Tests for WorkTracker — in-flight work tracking."""
import threading
from agent.work_tracker import WorkTracker


def test_register_and_list():
    wt = WorkTracker()
    ev = threading.Event()
    wt.register("w1", "envoy:cdaweb:ace", "Fetching ACE data", ev)
    active = wt.list_active()
    assert len(active) == 1
    assert active[0]["work_id"] == "w1"
    assert active[0]["agent_id"] == "envoy:cdaweb:ace"


def test_mark_completed():
    wt = WorkTracker()
    ev = threading.Event()
    wt.register("w1", "agent1", "task", ev)
    wt.mark_completed("w1")
    assert len(wt.list_active()) == 0


def test_mark_failed():
    wt = WorkTracker()
    ev = threading.Event()
    wt.register("w1", "agent1", "task", ev)
    wt.mark_failed("w1", "timeout")
    assert len(wt.list_active()) == 0


def test_cancel_sets_event():
    wt = WorkTracker()
    ev = threading.Event()
    wt.register("w1", "agent1", "task", ev)
    assert wt.cancel("w1") is True
    assert ev.is_set()
    assert len(wt.list_active()) == 0


def test_cancel_nonexistent():
    wt = WorkTracker()
    assert wt.cancel("nope") is False


def test_cancel_all():
    wt = WorkTracker()
    e1, e2 = threading.Event(), threading.Event()
    wt.register("w1", "a1", "t1", e1)
    wt.register("w2", "a2", "t2", e2)
    count = wt.cancel_all()
    assert count == 2
    assert e1.is_set() and e2.is_set()


def test_clear_removes_completed():
    wt = WorkTracker()
    ev = threading.Event()
    wt.register("w1", "a1", "t1", ev)
    wt.mark_completed("w1")
    wt.clear()
    # No error, completed entries cleaned up
