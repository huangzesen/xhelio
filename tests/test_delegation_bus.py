"""Tests for DelegationBus."""

import threading
from unittest.mock import MagicMock, patch
from agent.delegation import DelegationBus


def _make_bus():
    """Create a DelegationBus with a minimal mock context."""
    ctx = MagicMock()
    ctx._event_bus = MagicMock()
    ctx._ctx_tracker = MagicMock()
    return DelegationBus(ctx=ctx)


def test_agents_dict_starts_empty():
    bus = _make_bus()
    assert len(bus._agents) == 0


def test_reset_clears_agents():
    bus = _make_bus()
    mock_agent = MagicMock()
    bus._agents["test"] = mock_agent
    with patch("agent.envoy_kinds.registry.ENVOY_KIND_REGISTRY", MagicMock()):
        bus.reset()
    assert len(bus._agents) == 0
    mock_agent.stop.assert_called_once_with(timeout=2.0)


def test_retired_usage_starts_empty():
    bus = _make_bus()
    assert len(bus._retired_usage) == 0


def test_cleanup_ephemeral_retires_usage():
    bus = _make_bus()
    mock_agent = MagicMock()
    mock_agent.agent_id = "ephemeral_test"
    mock_agent.get_token_usage.return_value = {
        "input_tokens": 100,
        "output_tokens": 50,
        "thinking_tokens": 10,
        "cached_tokens": 0,
        "api_calls": 2,
    }
    bus._agents["ephemeral_test"] = mock_agent
    bus.cleanup_ephemeral("ephemeral_test")
    assert "ephemeral_test" not in bus._agents
    assert len(bus._retired_usage) == 1
    assert bus._retired_usage[0]["input_tokens"] == 100
    assert bus._retired_usage[0]["api_calls"] == 2


def test_cleanup_ephemeral_no_usage_if_zero_calls():
    bus = _make_bus()
    mock_agent = MagicMock()
    mock_agent.agent_id = "ephemeral_zero"
    mock_agent.get_token_usage.return_value = {
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "cached_tokens": 0,
        "api_calls": 0,
    }
    bus._agents["ephemeral_zero"] = mock_agent
    bus.cleanup_ephemeral("ephemeral_zero")
    assert "ephemeral_zero" not in bus._agents
    assert len(bus._retired_usage) == 0


def test_cleanup_nonexistent_agent_is_noop():
    bus = _make_bus()
    bus.cleanup_ephemeral("nonexistent")
    assert len(bus._retired_usage) == 0


def test_all_work_subagents_idle_empty():
    bus = _make_bus()
    assert bus.all_work_subagents_idle() is True


def test_all_work_subagents_idle_skips_eureka_and_memory():
    bus = _make_bus()
    eureka = MagicMock()
    eureka.is_idle = False  # busy, but should be skipped
    bus._agents["EurekaAgent"] = eureka
    memory = MagicMock()
    memory.is_idle = False
    bus._agents["MemoryAgent"] = memory
    assert bus.all_work_subagents_idle() is True


def test_all_work_subagents_idle_returns_false_when_busy():
    bus = _make_bus()
    busy_agent = MagicMock()
    busy_agent.is_idle = False
    bus._agents["DataOpsAgent"] = busy_agent
    assert bus.all_work_subagents_idle() is False


def test_get_active_envoy_ids():
    bus = _make_bus()
    bus._agents["EnvoyAgent[ACE]"] = MagicMock()
    bus._agents["EnvoyAgent[WIND]"] = MagicMock()
    bus._agents["VizAgent[Plotly]"] = MagicMock()
    ids = bus.get_active_envoy_ids()
    assert ids == {"ACE", "WIND"}


def test_register_agent():
    bus = _make_bus()
    mock_agent = MagicMock()
    bus.register_agent("TestAgent", mock_agent)
    assert bus._agents["TestAgent"] is mock_agent


def test_reset_full_clears_counters():
    bus = _make_bus()
    bus._dataops_seq = 5
    bus._mission_seq = 3
    bus._agents["test"] = MagicMock()
    with patch("agent.envoy_kinds.registry.ENVOY_KIND_REGISTRY", MagicMock()):
        bus.reset_full()
    assert len(bus._agents) == 0
    assert bus._dataops_seq == 0
    assert bus._mission_seq == 0


def test_stop_all():
    bus = _make_bus()
    a1 = MagicMock()
    a2 = MagicMock()
    bus._agents["a1"] = a1
    bus._agents["a2"] = a2
    bus.stop_all()
    assert len(bus._agents) == 0
    a1.stop.assert_called_once_with(timeout=2.0)
    a2.stop.assert_called_once_with(timeout=2.0)


def test_get_all_agent_usages():
    bus = _make_bus()
    agent1 = MagicMock()
    agent1.get_token_usage.return_value = {"input_tokens": 10, "output_tokens": 5, "api_calls": 1}
    bus._agents["Agent1"] = agent1
    bus._retired_usage.append({"agent": "OldAgent", "input_tokens": 20, "output_tokens": 10, "api_calls": 1})
    active, retired = bus.get_all_agent_usages()
    assert len(active) == 1
    assert active[0][0] == "Agent1"
    assert len(retired) == 1
    assert retired[0]["agent"] == "OldAgent"


def test_has_agent():
    bus = _make_bus()
    bus._agents["X"] = MagicMock()
    assert bus.has_agent("X") is True
    assert bus.has_agent("Y") is False


def test_is_stale():
    bus = _make_bus()
    assert bus.is_stale() is False
    bus._agents["X"] = MagicMock()
    assert bus.is_stale() is True


def test_wrap_delegation_result_success():
    result = DelegationBus.wrap_delegation_result(
        {"text": "Done", "failed": False, "errors": []}
    )
    assert result["status"] == "success"
    assert result["result"] == "Done"


def test_wrap_delegation_result_failure():
    result = DelegationBus.wrap_delegation_result(
        {"text": "", "failed": True, "errors": ["boom"]}
    )
    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_wrap_delegation_result_with_store_snapshot():
    result = DelegationBus.wrap_delegation_result(
        {"text": "ok", "failed": False, "errors": []},
        store_snapshot=[
            {"label": "ACE_MAG", "columns": ["Bx"], "shape": "(100, 1)", "units": "nT", "num_points": 100},
        ],
    )
    assert result["status"] == "success"
    assert len(result["data_in_memory"]) == 1
    assert result["data_in_memory"][0]["label"] == "ACE_MAG"


def test_wrap_delegation_result_output_files():
    result = DelegationBus.wrap_delegation_result(
        {"text": "", "failed": True, "errors": ["minor"], "output_files": ["/tmp/plot.png"]}
    )
    # Has output files, so it's a success despite errors
    assert result["status"] == "success"
    assert result["output_files"] == ["/tmp/plot.png"]


def test_build_data_io_request_no_context():
    bus = _make_bus()
    req = bus.build_data_io_request("fetch something", "")
    assert req == "fetch something"


def test_build_data_io_request_with_context():
    bus = _make_bus()
    req = bus.build_data_io_request("fetch something", "extra info")
    assert "fetch something" in req
    assert "extra info" in req
