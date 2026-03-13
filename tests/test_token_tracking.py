"""Tests for agent/token_tracking.py — aggregation via DelegationBus."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

from agent.token_tracking import get_token_usage, get_token_usage_breakdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sub_agent(input_t=0, output_t=0, thinking_t=0, cached_t=0, api_calls=0):
    """Return a mock sub-agent with a get_token_usage() method."""
    agent = MagicMock()
    agent.get_token_usage.return_value = {
        "input_tokens": input_t,
        "output_tokens": output_t,
        "thinking_tokens": thinking_t,
        "cached_tokens": cached_t,
        "api_calls": api_calls,
        "ctx_system_tokens": 0,
        "ctx_tools_tokens": 0,
        "ctx_history_tokens": 0,
        "ctx_total_tokens": 0,
    }
    return agent


def _make_delegation(active_agents=None, retired=None):
    """Return a mock DelegationBus."""
    delegation = MagicMock()
    active = [
        (agent_id, agent.get_token_usage())
        for agent_id, agent in (active_agents or {}).items()
    ]
    delegation.get_all_agent_usages.return_value = (active, retired or [])
    return delegation


def _make_orch(
    input_t=0, output_t=0, thinking_t=0, cached_t=0, api_calls=0,
    active_agents=None, retired=None,
    system_prompt_tokens=0, tools_tokens=0, latest_input=0,
):
    """Construct a minimal mock OrchestratorAgent for token tracking tests."""
    orch = MagicMock()
    orch.get_token_usage.return_value = {
        "input_tokens": input_t,
        "output_tokens": output_t,
        "thinking_tokens": thinking_t,
        "cached_tokens": cached_t,
        "api_calls": api_calls,
        "ctx_system_tokens": system_prompt_tokens,
        "ctx_tools_tokens": tools_tokens,
        "ctx_history_tokens": max(0, latest_input - system_prompt_tokens - tools_tokens),
        "ctx_total_tokens": latest_input,
    }
    orch._delegation = _make_delegation(active_agents, retired)
    return orch


# ---------------------------------------------------------------------------
# get_token_usage tests
# ---------------------------------------------------------------------------

class TestGetTokenUsage:
    def test_empty_state_returns_zeros(self):
        orch = _make_orch()
        result = get_token_usage(orch)
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["thinking_tokens"] == 0
        assert result["cached_tokens"] == 0
        assert result["total_tokens"] == 0
        assert result["api_calls"] == 0

    def test_orch_usage_included(self):
        orch = _make_orch(input_t=100, output_t=50, thinking_t=10,
                          cached_t=5, api_calls=2)
        result = get_token_usage(orch)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["thinking_tokens"] == 10
        assert result["cached_tokens"] == 5
        assert result["api_calls"] == 2
        assert result["total_tokens"] == 100 + 50 + 10

    def test_sub_agents_included(self):
        sub = _make_sub_agent(input_t=200, output_t=80, thinking_t=5,
                              cached_t=3, api_calls=2)
        orch = _make_orch(
            input_t=100, output_t=50, api_calls=1,
            active_agents={"viz": sub},
        )
        result = get_token_usage(orch)
        assert result["input_tokens"] == 300
        assert result["output_tokens"] == 130
        assert result["thinking_tokens"] == 5
        assert result["cached_tokens"] == 3
        assert result["api_calls"] == 3

    def test_retired_agents_included(self):
        retired = [
            {"agent": "old_viz", "input_tokens": 50, "output_tokens": 20,
             "thinking_tokens": 3, "cached_tokens": 1, "api_calls": 1},
            {"agent": "old_data", "input_tokens": 30, "output_tokens": 10,
             "thinking_tokens": 0, "cached_tokens": 0, "api_calls": 1},
        ]
        orch = _make_orch(input_t=100, output_t=50, api_calls=1, retired=retired)
        result = get_token_usage(orch)
        assert result["input_tokens"] == 100 + 50 + 30
        assert result["output_tokens"] == 50 + 20 + 10
        assert result["thinking_tokens"] == 3
        assert result["api_calls"] == 3

    def test_total_tokens_excludes_cached(self):
        orch = _make_orch(input_t=100, output_t=50, thinking_t=10,
                          cached_t=999, api_calls=1)
        result = get_token_usage(orch)
        assert result["total_tokens"] == 100 + 50 + 10

    def test_multiple_sub_agents_all_summed(self):
        sub_agents = {
            "viz": _make_sub_agent(input_t=100, output_t=40, api_calls=2),
            "data_ops": _make_sub_agent(input_t=200, output_t=80, api_calls=3),
            "memory": _make_sub_agent(input_t=50, output_t=20, api_calls=1),
        }
        orch = _make_orch(api_calls=1, active_agents=sub_agents)
        result = get_token_usage(orch)
        assert result["input_tokens"] == 350
        assert result["output_tokens"] == 140
        assert result["api_calls"] == 7

    def test_no_delegation_attribute(self):
        """Works when _delegation is None (e.g., agent not fully initialized)."""
        orch = MagicMock()
        orch._delegation = None
        orch.get_token_usage.return_value = {
            "input_tokens": 100, "output_tokens": 50,
            "thinking_tokens": 10, "cached_tokens": 5, "api_calls": 1,
        }
        result = get_token_usage(orch)
        assert result["input_tokens"] == 100
        assert result["api_calls"] == 1


# ---------------------------------------------------------------------------
# get_token_usage_breakdown tests
# ---------------------------------------------------------------------------

class TestGetTokenUsageBreakdown:
    def test_empty_state_returns_empty_list(self):
        orch = _make_orch()
        result = get_token_usage_breakdown(orch)
        assert result == []

    def test_orchestrator_row_included_when_calls_made(self):
        orch = _make_orch(
            input_t=500, output_t=200, thinking_t=50, cached_t=10, api_calls=5,
            system_prompt_tokens=100, tools_tokens=80, latest_input=500,
        )
        result = get_token_usage_breakdown(orch)
        assert len(result) == 1
        row = result[0]
        assert row["agent"] == "Orchestrator"
        assert row["input"] == 500
        assert row["output"] == 200
        assert row["thinking"] == 50
        assert row["cached"] == 10
        assert row["calls"] == 5
        assert row["ctx_system"] == 100
        assert row["ctx_tools"] == 80
        assert row["ctx_history"] == 500 - 100 - 80
        assert row["ctx_total"] == 500

    def test_ctx_history_clamped_to_zero(self):
        orch = _make_orch(
            api_calls=1,
            system_prompt_tokens=100, tools_tokens=80, latest_input=50,
        )
        result = get_token_usage_breakdown(orch)
        row = result[0]
        assert row["ctx_history"] == 0

    def test_sub_agent_rows_included(self):
        sub_agents = {
            "viz": _make_sub_agent(input_t=100, output_t=40, api_calls=2),
            "data_ops": _make_sub_agent(input_t=200, output_t=80, api_calls=3),
        }
        orch = _make_orch(active_agents=sub_agents)
        result = get_token_usage_breakdown(orch)
        agents = {r["agent"] for r in result}
        assert "viz" in agents
        assert "data_ops" in agents

        viz_row = next(r for r in result if r["agent"] == "viz")
        assert viz_row["input"] == 100
        assert viz_row["calls"] == 2

    def test_zero_call_agents_excluded(self):
        sub_agents = {"viz": _make_sub_agent(api_calls=0)}
        orch = _make_orch(active_agents=sub_agents)
        result = get_token_usage_breakdown(orch)
        assert result == []

    def test_retired_agents_appended(self):
        retired = [
            {
                "agent": "old_viz",
                "input_tokens": 50, "output_tokens": 20,
                "thinking_tokens": 3, "cached_tokens": 1,
                "api_calls": 1,
                "ctx_system_tokens": 10, "ctx_tools_tokens": 5,
                "ctx_history_tokens": 35, "ctx_total_tokens": 50,
            }
        ]
        orch = _make_orch(retired=retired)
        result = get_token_usage_breakdown(orch)
        assert len(result) == 1
        row = result[0]
        assert row["agent"] == "old_viz"
        assert row["input"] == 50
        assert row["output"] == 20
        assert row["thinking"] == 3
        assert row["cached"] == 1
        assert row["calls"] == 1
        assert row["ctx_system"] == 10
        assert row["ctx_tools"] == 5
        assert row["ctx_history"] == 35
        assert row["ctx_total"] == 50

    def test_retired_missing_ctx_fields_default_zero(self):
        retired = [
            {
                "agent": "old_agent",
                "input_tokens": 10, "output_tokens": 5,
                "api_calls": 1,
            }
        ]
        orch = _make_orch(retired=retired)
        result = get_token_usage_breakdown(orch)
        assert len(result) == 1
        row = result[0]
        assert row["thinking"] == 0
        assert row["cached"] == 0
        assert row["ctx_system"] == 0
        assert row["ctx_tools"] == 0
        assert row["ctx_history"] == 0
        assert row["ctx_total"] == 0

    def test_breakdown_row_keys(self):
        orch = _make_orch(api_calls=1, latest_input=100)
        result = get_token_usage_breakdown(orch)
        expected_keys = {"agent", "input", "output", "thinking", "cached", "calls",
                         "ctx_system", "ctx_tools", "ctx_history", "ctx_total"}
        for row in result:
            assert set(row.keys()) == expected_keys
