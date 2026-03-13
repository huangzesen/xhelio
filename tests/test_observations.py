"""
Tests for agent.observations — structured observation summaries.

Run with: python -m pytest tests/test_observations.py -v
"""

import pytest
from unittest.mock import patch

from agent.observations import generate_observation


class TestRunCodeObservation:
    def test_success(self):
        result = {
            "status": "success",
            "label": "Bmag",
            "num_points": 10000,
            "units": "nT",
        }
        obs = generate_observation("run_code", {}, result)
        assert "Bmag" in obs
        assert "10,000" in obs
        assert "nT" in obs

    def test_error(self):
        result = {"status": "error", "message": "NameError: 'Bx' not defined"}
        obs = generate_observation("run_code", {}, result)
        assert "FAILED" in obs
        assert "assets" in obs


class TestVisualizationObservation:
    def test_render_plotly(self):
        result = {"status": "success"}
        obs = generate_observation("render_plotly_json", {}, result)
        assert "Plot rendered" in obs

    def test_manage_plot(self):
        result = {"status": "success"}
        obs = generate_observation("manage_plot", {"action": "update_layout"}, result)
        assert "update_layout" in obs

    def test_render_error(self):
        result = {"status": "error", "message": "Invalid trace"}
        obs = generate_observation("render_plotly_json", {}, result)
        assert "FAILED" in obs
        assert "data_labels" in obs or "assets" in obs


class TestAssetsObservation:
    def test_list_with_assets(self):
        result = {"status": "success", "assets": [{"kind": "data", "name": "a"}, {"kind": "data", "name": "b"}], "count": 2}
        obs = generate_observation("assets", {"action": "list"}, result)
        assert "2" in obs
        assert "asset" in obs.lower()
        assert "data" in obs

    def test_list_empty(self):
        result = {"status": "success", "assets": [], "count": 0}
        obs = generate_observation("assets", {"action": "list"}, result)
        assert "No assets" in obs

    def test_status(self):
        result = {"status": "success", "plot": {"state": "active"}, "data": {"total_entries": 3}, "operations_count": 5}
        obs = generate_observation("assets", {"action": "status"}, result)
        assert "active" in obs
        assert "3" in obs

    def test_restore_plot(self):
        result = {"status": "success", "message": "Plot restored successfully."}
        obs = generate_observation("assets", {"action": "restore_plot"}, result)
        assert "restored" in obs.lower()


class TestDelegationObservation:
    def test_success(self):
        result = {
            "status": "success",
            "result": "Found 3 magnetic field datasets for ACE.",
        }
        obs = generate_observation("delegate_to_envoy_agent", {}, result)
        assert "Sub-agent completed" in obs
        assert "magnetic field" in obs

    def test_long_result_truncated(self):
        result = {"status": "success", "result": "x" * 200}
        obs = generate_observation("delegate_to_envoy_agent", {}, result)
        assert "..." in obs

    def test_error(self):
        result = {"status": "error", "message": "Sub-agent failed"}
        obs = generate_observation("delegate_to_viz_agent", {}, result)
        assert "FAILED" in obs
        assert "directly" in obs.lower()


class TestGenericObservation:
    def test_unknown_tool(self):
        result = {"status": "success"}
        obs = generate_observation("some_future_tool", {}, result)
        assert "some_future_tool" in obs
        assert "completed successfully" in obs

    def test_unknown_tool_error(self):
        result = {"status": "error", "message": "oops"}
        obs = generate_observation("some_future_tool", {}, result)
        assert "FAILED" in obs
        assert "different approach" in obs


class TestErrorReflectionHints:
    def test_run_code_hint(self):
        result = {"status": "error", "message": "syntax error"}
        obs = generate_observation("run_code", {}, result)
        assert "assets" in obs

    def test_delegation_hint(self):
        result = {"status": "error", "message": "failed"}
        obs = generate_observation("delegate_to_envoy_agent", {}, result)
        assert "directly" in obs.lower()

    def test_generic_hint(self):
        result = {"status": "error", "message": "boom"}
        obs = generate_observation("unknown_tool", {}, result)
        assert "different approach" in obs


class TestConfigToggle:
    @patch("config.OBSERVATION_SUMMARIES", False)
    def test_observations_disabled_does_not_crash(self):
        """generate_observation still works when called directly, config only
        guards injection in the agent loops."""
        result = {"status": "success"}
        obs = generate_observation("assets", {"action": "list"}, result)
        assert isinstance(obs, str)
