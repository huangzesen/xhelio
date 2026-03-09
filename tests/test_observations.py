"""
Tests for agent.observations — structured observation summaries.

Run with: python -m pytest tests/test_observations.py -v
"""

import pytest
from unittest.mock import patch

from agent.observations import generate_observation, register_spice_tool_names

# Register SPICE tool names so observation routing works in tests
register_spice_tool_names(
    [
        "get_spacecraft_position",
        "get_spacecraft_trajectory",
        "get_spacecraft_velocity",
        "compute_distance",
        "transform_coordinates",
        "list_spice_missions",
        "list_coordinate_frames",
        "manage_kernels",
    ]
)


class TestFetchDataObservation:
    def test_success_with_details(self):
        result = {
            "status": "success",
            "label": "AC_H2_MFI.BGSEc",
            "num_points": 10000,
            "columns": ["Bx", "By", "Bz"],
            "units": "nT",
            "nan_percentage": 5.2,
        }
        obs = generate_observation("fetch_data", {"dataset_id": "AC_H2_MFI"}, result)
        assert "10,000" in obs
        assert "AC_H2_MFI.BGSEc" in obs
        assert "Bx" in obs
        assert "nT" in obs
        assert "5.2% NaN" in obs

    def test_already_loaded(self):
        result = {"status": "already_loaded", "label": "AC_H2_MFI.BGSEc"}
        obs = generate_observation("fetch_data", {}, result)
        assert "Already in memory" in obs
        assert "AC_H2_MFI.BGSEc" in obs

    def test_no_nan(self):
        result = {
            "status": "success",
            "label": "test",
            "num_points": 100,
            "columns": ["val"],
            "units": "km",
        }
        obs = generate_observation("fetch_data", {}, result)
        assert "NaN" not in obs

    def test_quality_warning(self):
        result = {
            "status": "success",
            "label": "test",
            "num_points": 50,
            "quality_warning": "Large gap detected",
        }
        obs = generate_observation("fetch_data", {}, result)
        assert "Large gap detected" in obs

    def test_error(self):
        result = {"status": "error", "message": "Dataset XYZ not found"}
        obs = generate_observation("fetch_data", {}, result)
        assert "FAILED" in obs
        assert "Dataset XYZ not found" in obs
        assert "search_datasets" in obs

    def test_label_fallback_to_dataset_id(self):
        result = {"status": "success", "num_points": 5}
        obs = generate_observation("fetch_data", {"dataset_id": "MY_DS"}, result)
        assert "MY_DS" in obs


class TestSearchDatasetsObservation:
    def test_found_results(self):
        result = {"status": "success", "count": 5}
        obs = generate_observation("search_datasets", {"query": "ACE magnetic"}, result)
        assert "5" in obs
        assert "ACE magnetic" in obs

    def test_no_results(self):
        result = {"status": "success", "count": 0}
        obs = generate_observation("search_datasets", {"query": "nonexistent"}, result)
        assert "No datasets found" in obs

    def test_error(self):
        result = {"status": "error", "message": "timeout"}
        obs = generate_observation("search_datasets", {"query": "test"}, result)
        assert "FAILED" in obs
        assert "broader" in obs.lower() or "different" in obs.lower()


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
        assert "list_fetched_data" in obs


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
        assert "data_labels" in obs or "list_fetched_data" in obs


class TestListFetchedDataObservation:
    def test_with_entries(self):
        result = {"status": "success", "entries": [{"label": "a"}, {"label": "b"}]}
        obs = generate_observation("list_fetched_data", {}, result)
        assert "2" in obs

    def test_empty(self):
        result = {"status": "success", "entries": []}
        obs = generate_observation("list_fetched_data", {}, result)
        assert "No data" in obs


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


class TestSpiceObservation:
    def test_position(self):
        result = {"status": "success", "r_au": 0.071}
        obs = generate_observation(
            "get_spacecraft_position",
            {"spacecraft": "PSP", "observer": "SUN"},
            result,
        )
        assert "PSP" in obs
        assert "0.071" in obs
        assert "SUN" in obs

    def test_trajectory(self):
        result = {"status": "success", "num_points": 720}
        obs = generate_observation(
            "get_spacecraft_trajectory", {"spacecraft": "PSP"}, result
        )
        assert "PSP" in obs
        assert "720" in obs

    def test_distance(self):
        result = {"status": "success", "min_distance_au": 0.05}
        obs = generate_observation(
            "compute_distance",
            {"target1": "PSP", "target2": "SUN"},
            result,
        )
        assert "PSP" in obs
        assert "SUN" in obs
        assert "0.050" in obs

    def test_transform(self):
        result = {"status": "success"}
        obs = generate_observation(
            "transform_coordinates",
            {"from_frame": "RTN", "to_frame": "GSE"},
            result,
        )
        assert "RTN" in obs
        assert "GSE" in obs

    def test_list_missions(self):
        result = {"status": "success"}
        obs = generate_observation("list_spice_missions", {}, result)
        assert "mission" in obs.lower()


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
    def test_fetch_data_hint(self):
        result = {"status": "error", "message": "not found"}
        obs = generate_observation("fetch_data", {}, result)
        assert "search_datasets" in obs

    def test_run_code_hint(self):
        result = {"status": "error", "message": "syntax error"}
        obs = generate_observation("run_code", {}, result)
        assert "list_fetched_data" in obs

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
        obs = generate_observation("fetch_data", {}, result)
        assert isinstance(obs, str)
