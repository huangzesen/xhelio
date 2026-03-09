"""Tests for envoy_query tool handler."""
import re
import pytest
from unittest.mock import patch, MagicMock


def _make_cdaweb_mission():
    """Minimal CDAWeb mission JSON (PSP-like)."""
    return {
        "id": "PSP",
        "name": "Parker Solar Probe",
        "keywords": ["parker", "psp", "solar"],
        "profile": {
            "description": "Parker Solar Probe spacecraft data from CDAWeb.",
        },
        "instruments": {
            "FIELDS/MAG": {
                "name": "FIELDS/MAG",
                "keywords": ["magnetic", "field", "mag"],
                "datasets": {
                    "PSP_FLD_L2_MAG_RTN_1MIN": {
                        "description": "PSP FIELDS 1 minute cadence Fluxgate Magnetometer (MAG) data in RTN coordinates",
                        "start_date": "2018-10-02T03:48:30.000Z",
                        "stop_date": "2025-07-31T23:59:30.000Z",
                        "pi_name": "Stuart D. Bale",
                    },
                    "PSP_FLD_L2_MAG_SC_4SA": {
                        "description": "PSP FIELDS Level 2 Magnetometer SC 4 Samples/Cycle",
                        "start_date": "2018-10-02T00:00:00.000Z",
                        "stop_date": "2025-07-31T23:59:59.000Z",
                    },
                },
            },
            "SWEAP": {
                "name": "SWEAP",
                "keywords": ["solar", "wind", "plasma"],
                "datasets": {
                    "PSP_SWP_SPC_L3I": {
                        "description": "PSP SWEAP SPC Level 3 Ion Moments",
                        "start_date": "2018-11-01T00:00:00.000Z",
                        "stop_date": "2025-06-30T23:59:59.000Z",
                    },
                },
            },
        },
    }


@pytest.fixture
def mock_missions():
    missions = {
        "PSP": _make_cdaweb_mission(),
    }
    with patch("agent.tool_handlers.envoy_query.get_all_missions", return_value=missions):
        with patch("agent.tool_handlers.envoy_query.load_mission_json", side_effect=lambda mid: missions[mid]):
            with patch("agent.tool_handlers.envoy_query.get_envoy_kind", return_value="cdaweb"):
                yield missions


@pytest.fixture
def mock_orch():
    orch = MagicMock()
    orch._event_bus = MagicMock()
    return orch


def test_list_all_envoys(mock_orch, mock_missions):
    from agent.tool_handlers.envoy_query import handle_envoy_query
    result = handle_envoy_query(mock_orch, {})
    assert result["status"] == "success"
    assert result["count"] == 1
    envoys = {e["id"]: e for e in result["envoys"]}
    assert "PSP" in envoys
    assert envoys["PSP"]["name"] == "Parker Solar Probe"
    assert "description" in envoys["PSP"]


def test_navigate_envoy_top_level(mock_orch, mock_missions):
    from agent.tool_handlers.envoy_query import handle_envoy_query
    result = handle_envoy_query(mock_orch, {"envoy": "PSP"})
    assert result["status"] == "success"
    assert result["id"] == "PSP"
    assert "catalog" in result
    assert "FIELDS/MAG" in result["catalog"]


def test_navigate_envoy_not_found(mock_orch, mock_missions):
    from agent.tool_handlers.envoy_query import handle_envoy_query
    result = handle_envoy_query(mock_orch, {"envoy": "NONEXISTENT"})
    assert result["status"] == "error"


def test_navigate_instruments(mock_orch, mock_missions):
    from agent.tool_handlers.envoy_query import handle_envoy_query
    result = handle_envoy_query(mock_orch, {"envoy": "PSP", "path": "instruments"})
    assert result["status"] == "success"
    children = result["children"]
    assert "FIELDS/MAG" in children
    assert "SWEAP" in children
    assert children["FIELDS/MAG"]["dataset_count"] == 2


def test_navigate_instrument_detail(mock_orch, mock_missions):
    from agent.tool_handlers.envoy_query import handle_envoy_query
    result = handle_envoy_query(mock_orch, {"envoy": "PSP", "path": "instruments.FIELDS/MAG"})
    assert result["status"] == "success"
    assert result["name"] == "FIELDS/MAG"
    assert "children" in result
    assert "datasets" in result["children"]


def test_navigate_datasets(mock_orch, mock_missions):
    from agent.tool_handlers.envoy_query import handle_envoy_query
    result = handle_envoy_query(mock_orch, {"envoy": "PSP", "path": "instruments.FIELDS/MAG.datasets"})
    assert result["status"] == "success"
    children = result["children"]
    assert "PSP_FLD_L2_MAG_RTN_1MIN" in children
    assert "description" in children["PSP_FLD_L2_MAG_RTN_1MIN"]


def test_navigate_single_dataset(mock_orch, mock_missions):
    from agent.tool_handlers.envoy_query import handle_envoy_query
    result = handle_envoy_query(mock_orch, {
        "envoy": "PSP",
        "path": "instruments.FIELDS/MAG.datasets.PSP_FLD_L2_MAG_RTN_1MIN",
    })
    assert result["status"] == "success"
    assert result["description"].startswith("PSP FIELDS 1 minute")
    assert "start_date" in result
    assert "stop_date" in result


def test_navigate_invalid_path(mock_orch, mock_missions):
    from agent.tool_handlers.envoy_query import handle_envoy_query
    result = handle_envoy_query(mock_orch, {"envoy": "PSP", "path": "instruments.NONEXISTENT"})
    assert result["status"] == "error"
    assert "not found" in result["message"].lower() or "invalid" in result["message"].lower()


def test_search_all_envoys(mock_orch, mock_missions):
    from agent.tool_handlers.envoy_query import handle_envoy_query
    result = handle_envoy_query(mock_orch, {"search": "(?i)magnetometer"})
    assert result["status"] == "success"
    assert len(result["matches"]) > 0
    envoys_found = {m["envoy"] for m in result["matches"]}
    assert "PSP" in envoys_found


def test_search_scoped_to_envoy(mock_orch, mock_missions):
    from agent.tool_handlers.envoy_query import handle_envoy_query
    result = handle_envoy_query(mock_orch, {"envoy": "PSP", "search": "(?i)sweap.*ion"})
    assert result["status"] == "success"
    assert len(result["matches"]) > 0
    assert all(m["envoy"] == "PSP" for m in result["matches"])


def test_search_no_matches(mock_orch, mock_missions):
    from agent.tool_handlers.envoy_query import handle_envoy_query
    result = handle_envoy_query(mock_orch, {"search": "zzz_no_match_zzz"})
    assert result["status"] == "success"
    assert result["count"] == 0


def test_search_invalid_regex(mock_orch, mock_missions):
    from agent.tool_handlers.envoy_query import handle_envoy_query
    result = handle_envoy_query(mock_orch, {"search": "[invalid"})
    assert result["status"] == "error"
    assert "regex" in result["message"].lower() or "pattern" in result["message"].lower()


def test_list_envoys_shows_correct_kind(mock_orch, mock_missions):
    from agent.tool_handlers.envoy_query import handle_envoy_query

    with patch("agent.tool_handlers.envoy_query.get_envoy_kind", return_value="cdaweb"):
        result = handle_envoy_query(mock_orch, {})

    envoys = {e["id"]: e for e in result["envoys"]}
    assert envoys["PSP"]["kind"] == "cdaweb"


def test_dataset_leaf_fetches_parameters(mock_orch, mock_missions):
    from agent.tool_handlers.envoy_query import handle_envoy_query
    mock_params = [
        {"name": "B_RTN", "description": "Magnetic field RTN", "units": "nT", "size": [3]},
        {"name": "B_MAG", "description": "Magnetic field magnitude", "units": "nT", "size": [1]},
    ]
    with patch("agent.tool_handlers.envoy_query.fetch_parameters", return_value=mock_params):
        result = handle_envoy_query(mock_orch, {
            "envoy": "PSP",
            "path": "instruments.FIELDS/MAG.datasets.PSP_FLD_L2_MAG_RTN_1MIN",
        })
    assert result["status"] == "success"
    assert "parameters" in result
    assert len(result["parameters"]) == 2
