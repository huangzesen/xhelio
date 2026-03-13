"""Tests for visualization tool handlers in agent/tool_handlers/visualization.py."""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Import verification
# ---------------------------------------------------------------------------


def test_import_all_handlers():
    """All handler functions can be imported from the module."""
    from agent.tool_handlers.visualization import (
        handle_render_plotly_json,
        handle_manage_plot,
        handle_generate_mpl_script,
        handle_manage_mpl_output,
        handle_generate_jsx_component,
        handle_manage_jsx_output,
    )
    assert callable(handle_render_plotly_json)
    assert callable(handle_manage_plot)
    assert callable(handle_generate_mpl_script)
    assert callable(handle_manage_mpl_output)
    assert callable(handle_generate_jsx_component)
    assert callable(handle_manage_jsx_output)


# ---------------------------------------------------------------------------
# Mock orchestrator helper
# ---------------------------------------------------------------------------


def _make_orch():
    """Return a minimal mock ToolContext (used to be OrchestratorAgent)."""
    orch = MagicMock()
    orch.gui_mode = False
    orch.web_mode = False
    # Handlers now use ctx.store, ctx.renderer, ctx.event_bus (not _store, _renderer, _event_bus)
    return orch


# ---------------------------------------------------------------------------
# handle_render_plotly_json — error cases
# ---------------------------------------------------------------------------


def test_render_plotly_json_empty_data():
    """Returns error when figure_json.data is empty or missing."""
    from agent.tool_handlers.visualization import handle_render_plotly_json

    orch = _make_orch()

    # No figure_json at all
    result = handle_render_plotly_json(orch, {})
    assert result["status"] == "error"
    assert "figure_json.data" in result["message"]

    # Empty data array
    result = handle_render_plotly_json(orch, {"figure_json": {"data": []}})
    assert result["status"] == "error"
    assert "figure_json.data" in result["message"]


def test_render_plotly_json_too_many_layout_objects():
    """Returns error when shapes + annotations exceed limit."""
    from agent.tool_handlers.visualization import handle_render_plotly_json

    orch = _make_orch()
    # 31 shapes should exceed the limit of 30
    fig_json = {
        "data": [{"type": "scatter", "data_label": "some_label"}],
        "layout": {
            "shapes": [{"type": "line"}] * 31,
            "annotations": [],
        },
    }
    result = handle_render_plotly_json(orch, {"figure_json": fig_json})
    assert result["status"] == "error"
    assert "Too many layout objects" in result["message"]


def test_render_plotly_json_label_not_found():
    """Returns error when data_label is not in the store."""
    from agent.tool_handlers.visualization import handle_render_plotly_json

    orch = _make_orch()
    orch.store.list_entries.return_value = []

    # resolve_entry returns (None, None) when label is not found
    with patch("agent.tool_handlers.visualization.resolve_entry", return_value=(None, None)):
        fig_json = {
            "data": [{"type": "scatter", "data_label": "MISSING.Label"}],
        }
        result = handle_render_plotly_json(orch, {"figure_json": fig_json})

    assert result["status"] == "error"
    assert "MISSING.Label" in result["message"]
    assert "not found" in result["message"]


# ---------------------------------------------------------------------------
# handle_manage_plot — error cases
# ---------------------------------------------------------------------------


def test_manage_plot_missing_action():
    """Returns error when action is not provided."""
    from agent.tool_handlers.visualization import handle_manage_plot

    orch = _make_orch()
    result = handle_manage_plot(orch, {})
    assert result["status"] == "error"
    assert "action is required" in result["message"]


def test_manage_plot_unknown_action():
    """Returns error for an unrecognised action."""
    from agent.tool_handlers.visualization import handle_manage_plot

    orch = _make_orch()
    result = handle_manage_plot(orch, {"action": "fly_to_the_moon"})
    assert result["status"] == "error"
    assert "Unknown action" in result["message"]


def test_manage_plot_reset():
    """Delegates reset to renderer and emits event."""
    from agent.tool_handlers.visualization import handle_manage_plot

    orch = _make_orch()
    orch.renderer.reset.return_value = {"status": "success"}
    result = handle_manage_plot(orch, {"action": "reset"})
    assert result["status"] == "success"
    orch.renderer.reset.assert_called_once()
    orch.event_bus.emit.assert_called_once()


def test_manage_plot_get_state():
    """Delegates get_state to renderer."""
    from agent.tool_handlers.visualization import handle_manage_plot

    orch = _make_orch()
    orch.renderer.get_current_state.return_value = {"status": "success", "state": {}}
    result = handle_manage_plot(orch, {"action": "get_state"})
    assert result["status"] == "success"
    orch.renderer.get_current_state.assert_called_once()


# ---------------------------------------------------------------------------
# handle_generate_mpl_script — error cases
# ---------------------------------------------------------------------------


def test_generate_mpl_script_missing_script():
    """Returns error when script parameter is absent."""
    from agent.tool_handlers.visualization import handle_generate_mpl_script

    orch = _make_orch()
    result = handle_generate_mpl_script(orch, {})
    assert result["status"] == "error"
    assert "script parameter is required" in result["message"]


# ---------------------------------------------------------------------------
# handle_manage_mpl_output — error cases
# ---------------------------------------------------------------------------


def test_manage_mpl_output_missing_action():
    """Returns error when action is not provided."""
    from agent.tool_handlers.visualization import handle_manage_mpl_output

    orch = _make_orch()
    result = handle_manage_mpl_output(orch, {})
    assert result["status"] == "error"
    assert "action is required" in result["message"]


def test_manage_mpl_output_unknown_action():
    """Returns error for unknown action."""
    from agent.tool_handlers.visualization import handle_manage_mpl_output

    orch = _make_orch()
    result = handle_manage_mpl_output(orch, {"action": "unknown_action"})
    assert result["status"] == "error"
    assert "Unknown action" in result["message"]


# ---------------------------------------------------------------------------
# handle_generate_jsx_component — error cases
# ---------------------------------------------------------------------------


def test_generate_jsx_component_missing_code():
    """Returns error when code parameter is absent."""
    from agent.tool_handlers.visualization import handle_generate_jsx_component

    orch = _make_orch()
    result = handle_generate_jsx_component(orch, {})
    assert result["status"] == "error"
    assert "code parameter is required" in result["message"]


# ---------------------------------------------------------------------------
# handle_manage_jsx_output — error cases
# ---------------------------------------------------------------------------


def test_manage_jsx_output_missing_action():
    """Returns error when action is not provided."""
    from agent.tool_handlers.visualization import handle_manage_jsx_output

    orch = _make_orch()
    result = handle_manage_jsx_output(orch, {})
    assert result["status"] == "error"
    assert "action is required" in result["message"]


def test_manage_jsx_output_unknown_action():
    """Returns error for unknown action."""
    from agent.tool_handlers.visualization import handle_manage_jsx_output

    orch = _make_orch()
    result = handle_manage_jsx_output(orch, {"action": "unknown_action"})
    assert result["status"] == "error"
    assert "Unknown action" in result["message"]
