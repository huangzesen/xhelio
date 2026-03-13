"""Tests for agent.tool_handlers.figure — manage_figure handler."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from data_ops.store import DataStore
from data_ops.asset_registry import AssetRegistry


@pytest.fixture
def session_dir(tmp_path):
    d = tmp_path / "session_fig"
    d.mkdir()
    return d


@pytest.fixture
def data_store(session_dir):
    return DataStore(session_dir / "data")


@pytest.fixture
def registry(session_dir, data_store):
    return AssetRegistry(session_dir, data_store)


@pytest.fixture
def ctx(session_dir, data_store, registry):
    mock = MagicMock()
    mock.session_dir = session_dir
    mock.store = data_store
    mock.event_bus = MagicMock()
    mock.renderer = MagicMock()
    mock.asset_registry = registry
    mock.agent_state = {"orchestrator": MagicMock()}
    return mock


class TestManageFigureList:
    def test_list_empty(self, ctx):
        from agent.tool_handlers.figure import handle_manage_figure

        result = handle_manage_figure(ctx, {"action": "list"})
        assert result["status"] == "success"
        assert result["figures"] == []

    def test_list_after_register(self, ctx, registry):
        from agent.tool_handlers.figure import handle_manage_figure

        registry.register_figure(
            fig_json={}, trace_labels=["a"], panel_count=1, op_id="op_1",
        )
        result = handle_manage_figure(ctx, {"action": "list"})
        assert len(result["figures"]) == 1

    def test_list_filter_by_kind(self, ctx, registry):
        from agent.tool_handlers.figure import handle_manage_figure

        asset1 = registry.register_figure(
            fig_json={}, trace_labels=[], panel_count=1, op_id="op_1",
        )
        asset1.figure_kind = "plotly"
        asset2 = registry.register_image(name="Web", image_path="/tmp/img.png")
        asset2.figure_kind = "image"

        result = handle_manage_figure(ctx, {"action": "list", "figure_kind": "plotly"})
        assert len(result["figures"]) == 1
        assert result["figures"][0]["asset_id"] == "fig_001"


class TestManageFigureShow:
    def test_show_image(self, ctx, registry, session_dir):
        from agent.tool_handlers.figure import handle_manage_figure

        # Create a fake image file
        output_dir = session_dir / "mpl_outputs"
        output_dir.mkdir()
        img = output_dir / "test_123.png"
        img.write_bytes(b"fake png data")

        asset = registry.register_image(name="Test", image_path=str(img))

        result = handle_manage_figure(ctx, {"action": "show", "asset_id": asset.asset_id})
        assert result["status"] == "success"
        ctx.event_bus.emit.assert_called()

    def test_show_missing_asset(self, ctx):
        from agent.tool_handlers.figure import handle_manage_figure

        result = handle_manage_figure(ctx, {"action": "show", "asset_id": "fig_999"})
        assert result["status"] == "error"


class TestManageFigureDelete:
    def test_delete_removes_asset(self, ctx, registry):
        from agent.tool_handlers.figure import handle_manage_figure

        asset = registry.register_figure(
            fig_json={}, trace_labels=[], panel_count=1, op_id="op_1",
        )
        result = handle_manage_figure(ctx, {"action": "delete", "asset_id": asset.asset_id})
        assert result["status"] == "success"
        assert registry.get_asset(asset.asset_id) is None

    def test_delete_missing(self, ctx):
        from agent.tool_handlers.figure import handle_manage_figure

        result = handle_manage_figure(ctx, {"action": "delete", "asset_id": "fig_999"})
        assert result["status"] == "error"


class TestManageFigureRestore:
    def test_restore_delegates_to_session_persistence(self, ctx):
        from agent.tool_handlers.figure import handle_manage_figure

        with patch("agent.session_persistence.handle_restore_plot") as mock_restore:
            mock_restore.return_value = {"status": "success"}
            result = handle_manage_figure(ctx, {"action": "restore"})
            mock_restore.assert_called_once_with(ctx)
            assert result["status"] == "success"
