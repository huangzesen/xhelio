"""Tests for manage_figure handler — save_from_url and show actions.

Migrated from test_save_figure.py (handle_save_figure, handle_show_figure)
to use handle_manage_figure with action-based dispatch.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

import pytest


@dataclass
class FakeOrchestratorState:
    eureka_hooks: object = None


@dataclass
class FakeToolContext:
    session_dir: Path = None
    store: object = None
    renderer: object = None
    event_bus: object = None
    asset_registry: object = None
    agent_state: dict = field(default_factory=dict)
    web_mode: bool = False
    mcp_client: object = None


def _make_ctx(tmp_path):
    """Create a minimal ToolContext for testing."""
    from data_ops.store import DataStore
    from data_ops.asset_registry import AssetRegistry

    session_dir = tmp_path / "session_test"
    session_dir.mkdir()
    store = DataStore(session_dir / "data")
    ar = AssetRegistry(session_dir, store)
    orch = FakeOrchestratorState()

    ctx = FakeToolContext(
        session_dir=session_dir,
        store=store,
        asset_registry=ar,
        agent_state={"orchestrator": orch},
    )
    return ctx


class TestSaveFromUrlValidation:
    def test_missing_url(self, tmp_path):
        from agent.tool_handlers.figure import handle_manage_figure

        ctx = _make_ctx(tmp_path)
        result = handle_manage_figure(ctx, {"action": "save_from_url", "name": "test"})
        assert result["status"] == "error"
        assert "url" in result["message"].lower()

    def test_missing_name(self, tmp_path):
        from agent.tool_handlers.figure import handle_manage_figure

        ctx = _make_ctx(tmp_path)
        result = handle_manage_figure(ctx, {"action": "save_from_url", "url": "https://example.com/img.png"})
        assert result["status"] == "error"
        assert "name" in result["message"].lower()

    def test_invalid_scheme_file(self, tmp_path):
        from agent.tool_handlers.figure import handle_manage_figure

        ctx = _make_ctx(tmp_path)
        result = handle_manage_figure(ctx, {"action": "save_from_url", "url": "file:///etc/passwd", "name": "hack"})
        assert result["status"] == "error"

    def test_invalid_scheme_ftp(self, tmp_path):
        from agent.tool_handlers.figure import handle_manage_figure

        ctx = _make_ctx(tmp_path)
        result = handle_manage_figure(ctx, {"action": "save_from_url", "url": "ftp://example.com/img.png", "name": "ftp"})
        assert result["status"] == "error"

    def test_private_ip_blocked(self, tmp_path):
        from agent.tool_handlers.figure import handle_manage_figure

        ctx = _make_ctx(tmp_path)
        result = handle_manage_figure(ctx, {"action": "save_from_url", "url": "http://169.254.169.254/latest/meta-data", "name": "metadata"})
        assert result["status"] == "error"


class TestSaveFromUrlDownload:
    @patch("requests.get")
    def test_successful_download(self, mock_get, tmp_path):
        from agent.tool_handlers.figure import handle_manage_figure

        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "image/png", "Content-Length": str(len(png_bytes))}
        mock_resp.content = png_bytes
        mock_get.return_value = mock_resp

        ctx = _make_ctx(tmp_path)
        result = handle_manage_figure(ctx, {
            "action": "save_from_url",
            "url": "https://example.com/image.png",
            "name": "Test Image",
        })

        assert result["status"] == "success"
        assert result["asset_id"] == "fig_001"
        assert "script_id" in result
        output_path = ctx.session_dir / "mpl_outputs" / f"{result['script_id']}.png"
        assert output_path.exists()
        assert output_path.read_bytes() == png_bytes

    @patch("requests.get")
    def test_invalid_content_type(self, mock_get, tmp_path):
        from agent.tool_handlers.figure import handle_manage_figure

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.content = b"<html>not an image</html>"
        mock_get.return_value = mock_resp

        ctx = _make_ctx(tmp_path)
        result = handle_manage_figure(ctx, {
            "action": "save_from_url",
            "url": "https://example.com/page.html",
            "name": "NotImage",
        })
        assert result["status"] == "error"

    @patch("requests.get")
    def test_http_error_status(self, mock_get, tmp_path):
        import requests as _requests
        from agent.tool_handlers.figure import handle_manage_figure

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = _requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_resp

        ctx = _make_ctx(tmp_path)
        result = handle_manage_figure(ctx, {
            "action": "save_from_url",
            "url": "https://example.com/missing.png",
            "name": "Missing",
        })
        assert result["status"] == "error"


class TestShowFigure:
    def test_missing_asset_id(self, tmp_path):
        from agent.tool_handlers.figure import handle_manage_figure

        ctx = _make_ctx(tmp_path)
        result = handle_manage_figure(ctx, {"action": "show"})
        assert result["status"] == "error"

    def test_asset_not_found(self, tmp_path):
        from agent.tool_handlers.figure import handle_manage_figure

        ctx = _make_ctx(tmp_path)
        result = handle_manage_figure(ctx, {"action": "show", "asset_id": "fig_999"})
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    def test_show_registered_image(self, tmp_path):
        from agent.tool_handlers.figure import handle_manage_figure

        ctx = _make_ctx(tmp_path)
        output_dir = ctx.session_dir / "mpl_outputs"
        output_dir.mkdir()
        png_path = output_dir / "20260312_120000_abc123.png"
        png_path.write_bytes(b"fake png")
        ctx.asset_registry.register_image(
            name="Test",
            image_path=str(png_path),
            source_url="https://example.com/img.png",
        )

        result = handle_manage_figure(ctx, {"action": "show", "asset_id": "fig_001"})
        assert result["status"] == "success"
        assert result["script_id"] == "20260312_120000_abc123"
