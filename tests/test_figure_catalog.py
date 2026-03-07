"""Tests for figure catalog and disk fallback for insight agent."""

import time

import pytest


def test_find_latest_png_reads_from_disk(tmp_path):
    """After session reload, get_latest_figure_png should find PNGs on disk."""
    from agent.core import OrchestratorAgent

    session_dir = tmp_path / "sessions" / "test_session"
    mpl_dir = session_dir / "mpl_outputs"
    mpl_dir.mkdir(parents=True)

    fake_png = b"\x89PNG\r\n\x1a\nfake_image_data"
    (mpl_dir / "20260303_213236_a51849.png").write_bytes(fake_png)

    latest = OrchestratorAgent._find_latest_png(session_dir)
    assert latest is not None
    assert latest.read_bytes() == fake_png


def test_find_latest_png_prefers_newest(tmp_path):
    """When multiple PNGs exist, return the most recently modified one."""
    from agent.core import OrchestratorAgent

    session_dir = tmp_path / "sessions" / "test_session"
    mpl_dir = session_dir / "mpl_outputs"
    plotly_dir = session_dir / "plotly_outputs"
    mpl_dir.mkdir(parents=True)
    plotly_dir.mkdir(parents=True)

    old_png = mpl_dir / "old.png"
    old_png.write_bytes(b"\x89PNG\r\n\x1a\nold")

    time.sleep(0.05)

    new_png = plotly_dir / "new.png"
    new_png.write_bytes(b"\x89PNG\r\n\x1a\nnew")

    latest = OrchestratorAgent._find_latest_png(session_dir)
    assert latest == new_png


def test_find_latest_png_empty_session(tmp_path):
    """No PNG files → returns None."""
    from agent.core import OrchestratorAgent

    session_dir = tmp_path / "empty_session"
    session_dir.mkdir(parents=True)

    assert OrchestratorAgent._find_latest_png(session_dir) is None


def test_find_latest_png_ignores_empty_files(tmp_path):
    """Empty PNG files should be ignored."""
    from agent.core import OrchestratorAgent

    session_dir = tmp_path / "sessions" / "test_session"
    mpl_dir = session_dir / "mpl_outputs"
    mpl_dir.mkdir(parents=True)

    (mpl_dir / "empty.png").write_bytes(b"")
    (mpl_dir / "valid.png").write_bytes(b"\x89PNG\r\n\x1a\nvalid")

    latest = OrchestratorAgent._find_latest_png(session_dir)
    assert latest is not None
    assert latest.name == "valid.png"


def test_find_latest_png_ignores_non_png_files(tmp_path):
    """Non-PNG files should be ignored."""
    from agent.core import OrchestratorAgent

    session_dir = tmp_path / "sessions" / "test_session"
    mpl_dir = session_dir / "mpl_outputs"
    mpl_dir.mkdir(parents=True)

    (mpl_dir / "data.json").write_bytes(b'{"foo": "bar"}')
    (mpl_dir / "figure.png").write_bytes(b"\x89PNG\r\n\x1a\nreal png")

    latest = OrchestratorAgent._find_latest_png(session_dir)
    assert latest is not None
    assert latest.name == "figure.png"
