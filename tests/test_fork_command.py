"""Tests for the /fork command."""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agent.session import SessionManager


class TestForkSessionManager:
    def test_fork_creates_new_session_directory(self, tmp_path):
        """Forking should create a new session directory with copied data."""
        sm = SessionManager(base_dir=tmp_path)
        original_id = sm.create_session("test-model")

        # Add some data to the original session
        data_dir = tmp_path / original_id / "data"
        (data_dir / "test.txt").write_text("test data")

        # Fork
        new_id = sm.generate_session_id()
        src_dir = tmp_path / original_id
        dst_dir = tmp_path / new_id
        shutil.copytree(str(src_dir), str(dst_dir))

        assert dst_dir.exists()
        assert (dst_dir / "data" / "test.txt").read_text() == "test data"

    def test_fork_metadata_includes_fork_origin(self, tmp_path):
        """Forked session metadata should record where it came from."""
        sm = SessionManager(base_dir=tmp_path)
        original_id = sm.create_session("test-model")

        new_id = sm.generate_session_id()
        src_dir = tmp_path / original_id
        dst_dir = tmp_path / new_id
        shutil.copytree(str(src_dir), str(dst_dir))

        # Update metadata with fork info
        meta_path = dst_dir / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["id"] = new_id
        meta["forked_from"] = original_id
        meta_path.write_text(json.dumps(meta, indent=2))

        loaded_meta = json.loads(meta_path.read_text())
        assert loaded_meta["forked_from"] == original_id
        assert loaded_meta["id"] == new_id

    def test_fork_session_id_format(self, tmp_path):
        """Forked session should use proper YYYYMMDD_HHMMSS_XXXXXXXX format."""
        sm = SessionManager(base_dir=tmp_path)
        new_id = sm.generate_session_id()
        import re

        assert re.match(r"\d{8}_\d{6}_[0-9a-f]{8}", new_id)
