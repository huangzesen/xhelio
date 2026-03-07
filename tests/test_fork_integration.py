"""Integration test for the full /fork flow."""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agent.session import SessionManager


class TestForkIntegration:
    def test_full_fork_flow(self, tmp_path):
        """Test the complete fork flow: save -> copy -> resume with fresh chat."""
        sm = SessionManager(base_dir=tmp_path)

        # Create original session with history
        original_id = sm.create_session("gemini-test")
        original_dir = tmp_path / original_id

        # Simulate an Interactions API session: history has _interaction_id + _client_history
        client_history = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Show me solar wind data"}],
            },
            {
                "role": "model",
                "content": [{"type": "text", "text": "I'll fetch ACE data for you."}],
            },
            {
                "role": "model",
                "content": [
                    {
                        "type": "function_call",
                        "id": "call_1",
                        "name": "fetch_data",
                        "arguments": {"dataset": "AC_H0_MFI"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "function_result",
                        "call_id": "call_1",
                        "result": '{"status": "ok"}',
                        "name": "fetch_data",
                    }
                ],
            },
            {
                "role": "model",
                "content": [
                    {"type": "text", "text": "Here's the solar wind data plot."}
                ],
            },
        ]

        history = [
            {"_interaction_id": "int_original_abc"},
            {"_client_history": client_history},
        ]
        sm.save_session(
            session_id=original_id,
            chat_history=history,
            data_store=None,
            metadata_updates={
                "turn_count": 2,
                "interaction_id": "int_original_abc",
            },
        )

        # Fork: copy directory
        new_id = sm.generate_session_id()
        dst_dir = tmp_path / new_id
        shutil.copytree(str(original_dir), str(dst_dir))

        # Update metadata: clear interaction_id, add fork lineage
        meta_path = dst_dir / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["id"] = new_id
        meta["forked_from"] = original_id
        meta.pop("interaction_id", None)
        meta_path.write_text(json.dumps(meta, indent=2))

        # Load the forked session
        loaded = sm.load_session(new_id)
        loaded_history = loaded[0]
        loaded_meta = loaded[2]

        # Verify: no interaction_id in metadata (fresh chat)
        assert "interaction_id" not in loaded_meta

        # Verify: client history is present for seeding
        ch_entry = next(
            (
                e
                for e in loaded_history
                if isinstance(e, dict) and "_client_history" in e
            ),
            None,
        )
        assert ch_entry is not None
        assert len(ch_entry["_client_history"]) == 5  # all turns preserved

        # Verify: fork lineage recorded
        assert loaded_meta["forked_from"] == original_id
