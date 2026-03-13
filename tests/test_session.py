"""
Tests for agent.session — SessionManager and DataStore persistence.

Run with: python -m pytest tests/test_session.py -v
"""

import json
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.session import SessionManager
from data_ops.store import DataEntry, DataStore


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for session storage."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sm(tmp_dir):
    """Provide a SessionManager backed by a temp directory."""
    return SessionManager(base_dir=tmp_dir)


def _make_store_with_entries(data_dir: Path) -> DataStore:
    """Create a DataStore with sample scalar and vector entries."""
    store = DataStore(data_dir)
    idx = pd.date_range("2024-01-01", periods=100, freq="1min")

    scalar_df = pd.DataFrame(np.random.randn(100), index=idx, columns=["value"])
    store.put(
        DataEntry(
            label="AC_H2_MFI.Magnitude",
            data=scalar_df,
            units="nT",
            description="Magnetic field magnitude",
            source="cdf",
        )
    )

    vector_df = pd.DataFrame(
        np.random.randn(100, 3), index=idx, columns=["Bx", "By", "Bz"]
    )
    store.put(
        DataEntry(
            label="AC_H2_MFI.BGSEc",
            data=vector_df,
            units="nT",
            description="Magnetic field vector",
            source="cdf",
        )
    )

    return store


class TestSessionManager:
    def test_generate_session_id_format(self, sm):
        """generate_session_id returns YYYYMMDD_HHMMSS_XXXXXXXX format."""
        sid = sm.generate_session_id()
        assert re.match(r"^\d{8}_\d{6}_[0-9a-f]{8}$", sid)

    def test_generate_session_id_no_files(self, sm, tmp_dir):
        """generate_session_id creates no files or directories."""
        sid = sm.generate_session_id()
        session_dir = tmp_dir / sid
        assert not session_dir.exists()

    def test_generate_session_id_unique(self, sm):
        """Two calls produce different IDs (UUID suffix guarantees uniqueness)."""
        sid1 = sm.generate_session_id()
        sid2 = sm.generate_session_id()
        assert sid1 != sid2

    def test_list_sessions_sorted(self, sm, tmp_dir):
        """list_sessions returns sorted by updated_at descending."""
        sid1 = sm.generate_session_id()
        store1 = DataStore(tmp_dir / f"{sid1}_data")
        sm.save_session(sid1, [], store1, {"model": "model-a"})
        time.sleep(0.05)  # ensure distinct timestamps
        sid2 = sm.generate_session_id()
        store2 = DataStore(tmp_dir / f"{sid2}_data")
        sm.save_session(sid2, [], store2, {"model": "model-b"})

        sessions = sm.list_sessions()
        assert len(sessions) == 2
        # Most recent first
        assert sessions[0]["id"] == sid2
        assert sessions[1]["id"] == sid1

    def test_delete_session(self, sm, tmp_dir):
        """delete_session removes the directory."""
        sid = sm.generate_session_id()
        store = DataStore(tmp_dir / f"{sid}_data")
        sm.save_session(sid, [], store)
        assert (tmp_dir / sid).exists()

        result = sm.delete_session(sid)
        assert result is True
        assert not (tmp_dir / sid).exists()

        # Deleting non-existent returns False
        assert sm.delete_session("nonexistent") is False

    def test_get_most_recent_session(self, sm, tmp_dir):
        """get_most_recent_session returns the latest session."""
        sid1 = sm.generate_session_id()
        store1 = DataStore(tmp_dir / f"{sid1}_data")
        sm.save_session(sid1, [], store1)
        time.sleep(0.05)
        sid2 = sm.generate_session_id()
        store2 = DataStore(tmp_dir / f"{sid2}_data")
        sm.save_session(sid2, [], store2)

        assert sm.get_most_recent_session() == sid2

    def test_get_most_recent_empty(self, sm):
        """get_most_recent_session returns None when no sessions exist."""
        assert sm.get_most_recent_session() is None

    def test_save_and_load_history(self, sm, tmp_dir):
        """Round-trip Content dicts through JSON."""
        sid = sm.generate_session_id()
        store = DataStore(tmp_dir / f"{sid}_data")

        history = [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi there!"}]},
            {"role": "user", "parts": [{"text": "Show me ACE data"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "name": "search_datasets",
                            "args": {"query": "ACE"},
                        }
                    },
                ],
            },
        ]

        sm.save_session(sid, history, store, {"turn_count": 2})

        loaded_history, data_dir, meta, _fig, _ops, _dlog, _elog = sm.load_session(sid)
        assert len(loaded_history) == 4
        assert loaded_history[0]["role"] == "user"
        assert loaded_history[0]["parts"][0]["text"] == "Hello"
        assert (
            loaded_history[3]["parts"][0]["function_call"]["name"] == "search_datasets"
        )
        assert meta["turn_count"] == 2

    def test_save_and_load_dataframes(self, sm, tmp_dir):
        """Round-trip DataFrames through the disk-backed DataStore."""
        sid = sm.generate_session_id()
        session_dir = sm.base_dir / sid
        session_dir.mkdir(parents=True, exist_ok=True)
        data_dir = session_dir / "data"
        store = _make_store_with_entries(data_dir)

        # Save
        sm.save_session(sid, [], store)

        # Load into a fresh store (from the same data_dir)
        _, loaded_data_dir, _, _, _, _, _ = sm.load_session(sid)
        restored_store = DataStore(loaded_data_dir)

        assert len(restored_store) == 2

        # Check scalar entry
        orig = store.get("AC_H2_MFI.Magnitude")
        rest = restored_store.get("AC_H2_MFI.Magnitude")
        assert rest is not None
        assert rest.units == orig.units
        assert rest.description == orig.description
        assert rest.source == orig.source
        pd.testing.assert_frame_equal(orig.data, rest.data, check_freq=False)

        # Check vector entry
        orig_v = store.get("AC_H2_MFI.BGSEc")
        rest_v = restored_store.get("AC_H2_MFI.BGSEc")
        assert rest_v is not None
        pd.testing.assert_frame_equal(orig_v.data, rest_v.data, check_freq=False)

    def test_save_updates_metadata(self, sm, tmp_dir):
        """save_session merges metadata_updates into metadata.json."""
        sid = sm.generate_session_id()
        store = DataStore(tmp_dir / f"{sid}_data")

        sm.save_session(
            sid,
            [],
            store,
            {
                "turn_count": 5,
                "last_message_preview": "Show me data",
                "token_usage": {"input_tokens": 100, "output_tokens": 50},
            },
        )

        _, _, meta, _, _, _, _ = sm.load_session(sid)
        assert meta["turn_count"] == 5
        assert meta["last_message_preview"] == "Show me data"
        assert meta["token_usage"]["input_tokens"] == 100

    def test_thinking_tokens_in_metadata(self, sm, tmp_dir):
        """thinking_tokens appears in initial and updated session metadata."""
        sid = sm.generate_session_id()
        store = DataStore(tmp_dir / f"{sid}_data")

        # First save creates initial metadata with thinking_tokens
        sm.save_session(sid, [], store)
        _, _, meta_init, _, _, _, _ = sm.load_session(sid)
        assert "thinking_tokens" in meta_init["token_usage"]
        assert meta_init["token_usage"]["thinking_tokens"] == 0

        # Save with thinking_tokens
        sm.save_session(
            sid,
            [],
            store,
            {
                "token_usage": {
                    "input_tokens": 200,
                    "output_tokens": 80,
                    "thinking_tokens": 150,
                    "api_calls": 3,
                },
            },
        )
        _, _, meta, _, _, _, _ = sm.load_session(sid)
        assert meta["token_usage"]["thinking_tokens"] == 150

    def test_save_and_load_display_log(self, sm, tmp_dir):
        """Round-trip display_log through JSON."""
        sid = sm.generate_session_id()
        store = DataStore(tmp_dir / f"{sid}_data")

        display_log = [
            {
                "user": "User",
                "avatar": "👤",
                "content": "Hello",
                "timestamp": "2024-01-01T00:00:00",
            },
            {
                "user": "Helio Agent",
                "avatar": "⚙️",
                "content": "Thinking...",
                "timestamp": "2024-01-01T00:00:01",
            },
            {
                "user": "Helio Agent",
                "avatar": "🤖",
                "content": "Hi there!",
                "timestamp": "2024-01-01T00:00:02",
            },
        ]

        sm.save_session(sid, [], store, display_log=display_log)

        _, _, _, _, _, loaded_log, _ = sm.load_session(sid)
        assert loaded_log is not None
        assert len(loaded_log) == 3
        assert loaded_log[0]["user"] == "User"
        assert loaded_log[0]["content"] == "Hello"
        assert loaded_log[1]["avatar"] == "⚙️"
        assert loaded_log[2]["user"] == "Helio Agent"
        assert loaded_log[2]["timestamp"] == "2024-01-01T00:00:02"

    def test_load_session_without_display_log(self, sm, tmp_dir):
        """load_session returns None for display_log when not saved."""
        sid = sm.generate_session_id()
        store = DataStore(tmp_dir / f"{sid}_data")

        sm.save_session(sid, [], store)

        _, _, _, _, _, loaded_log, _ = sm.load_session(sid)
        assert loaded_log is None

    def test_load_nonexistent_raises(self, sm):
        """load_session raises FileNotFoundError for missing session."""
        with pytest.raises(FileNotFoundError):
            sm.load_session("nonexistent_session_id")

    def test_save_nonexistent_creates_dir(self, sm, tmp_dir):
        """save_session auto-creates session dir if missing."""
        store = DataStore(tmp_dir / "auto_data")
        sm.save_session("auto_created", [], store)
        assert (sm.base_dir / "auto_created").exists()
        assert (sm.base_dir / "auto_created" / "metadata.json").exists()

    def test_get_tmp_dir(self, sm):
        """Session tmp dir is created on demand and scoped to session."""
        session_id = sm.create_session()
        tmp_dir = sm.get_tmp_dir(session_id)
        assert tmp_dir.exists()
        assert tmp_dir.name == "tmp"
        assert tmp_dir.parent.name == session_id
        # Calling again returns same path
        assert sm.get_tmp_dir(session_id) == tmp_dir

    def test_first_save_creates_initial_metadata(self, sm, tmp_dir):
        """First save_session creates metadata.json with default fields."""
        sid = sm.generate_session_id()
        store = DataStore(tmp_dir / f"{sid}_data")
        sm.save_session(sid, [], store)

        session_dir = tmp_dir / sid
        assert session_dir.exists()
        assert (session_dir / "metadata.json").exists()
        assert (session_dir / "data").is_dir()

        with open(session_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["id"] == sid
        assert meta["turn_count"] == 0


class TestDataStorePersistence:
    def test_save_empty_store(self, tmp_dir):
        """An empty store has no _labels.json until something is put."""
        data_dir = tmp_dir / "data"
        store = DataStore(data_dir)
        # Empty store — _labels.json doesn't exist yet (no entries to index)
        assert len(store) == 0

    def test_load_empty_directory(self, tmp_dir):
        """Opening a store at an empty directory starts with 0 entries."""
        store = DataStore(tmp_dir / "data")
        assert len(store) == 0

    def test_label_with_special_chars(self, tmp_dir):
        """Labels with special characters are handled via hash-based folders."""
        store = DataStore(tmp_dir / "data")
        idx = pd.date_range("2024-01-01", periods=10, freq="1s")
        df = pd.DataFrame(np.random.randn(10), index=idx, columns=["v"])
        store.put(
            DataEntry(
                label="DS/PARAM:special*chars",
                data=df,
                units="",
                source="computed",
            )
        )

        assert store.has("DS/PARAM:special*chars")

        # Round-trip via new DataStore instance
        store2 = DataStore(tmp_dir / "data")
        assert len(store2) == 1
        entry = store2.get("DS/PARAM:special*chars")
        assert entry is not None
        pd.testing.assert_frame_equal(df, entry.data, check_freq=False)

    def test_corrupted_labels_json(self, tmp_dir):
        """Corrupted _labels.json causes json.JSONDecodeError on construction."""
        data_dir = tmp_dir / "data"
        data_dir.mkdir()
        with open(data_dir / "_labels.json", "w") as f:
            f.write("{corrupted json")

        with pytest.raises(json.JSONDecodeError):
            DataStore(data_dir)


class TestParseChatEntries:
    """Tests for agent.logging.parse_chat_entries()."""

    def test_basic_user_and_agent(self, tmp_dir):
        """Parse a simple [User] + [Agent] pair (new 6-field format)."""
        from agent.logging import parse_chat_entries

        log = tmp_dir / "agent.log"
        log.write_text(
            "2026-02-17 10:00:00 | INFO     | helio-agent | sess1 |  | [User] Hello\n"
            "2026-02-17 10:00:05 | INFO     | helio-agent | sess1 |  | [Agent] Hi there!\n"
        )
        entries = parse_chat_entries(log, "sess1")
        assert len(entries) == 2
        assert entries[0] == {
            "role": "user",
            "content": "Hello",
            "timestamp": "2026-02-17 10:00:00",
        }
        assert entries[1] == {
            "role": "agent",
            "content": "Hi there!",
            "timestamp": "2026-02-17 10:00:05",
        }

    def test_multiline_agent_response(self, tmp_dir):
        """Multi-line [Agent] response collects continuation lines."""
        from agent.logging import parse_chat_entries

        log = tmp_dir / "agent.log"
        log.write_text(
            "2026-02-17 10:00:00 | INFO     | helio-agent | sess1 |  | [User] Show data\n"
            "2026-02-17 10:00:05 | INFO     | helio-agent | sess1 |  | [Agent] Here is your data:\n"
            "- ACE magnetic field\n"
            "- Wind proton density\n"
            "2026-02-17 10:00:10 | DEBUG    | helio-agent | sess1 |  | Tool call: something\n"
        )
        entries = parse_chat_entries(log, "sess1")
        assert len(entries) == 2
        assert (
            entries[1]["content"]
            == "Here is your data:\n- ACE magnetic field\n- Wind proton density"
        )

    def test_milestones_as_separate_entries(self, tmp_dir):
        """Tagged milestone lines appear as separate role='milestone' entries."""
        from agent.logging import parse_chat_entries

        log = tmp_dir / "agent.log"
        log.write_text(
            "2026-02-17 10:00:00 | INFO     | helio-agent | s1 |  | [User] Show ACE data\n"
            "2026-02-17 10:00:01 | DEBUG    | helio-agent | s1 | progress | [Planner] Think phase: researching datasets...\n"
            "2026-02-17 10:00:02 | DEBUG    | helio-agent | s1 | data_fetched | [DataOps] Stored 'ACE.Bmag' (1000 points)\n"
            "2026-02-17 10:00:03 | DEBUG    | helio-agent | s1 |  | Tool call: search_datasets\n"
            "2026-02-17 10:00:05 | INFO     | helio-agent | s1 |  | [Agent] Done.\n"
        )
        entries = parse_chat_entries(log, "s1")
        assert len(entries) == 4
        assert entries[0] == {
            "role": "user",
            "content": "Show ACE data",
            "timestamp": "2026-02-17 10:00:00",
        }
        assert entries[1] == {
            "role": "milestone",
            "content": "[Planner] Think phase: researching datasets...",
            "timestamp": "2026-02-17 10:00:01",
        }
        assert entries[2] == {
            "role": "milestone",
            "content": "[DataOps] Stored 'ACE.Bmag' (1000 points)",
            "timestamp": "2026-02-17 10:00:02",
        }
        assert entries[3] == {
            "role": "agent",
            "content": "Done.",
            "timestamp": "2026-02-17 10:00:05",
        }

    def test_milestones_scoped_to_turn(self, tmp_dir):
        """Milestones appear between their [User] and [Agent], not leaking across turns."""
        from agent.logging import parse_chat_entries

        log = tmp_dir / "agent.log"
        log.write_text(
            "2026-02-17 10:00:00 | INFO     | helio-agent | s1 |  | [User] Q1\n"
            "2026-02-17 10:00:01 | DEBUG    | helio-agent | s1 | data_fetched | [DataOps] Stored 'X' (100 pts)\n"
            "2026-02-17 10:00:02 | INFO     | helio-agent | s1 |  | [Agent] A1\n"
            "2026-02-17 10:00:10 | INFO     | helio-agent | s1 |  | [User] Q2\n"
            "2026-02-17 10:00:15 | INFO     | helio-agent | s1 |  | [Agent] A2\n"
        )
        entries = parse_chat_entries(log, "s1")
        assert len(entries) == 5
        assert entries[0] == {
            "role": "user",
            "content": "Q1",
            "timestamp": "2026-02-17 10:00:00",
        }
        assert entries[1] == {
            "role": "milestone",
            "content": "[DataOps] Stored 'X' (100 pts)",
            "timestamp": "2026-02-17 10:00:01",
        }
        assert entries[2] == {
            "role": "agent",
            "content": "A1",
            "timestamp": "2026-02-17 10:00:02",
        }
        assert entries[3] == {
            "role": "user",
            "content": "Q2",
            "timestamp": "2026-02-17 10:00:10",
        }
        assert entries[4] == {
            "role": "agent",
            "content": "A2",
            "timestamp": "2026-02-17 10:00:15",
        }

    def test_filters_by_session_id(self, tmp_dir):
        """Only entries matching the given session_id are returned."""
        from agent.logging import parse_chat_entries

        log = tmp_dir / "agent.log"
        log.write_text(
            "2026-02-17 10:00:00 | INFO     | helio-agent | sess1 |  | [User] Hello\n"
            "2026-02-17 10:00:01 | INFO     | helio-agent | sess2 |  | [User] World\n"
            "2026-02-17 10:00:02 | INFO     | helio-agent | sess1 |  | [Agent] Hi\n"
        )
        entries = parse_chat_entries(log, "sess1")
        assert len(entries) == 2
        assert entries[0]["content"] == "Hello"
        assert entries[1]["content"] == "Hi"

    def test_ignores_non_chat_lines(self, tmp_dir):
        """DEBUG/tool lines without [User]/[Agent] prefix or tag are ignored."""
        from agent.logging import parse_chat_entries

        log = tmp_dir / "agent.log"
        log.write_text(
            "2026-02-17 10:00:00 | INFO     | helio-agent | sess1 |  | Session started\n"
            "2026-02-17 10:00:01 | DEBUG    | helio-agent | sess1 |  | Tool call: search_datasets\n"
            "2026-02-17 10:00:02 | INFO     | helio-agent | sess1 |  | [User] Hello\n"
            "2026-02-17 10:00:03 | INFO     | helio-agent | sess1 |  | [Agent] Hi\n"
        )
        entries = parse_chat_entries(log, "sess1")
        assert len(entries) == 2

    def test_missing_file_returns_empty(self, tmp_dir):
        """Non-existent log file returns an empty list."""
        from agent.logging import parse_chat_entries

        entries = parse_chat_entries(tmp_dir / "nonexistent.log", "sess1")
        assert entries == []

    def test_empty_file_returns_empty(self, tmp_dir):
        """Empty log file returns an empty list."""
        from agent.logging import parse_chat_entries

        log = tmp_dir / "agent.log"
        log.write_text("")
        entries = parse_chat_entries(log, "sess1")
        assert entries == []

    def test_multiple_turns(self, tmp_dir):
        """Multiple user-agent turn pairs are all captured in order."""
        from agent.logging import parse_chat_entries

        log = tmp_dir / "agent.log"
        log.write_text(
            "2026-02-17 10:00:00 | INFO     | helio-agent | s1 |  | [User] Q1\n"
            "2026-02-17 10:00:05 | INFO     | helio-agent | s1 |  | [Agent] A1\n"
            "2026-02-17 10:01:00 | INFO     | helio-agent | s1 |  | [User] Q2\n"
            "2026-02-17 10:01:05 | INFO     | helio-agent | s1 |  | [Agent] A2\n"
        )
        entries = parse_chat_entries(log, "s1")
        assert len(entries) == 4
        assert [e["role"] for e in entries] == ["user", "agent", "user", "agent"]
        assert entries[2]["content"] == "Q2"
        assert entries[3]["content"] == "A2"

    def test_tail_lines(self, tmp_dir):
        """tail_lines parses only the last N lines of the file."""
        from agent.logging import parse_chat_entries

        log = tmp_dir / "agent.log"
        log.write_text(
            "2026-02-17 10:00:00 | INFO     | helio-agent | s1 |  | [User] Q1\n"
            "2026-02-17 10:00:05 | INFO     | helio-agent | s1 |  | [Agent] A1\n"
            "2026-02-17 10:01:00 | INFO     | helio-agent | s1 |  | [User] Q2\n"
            "2026-02-17 10:01:05 | INFO     | helio-agent | s1 |  | [Agent] A2\n"
        )
        # Only parse last 2 lines — should get Q2 + A2
        entries = parse_chat_entries(log, "s1", tail_lines=2)
        assert len(entries) == 2
        assert entries[0]["content"] == "Q2"
        assert entries[1]["content"] == "A2"

        # Full parse gets all 4
        entries_full = parse_chat_entries(log, "s1")
        assert len(entries_full) == 4


class TestForkHistory:
    def test_save_and_load_client_history(self, tmp_path):
        """Client history from Interactions API sessions should persist to disk."""
        sm = SessionManager(base_dir=tmp_path)
        session_id = sm.create_session("test-model")

        # Simulate saving with client history embedded in history_dicts
        client_history = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "model", "content": [{"type": "text", "text": "Hi!"}]},
        ]
        history_dicts = [
            {"_interaction_id": "int_abc123"},
            {"_client_history": client_history},
        ]

        sm.save_session(
            session_id=session_id,
            chat_history=history_dicts,
            data_store=None,
            metadata_updates={"turn_count": 1},
        )

        # Load and verify client history is preserved
        loaded = sm.load_session(session_id)
        loaded_history = loaded[0]  # history_dicts
        assert any("_client_history" in entry for entry in loaded_history)
        ch_entry = next(e for e in loaded_history if "_client_history" in e)
        assert len(ch_entry["_client_history"]) == 2
