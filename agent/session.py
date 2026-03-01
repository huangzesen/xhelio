"""
Session persistence for chat history and data store.

Saves and restores Gemini chat history + DataStore entries so users
can resume conversations across process restarts.

Storage layout:
    ~/.xhelio/sessions/{session_id}/
        metadata.json     — session info (model, turn_count, timestamps, etc.)
        history.json      — Gemini Content dicts (from Content.model_dump)
        data/
            _labels.json          — label -> hash folder name
            {hash}/data.pkl       — pickled DataFrame (or data.nc for xarray)
            {hash}/meta.json      — label, units, description, stats, etc.
"""

import base64
import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


# Characters unsafe for filenames on Windows
_UNSAFE_CHARS = re.compile(r'[/\\:*?"<>|]')


def _safe_filename(label: str) -> str:
    """Convert a DataStore label to a safe filename (without extension)."""
    return _UNSAFE_CHARS.sub("_", label)


_B64_PREFIX = "b64:"


def _encode_bytes_fields(history: list[dict]) -> list[dict]:
    """Encode bytes fields (e.g. thought_signature) as base64 for JSON safety.

    The Gemini SDK's Content.model_dump() produces raw bytes for
    thought_signature fields.  json.dump(default=str) would mangle these
    into Python repr strings like "b'\\x12\\x9c...'" which can't be
    deserialized back to bytes.  We base64-encode them instead.
    """
    for entry in history:
        for part in entry.get("parts", []):
            if isinstance(part, dict) and "thought_signature" in part:
                val = part["thought_signature"]
                if isinstance(val, bytes):
                    part["thought_signature"] = _B64_PREFIX + base64.b64encode(val).decode("ascii")
    return history


def _decode_bytes_fields(history: list[dict]) -> list[dict]:
    """Decode base64-encoded bytes fields back to raw bytes for the SDK.

    Also handles legacy sessions where bytes were mangled by ``default=str``
    into Python repr strings like ``"b'\\x12\\x9c...'"`` — these are
    re-parsed via ``ast.literal_eval``.
    """
    import ast

    for entry in history:
        for part in entry.get("parts", []):
            if isinstance(part, dict) and "thought_signature" in part:
                val = part["thought_signature"]
                if isinstance(val, (bytes, bytearray)):
                    continue  # already bytes
                if isinstance(val, str):
                    if val.startswith(_B64_PREFIX):
                        part["thought_signature"] = base64.b64decode(val[len(_B64_PREFIX):])
                    elif val.startswith("b'") or val.startswith('b"'):
                        # Legacy format from default=str mangling
                        try:
                            part["thought_signature"] = ast.literal_eval(val)
                        except (ValueError, SyntaxError):
                            pass
    return history


class SessionManager:
    """Manages session directories for chat history persistence."""

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            from config import get_data_dir
            base_dir = get_data_dir() / "sessions"
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def generate_session_id(self) -> str:
        """Generate a unique session ID without creating any files or directories.

        Returns:
            The session_id string (format: YYYYMMDD_HHMMSS_XXXXXXXX).
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    def create_session(self, model_name: str = "") -> str:
        """Create a new session directory with initial metadata on disk.

        This makes the session immediately visible in ``list_sessions()``
        (and therefore in the sidebar) even before the first message.

        Returns:
            The session_id string.
        """
        session_id = self.generate_session_id()
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "data").mkdir(exist_ok=True)

        metadata = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "model": model_name,
            "turn_count": 0,
            "round_count": 0,
            "last_message_preview": "",
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "api_calls": 0},
        }
        self._write_json(session_dir / "metadata.json", metadata)
        self._write_json(session_dir / "history.json", [])

        return session_id

    def cleanup_empty_sessions(self, exclude_ids: Optional[set[str]] = None, max_age_seconds: int = 300) -> int:
        """Remove sessions with no meaningful content (turn_count == 0, no history).

        Prevents accumulation of abandoned sessions (e.g. user opened the app
        and quit without sending a message).

        Args:
            exclude_ids: Session IDs to skip (e.g. the current active session).
            max_age_seconds: Only delete empty sessions older than this many
                seconds (default 300 = 5 minutes).  Prevents deleting sessions
                that were just created but haven't received a message yet.

        Returns:
            Number of sessions deleted.
        """
        exclude_ids = exclude_ids or set()
        now = datetime.now()
        deleted = 0
        for d in list(self.base_dir.iterdir()):
            if not d.is_dir():
                continue
            session_id = d.name
            if session_id in exclude_ids:
                continue
            meta_path = d / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                meta = self._read_json(meta_path)
                if not meta:
                    continue
                turn_count = meta.get("turn_count", 0)
                if turn_count == 0:
                    # Skip sessions created within the grace period
                    created_at_str = meta.get("created_at")
                    if created_at_str:
                        try:
                            created_at = datetime.fromisoformat(created_at_str)
                            age_seconds = (now - created_at).total_seconds()
                            if age_seconds < max_age_seconds:
                                continue
                        except (ValueError, TypeError):
                            pass
                    self.delete_session(session_id)
                    deleted += 1
            except Exception:
                continue
        return deleted

    def save_session(
        self,
        session_id: str,
        chat_history: list[dict],
        data_store,
        metadata_updates: Optional[dict] = None,
        figure_state: Optional[dict] = None,
        figure_obj=None,
        operations: Optional[list[dict]] = None,
        display_log: Optional[list[dict]] = None,
    ) -> None:
        """Save chat history, DataStore, and plot figure to disk.

        Args:
            session_id: The session to save.
            chat_history: List of Content dicts (from Content.model_dump).
            data_store: A DataStore instance to persist.
            metadata_updates: Optional dict to merge into metadata
                (e.g. token_usage, turn_count).
            figure_state: Optional renderer state dict from
                ``PlotlyRenderer.save_state()``.
            figure_obj: Optional Plotly Figure object for thumbnail export.
            operations: Optional list of operation records from
                ``OperationsLog.get_records()``.
            display_log: Optional list of display log entries
                (role/content/timestamp dicts for UI replay).
        """
        session_dir = self.base_dir / session_id
        is_new_dir = not session_dir.exists()
        session_dir.mkdir(parents=True, exist_ok=True)

        # First save — create initial metadata so list_sessions() can find this session
        if is_new_dir:
            (session_dir / "data").mkdir(exist_ok=True)
            initial_metadata = {
                "id": session_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "model": "",
                "turn_count": 0,
                "round_count": 0,
                "last_message_preview": "",
                "token_usage": {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "api_calls": 0},
            }
            self._write_json(session_dir / "metadata.json", initial_metadata)

        # Save history (encode bytes fields so JSON round-trips cleanly)
        self._write_json(session_dir / "history.json", _encode_bytes_fields(chat_history))

        # DataStore is disk-backed — data is already persisted by put() calls.
        # Ensure the data directory exists for legacy callers.
        data_dir = session_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # Save figure state (or remove stale file if no figure)
        figure_path = session_dir / "figure.json"
        if figure_state:
            self._write_json(figure_path, figure_state)
        elif figure_path.exists():
            figure_path.unlink()

        # Export a PNG thumbnail for fast resume preview
        thumb_path = session_dir / "figure_thumbnail.png"
        if figure_state and figure_obj is not None:
            try:
                figure_obj.write_image(
                    str(thumb_path), format="png",
                    width=800, height=400, scale=1,
                )
            except Exception:
                pass  # kaleido not available or figure too complex
        elif not figure_state and thumb_path.exists():
            thumb_path.unlink()

        # Save operations log
        if operations is not None:
            self._write_json(session_dir / "operations.json", operations)

        # Generate per-render thumbnails (one PNG per render_plotly_json op)
        if operations is not None and data_store is not None:
            self._generate_render_thumbnails(session_dir, operations, data_store)

        # Save display log
        if display_log is not None:
            self._write_json(session_dir / "display_log.json", display_log)

        # Update metadata
        metadata = self._read_json(session_dir / "metadata.json") or {}
        metadata["updated_at"] = datetime.now().isoformat()
        if metadata_updates:
            metadata.update(metadata_updates)
        self._write_json(session_dir / "metadata.json", metadata)

    def _generate_render_thumbnails(
        self, session_dir: Path, operations: list[dict], data_store
    ) -> None:
        """Generate per-render PNG thumbnails for all render_plotly_json ops.

        Each successful render op gets a thumbnail at
        ``session_dir/thumbnails/{op_id}.png``.  Existing thumbnails are
        skipped (incremental saves).  Failures are silently ignored.
        """
        render_ops = [
            op for op in operations
            if op.get("tool") == "render_plotly_json"
            and op.get("status") == "success"
            and op.get("id")
        ]
        if not render_ops:
            return

        thumb_dir = session_dir / "thumbnails"
        thumb_dir.mkdir(exist_ok=True)

        for op in render_ops:
            op_id = op["id"]
            thumb_path = thumb_dir / f"{op_id}.png"
            if thumb_path.exists():
                continue  # incremental save — already generated

            try:
                fig_json = op.get("args", {}).get("figure_json", {})
                input_labels = op.get("inputs", [])
                if not fig_json or not input_labels:
                    continue

                # Resolve all input labels to DataEntry objects
                from data_ops.store import resolve_entry
                entry_map: dict = {}
                skip = False
                for label in input_labels:
                    entry, _ = resolve_entry(data_store, label)
                    if entry is None:
                        skip = True
                        break
                    entry_map[label] = entry
                if skip:
                    continue

                # Build the figure from JSON + data
                from rendering.plotly_renderer import fill_figure_data
                result = fill_figure_data(fig_json, entry_map)
                figure = result.figure
                if figure is None:
                    continue

                figure.write_image(
                    str(thumb_path), format="png",
                    width=800, height=400, scale=1,
                )
            except Exception:
                pass  # best-effort — skip failures silently

    def load_session(self, session_id: str) -> tuple[list[dict], Path, dict, Optional[dict], Optional[list[dict]], Optional[list[dict]], Optional[list[dict]]]:
        """Load a session from disk.

        Args:
            session_id: The session to load.

        Returns:
            Tuple of (history_dicts, data_dir_path, metadata, figure_state,
            operations, display_log, event_log).
            ``figure_state`` is the dict saved by ``PlotlyRenderer.save_state()``,
            or ``None`` if no figure was saved.
            ``operations`` is the list of operation records, or ``None`` if
            no operations log was saved.
            ``display_log`` is the list of display log entries for UI replay,
            or ``None`` if not saved.
            ``event_log`` is the list of structured event records from
            ``events.jsonl``, or ``None`` for old sessions without the file.

        Raises:
            FileNotFoundError: If the session does not exist.
        """
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        history = self._read_json(session_dir / "history.json") or []
        history = _decode_bytes_fields(history)
        metadata = self._read_json(session_dir / "metadata.json") or {}
        data_dir = session_dir / "data"
        figure_state = self._read_json(session_dir / "figure.json")
        operations = self._read_json(session_dir / "operations.json")
        display_log = self._read_json(session_dir / "display_log.json")

        # Load structured event log (JSONL format, may not exist for old sessions)
        from .event_bus import load_event_log
        event_log = load_event_log(session_dir / "events.jsonl") or None

        return history, data_dir, metadata, figure_state, operations, display_log, event_log

    def list_sessions(self) -> list[dict]:
        """List all sessions, sorted by updated_at descending.

        Returns:
            List of metadata dicts (with id, created_at, updated_at, etc.).
        """
        sessions = []
        for d in self.base_dir.iterdir():
            if not d.is_dir():
                continue
            meta_path = d / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                meta = self._read_json(meta_path)
                if meta and "id" in meta:
                    sessions.append(meta)
            except Exception:
                continue

        sessions.sort(key=lambda m: m.get("updated_at", ""), reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session directory.

        Returns:
            True if deleted, False if not found.
        """
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            return False
        shutil.rmtree(session_dir)
        return True

    def get_most_recent_session(self) -> Optional[str]:
        """Return the session_id of the most recently updated session, or None."""
        sessions = self.list_sessions()
        if not sessions:
            return None
        return sessions[0]["id"]

    # ---- Internal helpers ----

    @staticmethod
    def _write_json(path: Path, data) -> None:
        import os
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)

    @staticmethod
    def _read_json(path: Path):
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_event_log(path: Path) -> Optional[list[dict]]:
        """Read a JSONL event log file. Returns None if file doesn't exist."""
        from .event_bus import load_event_log
        return load_event_log(path) or None
