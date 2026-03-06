"""Eureka store — persistent JSON store for eureka findings and suggestions."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class EurekaEntry:
    id: str
    session_id: str
    timestamp: str
    title: str
    observation: str
    hypothesis: str
    evidence: list[str]
    confidence: float
    tags: list[str]
    status: str


@dataclass
class EurekaSuggestion:
    """An actionable follow-up suggestion linked to a eureka finding."""

    id: str
    session_id: str
    timestamp: str
    action: str  # "fetch_data", "visualize", "compute", "zoom", "compare"
    description: str  # Human-readable description of what to do
    rationale: str  # Why this suggestion matters
    parameters: dict  # Action-specific parameters
    priority: str  # "high", "medium", "low"
    linked_eureka_id: str  # Which eureka this aims to validate
    status: str  # "proposed", "approved", "executed", "rejected"


@dataclass
class EurekaChatMessage:
    """A chat message between user and EurekaAgent."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: str


class EurekaStore:
    def __init__(self, path: Optional[Path] = None):
        if path is None:
            xhelio_dir = Path.home() / ".xhelio"
            xhelio_dir.mkdir(exist_ok=True)
            path = xhelio_dir / "eurekas.json"
        self._path = Path(path) if not isinstance(path, Path) else path
        self._lock = threading.Lock()

    def _load(self) -> dict:
        if not self._path.exists():
            return {"eurekas": [], "suggestions": [], "chat_history": []}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Backward compatibility: old files may lack "suggestions" or "chat_history"
            if "suggestions" not in data:
                data["suggestions"] = []
            if "chat_history" not in data:
                data["chat_history"] = []
            return data
        except (json.JSONDecodeError, IOError):
            return {"eurekas": [], "suggestions": [], "chat_history": []}

    def _save(self, data: dict) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # -- Eureka entries --

    def add(self, entry: EurekaEntry) -> None:
        with self._lock:
            data = self._load()
            data["eurekas"].append(asdict(entry))
            self._save(data)

    def get(self, id: str) -> Optional[EurekaEntry]:
        with self._lock:
            data = self._load()
            for e in data["eurekas"]:
                if e["id"] == id:
                    return EurekaEntry(**e)
            return None

    def list(
        self,
        session_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[EurekaEntry]:
        with self._lock:
            data = self._load()
            results = []
            for e in data["eurekas"]:
                if session_id is not None and e["session_id"] != session_id:
                    continue
                if status is not None and e["status"] != status:
                    continue
                results.append(EurekaEntry(**e))
            return results

    def update_status(self, id: str, status: str) -> bool:
        with self._lock:
            data = self._load()
            for e in data["eurekas"]:
                if e["id"] == id:
                    e["status"] = status
                    self._save(data)
                    return True
            return False

    # -- Suggestions --

    def add_suggestion(self, suggestion: EurekaSuggestion) -> None:
        with self._lock:
            data = self._load()
            data["suggestions"].append(asdict(suggestion))
            self._save(data)

    def list_suggestions(
        self,
        session_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[EurekaSuggestion]:
        with self._lock:
            data = self._load()
            results = []
            for s in data["suggestions"]:
                if session_id is not None and s["session_id"] != session_id:
                    continue
                if status is not None and s["status"] != status:
                    continue
                results.append(EurekaSuggestion(**s))
            return results

    def update_suggestion_status(self, id: str, status: str) -> bool:
        with self._lock:
            data = self._load()
            for s in data["suggestions"]:
                if s["id"] == id:
                    s["status"] = status
                    self._save(data)
                    return True
            return False

    def get_session_history(self, session_id: str) -> dict:
        """Return combined findings + suggestions for a session.

        Returns a dict with 'eurekas' and 'suggestions' lists for use
        by the eureka agent's read_eureka_history tool.
        """
        with self._lock:
            data = self._load()
            eurekas = [e for e in data["eurekas"] if e["session_id"] == session_id]
            suggestions = [
                s for s in data["suggestions"] if s["session_id"] == session_id
            ]
        return {"eurekas": eurekas, "suggestions": suggestions}

    # -- Chat history --

    def add_chat_message(self, message: EurekaChatMessage) -> None:
        """Add a chat message to the global chat history."""
        with self._lock:
            data = self._load()
            if "chat_history" not in data:
                data["chat_history"] = []
            data["chat_history"].append(asdict(message))
            # Keep only last 100 messages to prevent file bloat
            data["chat_history"] = data["chat_history"][-100:]
            self._save(data)

    def get_chat_history(self, limit: int = 50) -> list[EurekaChatMessage]:
        """Get the most recent chat messages."""
        with self._lock:
            data = self._load()
            messages = data.get("chat_history", [])[-limit:]
            return [EurekaChatMessage(**m) for m in messages]
