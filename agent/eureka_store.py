"""Eureka store — persistent JSON store for eureka findings."""

import json
import os
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
            return {"eurekas": []}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"eurekas": []}

    def _save(self, data: dict) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

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
