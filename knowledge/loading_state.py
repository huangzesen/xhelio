"""Thread-safe singleton tracking mission data loading lifecycle.

Used by the background loader (startup.py) to report progress and by
the API layer (routes.py) to stream state to the frontend via SSE.
"""

import threading
from enum import Enum
from typing import Callable, Optional


class LoadingPhase(Enum):
    NOT_STARTED = "not_started"
    CHECKING = "checking"
    BOOTSTRAPPING_CDAWEB = "bootstrapping_cdaweb"
    BOOTSTRAPPING_PPI = "bootstrapping_ppi"
    LOADING_JSON = "loading_json"
    COMPLETE = "complete"
    FAILED = "failed"


class MissionLoadingState:
    """Singleton tracking mission data loading progress.

    Thread-safe: all mutations go through ``update()`` which holds a lock
    and notifies subscribers synchronously.
    """

    _instance: Optional["MissionLoadingState"] = None
    _lock_cls = threading.Lock()  # class-level lock for singleton creation

    def __new__(cls) -> "MissionLoadingState":
        with cls._lock_cls:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._lock = threading.Lock()
                inst._condition = threading.Condition(inst._lock)
                inst._phase = LoadingPhase.NOT_STARTED
                inst._progress_pct = 0.0
                inst._progress_message = ""
                inst._error: Optional[str] = None
                inst._subscribers: list[Callable[[dict], None]] = []
                cls._instance = inst
            return cls._instance

    # -- read-only properties --

    @property
    def phase(self) -> LoadingPhase:
        return self._phase

    @property
    def progress_pct(self) -> float:
        return self._progress_pct

    @property
    def progress_message(self) -> str:
        return self._progress_message

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def is_ready(self) -> bool:
        return self._phase == LoadingPhase.COMPLETE

    @property
    def is_loading(self) -> bool:
        return self._phase not in (
            LoadingPhase.NOT_STARTED,
            LoadingPhase.COMPLETE,
            LoadingPhase.FAILED,
        )

    # -- mutations --

    def update(
        self,
        phase: Optional[LoadingPhase] = None,
        pct: Optional[float] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Thread-safe state update.  Notifies all subscribers."""
        with self._condition:
            if phase is not None:
                self._phase = phase
            if pct is not None:
                self._progress_pct = pct
            if message is not None:
                self._progress_message = message
            if error is not None:
                self._error = error
            if self._phase in (LoadingPhase.COMPLETE, LoadingPhase.FAILED):
                self._condition.notify_all()
            snapshot = self.to_dict()
        # Notify outside the lock to avoid deadlocks in subscribers
        for cb in list(self._subscribers):
            try:
                cb(snapshot)
            except Exception as exc:
                import logging
                logging.getLogger("xhelio").debug("LoadingState subscriber error: %s", exc)

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """Block until loading is complete or failed.

        Returns True if ready, False on timeout.
        """
        with self._condition:
            if self._phase in (LoadingPhase.COMPLETE, LoadingPhase.FAILED):
                return self._phase == LoadingPhase.COMPLETE
            self._condition.wait_for(
                lambda: self._phase in (LoadingPhase.COMPLETE, LoadingPhase.FAILED),
                timeout=timeout,
            )
            return self._phase == LoadingPhase.COMPLETE

    # -- pub/sub --

    def subscribe(self, callback: Callable[[dict], None]) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[dict], None]) -> None:
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    # -- serialization --

    def to_dict(self) -> dict:
        return {
            "phase": self._phase.value,
            "is_ready": self.is_ready,
            "is_loading": self.is_loading,
            "progress_pct": round(self._progress_pct, 1),
            "progress_message": self._progress_message,
            "error": self._error,
        }

    # -- testing support --

    @classmethod
    def _reset(cls) -> None:
        """Reset singleton for testing. Not for production use."""
        with cls._lock_cls:
            cls._instance = None


def get_loading_state() -> MissionLoadingState:
    """Module-level accessor for the singleton."""
    return MissionLoadingState()
