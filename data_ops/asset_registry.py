"""Unified session asset registry.

Provides a single interface for three kinds of session artifacts:
  - **data**: fetched/computed DataEntries (delegated to DataStore)
  - **file**: user-uploaded files (tracked here, stored on disk)
  - **figure**: rendered Plotly figures (tracked here, JSON + thumbnail on disk)

The registry is a thin layer on top of DataStore for data assets and
maintains its own lightweight dicts for files and figures, persisted
to ``assets.json`` in the session directory.
"""

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_ops.store import DataStore


@dataclass
class AssetMeta:
    """Metadata for a single session asset."""

    asset_id: str  # "file_abc123", "fig_001", or DataEntry.id
    kind: str  # "data" | "file" | "figure"
    name: str  # human-readable (filename, label, figure title)
    created_at: str  # ISO timestamp
    metadata: dict = field(default_factory=dict)  # kind-specific


class AssetRegistry:
    """Unified registry for session assets.

    Parameters
    ----------
    session_dir : Path
        Root directory for the session (``~/.xhelio/sessions/<id>``).
    data_store : DataStore
        The session's DataStore instance (for ``kind="data"`` queries).
    """

    def __init__(self, session_dir: Path, data_store: "DataStore") -> None:
        self._session_dir = session_dir
        self._store = data_store
        self._files: dict[str, AssetMeta] = {}
        self._figures: dict[str, AssetMeta] = {}
        self._figure_counter: int = 0
        self._lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_file(
        self,
        filename: str,
        path: Path | str,
        size_bytes: int,
        mime_type: str = "",
    ) -> AssetMeta:
        """Register an uploaded file as a session asset.

        Returns the created AssetMeta.
        """
        path = Path(path)
        ext = path.suffix.lower()
        asset_id = f"file_{_short_hash(str(path))}"
        meta = AssetMeta(
            asset_id=asset_id,
            kind="file",
            name=filename,
            created_at=_now_iso(),
            metadata={
                "path": str(path),
                "size_bytes": size_bytes,
                "mime_type": mime_type,
                "extension": ext,
            },
        )
        with self._lock:
            self._files[asset_id] = meta
        return meta

    def register_figure(
        self,
        fig_json: dict,
        trace_labels: list[str],
        panel_count: int,
        op_id: str,
        thumbnail_path: str | None = None,
    ) -> AssetMeta:
        """Register a rendered Plotly figure as a session asset.

        Returns the created AssetMeta.
        """
        with self._lock:
            self._figure_counter += 1
            num = self._figure_counter
        asset_id = f"fig_{num:03d}"
        title = (fig_json.get("layout") or {}).get("title", "")
        if isinstance(title, dict):
            title = title.get("text", "")
        name = title or f"Figure {num}"
        meta = AssetMeta(
            asset_id=asset_id,
            kind="figure",
            name=name,
            created_at=_now_iso(),
            metadata={
                "op_id": op_id,
                "fig_json": fig_json,
                "trace_labels": trace_labels,
                "panel_count": panel_count,
                "thumbnail_path": thumbnail_path,
            },
        )
        with self._lock:
            self._figures[asset_id] = meta
        return meta

    def list_assets(self, kind: str | None = None) -> list[dict]:
        """Return a unified list of all session assets.

        Parameters
        ----------
        kind : str, optional
            Filter by ``"data"``, ``"file"``, or ``"figure"``.
            If None, returns all kinds.
        """
        results: list[dict] = []

        if kind is None or kind == "data":
            for entry_dict in self._store.list_entries():
                results.append(
                    {
                        "asset_id": entry_dict.get("id", ""),
                        "kind": "data",
                        "name": entry_dict.get("label", ""),
                        "created_at": "",
                        "metadata": entry_dict,
                    }
                )

        with self._lock:
            if kind is None or kind == "file":
                for meta in self._files.values():
                    results.append(asdict(meta))

            if kind is None or kind == "figure":
                for meta in self._figures.values():
                    # Return lightweight summary (omit full fig_json)
                    d = asdict(meta)
                    fig_meta = dict(d["metadata"])
                    fig_meta.pop("fig_json", None)
                    d["metadata"] = fig_meta
                    results.append(d)

        return results

    def get_asset(self, asset_id: str) -> AssetMeta | None:
        """Look up a single asset by ID.

        For ``kind="data"``, returns None (use DataStore directly).
        """
        with self._lock:
            if asset_id in self._files:
                return self._files[asset_id]
            if asset_id in self._figures:
                return self._figures[asset_id]
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write file and figure assets to ``assets.json``."""
        with self._lock:
            data = {
                "figure_counter": self._figure_counter,
                "files": {k: asdict(v) for k, v in self._files.items()},
                "figures": {k: asdict(v) for k, v in self._figures.items()},
            }
        path = self._session_dir / "assets.json"
        path.write_text(json.dumps(data, default=str), encoding="utf-8")

    def _load(self) -> None:
        """Load assets.json if it exists. Graceful if missing or corrupt."""
        path = self._session_dir / "assets.json"
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        self._figure_counter = raw.get("figure_counter", 0)

        for aid, d in raw.get("files", {}).items():
            self._files[aid] = AssetMeta(
                asset_id=d["asset_id"],
                kind=d["kind"],
                name=d["name"],
                created_at=d["created_at"],
                metadata=d.get("metadata", {}),
            )

        for aid, d in raw.get("figures", {}).items():
            self._figures[aid] = AssetMeta(
                asset_id=d["asset_id"],
                kind=d["kind"],
                name=d["name"],
                created_at=d["created_at"],
                metadata=d.get("metadata", {}),
            )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _short_hash(s: str) -> str:
    """Return a short hex hash for generating asset IDs."""
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest()[:8]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
