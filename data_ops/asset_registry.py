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
    source_path: str | None = None  # original file path (path-registered files)
    session_path: str | None = None  # safe copy in session dir (after register/upload)
    lineage: dict | None = None  # e.g. {"derived_from": "file_abc"}
    figure_kind: str | None = None  # "plotly" | "mpl" | "jsx" | "image" (figures only)


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
        source_path: str | None = None,
        asset_id: str | None = None,
    ) -> AssetMeta:
        """Register an uploaded file as a session asset.

        Returns the created AssetMeta.
        """
        path = Path(path)
        ext = path.suffix.lower()
        asset_id = asset_id or f"file_{_short_hash(str(path))}"
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
            source_path=source_path or str(path),
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

    def register_image(
        self,
        name: str,
        image_path: str,
        source_url: str = "",
    ) -> AssetMeta:
        """Register an externally-fetched image as a figure asset.

        Unlike ``register_figure`` (which stores Plotly fig_json), this method
        stores a simple image reference — no trace labels, panel count, or
        fig_json.  The image is served via the same ``mpl_outputs/`` endpoint.
        """
        with self._lock:
            self._figure_counter += 1
            num = self._figure_counter
        asset_id = f"fig_{num:03d}"
        meta = AssetMeta(
            asset_id=asset_id,
            kind="figure",
            name=name,
            created_at=_now_iso(),
            metadata={
                "image_path": image_path,
                "source_url": source_url,
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

    def list_assets_enriched(self, kind: str | None = None) -> dict:
        """Return enriched asset listing with summary counts and lineage.

        Parameters
        ----------
        kind : str, optional
            Filter to "data", "file", or "figure".

        Returns
        -------
        dict with keys:
            summary: {data: int, files: int, figures: int}
            data: list[dict]  (if kind is None or "data")
            files: list[dict] (if kind is None or "file")
            figures: list[dict] (if kind is None or "figure")
        """
        result: dict = {}

        # Build file_asset_id → [data_entry_id] mapping for lineage
        file_to_data: dict[str, list[str]] = {}

        data_list = []
        if kind is None or kind == "data":
            for entry_dict in self._store.list_entries():
                data_id = entry_dict.get("id", "")
                file_asset_id = None
                entry_obj = self._store.get(data_id)
                if entry_obj and entry_obj.metadata:
                    file_asset_id = entry_obj.metadata.get("file_asset_id")
                if file_asset_id:
                    file_to_data.setdefault(file_asset_id, []).append(data_id)

                columns = entry_dict.get("columns", [])
                data_list.append({
                    "asset_id": data_id,
                    "name": entry_dict.get("label", ""),
                    "created_at": "",
                    "shape": [
                        entry_dict.get("num_points", 0),
                        len(columns),
                    ],
                    "source": entry_dict.get("source", ""),
                    "derived_from": file_asset_id,
                })
            result["data"] = data_list

        file_list = []
        if kind is None or kind == "file":
            with self._lock:
                for meta in self._files.values():
                    file_list.append({
                        "asset_id": meta.asset_id,
                        "name": meta.name,
                        "created_at": meta.created_at,
                        "size_bytes": meta.metadata.get("size_bytes", 0),
                        "mime_type": meta.metadata.get("mime_type", ""),
                        "has_local_copy": meta.session_path is not None,
                        "derived_data": file_to_data.get(meta.asset_id, []),
                    })
            result["files"] = file_list

        fig_list = []
        if kind is None or kind == "figure":
            with self._lock:
                for meta in self._figures.values():
                    fig_list.append({
                        "asset_id": meta.asset_id,
                        "name": meta.name,
                        "created_at": meta.created_at,
                        "figure_kind": meta.figure_kind,
                        "panel_count": meta.metadata.get("panel_count"),
                        "has_thumbnail": bool(meta.metadata.get("thumbnail_path")),
                        "data_sources": meta.metadata.get("trace_labels", []),
                    })
            result["figures"] = fig_list

        # Summary always present
        result["summary"] = {
            "data": len(data_list) if kind is None or kind == "data" else len(self._store.list_entries()),
            "files": len(file_list) if kind is None or kind == "file" else len(self._files),
            "figures": len(fig_list) if kind is None or kind == "figure" else len(self._figures),
        }

        return result

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

    def remove_file(self, asset_id: str) -> bool:
        """Remove a file asset from the registry.

        Deletes session copy if it exists. Returns True if found and removed.
        """
        with self._lock:
            meta = self._files.pop(asset_id, None)
        if meta is None:
            return False
        # Delete session copy if it exists
        if meta.session_path:
            p = Path(meta.session_path)
            if p.exists():
                p.unlink()
        return True

    def remove_figure(self, asset_id: str) -> bool:
        """Remove a figure asset from the registry.

        Deletes associated files (thumbnail, JSON, image). Returns True if found.
        """
        with self._lock:
            meta = self._figures.pop(asset_id, None)
        if meta is None:
            return False
        # Delete associated files
        for key in ("thumbnail_path", "image_path"):
            path_str = meta.metadata.get(key)
            if path_str:
                p = Path(path_str)
                if p.exists():
                    p.unlink()
        # Delete JSON file if it exists (same dir as thumbnail, same op_id)
        op_id = meta.metadata.get("op_id")
        if op_id and meta.metadata.get("thumbnail_path"):
            json_path = Path(meta.metadata["thumbnail_path"]).with_suffix(".json")
            if json_path.exists():
                json_path.unlink()
        return True

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
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
        tmp.replace(path)

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
                source_path=d.get("source_path"),
                session_path=d.get("session_path"),
                lineage=d.get("lineage"),
                figure_kind=d.get("figure_kind"),
            )

        for aid, d in raw.get("figures", {}).items():
            self._figures[aid] = AssetMeta(
                asset_id=d["asset_id"],
                kind=d["kind"],
                name=d["name"],
                created_at=d["created_at"],
                metadata=d.get("metadata", {}),
                source_path=d.get("source_path"),
                session_path=d.get("session_path"),
                lineage=d.get("lineage"),
                figure_kind=d.get("figure_kind"),
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
