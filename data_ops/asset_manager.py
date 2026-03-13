"""Disk asset management — scanning and cleanup for session data.

Covers one category:
  - Sessions    (~/.xhelio/sessions/)

External asset management (CDF, SPICE kernels, etc.) is handled by standalone MCP packages.
"""

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _sessions_dir() -> Path:
    from config import get_data_dir
    return get_data_dir() / "sessions"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DirStats:
    name: str
    path: str
    total_bytes: int
    file_count: int
    oldest_mtime: Optional[float] = None
    newest_mtime: Optional[float] = None
    # Sessions only
    turn_count: Optional[int] = None
    session_name: Optional[str] = None


@dataclass
class AssetCategory:
    name: str
    path: str
    total_bytes: int
    file_count: int
    subcategories: list[DirStats] = field(default_factory=list)


@dataclass
class AssetOverview:
    categories: list[AssetCategory]
    total_bytes: int
    scan_time_ms: int


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_name(name: str) -> str:
    """Validate a single path component (no traversal)."""
    if not name or not _SAFE_NAME_RE.match(name) or ".." in name:
        raise ValueError(f"Invalid name: {name!r}")
    return name


# ---------------------------------------------------------------------------
# Low-level scanning
# ---------------------------------------------------------------------------

def _scan_directory(path: Path) -> DirStats:
    """Recursively scan a directory using os.scandir() for speed."""
    total_bytes = 0
    file_count = 0
    oldest_mtime: Optional[float] = None
    newest_mtime: Optional[float] = None

    def _walk(p: str) -> None:
        nonlocal total_bytes, file_count, oldest_mtime, newest_mtime
        try:
            with os.scandir(p) as it:
                for entry in it:
                    if entry.is_file(follow_symlinks=False):
                        try:
                            stat = entry.stat(follow_symlinks=False)
                        except OSError:
                            continue
                        total_bytes += stat.st_size
                        file_count += 1
                        mt = stat.st_mtime
                        if oldest_mtime is None or mt < oldest_mtime:
                            oldest_mtime = mt
                        if newest_mtime is None or mt > newest_mtime:
                            newest_mtime = mt
                    elif entry.is_dir(follow_symlinks=False):
                        _walk(entry.path)
        except PermissionError:
            pass

    if path.exists():
        _walk(str(path))

    return DirStats(
        name=path.name,
        path=str(path),
        total_bytes=total_bytes,
        file_count=file_count,
        oldest_mtime=oldest_mtime,
        newest_mtime=newest_mtime,
    )


# ---------------------------------------------------------------------------
# Category detail functions
# ---------------------------------------------------------------------------

def get_sessions_detail() -> AssetCategory:
    """Per-session breakdown with metadata (turn_count, name)."""
    root = _sessions_dir()
    subs: list[DirStats] = []
    if root.exists():
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            stats = _scan_directory(child)
            # Try to read session metadata
            meta_path = child / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    stats.session_name = meta.get("name") or meta.get("session_name")
                    stats.turn_count = meta.get("turn_count")
                except (json.JSONDecodeError, OSError):
                    pass
            # Estimate turn_count from events.jsonl if not in metadata
            if stats.turn_count is None:
                events_path = child / "events.jsonl"
                if events_path.exists():
                    try:
                        count = 0
                        with open(events_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if '"user_message"' in line or '"USER_MESSAGE"' in line:
                                    count += 1
                        stats.turn_count = count
                    except OSError:
                        pass
            subs.append(stats)
    return AssetCategory(
        name="sessions",
        path=str(root),
        total_bytes=sum(s.total_bytes for s in subs),
        file_count=sum(s.file_count for s in subs),
        subcategories=subs,
    )


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def get_asset_overview() -> AssetOverview:
    """Scan session assets and return aggregated stats."""
    t0 = time.monotonic()
    categories = [
        get_sessions_detail(),
    ]
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    total = sum(c.total_bytes for c in categories)
    return AssetOverview(
        categories=categories,
        total_bytes=total,
        scan_time_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def _remove_empty_dirs(root: Path) -> None:
    """Remove empty leaf directories under root (bottom-up)."""
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(str(root), topdown=False):
        if not filenames and not dirnames:
            p = Path(dirpath)
            if p != root:
                try:
                    p.rmdir()
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Cleanup functions
# ---------------------------------------------------------------------------

def clean_sessions(
    session_ids: Optional[list[str]] = None,
    older_than_days: Optional[int] = None,
    empty_only: bool = False,
    exclude_ids: Optional[set[str]] = None,
    dry_run: bool = False,
) -> dict:
    """Delete sessions; filter by ID, age, or emptiness."""
    root = _sessions_dir()
    if not root.exists():
        return {"deleted_count": 0, "freed_bytes": 0, "freed_human": "0 B", "dry_run": dry_run}

    exclude = exclude_ids or set()
    deleted_count = 0
    freed_bytes = 0

    if session_ids:
        # Delete specific sessions
        for sid in session_ids:
            _validate_name(sid)
            if sid in exclude:
                continue
            session_path = root / sid
            if not session_path.exists() or not session_path.is_dir():
                continue
            stats = _scan_directory(session_path)
            deleted_count += stats.file_count
            freed_bytes += stats.total_bytes
            if not dry_run:
                shutil.rmtree(session_path, ignore_errors=True)
    else:
        # Iterate all sessions
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name in exclude:
                continue

            # Age filter
            if older_than_days is not None:
                stats = _scan_directory(child)
                if stats.newest_mtime is not None:
                    cutoff = time.time() - older_than_days * 86400
                    if stats.newest_mtime >= cutoff:
                        continue

            # Empty filter
            if empty_only:
                turn_count = _get_turn_count(child)
                if turn_count is None or turn_count > 0:
                    continue

            stats = _scan_directory(child)
            deleted_count += stats.file_count
            freed_bytes += stats.total_bytes
            if not dry_run:
                shutil.rmtree(child, ignore_errors=True)

    return {
        "deleted_count": deleted_count,
        "freed_bytes": freed_bytes,
        "freed_human": format_bytes(freed_bytes),
        "dry_run": dry_run,
    }


def _get_turn_count(session_path: Path) -> Optional[int]:
    """Read turn_count from metadata or count user messages in events."""
    meta_path = session_path / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            tc = meta.get("turn_count")
            if tc is not None:
                return tc
        except (json.JSONDecodeError, OSError):
            pass
    events_path = session_path / "events.jsonl"
    if events_path.exists():
        try:
            count = 0
            with open(events_path, "r", encoding="utf-8") as f:
                for line in f:
                    if '"user_message"' in line or '"USER_MESSAGE"' in line:
                        count += 1
            return count
        except OSError:
            pass
    return None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_bytes(n: int) -> str:
    """Human-readable size (e.g., '2.3 GB')."""
    if n < 1024:
        return f"{n} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
    return f"{n:.1f} TB"
