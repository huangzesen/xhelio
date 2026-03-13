"""Tool handler for manage_files — file asset management."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING

from config import get_data_dir

if TYPE_CHECKING:
    from agent.tool_caller import ToolCaller
    from agent.tool_context import ToolContext


def _get_allowed_dirs(session_dir: Path | None = None) -> list[Path]:
    """Directories from which file registration is permitted."""
    dirs = [Path.home(), get_data_dir()]
    if session_dir is not None:
        dirs.append(session_dir)
    return dirs


def _validate_file_path(file_path: str, session_dir: Path | None = None) -> Path:
    """Validate that file exists and is within allowed directories.

    Returns resolved Path.
    Raises FileNotFoundError or ValueError.
    """
    p = Path(file_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    allowed = _get_allowed_dirs(session_dir)
    for d in allowed:
        try:
            p.relative_to(d)
            return p
        except ValueError:
            continue

    raise ValueError(
        f"File path '{p}' is outside allowed directories: "
        f"{[str(d) for d in allowed]}"
    )


def _is_subpath(path: Path, parent: Path) -> bool:
    """Check if path is inside parent directory."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def handle_manage_files(
    ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None
) -> dict:
    """Dispatch manage_files actions."""
    action = tool_args.get("action", "")

    if action == "list":
        return _handle_list(ctx)
    elif action == "register":
        return _handle_register(ctx, tool_args)
    elif action == "info":
        return _handle_info(ctx, tool_args)
    elif action == "prepare":
        return {
            "status": "error",
            "message": "The 'prepare' action has been removed. Use 'register' instead — it now copies files to session storage automatically.",
        }
    elif action == "delete":
        return _handle_delete(ctx, tool_args)

    return {
        "status": "error",
        "message": f"Unknown manage_files action: {action!r}. "
        f"Use 'list', 'register', 'info', or 'delete'.",
    }


def _handle_list(ctx) -> dict:
    ar = ctx.asset_registry
    if ar is None:
        return {"status": "error", "message": "Asset registry not available"}

    files = ar.list_assets(kind="file")
    return {"status": "success", "files": files, "count": len(files)}


def _handle_register(ctx, tool_args: dict) -> dict:
    import hashlib
    import shutil
    from datetime import datetime, timezone

    file_path = tool_args.get("file_path", "")
    if not file_path:
        return {"status": "error", "message": "file_path is required"}

    session_dir = ctx.session_dir
    try:
        resolved = _validate_file_path(file_path, session_dir)
    except (FileNotFoundError, ValueError) as e:
        return {"status": "error", "message": str(e)}

    ar = ctx.asset_registry
    if ar is None:
        return {"status": "error", "message": "Asset registry not available"}

    mime_type, _ = mimetypes.guess_type(str(resolved))
    size_bytes = resolved.stat().st_size
    name = tool_args.get("name") or resolved.name
    source_url = tool_args.get("source_url", "")
    original_filename = resolved.name

    # Check if file is inside session sandbox — if so, move to files/
    is_sandbox_file = (
        session_dir is not None
        and _is_subpath(resolved, session_dir / "sandbox")
    )

    if is_sandbox_file:
        # Generate hash-based asset ID
        hash_input = (source_url or str(resolved)) + datetime.now(timezone.utc).isoformat()
        short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        asset_id = f"file_{short_hash}"
        ext = resolved.suffix

        files_dir = session_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        dest = files_dir / f"{asset_id}{ext}"
        shutil.move(str(resolved), str(dest))

        asset = ar.register_file(
            filename=name,
            path=dest,
            size_bytes=size_bytes,
            mime_type=mime_type or "",
            source_path=str(dest),
            asset_id=asset_id,
        )
        asset.session_path = str(dest)
        asset.metadata["original_filename"] = original_filename
        if source_url:
            asset.metadata["source_url"] = source_url
        ar.save()

        return {
            "status": "success",
            "asset_id": asset.asset_id,
            "name": asset.name,
            "session_path": str(dest),
            "original_filename": original_filename,
            "size_bytes": size_bytes,
            "mime_type": mime_type or "",
        }
    else:
        # Copy external file to session files/ dir
        hash_input = str(resolved) + datetime.now(timezone.utc).isoformat()
        short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        asset_id = f"file_{short_hash}"
        ext = resolved.suffix

        if session_dir is not None:
            files_dir = session_dir / "files"
            files_dir.mkdir(parents=True, exist_ok=True)
            dest = files_dir / f"{asset_id}{ext}"
            shutil.copy2(str(resolved), str(dest))
            session_path = str(dest)
        else:
            dest = resolved
            session_path = None

        asset = ar.register_file(
            filename=name,
            path=dest,
            size_bytes=size_bytes,
            mime_type=mime_type or "",
            source_path=str(resolved),
            asset_id=asset_id,
        )
        if session_path:
            asset.session_path = session_path
            asset.metadata["original_filename"] = original_filename
            ar.save()

        result = {
            "status": "success",
            "asset_id": asset.asset_id,
            "name": asset.name,
            "size_bytes": size_bytes,
            "mime_type": mime_type or "",
            "extension": resolved.suffix.lower(),
        }
        if session_path:
            result["session_path"] = session_path
            result["original_filename"] = original_filename
        return result


def _handle_info(ctx, tool_args: dict) -> dict:
    asset_id = tool_args.get("asset_id", "")
    if not asset_id:
        return {"status": "error", "message": "asset_id is required"}

    ar = ctx.asset_registry
    if ar is None:
        return {"status": "error", "message": "Asset registry not available"}

    asset = ar.get_asset(asset_id)
    if asset is None or asset.kind != "file":
        return {"status": "error", "message": f"File asset '{asset_id}' not found"}

    return {
        "status": "success",
        "asset_id": asset.asset_id,
        "name": asset.name,
        "created_at": asset.created_at,
        "source_path": asset.source_path,
        "session_path": asset.session_path,
        "has_local_copy": asset.session_path is not None,
        "size_bytes": asset.metadata.get("size_bytes", 0),
        "mime_type": asset.metadata.get("mime_type", ""),
        "extension": asset.metadata.get("extension", ""),
    }


def _handle_delete(ctx, tool_args: dict) -> dict:
    asset_id = tool_args.get("asset_id", "")
    if not asset_id:
        return {"status": "error", "message": "asset_id is required"}

    ar = ctx.asset_registry
    if ar is None:
        return {"status": "error", "message": "Asset registry not available"}

    removed = ar.remove_file(asset_id)
    if not removed:
        return {"status": "error", "message": f"File asset '{asset_id}' not found"}

    return {"status": "success", "deleted": asset_id}
