"""Tool handler for manage_figure — figure asset management."""

from __future__ import annotations

import ipaddress
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from agent.event_bus import MPL_RENDER_EXECUTED

if TYPE_CHECKING:
    from agent.tool_caller import ToolCaller
    from agent.tool_context import ToolContext

_MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB
_ALLOWED_MIME_TYPES = {
    "image/png", "image/jpeg", "image/webp", "image/gif",
}
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _is_private_url(url: str) -> bool:
    """Check if URL targets a private/reserved IP address."""
    try:
        hostname = urlparse(url).hostname
        if hostname is None:
            return True
        import socket
        addr_info = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in addr_info:
            ip = ipaddress.ip_address(sockaddr[0])
            for net in _BLOCKED_NETWORKS:
                if ip in net:
                    return True
    except (socket.gaierror, ValueError):
        return True
    return False


def handle_manage_figure(
    ctx: "ToolContext", tool_args: dict, caller: "ToolCaller" = None
) -> dict:
    """Dispatch manage_figure actions."""
    action = tool_args.get("action", "")

    if action == "list":
        return _handle_list(ctx, tool_args)
    elif action == "show":
        return _handle_show(ctx, tool_args)
    elif action == "save_from_url":
        return _handle_save_from_url(ctx, tool_args)
    elif action == "export":
        return _handle_export(ctx, tool_args)
    elif action == "delete":
        return _handle_delete(ctx, tool_args)
    elif action == "restore":
        return _handle_restore(ctx)

    return {
        "status": "error",
        "message": f"Unknown manage_figure action: {action!r}. "
        f"Use 'list', 'show', 'save_from_url', 'export', 'delete', or 'restore'.",
    }


def _handle_list(ctx, tool_args: dict) -> dict:
    ar = ctx.asset_registry
    if ar is None:
        return {"status": "error", "message": "Asset registry not available"}

    figure_kind = tool_args.get("figure_kind")
    figures = ar.list_assets(kind="figure")

    if figure_kind:
        # Filter by figure_kind
        filtered = []
        for fig in figures:
            asset = ar.get_asset(fig["asset_id"])
            if asset and asset.figure_kind == figure_kind:
                filtered.append(fig)
        figures = filtered

    return {"status": "success", "figures": figures, "count": len(figures)}


def _handle_show(ctx, tool_args: dict) -> dict:
    asset_id = tool_args.get("asset_id", "")
    if not asset_id:
        return {"status": "error", "message": "asset_id is required"}

    ar = ctx.asset_registry
    if ar is None:
        return {"status": "error", "message": "Asset registry not available"}

    asset = ar.get_asset(asset_id)
    if asset is None:
        return {"status": "error", "message": f"Asset '{asset_id}' not found"}

    image_path = asset.metadata.get("image_path") or asset.metadata.get("thumbnail_path")
    if not image_path:
        return {"status": "error", "message": f"No image file for asset '{asset_id}'"}

    image_path = Path(image_path)
    if not image_path.exists():
        return {"status": "error", "message": f"Image file not found: {image_path}"}

    script_id = image_path.stem
    if ctx.event_bus is not None:
        ctx.event_bus.emit(
            MPL_RENDER_EXECUTED,
            agent="orchestrator",
            msg=f"[Figure] Showing: {asset.name}",
            data={
                "script_id": script_id,
                "description": asset.name,
                "output_path": str(image_path),
                "args": {"asset_id": asset_id},
                "inputs": [],
                "outputs": [],
                "status": "success",
            },
        )

    return {
        "status": "success",
        "asset_id": asset_id,
        "script_id": script_id,
        "name": asset.name,
    }


def _handle_save_from_url(ctx, tool_args: dict) -> dict:
    import requests

    url = tool_args.get("url")
    name = tool_args.get("name")

    if not url:
        return {"status": "error", "message": "url is required"}
    if not name:
        return {"status": "error", "message": "name is required"}

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return {"status": "error", "message": f"Only http/https allowed, got: {parsed.scheme}"}

    if _is_private_url(url):
        return {"status": "error", "message": "URLs targeting private/reserved IPs are blocked"}

    try:
        resp = requests.get(url, timeout=(10, 30))
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"status": "error", "message": f"Download failed: {e}"}

    content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if content_type not in _ALLOWED_MIME_TYPES:
        return {"status": "error", "message": f"Unsupported type: {content_type}"}

    image_bytes = resp.content
    if len(image_bytes) > _MAX_IMAGE_SIZE:
        return {"status": "error", "message": f"Image too large ({len(image_bytes)} bytes)"}

    script_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
    output_dir = ctx.session_dir / "mpl_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{script_id}.png"
    output_path.write_bytes(image_bytes)

    ar = ctx.asset_registry
    asset_id = None
    if ar is not None:
        meta = ar.register_image(name=name, image_path=str(output_path), source_url=url)
        meta.figure_kind = "image"
        asset_id = meta.asset_id

    if ctx.event_bus is not None:
        ctx.event_bus.emit(
            MPL_RENDER_EXECUTED,
            agent="orchestrator",
            msg=f"[Figure] Saved: {name}",
            data={
                "script_id": script_id,
                "description": name,
                "output_path": str(output_path),
                "args": {"url": url, "name": name},
                "inputs": [],
                "outputs": [],
                "status": "success",
            },
        )

    return {
        "status": "success",
        "asset_id": asset_id,
        "script_id": script_id,
        "name": name,
        "output_path": str(output_path),
    }


def _handle_export(ctx, tool_args: dict) -> dict:
    asset_id = tool_args.get("asset_id", "")
    if not asset_id:
        return {"status": "error", "message": "asset_id is required"}

    ar = ctx.asset_registry
    if ar is None:
        return {"status": "error", "message": "Asset registry not available"}

    asset = ar.get_asset(asset_id)
    if asset is None:
        return {"status": "error", "message": f"Asset '{asset_id}' not found"}

    fmt = tool_args.get("format", "png")

    # Resolve safe export destination — strip directory components
    export_dir = ctx.session_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    # For image assets — direct file copy
    image_path = asset.metadata.get("image_path")
    if image_path:
        src = Path(image_path)
        if not src.exists():
            return {"status": "error", "message": f"Image file not found: {src}"}
        safe_name = Path(tool_args.get("filename", f"{asset.name}.{fmt}")).name
        dest = export_dir / safe_name
        import shutil
        shutil.copy2(str(src), str(dest))
        return {
            "status": "success",
            "filepath": str(dest.resolve()),
            "format": fmt,
        }

    # For Plotly figures — export via renderer
    if asset.figure_kind == "plotly" or asset.metadata.get("fig_json"):
        safe_name = Path(tool_args.get("filename", f"figure_{asset_id}.{fmt}")).name
        dest = export_dir / safe_name
        result = ctx.renderer.export(str(dest), format=fmt)
        return result

    # For thumbnail-only — copy thumbnail
    thumb = asset.metadata.get("thumbnail_path")
    if thumb:
        src = Path(thumb)
        if src.exists():
            safe_name = Path(tool_args.get("filename", f"{asset.name}.png")).name
            dest = export_dir / safe_name
            import shutil
            shutil.copy2(str(src), str(dest))
            return {"status": "success", "filepath": str(dest.resolve()), "format": "png"}

    return {"status": "error", "message": f"No exportable content for asset '{asset_id}'"}


def _handle_delete(ctx, tool_args: dict) -> dict:
    asset_id = tool_args.get("asset_id", "")
    if not asset_id:
        return {"status": "error", "message": "asset_id is required"}

    ar = ctx.asset_registry
    if ar is None:
        return {"status": "error", "message": "Asset registry not available"}

    removed = ar.remove_figure(asset_id)
    if not removed:
        return {"status": "error", "message": f"Figure asset '{asset_id}' not found"}

    return {"status": "success", "deleted": asset_id}


def _handle_restore(ctx) -> dict:
    from agent.session_persistence import handle_restore_plot
    return handle_restore_plot(ctx)
