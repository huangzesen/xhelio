"""Memory tool handlers."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent


def handle_review_memory(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from datetime import datetime as _dt
    from agent.memory import Memory, generate_tags
    import threading

    reviewer_agent_id = tool_args.pop("_reviewer_agent_id", None)
    memory_id = tool_args.get("memory_id", "")
    stars_raw = tool_args.get("stars")
    try:
        stars = int(stars_raw)
    except (TypeError, ValueError):
        stars = 0
    if not memory_id or stars < 1 or stars > 5:
        return {
            "status": "error",
            "message": "Invalid memory_id or stars (must be 1-5)",
        }

    # Assemble comment from structured fields
    rating = tool_args.get("rating", "").strip()
    criticism = tool_args.get("criticism", "").strip()
    suggestion = tool_args.get("suggestion", "").strip()
    comment_text = tool_args.get("comment", "").strip()

    if not all([rating, criticism, suggestion, comment_text]):
        return {
            "status": "error",
            "message": "All four fields required: rating, criticism, suggestion, comment",
        }

    comment = (
        f"(1) Rating: {rating}\n"
        f"(2) Criticism: {criticism}\n"
        f"(3) Suggestion: {suggestion}\n"
        f"(4) Comment: {comment_text}"
    )

    entry = orch._memory_store.get_by_id(memory_id)
    if entry is None or entry.archived:
        return {"status": "error", "message": f"Memory {memory_id} not found"}
    agent_name = reviewer_agent_id or orch._active_agent_name
    from agent.model_fallback import get_active_model

    model_name = get_active_model(orch.model_name)
    content = f"{stars}★ {comment.strip()}"
    tags = [f"review:{memory_id}", agent_name, f"stars:{stars}"]
    existing_review = orch._memory_store.get_review_for(memory_id, agent=agent_name)
    supersedes = ""
    version = 1
    if existing_review:
        supersedes = existing_review.id
        version = existing_review.version + 1
        existing_review.archived = True
    review_memory = Memory(
        type="review",
        scopes=list(entry.scopes),
        content=content,
        source="extracted",
        source_session=orch._session_id or "",
        tags=tags,
        review_of=memory_id,
        supersedes=supersedes,
        version=version,
    )
    orch._memory_store.add_no_save(review_memory)
    threading.Thread(target=orch._memory_store.save, daemon=False).start()
    from agent.event_bus import DEBUG
    orch._event_bus.emit(
        DEBUG,
        level="debug",
        msg=f"[Memory] Review {memory_id} by {agent_name}: {stars}★",
    )
    return {
        "status": "success",
        "memory_id": memory_id,
        "stars": stars,
        "reviewer": agent_name,
    }
