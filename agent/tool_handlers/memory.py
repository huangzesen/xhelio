"""Memory tool handlers."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent


def handle_recall_memories(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from dataclasses import asdict as _asdict

    query = tool_args.get("query", "")
    mem_type = tool_args.get("type")
    scope = tool_args.get("scope")
    limit = tool_args.get("limit", 20)
    if query:
        results = orch._memory_store.search(
            query,
            mem_type=mem_type,
            scope=scope,
            limit=limit,
        )
        results = [_asdict(m) for m in results]
    else:
        all_memories = [_asdict(m) for m in orch._memory_store.get_enabled()]
        if mem_type:
            all_memories = [m for m in all_memories if m.get("type") == mem_type]
        if scope:
            all_memories = [m for m in all_memories if scope in m.get("scopes", [])]
        results = all_memories[-limit:]
    return {
        "status": "success",
        "count": len(results),
        "memories": results,
    }


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
    comment = tool_args.get("comment", "")
    if not memory_id or stars < 1 or stars > 5:
        return {
            "status": "error",
            "message": "Invalid memory_id or stars (must be 1-5)",
        }
    if not isinstance(comment, str) or not comment.strip():
        return {"status": "error", "message": "Comment required"}
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
