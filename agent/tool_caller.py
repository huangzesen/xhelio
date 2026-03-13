"""Tool caller identity and per-agent-type state dataclasses.

ToolCaller is passed as the third argument to every tool handler,
identifying which agent is calling. Agent-specific state lives on
ctx.agent_state[agent_type], not on the caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCaller:
    """Identity of the agent invoking a tool.

    Passed as the third argument to every tool handler function.
    Carries agent identity and tool call ID for logging/pairing.
    Agent-specific state lives on ctx.agent_state[agent_type].
    """
    agent_id: str
    agent_type: str
    tool_call_id: str | None = None


@dataclass
class OrchestratorState:
    """State owned by the orchestrator agent.

    Stored in ctx.agent_state["orchestrator"]. Accessed by
    orchestrator tools via ctx.agent_state["orchestrator"].
    """
    # Planning
    current_plan: Any | None = None  # dict or None

    # Inline completions (titles, follow-ups)
    inline: Any | None = None  # InlineCompletions

    # Context tracking
    ctx_tracker: Any | None = None  # ContextTracker

    # Event feed (user event polling)
    event_feed: Any | None = None  # EventFeed

    # Session persistence state
    auto_save: bool = False
    session_title_generated: bool = False
    token_log_listener: Any | None = None  # TokenLogListener
    event_log_writer: Any | None = None  # EventLogWriter
