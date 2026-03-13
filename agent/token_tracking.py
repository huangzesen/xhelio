"""Token tracking helpers — aggregates usage across orchestrator + sub-agents.

Free functions taking an OrchestratorAgent instance as the first argument.
Uses DelegationBus for sub-agent iteration (no direct _sub_agents access).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.orchestrator_agent import OrchestratorAgent


def get_token_usage(orch: "OrchestratorAgent") -> dict:
    """Return cumulative token usage for this session (orchestrator + all sub-agents)."""
    # Orchestrator's own usage (from BaseAgent tracking)
    orch_usage = orch.get_token_usage()
    input_tokens = orch_usage["input_tokens"]
    output_tokens = orch_usage["output_tokens"]
    thinking_tokens = orch_usage["thinking_tokens"]
    cached_tokens = orch_usage["cached_tokens"]
    api_calls = orch_usage["api_calls"]

    # Include usage from all active sub-agents + retired ephemerals
    delegation = getattr(orch.session_ctx, "delegation", None) if orch.session_ctx else None
    if delegation is not None:
        active_agents, retired = delegation.get_all_agent_usages()
        for _agent_id, usage in active_agents:
            input_tokens += usage["input_tokens"]
            output_tokens += usage["output_tokens"]
            thinking_tokens += usage.get("thinking_tokens", 0)
            cached_tokens += usage.get("cached_tokens", 0)
            api_calls += usage["api_calls"]
        for retired_usage in retired:
            input_tokens += retired_usage["input_tokens"]
            output_tokens += retired_usage["output_tokens"]
            thinking_tokens += retired_usage.get("thinking_tokens", 0)
            cached_tokens += retired_usage.get("cached_tokens", 0)
            api_calls += retired_usage["api_calls"]

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": input_tokens + output_tokens + thinking_tokens,
        "api_calls": api_calls,
    }


def get_token_usage_breakdown(orch: "OrchestratorAgent") -> list[dict]:
    """Return per-agent token usage breakdown.

    Returns a list of dicts with keys: agent, input, output, thinking, cached, calls.
    Only includes agents that have made at least one API call.
    """
    rows = []

    def _add(name, usage):
        if usage["api_calls"] > 0:
            rows.append(
                {
                    "agent": name,
                    "input": usage["input_tokens"],
                    "output": usage["output_tokens"],
                    "thinking": usage.get("thinking_tokens", 0),
                    "cached": usage.get("cached_tokens", 0),
                    "calls": usage["api_calls"],
                    "ctx_system": usage.get("ctx_system_tokens", 0),
                    "ctx_tools": usage.get("ctx_tools_tokens", 0),
                    "ctx_history": usage.get("ctx_history_tokens", 0),
                    "ctx_total": usage.get("ctx_total_tokens", 0),
                }
            )

    # Orchestrator's own usage (from BaseAgent tracking)
    _add("Orchestrator", orch.get_token_usage())

    # Sub-agents (active + retired)
    delegation = getattr(orch.session_ctx, "delegation", None) if orch.session_ctx else None
    if delegation is not None:
        active_agents, retired = delegation.get_all_agent_usages()
        for agent_id, usage in active_agents:
            _add(agent_id, usage)
        for retired_usage in retired:
            rows.append(
                {
                    "agent": retired_usage.get("agent", "retired"),
                    "input": retired_usage["input_tokens"],
                    "output": retired_usage["output_tokens"],
                    "thinking": retired_usage.get("thinking_tokens", 0),
                    "cached": retired_usage.get("cached_tokens", 0),
                    "calls": retired_usage["api_calls"],
                    "ctx_system": retired_usage.get("ctx_system_tokens", 0),
                    "ctx_tools": retired_usage.get("ctx_tools_tokens", 0),
                    "ctx_history": retired_usage.get("ctx_history_tokens", 0),
                    "ctx_total": retired_usage.get("ctx_total_tokens", 0),
                }
            )

    return rows
