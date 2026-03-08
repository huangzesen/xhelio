"""Centralized token usage tracking for orchestrator and sub-agents.

Encapsulates the 5-counter accumulation pattern (input, output, thinking,
cached, api_calls) that was previously repeated 4 times in core.py.
"""
from __future__ import annotations

from .llm_utils import track_llm_usage
from .llm import LLMResponse


class TokenTracker:
    """Tracks LLM token usage for one agent tier.

    Not thread-safe — intended to be called from a single thread (the
    agent's run-loop).  For cross-thread aggregation, use get_usage()
    which returns an immutable snapshot.
    """

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self._input = 0
        self._output = 0
        self._thinking = 0
        self._cached = 0
        self._api_calls = 0
        self._latest_input = 0

    def track(
        self,
        response: LLMResponse,
        last_tool_context: str = "",
        system_tokens: int = 0,
        tools_tokens: int = 0,
    ) -> None:
        """Accumulate usage from an LLM response."""
        token_state = {
            "input": self._input,
            "output": self._output,
            "thinking": self._thinking,
            "cached": self._cached,
            "api_calls": self._api_calls,
        }
        track_llm_usage(
            response=response,
            token_state=token_state,
            agent_name=self.agent_name,
            last_tool_context=last_tool_context,
            system_tokens=system_tokens,
            tools_tokens=tools_tokens,
        )
        self._input = token_state["input"]
        self._output = token_state["output"]
        self._thinking = token_state["thinking"]
        self._cached = token_state["cached"]
        self._api_calls = token_state["api_calls"]
        if response.usage:
            self._latest_input = response.usage.input_tokens

    def get_usage(self) -> dict[str, int]:
        """Return current cumulative usage as a dict."""
        return {
            "input_tokens": self._input,
            "output_tokens": self._output,
            "thinking_tokens": self._thinking,
            "cached_tokens": self._cached,
            "api_calls": self._api_calls,
        }

    @property
    def latest_input_tokens(self) -> int:
        """The input token count from the most recent LLM call."""
        return self._latest_input

    @property
    def api_calls(self) -> int:
        """Total number of API calls tracked."""
        return self._api_calls

    def restore(self, data: dict[str, int]) -> None:
        """Restore counters from persisted session data."""
        self._input = data.get("input_tokens", 0)
        self._output = data.get("output_tokens", 0)
        self._thinking = data.get("thinking_tokens", 0)
        self._cached = data.get("cached_tokens", 0)
        self._api_calls = data.get("api_calls", 0)
