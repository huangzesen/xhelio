"""LLM abstraction layer — provider-agnostic interface for LLM interactions.

Re-exports the public API so consumers can write:
    from agent.llm import LLMService, ChatSession, LLMResponse, ...
"""

from .base import (
    LLMAdapter,
    LLMResponse,
    ToolCall,
    UsageMetadata,
    ChatSession,
    FunctionSchema,
)
from .interface import ToolResultBlock
from .service import LLMService

# Concrete adapters — prefer LLMService for new code; these re-exports
# exist for tests and scripts that construct adapters directly.
from .gemini_adapter import GeminiAdapter
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .minimax_adapter import MiniMaxAdapter
from .rate_limiter import RateLimiter
