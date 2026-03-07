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
from .interface import ChatInterface, ToolResultBlock
from .service import LLMService
from .rate_limiter import RateLimiter

# Concrete adapters — prefer LLMService for new code; these re-exports
# exist for tests and scripts that construct adapters directly.
from .gemini.adapter import GeminiAdapter
from .openai.adapter import OpenAIAdapter
from .anthropic.adapter import AnthropicAdapter
from .minimax.adapter import MiniMaxAdapter
