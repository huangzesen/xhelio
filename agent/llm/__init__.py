"""LLM abstraction layer — provider-agnostic interface for LLM interactions.

Re-exports the public API so consumers can write:
    from agent.llm import LLMAdapter, GeminiAdapter, OpenAIAdapter, LLMResponse, ...
"""

from .base import LLMAdapter, LLMResponse, ToolCall, UsageMetadata, ChatSession, FunctionSchema
from .gemini_adapter import GeminiAdapter
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
