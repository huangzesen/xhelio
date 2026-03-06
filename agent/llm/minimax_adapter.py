from agent.logging import get_logger
from .anthropic_adapter import AnthropicAdapter
from .base import LLMResponse
from .rate_limiter import RateLimiter

logger = get_logger()


class _RateLimitedSession:
    """Wraps a chat session, adding rate limiting before API calls."""

    def __init__(self, inner, limiter: RateLimiter):
        self._inner = inner
        self._limiter = limiter

    def send(self, message):
        self._limiter.wait()
        return self._inner.send(message)

    def send_stream(self, message, on_chunk=None):
        self._limiter.wait()
        return self._inner.send_stream(message, on_chunk=on_chunk)

    def get_history(self):
        return self._inner.get_history()

    def update_tools(self, tools):
        self._inner.update_tools(tools)

    def update_system_prompt(self, system_prompt: str) -> None:
        self._inner.update_system_prompt(system_prompt)

    def reset(self) -> None:
        """Delegate reset to the inner session."""
        self._inner.reset()

    def context_window(self) -> int:
        return self._inner.context_window()

    def estimate_context_tokens(self) -> int:
        return self._inner.estimate_context_tokens()

    def compact(self, summarizer=None) -> bool:
        return self._inner.compact(summarizer=summarizer)

    @property
    def interaction_id(self) -> str | None:
        return getattr(self._inner, "interaction_id", None)


class MiniMaxAdapter(AnthropicAdapter):
    def __init__(
        self, api_key: str, *, base_url: str | None = None, timeout_ms: int = 300_000
    ):
        import config as cfg

        effective_url = (
            base_url
            or cfg._provider_get("base_url")
            or "https://api.minimaxi.com/anthropic"
        )
        super().__init__(api_key=api_key, base_url=effective_url, timeout_ms=timeout_ms)

        # Set up rate limiting using base class
        interval = cfg._provider_get("rate_limit_interval", 2.0)
        self._setup_rate_limiter(float(interval))

    def create_chat(self, *args, **kwargs):
        session = super().create_chat(*args, **kwargs)
        # Wrap session with rate limiting if enabled
        if self._rate_limiter:
            return _RateLimitedSession(session, self._rate_limiter)
        return session

    def generate(self, *args, **kwargs) -> LLMResponse:
        if self._rate_limiter:
            self._rate_limiter.wait()
        return super().generate(*args, **kwargs)

    def web_search(self, query: str, model: str) -> LLMResponse:
        """Execute web search via MiniMax Coding Plan MCP server."""
        import config as cfg

        if not cfg._provider_get("web_search", True):
            return LLMResponse(text="")

        if self._rate_limiter:
            self._rate_limiter.wait()
        try:
            from agent.minimax_mcp_client import get_minimax_mcp_client

            client = get_minimax_mcp_client()
            result = client.call_tool("web_search", {"query": query})

            if result.get("status") == "error":
                logger.warning(
                    "MiniMax MCP web search error: %s", result.get("message")
                )
                return LLMResponse(text="")

            text = result.get("text", "") or result.get("answer", "")
            if not text and "organic" in result:
                parts = []
                for item in result["organic"]:
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")
                    link = item.get("link", "")
                    parts.append(f"**{title}**\n{snippet}\nSource: {link}")
                text = "\n\n".join(parts)
            if not text:
                text = str(result)
            return LLMResponse(text=text)
        except Exception as e:
            logger.warning("MiniMax MCP web search failed: %s", e)
            return LLMResponse(text="")

    def make_multimodal_message(
        self, text: str, image_bytes: bytes, mime_type: str = "image/png"
    ) -> dict:
        logger.warning("MiniMax Anthropic-compatible API does not support image input")
        return {"role": "user", "content": [{"type": "text", "text": text}]}
