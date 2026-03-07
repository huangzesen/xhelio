"""
Tests for agent.token_counter — centralized Gemini-based token counting.

Run with: python -m pytest tests/test_token_counter.py -v
"""

from unittest.mock import MagicMock

import pytest

from agent.token_counter import count_tokens


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_basic_text(self):
        result = count_tokens("hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_longer_text(self):
        short = count_tokens("hi")
        long = count_tokens("This is a much longer sentence with many more tokens")
        assert long > short

    def test_returns_consistent_results(self):
        text = "The quick brown fox jumps over the lazy dog."
        a = count_tokens(text)
        b = count_tokens(text)
        assert a == b

    def test_whitespace_only(self):
        result = count_tokens("   ")
        assert isinstance(result, int)
        # Whitespace-only should still produce a count (tokenizer dependent)
        assert result >= 0

    def test_special_characters(self):
        result = count_tokens("Hello! @#$%^&*() 你好世界")
        assert isinstance(result, int)
        assert result > 0

    def test_multiline(self):
        text = "Line one\nLine two\nLine three"
        result = count_tokens(text)
        assert result > 0


class TestBackwardCompat:
    """Verify backward compatibility through agent.memory re-exports."""

    def test_memory_estimate_tokens(self):
        from agent.memory import estimate_tokens
        result = estimate_tokens("hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_memory_estimate_memory_tokens(self):
        from agent.memory import estimate_memory_tokens, Memory
        m = Memory(type="preference", content="test content")
        result = estimate_memory_tokens(m)
        assert isinstance(result, int)
        assert result > 0

    def test_consistency_with_count_tokens(self):
        from agent.memory import estimate_tokens
        text = "The quick brown fox"
        assert estimate_tokens(text) == count_tokens(text)
