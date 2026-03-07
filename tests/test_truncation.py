"""Tests for agent/truncation.py — central truncation registry."""

import pytest
from unittest.mock import patch

from agent.truncation import (
    DEFAULTS,
    ITEM_DEFAULTS,
    trunc,
    trunc_items,
    join_labels,
    child_summaries,
    get_limit,
    get_item_limit,
    reload,
)


# ---------------------------------------------------------------------------
# get_limit / get_item_limit — defaults and KeyError
# ---------------------------------------------------------------------------


class TestGetLimit:
    def test_returns_default(self):
        assert get_limit("console.summary") == DEFAULTS["console.summary"]

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="Unknown truncation limit"):
            get_limit("nonexistent.limit")

    def test_all_defaults_are_positive_int(self):
        for name, value in DEFAULTS.items():
            assert isinstance(value, int), f"{name} default is not int"
            assert value > 0, f"{name} default is not positive"


class TestGetItemLimit:
    def test_returns_default(self):
        assert get_item_limit("items.tool_args") == ITEM_DEFAULTS["items.tool_args"]

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="Unknown item limit"):
            get_item_limit("nonexistent.limit")

    def test_all_defaults_are_positive_int(self):
        for name, value in ITEM_DEFAULTS.items():
            assert isinstance(value, int), f"{name} default is not int"
            assert value > 0, f"{name} default is not positive"


# ---------------------------------------------------------------------------
# trunc
# ---------------------------------------------------------------------------


class TestTrunc:
    def test_short_text_unchanged(self):
        assert trunc("hello", "console.summary") == "hello"

    def test_exact_length_unchanged(self):
        text = "x" * 300
        assert trunc(text, "console.summary") == text

    def test_long_text_truncated_with_ellipsis(self):
        text = "x" * 500
        result = trunc(text, "console.summary")
        assert result.endswith("...")
        assert len(result) == 300  # default for console.summary

    def test_unknown_limit_raises(self):
        with pytest.raises(KeyError):
            trunc("hello", "bogus.limit")

    def test_zero_limit_disables_truncation(self):
        """Config override of 0 means no truncation."""
        import agent.truncation as mod
        old = mod._text_overrides.copy()
        try:
            mod._text_overrides["console.summary"] = 0
            long_text = "x" * 1000
            assert trunc(long_text, "console.summary") == long_text
        finally:
            mod._text_overrides.clear()
            mod._text_overrides.update(old)

    def test_empty_string(self):
        assert trunc("", "console.summary") == ""


# ---------------------------------------------------------------------------
# trunc_items
# ---------------------------------------------------------------------------


class TestTruncItems:
    def test_short_list_unchanged(self):
        items = [1, 2]
        result, total = trunc_items(items, "items.tool_args")
        assert result == [1, 2]
        assert total == 2

    def test_long_list_truncated(self):
        items = list(range(10))
        result, total = trunc_items(items, "items.tool_args")
        assert len(result) == 3  # default for items.tool_args
        assert total == 10

    def test_exact_length_unchanged(self):
        items = [1, 2, 3]
        result, total = trunc_items(items, "items.tool_args")
        assert result == [1, 2, 3]
        assert total == 3

    def test_unknown_limit_raises(self):
        with pytest.raises(KeyError):
            trunc_items([1], "bogus.limit")

    def test_zero_limit_returns_all(self):
        import agent.truncation as mod
        old = mod._item_overrides.copy()
        try:
            mod._item_overrides["items.tool_args"] = 0
            items = list(range(100))
            result, total = trunc_items(items, "items.tool_args")
            assert result == items
            assert total == 100
        finally:
            mod._item_overrides.clear()
            mod._item_overrides.update(old)

    def test_empty_list(self):
        result, total = trunc_items([], "items.tool_args")
        assert result == []
        assert total == 0


# ---------------------------------------------------------------------------
# join_labels
# ---------------------------------------------------------------------------


class TestJoinLabels:
    def test_empty_returns_none_placeholder(self):
        assert join_labels([], "console.summary") == "(none)"

    def test_short_labels_joined(self):
        result = join_labels(["a", "b", "c"], "console.summary")
        assert result == "a, b, c"

    def test_long_labels_truncated(self):
        labels = [f"label_{i:03d}" for i in range(200)]
        result = join_labels(labels, "console.args.value")  # limit=300
        assert result.endswith("...")
        assert len(result) == 300


# ---------------------------------------------------------------------------
# child_summaries
# ---------------------------------------------------------------------------


class TestChildSummaries:
    def test_empty_children(self):
        assert child_summaries([]) == ""

    def test_within_limit(self):
        children = [
            {"summary": "Fetched data"},
            {"summary": "Computed magnitude"},
        ]
        result = child_summaries(children)
        assert "Fetched data" in result
        assert "Computed magnitude" in result

    def test_over_limit_shows_count(self):
        children = [{"summary": f"Step {i}"} for i in range(10)]
        result = child_summaries(children, "items.child_events")
        assert "and 5 more" in result

    def test_msg_fallback(self):
        children = [{"msg": "hello"}]
        result = child_summaries(children)
        assert "hello" in result


# ---------------------------------------------------------------------------
# Config overrides via reload()
# ---------------------------------------------------------------------------


class TestConfigOverrides:
    def test_text_override_takes_effect(self):
        import agent.truncation as mod
        old = mod._text_overrides.copy()
        try:
            mod._text_overrides["console.summary"] = 10
            assert get_limit("console.summary") == 10
            result = trunc("a" * 50, "console.summary")
            assert len(result) == 10
            assert result.endswith("...")
        finally:
            mod._text_overrides.clear()
            mod._text_overrides.update(old)

    def test_item_override_takes_effect(self):
        import agent.truncation as mod
        old = mod._item_overrides.copy()
        try:
            mod._item_overrides["items.tool_args"] = 1
            assert get_item_limit("items.tool_args") == 1
            result, total = trunc_items([1, 2, 3], "items.tool_args")
            assert result == [1]
            assert total == 3
        finally:
            mod._item_overrides.clear()
            mod._item_overrides.update(old)

    def test_reload_reads_config(self):
        """reload() populates overrides from config.get()."""
        import agent.truncation as mod
        old_text = mod._text_overrides.copy()
        old_item = mod._item_overrides.copy()
        try:
            with patch("config.get") as mock_get:
                mock_get.side_effect = lambda key, default=None: (
                    {"console.summary": 42} if key == "truncation" else
                    {"items.labels": 7} if key == "truncation_items" else
                    default
                )
                reload()
                assert mod._text_overrides == {"console.summary": 42}
                assert mod._item_overrides == {"items.labels": 7}
                assert get_limit("console.summary") == 42
                assert get_item_limit("items.labels") == 7
        finally:
            mod._text_overrides.clear()
            mod._text_overrides.update(old_text)
            mod._item_overrides.clear()
            mod._item_overrides.update(old_item)


# ---------------------------------------------------------------------------
# New context/output limits — existence and correctness
# ---------------------------------------------------------------------------


class TestNewTextLimits:
    """Verify all new text limits added for LLM context migration."""

    NEW_TEXT_LIMITS = [
        ("context.document", 50000),
        ("context.dataset_docs", 4000),
        ("context.turn_text", 300),
        ("context.turn_text.discovery", 500),
        ("context.turn_text.inline", 200),
        ("context.discovery_search", 500),
        ("context.param_description", 60),
        ("context.docstring_summary", 200),
        ("context.mission_example", 57),
        ("context.task_outcome_error", 100),
        ("context.tool_args_sanitize", 200),
        ("context.session_preview", 40),
        ("output.reflection_tokens", 100),
        ("output.inline_tokens", 100),
    ]

    @pytest.mark.parametrize("name,expected_default", NEW_TEXT_LIMITS)
    def test_default_value(self, name, expected_default):
        assert get_limit(name) == expected_default

    @pytest.mark.parametrize("name,_", NEW_TEXT_LIMITS)
    def test_trunc_works(self, name, _):
        """trunc() with these names doesn't raise."""
        trunc("hello world", name)

    def test_context_document_zero_override_disables(self):
        """Setting context.document to 0 disables truncation."""
        import agent.truncation as mod
        old = mod._text_overrides.copy()
        try:
            mod._text_overrides["context.document"] = 0
            long_text = "x" * 100_000
            assert trunc(long_text, "context.document") == long_text
        finally:
            mod._text_overrides.clear()
            mod._text_overrides.update(old)


class TestNewItemLimits:
    """Verify all new item limits added for LLM context migration."""

    NEW_ITEM_LIMITS = [
        ("items.parameters", 10),
        ("items.catalog_functions", 8),
        ("items.mission_keywords", 2),
        ("items.events", 200),
        ("items.query_event_log", 100),
        ("items.data_preview_rows", 50),
        ("items.data_preview_xr", 10),
        ("items.data_sample_points", 20),
        ("items.ops_log_reflection", 15),
        ("items.error_summary", 3),
        ("items.conversation_window", 20),
        ("items.follow_up_turns", 6),
        ("items.inline_turns", 4),
        ("items.sessions_shown", 10),
        ("items.api_data_preview", 10),
        ("items.api_input_history", 200),
    ]

    @pytest.mark.parametrize("name,expected_default", NEW_ITEM_LIMITS)
    def test_default_value(self, name, expected_default):
        assert get_item_limit(name) == expected_default

    @pytest.mark.parametrize("name,_", NEW_ITEM_LIMITS)
    def test_trunc_items_works(self, name, _):
        """trunc_items() with these names doesn't raise."""
        trunc_items([1, 2, 3], name)

    def test_output_tokens_accessible(self):
        """Output token limits are accessible via get_limit (not get_item_limit)."""
        assert get_limit("output.reflection_tokens") == 100
        assert get_limit("output.inline_tokens") == 100
