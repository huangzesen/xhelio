"""Tests for EurekaHooks — eureka/insight hook management."""

from unittest.mock import MagicMock, patch
import threading

from agent.eureka_hooks import EurekaHooks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(**overrides):
    """Build a minimal mock SessionContext for EurekaHooks."""
    ctx = MagicMock()
    ctx.store = MagicMock()
    ctx.renderer = MagicMock()
    ctx.event_bus = MagicMock()
    ctx.memory_store = MagicMock()
    ctx.session_id = "test-session"
    ctx.delegation = MagicMock()
    ctx.delegation.lock = threading.Lock()
    ctx.delegation.agents = {}
    ctx.delegation.register_agent = MagicMock()
    ctx.service = MagicMock()
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_defaults(self):
        hooks = EurekaHooks(ctx=_make_ctx(), eureka_mode=False)
        assert hooks._agent is None
        assert hooks._mode is False
        assert hooks._round_counter == 0
        assert hooks._turn_counter == 0
        assert hooks._pending_suggestion is None
        assert hooks._insight_review_iter == 0
        assert hooks._latest_render_png is None

    def test_eureka_enabled(self):
        hooks = EurekaHooks(ctx=_make_ctx(), eureka_mode=True)
        assert hooks._mode is True

    def test_properties_read_write(self):
        hooks = EurekaHooks(ctx=_make_ctx())
        hooks.eureka_mode = True
        assert hooks.eureka_mode is True

        hooks.eureka_round_counter = 3
        assert hooks.eureka_round_counter == 3

        hooks.eureka_turn_counter = 7
        assert hooks.eureka_turn_counter == 7

        hooks.eureka_pending_suggestion = "test"
        assert hooks.eureka_pending_suggestion == "test"

        hooks.insight_review_iter = 2
        assert hooks.insight_review_iter == 2

        hooks.latest_render_png = b"png-bytes"
        assert hooks.latest_render_png == b"png-bytes"


# ---------------------------------------------------------------------------
# build_insight_context
# ---------------------------------------------------------------------------

class TestBuildInsightContext:
    def test_no_data(self):
        ctx = _make_ctx()
        ctx.renderer.get_current_state.return_value = {}
        ctx.store.list_entries.return_value = []
        hooks = EurekaHooks(ctx=ctx)
        result = hooks.build_insight_context()
        assert result == "No data context available."

    def test_with_traces(self):
        ctx = _make_ctx()
        ctx.renderer.get_current_state.return_value = {
            "traces": ["Bx", "By", "Bz"],
            "num_panels": 2,
        }
        ctx.store.list_entries.return_value = []
        hooks = EurekaHooks(ctx=ctx)
        result = hooks.build_insight_context()
        assert "Traces on plot" in result
        assert "Number of panels: 2" in result

    def test_matplotlib_fallback(self):
        ctx = _make_ctx()
        ctx.renderer.get_current_state.return_value = {}
        ctx.store.list_entries.return_value = []
        hooks = EurekaHooks(ctx=ctx)
        hooks._latest_render_png = b"fake-png"
        result = hooks.build_insight_context()
        assert "matplotlib" in result

    def test_with_store_entries(self):
        ctx = _make_ctx()
        ctx.renderer.get_current_state.return_value = {}
        ctx.store.list_entries.return_value = [
            {
                "label": "ACE_MAG",
                "num_points": 1000,
                "units": "nT",
                "time_min": "2024-01-01",
                "time_max": "2024-01-02",
                "columns": ["Bx", "By", "Bz"],
            }
        ]
        hooks = EurekaHooks(ctx=ctx)
        result = hooks.build_insight_context()
        assert "ACE_MAG" in result
        assert "1000 pts" in result
        assert "units=nT" in result

    def test_many_columns_truncated(self):
        ctx = _make_ctx()
        ctx.renderer.get_current_state.return_value = {}
        cols = [f"col_{i}" for i in range(10)]
        ctx.store.list_entries.return_value = [
            {"label": "test", "columns": cols}
        ]
        hooks = EurekaHooks(ctx=ctx)
        result = hooks.build_insight_context()
        assert "10 cols" in result


# ---------------------------------------------------------------------------
# sync_insight_review
# ---------------------------------------------------------------------------

class TestSyncInsightReview:
    def test_disabled(self):
        hooks = EurekaHooks(ctx=_make_ctx())
        with patch("config.INSIGHT_FEEDBACK", False):
            result = hooks.sync_insight_review()
            assert result is None

    def test_max_iters_reached(self):
        hooks = EurekaHooks(ctx=_make_ctx())
        hooks._insight_review_iter = 999
        with patch("config.INSIGHT_FEEDBACK", True), \
             patch("config.INSIGHT_FEEDBACK_MAX_ITERS", 3):
            result = hooks.sync_insight_review()
            assert result is None

    def test_no_figure_png(self):
        ctx = _make_ctx()
        hooks = EurekaHooks(ctx=ctx)
        with patch("config.INSIGHT_FEEDBACK", True), \
             patch("config.INSIGHT_FEEDBACK_MAX_ITERS", 10), \
             patch("agent.session_persistence.get_latest_figure_png", return_value=None):
            result = hooks.sync_insight_review()
            assert result is None

    def test_successful_pass_review(self):
        ctx = _make_ctx()
        # Mock user message event
        user_event = MagicMock()
        user_event.data = {"text": "Plot solar wind speed"}
        user_event.msg = "Plot solar wind speed"
        ctx.event_bus.get_events.return_value = [user_event]

        # Mock generate_vision response
        vision_response = MagicMock()
        vision_response.text = "VERDICT: PASS\nThe figure correctly shows solar wind speed."
        ctx.service.generate_vision.return_value = vision_response

        hooks = EurekaHooks(ctx=ctx)
        with patch("config.INSIGHT_FEEDBACK", True), \
             patch("config.INSIGHT_FEEDBACK_MAX_ITERS", 10), \
             patch("agent.session_persistence.get_latest_figure_png", return_value=b"fake-png"):
            result = hooks.sync_insight_review()

        assert result is not None
        assert result["passed"] is True
        assert "VERDICT: PASS" in result["text"]
        assert result["suggestions"] == []
        assert hooks._insight_review_iter == 1
        ctx.service.generate_vision.assert_called_once()
        ctx.event_bus.emit.assert_called_once()

    def test_successful_needs_improvement_review(self):
        ctx = _make_ctx()
        user_event = MagicMock()
        user_event.data = {"text": "Plot magnetic field"}
        user_event.msg = "Plot magnetic field"
        ctx.event_bus.get_events.return_value = [user_event]

        vision_response = MagicMock()
        vision_response.text = "VERDICT: NEEDS_IMPROVEMENT\n- Missing axis labels\n- Wrong time range"
        ctx.service.generate_vision.return_value = vision_response

        hooks = EurekaHooks(ctx=ctx)
        with patch("config.INSIGHT_FEEDBACK", True), \
             patch("config.INSIGHT_FEEDBACK_MAX_ITERS", 10), \
             patch("agent.session_persistence.get_latest_figure_png", return_value=b"fake-png"):
            result = hooks.sync_insight_review()

        assert result is not None
        assert result["passed"] is False
        assert "NEEDS_IMPROVEMENT" in result["text"]
        assert result["suggestions"] == []

    def test_no_vision_provider(self):
        ctx = _make_ctx()
        user_event = MagicMock()
        user_event.data = {"text": "Plot data"}
        user_event.msg = "Plot data"
        ctx.event_bus.get_events.return_value = [user_event]

        vision_response = MagicMock()
        vision_response.text = ""
        ctx.service.generate_vision.return_value = vision_response

        hooks = EurekaHooks(ctx=ctx)
        with patch("config.INSIGHT_FEEDBACK", True), \
             patch("config.INSIGHT_FEEDBACK_MAX_ITERS", 10), \
             patch("agent.session_persistence.get_latest_figure_png", return_value=b"fake-png"):
            result = hooks.sync_insight_review()

        assert result is not None
        assert result["failed"] is True
        assert "unavailable" in result["text"].lower()


# ---------------------------------------------------------------------------
# Eureka context and formatting
# ---------------------------------------------------------------------------

class TestBuildEurekaContext:
    def test_basic(self):
        ctx = _make_ctx()
        ctx.store.list_entries.return_value = [
            {"label": "ACE_MAG"},
            {"label": "WIND_SWE"},
        ]
        ctx.renderer.get_figure.return_value = MagicMock()
        ctx.event_bus.get_events.return_value = []
        hooks = EurekaHooks(ctx=ctx)
        result = hooks.build_eureka_context()
        assert result["session_id"] == "test-session"
        assert result["data_store_keys"] == ["ACE_MAG", "WIND_SWE"]
        assert result["has_figure"] is True

    def test_no_store(self):
        ctx = _make_ctx()
        ctx.store = None
        ctx.renderer.get_figure.return_value = None
        ctx.event_bus.get_events.return_value = []
        hooks = EurekaHooks(ctx=ctx)
        result = hooks.build_eureka_context()
        assert result["data_store_keys"] == []
        assert result["has_figure"] is False


class TestFormatEurekaSuggestion:
    def test_basic(self):
        hooks = EurekaHooks(ctx=_make_ctx())
        suggestion = MagicMock()
        suggestion.description = "Try plotting solar wind speed"
        suggestion.rationale = "Speed data is available"
        suggestion.parameters = {"dataset": "ACE_SWE"}
        result = hooks.format_eureka_suggestion_as_user_msg(suggestion)
        assert "[eureka]" in result
        assert "Try plotting solar wind speed" in result
        assert "Speed data is available" in result
        assert "ACE_SWE" in result

    def test_no_rationale_or_params(self):
        hooks = EurekaHooks(ctx=_make_ctx())
        suggestion = MagicMock()
        suggestion.description = "Zoom in on the shock"
        suggestion.rationale = None
        suggestion.parameters = None
        result = hooks.format_eureka_suggestion_as_user_msg(suggestion)
        assert "[eureka] Zoom in on the shock" in result
        assert "Rationale" not in result
        assert "Parameters" not in result


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_full_reset(self):
        hooks = EurekaHooks(ctx=_make_ctx(), eureka_mode=True)
        hooks._agent = MagicMock()
        hooks._turn_counter = 5
        hooks._round_counter = 3
        hooks._pending_suggestion = "something"
        hooks._insight_review_iter = 2
        hooks._latest_render_png = b"png"

        with patch("config.get", return_value=False):
            hooks.reset()

        assert hooks._agent is None
        assert hooks._turn_counter == 0
        assert hooks._round_counter == 0
        assert hooks._pending_suggestion is None
        assert hooks._insight_review_iter == 0
        assert hooks._latest_render_png is None

    def test_reset_per_message(self):
        hooks = EurekaHooks(ctx=_make_ctx())
        hooks._insight_review_iter = 3
        hooks._latest_render_png = b"png"
        hooks.reset_per_message()
        assert hooks._insight_review_iter == 0
        assert hooks._latest_render_png is None

    def test_reset_eureka_on_real_user_message(self):
        hooks = EurekaHooks(ctx=_make_ctx(), eureka_mode=True)
        hooks._round_counter = 3
        hooks._pending_suggestion = "test"
        hooks.reset_eureka_on_user_message("Show me ACE data")
        assert hooks._round_counter == 0
        assert hooks._pending_suggestion is None

    def test_reset_eureka_skipped_for_eureka_message(self):
        hooks = EurekaHooks(ctx=_make_ctx(), eureka_mode=True)
        hooks._round_counter = 3
        hooks._pending_suggestion = "test"
        hooks.reset_eureka_on_user_message("[eureka] Try something")
        assert hooks._round_counter == 3
        assert hooks._pending_suggestion == "test"

    def test_reset_eureka_skipped_when_mode_off(self):
        hooks = EurekaHooks(ctx=_make_ctx(), eureka_mode=False)
        hooks._round_counter = 3
        hooks._pending_suggestion = "test"
        hooks.reset_eureka_on_user_message("Show me data")
        # mode is off, so no reset
        assert hooks._round_counter == 3
