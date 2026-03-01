import json
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Secret — stays in .env (per-provider env vars: GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY)

# User config — loaded from ~/.xhelio/config.json (primary)
# or project-root config.json (fallback).
CONFIG_PATH = Path.home() / ".xhelio" / "config.json"
_LOCAL_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
_user_config: dict = {}


def _load_config() -> dict:
    # Project-local config.json as base, user home config overlaid on top
    merged: dict = {}
    for path in (_LOCAL_CONFIG_PATH, CONFIG_PATH):
        if path is not None and path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    merged.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
    return merged


def get(key: str, default=None):
    """Get a config value by dot-separated key. E.g. get('memory_token_budget', 10000)"""
    keys = key.split(".")
    val = _user_config
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
    return val if val is not None else default


_user_config = _load_config()


# ---- Data directory -----------------------------------------------------------
# Single source of truth for the base data directory (logs, sessions, memory, etc.).
# Priority: XHELIO_DIR env var > "data_dir" config key > ~/.xhelio

_data_dir: Optional[Path] = None


def get_data_dir() -> Path:
    """Return the resolved base data directory.

    Resolution order:
    1. ``XHELIO_DIR`` environment variable (highest — useful for CI/Docker)
    2. ``"data_dir"`` key in config.json
    3. ``~/.xhelio`` (default)
    """
    global _data_dir
    if _data_dir is not None:
        return _data_dir
    env_val = os.environ.get("XHELIO_DIR")
    if env_val:
        _data_dir = Path(env_val).expanduser().resolve()
    else:
        configured = get("data_dir")
        if configured:
            _data_dir = Path(configured).expanduser().resolve()
        else:
            _data_dir = Path.home() / ".xhelio"
    return _data_dir


def _reset_data_dir() -> None:
    """Reset the cached data directory (for testing only)."""
    global _data_dir
    _data_dir = None


# ---- LLM provider config ------------------------------------------------------
LLM_PROVIDER = get("llm_provider", "gemini")  # "gemini", "openai", "anthropic"

_PROVIDER_ENV_KEYS = {
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def get_api_key(provider: str | None = None) -> str | None:
    """Return the API key for the given provider.

    Each provider uses its own env var:
      gemini    → GOOGLE_API_KEY
      openai    → OPENAI_API_KEY
      anthropic → ANTHROPIC_API_KEY
    """
    p = (provider or LLM_PROVIDER).lower()
    env_key = _PROVIDER_ENV_KEYS.get(p)
    if env_key:
        return os.getenv(env_key)
    return None


# ---- Per-provider defaults ---------------------------------------------------
# Hardcoded defaults per provider. Used as final fallback when neither the
# providers.<active>.key nor a top-level key is set in config.json.
_PROVIDER_DEFAULTS = {
    "gemini": {
        "model": "gemini-3-flash-preview",
        "sub_agent_model": "gemini-3-flash-preview",
        "insight_model": "gemini-3-flash-preview",
        "inline_model": "gemini-2.5-flash-lite",
        "planner_model": "gemini-3-flash-preview",
        "fallback_model": "gemini-3-flash-preview",
        "base_url": None,
        "thinking_model": "high",
        "thinking_sub_agent": "high",
        "thinking_insight": "low",
    },
    "openai": {
        "model": "",
        "sub_agent_model": "",
        "insight_model": "",
        "inline_model": "",
        "planner_model": "",
        "fallback_model": "",
        "base_url": "",
        "use_responses_api": True,
        "compact_threshold": 100000,
        "thinking_model": "high",
        "thinking_sub_agent": "low",
        "thinking_insight": "low",
    },
    "anthropic": {
        "model": "MiniMax-M2.5-highspeed",
        "sub_agent_model": "MiniMax-M2.5-highspeed",
        "insight_model": "MiniMax-M2.5-highspeed",
        "inline_model": "MiniMax-M2.1",
        "planner_model": "MiniMax-M2.5-highspeed",
        "fallback_model": "MiniMax-M2.5-highspeed",
        "base_url": "https://api.minimaxi.com/anthropic",
        "thinking_model": "high",
        "thinking_sub_agent": "low",
        "thinking_insight": "low",
    },
}


# Legacy top-level key names → provider section key names
_LEGACY_KEY_MAP = {"base_url": "llm_base_url"}


def _provider_get(key: str, default=None):
    """Get a config value with provider-section priority.

    Resolution order:
    1. providers.<active_provider>.key  (provider-specific)
    2. Top-level key                    (backward compat / override)
    3. _PROVIDER_DEFAULTS[provider].key (hardcoded defaults)
    4. default argument
    """
    provider = get("llm_provider", "gemini")
    # 1. Provider section
    val = get(f"providers.{provider}.{key}")
    if val is not None:
        return val
    # 2. Top-level key (check canonical name, then legacy alias)
    val = get(key)
    if val is not None:
        return val
    legacy_key = _LEGACY_KEY_MAP.get(key)
    if legacy_key:
        val = get(legacy_key)
        if val is not None:
            return val
    # 3. Hardcoded provider defaults
    provider_defaults = _PROVIDER_DEFAULTS.get(provider, {})
    if key in provider_defaults:
        return provider_defaults[key]
    return default


# ---- Model tier constants ------------------------------------------------
# Four tiers: smart (orchestrator + planner), sub-agent, insight (multimodal
# plot analysis), inline (cheapest).
LLM_BASE_URL = _provider_get("base_url")
SMART_MODEL = _provider_get("model")
SUB_AGENT_MODEL = _provider_get("sub_agent_model")
INSIGHT_MODEL = _provider_get("insight_model") or SUB_AGENT_MODEL
INLINE_MODEL = _provider_get("inline_model")
PLANNER_MODEL = _provider_get("planner_model") or SMART_MODEL
FALLBACK_MODEL = _provider_get("fallback_model") or SUB_AGENT_MODEL
DATA_BACKEND = get("data_backend", "cdf")  # "cdf" only
CATALOG_SEARCH_METHOD = get(
    "catalog_search_method", "semantic"
)  # "semantic" or "substring"
PARALLEL_FETCH = get("parallel_fetch", True)
PARALLEL_MAX_WORKERS = get("parallel_max_workers", 4)
MAX_PLOT_POINTS = get("max_plot_points", 10_000)
PREFER_VIZ_BACKEND = get(
    "prefer_viz_backend", "matplotlib"
)  # "plotly", "matplotlib", or "jsx"

# ---- Reasoning features -------------------------------------------------------
OBSERVATION_SUMMARIES = get("reasoning.observation_summaries", True)
SELF_REFLECTION = get("reasoning.self_reflection", True)
SHOW_THINKING = get("reasoning.show_thinking", False)
INSIGHT_FEEDBACK = get("reasoning.insight_feedback", False)
INSIGHT_FEEDBACK_MAX_ITERS = get("reasoning.insight_feedback_max_iterations", 2)
ASYNC_DELEGATION = get("reasoning.async_delegation", True)

TURNLESS_MODE = get("reasoning.turnless_mode", True)
PIPELINE_CONFIRMATION = get("reasoning.pipeline_confirmation", True)

# ---- Gemini-specific settings ------------------------------------------------
# Thinking levels for Gemini 3+ models. Ignored for Gemini < 3.
# "model" = orchestrator + planner (smart tier), "sub_agent" = mission/viz agents.
# Values: "off", "low", "high".
GEMINI_THINKING_MODEL = _provider_get("thinking_model", "high")
GEMINI_THINKING_SUB_AGENT = _provider_get("thinking_sub_agent", "high")
GEMINI_THINKING_INSIGHT = _provider_get("thinking_insight", "low")
# Interactions API: server-side conversation state eliminates quadratic context growth.
USE_INTERACTIONS_API = get("use_interactions_api", True)
# Tool store: browse-and-load pattern — model discovers tools on first turn, loads
# what it needs. Only effective when USE_INTERACTIONS_API is True (per-call tools).
USE_TOOL_STORE = get("use_tool_store", True)


# ---- Setting descriptions (single source of truth for UI) --------------------
# Keys match config.json keys. Nested keys use dot notation (e.g. "reasoning.show_thinking").
CONFIG_DESCRIPTIONS: dict[str, str] = {
    # Data & Search
    "catalog_search_method": "Dataset search algorithm: 'semantic' uses AI embeddings, 'substring' uses simple text matching.",
    "parallel_fetch": "Download multiple CDF files and run tool calls concurrently.",
    "parallel_max_workers": "Maximum concurrent threads per pool when parallel fetch is enabled.",
    "max_plot_points": "Maximum points per trace before stride-decimation. Larger datasets plot every Nth point to keep rendering fast.",
    "prefer_viz_backend": "Visualization backend. 'matplotlib' (static/publication-quality, default), 'plotly' (interactive), or 'jsx' (rich dashboards).",
    # Memory
    "memory_token_budget": "Global token cap for all memory injection. Sub-agents get 1/4 each.",
    "memory_extraction_interval": "Extract memories every N user rounds. Set to 0 to disable.",
    "ops_library_max_entries": "Saved operations in custom_ops library. Least-used entries evicted when full.",
    # Reasoning
    "reasoning.observation_summaries": "Inject human-readable summaries into tool results for better LLM reasoning.",
    "reasoning.self_reflection": "Add reflection hints on errors to steer the LLM toward alternatives.",
    "reasoning.show_thinking": "Display LLM thinking tokens in the UI. Always logged to file regardless.",
    "reasoning.insight_feedback": "Automatically review every rendered figure against the user's request. Synchronous quality gate — blocks render result until review completes. Adds ~4-9s per render.",
    "reasoning.insight_feedback_max_iterations": "Maximum review-then-re-render cycles per user turn. Prevents infinite review loops. Default: 2.",
    "reasoning.async_delegation": "Launch sub-agent delegations on threads. The orchestrator freezes (zero LLM cost) until results arrive, enabling parallel sub-agent work.",
    "reasoning.turnless_mode": "Persistent orchestrator event loop. The user can send messages at any time; the orchestrator absorbs them and can selectively cancel in-flight work. Implies async_delegation.",
    "reasoning.pipeline_confirmation": "Require explicit user confirmation before running a saved pipeline. When enabled, the orchestrator lists top matches and asks for permission via ask_clarification.",
    # Turn limits
    "turn_limits": "Override agent loop limits (max iterations, max tool calls, planner rounds). Keys are named limits (e.g. 'orchestrator.max_iterations', 'sub_agent.max_total_calls'). Values are integers. See agent/turn_limits.py DEFAULTS for all limit names.",
    # Session history
    "history_budget_orchestrator": "Token budget for the pull-based orchestrator event feed.",
    # Gemini Interactions API
    "use_interactions_api": "Use Gemini Interactions API for server-side conversation state. Eliminates quadratic context growth. Falls back to Chat API if interactions fail.",
    "use_tool_store": "Browse-and-load tool pattern. Model discovers tools on first turn and loads what it needs. Reduces per-call tool token cost. Only effective with Interactions API.",
    # Truncation
    "truncation": "Override text character limits for truncation. Keys are named limits (e.g. 'console.summary', 'history.error'). Values are integers; 0 means no truncation. See agent/truncation.py for all limit names.",
    "truncation_items": "Override item count limits for list truncation. Keys are named limits (e.g. 'items.tool_args', 'items.columns'). Values are integers; 0 means no truncation. See agent/truncation.py for all limit names.",
}


def reload_config() -> None:
    """Re-read config from disk and reassign all module-level constants.

    Call this after writing config.json to make new values take effect
    without restarting the server. Existing sessions keep their current
    adapter/model; only new sessions pick up changes.
    """
    global _user_config
    global LLM_PROVIDER, LLM_BASE_URL
    global \
        SMART_MODEL, \
        SUB_AGENT_MODEL, \
        INSIGHT_MODEL, \
        INLINE_MODEL, \
        PLANNER_MODEL, \
        FALLBACK_MODEL
    global \
        DATA_BACKEND, \
        CATALOG_SEARCH_METHOD, \
        PARALLEL_FETCH, \
        PARALLEL_MAX_WORKERS, \
        MAX_PLOT_POINTS, \
        PREFER_VIZ_BACKEND
    global \
        OBSERVATION_SUMMARIES, \
        SELF_REFLECTION, \
        SHOW_THINKING, \
        INSIGHT_FEEDBACK, \
        INSIGHT_FEEDBACK_MAX_ITERS
    global ASYNC_DELEGATION, TURNLESS_MODE, PIPELINE_CONFIRMATION
    global GEMINI_THINKING_MODEL, GEMINI_THINKING_SUB_AGENT, GEMINI_THINKING_INSIGHT
    global USE_INTERACTIONS_API, USE_TOOL_STORE

    load_dotenv(override=True)

    _user_config = _load_config()
    _reset_data_dir()

    LLM_PROVIDER = get("llm_provider", "gemini")
    LLM_BASE_URL = _provider_get("base_url")
    SMART_MODEL = _provider_get("model")
    SUB_AGENT_MODEL = _provider_get("sub_agent_model")
    INSIGHT_MODEL = _provider_get("insight_model") or SUB_AGENT_MODEL
    INLINE_MODEL = _provider_get("inline_model")
    PLANNER_MODEL = _provider_get("planner_model") or SMART_MODEL
    FALLBACK_MODEL = _provider_get("fallback_model") or SUB_AGENT_MODEL
    DATA_BACKEND = get("data_backend", "cdf")
    CATALOG_SEARCH_METHOD = get("catalog_search_method", "semantic")
    PARALLEL_FETCH = get("parallel_fetch", True)
    PARALLEL_MAX_WORKERS = get("parallel_max_workers", 4)
    MAX_PLOT_POINTS = get("max_plot_points", 10_000)
    PREFER_VIZ_BACKEND = get("prefer_viz_backend", "matplotlib")
    OBSERVATION_SUMMARIES = get("reasoning.observation_summaries", True)
    SELF_REFLECTION = get("reasoning.self_reflection", True)
    SHOW_THINKING = get("reasoning.show_thinking", False)
    INSIGHT_FEEDBACK = get("reasoning.insight_feedback", False)
    INSIGHT_FEEDBACK_MAX_ITERS = get("reasoning.insight_feedback_max_iterations", 2)
    ASYNC_DELEGATION = get("reasoning.async_delegation", True)
    TURNLESS_MODE = get("reasoning.turnless_mode", True)
    PIPELINE_CONFIRMATION = get("reasoning.pipeline_confirmation", True)
    GEMINI_THINKING_MODEL = _provider_get("thinking_model", "high")
    GEMINI_THINKING_SUB_AGENT = _provider_get("thinking_sub_agent", "high")
    GEMINI_THINKING_INSIGHT = _provider_get("thinking_insight", "low")
    USE_INTERACTIONS_API = get("use_interactions_api", True)
    USE_TOOL_STORE = get("use_tool_store", True)

    # Reload truncation overrides from config
    try:
        from agent.truncation import reload as _reload_truncation

        _reload_truncation()
    except ImportError:
        pass

    # Reload turn limits overrides from config
    try:
        from agent.turn_limits import reload as _reload_turn_limits

        _reload_turn_limits()
    except ImportError:
        pass
