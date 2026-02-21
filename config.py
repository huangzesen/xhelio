import json
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Secret — stays in .env
# Legacy alias — prefer LLM_API_KEY for all providers
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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

    Auto-migration: if ``~/.xhelio`` doesn't exist, checks for legacy
    directories ``~/.helion`` then ``~/.helio-agent`` and moves the first
    one found.
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
            new_dir = Path.home() / ".xhelio"
            if not new_dir.exists():
                # Chain migration: .xhelio ← .helion ← .helio-agent
                for legacy_name in (".helion", ".helio-agent"):
                    legacy_dir = Path.home() / legacy_name
                    if legacy_dir.exists():
                        import shutil
                        shutil.move(str(legacy_dir), str(new_dir))
                        break
            _data_dir = new_dir
    return _data_dir


def _reset_data_dir() -> None:
    """Reset the cached data directory (for testing only)."""
    global _data_dir
    _data_dir = None


# ---- LLM provider config ------------------------------------------------------
LLM_PROVIDER = get("llm_provider", "gemini")  # "gemini", "openai", "anthropic"

LLM_API_KEY = os.getenv("LLM_API_KEY") or GOOGLE_API_KEY


_PROVIDER_ENV_KEYS = {
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def get_api_key(provider: str | None = None) -> str | None:
    """Return the API key for the given provider.

    Resolution: provider-specific env var (e.g. GOOGLE_API_KEY) first,
    then generic LLM_API_KEY as fallback.
    """
    p = (provider or LLM_PROVIDER).lower()
    env_key = _PROVIDER_ENV_KEYS.get(p)
    if env_key:
        val = os.getenv(env_key)
        if val:
            return val
    return LLM_API_KEY


# ---- Per-provider defaults ---------------------------------------------------
# Hardcoded defaults per provider. Used as final fallback when neither the
# providers.<active>.key nor a top-level key is set in config.json.
_PROVIDER_DEFAULTS = {
    "gemini": {
        "model": "gemini-3-flash",
        "sub_agent_model": "gemini-3-flash",
        "inline_model": "gemini-2.5-flash-lite",
        "planner_model": "gemini-3-flash",
        "fallback_model": "gemini-3-flash",
        "base_url": None,
        "thinking_model": "high",
        "thinking_sub_agent": "low",
    },
    "openai": {
        "model": "minimax/minimax-m2.5",
        "sub_agent_model": "minimax/minimax-m2.1",
        "inline_model": "minimax/minimax-m2.1",
        "planner_model": "minimax/minimax-m2.5",
        "fallback_model": "minimax/minimax-m2.1",
        "base_url": "https://openrouter.ai/api/v1",
    },
    "anthropic": {
        "model": "claude-sonnet-4-5-20250514",
        "sub_agent_model": "claude-haiku-4-5-20251001",
        "inline_model": "claude-haiku-4-5-20251001",
        "planner_model": "claude-sonnet-4-5-20250514",
        "fallback_model": "claude-haiku-4-5-20251001",
        "base_url": None,
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
# Three tiers: smart (orchestrator + planner), sub-agent, inline (cheapest).
LLM_BASE_URL = _provider_get("base_url")
SMART_MODEL = _provider_get("model")
SUB_AGENT_MODEL = _provider_get("sub_agent_model")
INLINE_MODEL = _provider_get("inline_model")
PLANNER_MODEL = _provider_get("planner_model") or SMART_MODEL
FALLBACK_MODEL = _provider_get("fallback_model") or SUB_AGENT_MODEL
DATA_BACKEND = get("data_backend", "cdf")  # "cdf" only
CATALOG_SEARCH_METHOD = get("catalog_search_method", "semantic")  # "semantic" or "substring"
PARALLEL_FETCH = get("parallel_fetch", True)
PARALLEL_MAX_WORKERS = get("parallel_max_workers", 4)
MAX_PLOT_POINTS = get("max_plot_points", 10_000)

# ---- Reasoning features -------------------------------------------------------
OBSERVATION_SUMMARIES = get("reasoning.observation_summaries", True)
SELF_REFLECTION = get("reasoning.self_reflection", True)
SHOW_THINKING = get("reasoning.show_thinking", False)

# ---- Gemini-specific settings ------------------------------------------------
# Thinking levels for Gemini 3+ models. Ignored for Gemini < 3.
# "model" = orchestrator + planner (smart tier), "sub_agent" = mission/viz agents.
# Values: "off", "low", "high".
GEMINI_THINKING_MODEL = _provider_get("thinking_model", "high")
GEMINI_THINKING_SUB_AGENT = _provider_get("thinking_sub_agent", "low")


# ---- Setting descriptions (single source of truth for UI) --------------------
# Keys match config.json keys. Nested keys use dot notation (e.g. "reasoning.show_thinking").
CONFIG_DESCRIPTIONS: dict[str, str] = {
    # Data & Search
    "catalog_search_method": "Dataset search algorithm: 'semantic' uses AI embeddings, 'substring' uses simple text matching.",
    "parallel_fetch": "Download multiple CDF files and run tool calls concurrently.",
    "parallel_max_workers": "Maximum concurrent threads per pool when parallel fetch is enabled.",
    "max_plot_points": "Maximum points per trace before stride-decimation. Larger datasets plot every Nth point to keep rendering fast.",
    # Memory
    "memory_token_budget": "Global token cap for all memory injection. Sub-agents get 1/4 each.",
    "memory_extraction_interval": "Extract memories every N user rounds. Set to 0 to disable.",
    "ops_library_max_entries": "Saved operations in custom_ops library. Least-used entries evicted when full.",
    # Reasoning
    "reasoning.observation_summaries": "Inject human-readable summaries into tool results for better LLM reasoning.",
    "reasoning.self_reflection": "Add reflection hints on errors to steer the LLM toward alternatives.",
    "reasoning.show_thinking": "Display LLM thinking tokens in the UI. Always logged to file regardless.",
    # Session history
    "history_budget_sub_agent": "Token budget for session history injected into sub-agent delegations.",
    "history_budget_orchestrator": "Token budget for session history injected into orchestrator and planner.",
}


def reload_config() -> None:
    """Re-read config from disk and reassign all module-level constants.

    Call this after writing config.json to make new values take effect
    without restarting the server. Existing sessions keep their current
    adapter/model; only new sessions pick up changes.
    """
    global _user_config, GOOGLE_API_KEY, LLM_API_KEY
    global LLM_PROVIDER, LLM_BASE_URL
    global SMART_MODEL, SUB_AGENT_MODEL, INLINE_MODEL, PLANNER_MODEL, FALLBACK_MODEL
    global DATA_BACKEND, CATALOG_SEARCH_METHOD, PARALLEL_FETCH, PARALLEL_MAX_WORKERS, MAX_PLOT_POINTS
    global OBSERVATION_SUMMARIES, SELF_REFLECTION, SHOW_THINKING
    global GEMINI_THINKING_MODEL, GEMINI_THINKING_SUB_AGENT

    load_dotenv(override=True)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    LLM_API_KEY = os.getenv("LLM_API_KEY") or GOOGLE_API_KEY

    _user_config = _load_config()
    _reset_data_dir()

    LLM_PROVIDER = get("llm_provider", "gemini")
    LLM_BASE_URL = _provider_get("base_url")
    SMART_MODEL = _provider_get("model")
    SUB_AGENT_MODEL = _provider_get("sub_agent_model")
    INLINE_MODEL = _provider_get("inline_model")
    PLANNER_MODEL = _provider_get("planner_model") or SMART_MODEL
    FALLBACK_MODEL = _provider_get("fallback_model") or SUB_AGENT_MODEL
    DATA_BACKEND = get("data_backend", "cdf")
    CATALOG_SEARCH_METHOD = get("catalog_search_method", "semantic")
    PARALLEL_FETCH = get("parallel_fetch", True)
    PARALLEL_MAX_WORKERS = get("parallel_max_workers", 4)
    MAX_PLOT_POINTS = get("max_plot_points", 10_000)
    OBSERVATION_SUMMARIES = get("reasoning.observation_summaries", True)
    SELF_REFLECTION = get("reasoning.self_reflection", True)
    SHOW_THINKING = get("reasoning.show_thinking", False)
    GEMINI_THINKING_MODEL = _provider_get("thinking_model", "high")
    GEMINI_THINKING_SUB_AGENT = _provider_get("thinking_sub_agent", "low")
