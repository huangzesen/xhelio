"""agent/fallback_registry.py — Central registry for all fallback mechanisms.

Every fallback mechanism in the codebase must be registered here. This provides
a single source of truth for:
- Default values
- Fallback trigger conditions
- Code locations
- Documentation

Use this registry instead of scattered `or` fallbacks, try/except blocks, or
hardcoded defaults throughout the codebase.

## Registry Structure

Each fallback is registered as a dict with keys:
    - name: str — unique identifier
    - description: str — what this fallback does
    - category: str — one of CATEGORY_* constants below
    - default_value: Any — the fallback value
    - trigger_condition: str — when this fallback activates
    - code_location: str — file:line reference
    - is_legacy: bool — True if this should be removed (see Migration Policy)

## Categories

    CATEGORY_MODEL: LLM model fallback on quota/rate limit
    CATEGORY_CONFIG: Configuration defaults
    CATEGORY_TOKENIZER: Tokenization fallback
    CATEGORY_RENDERING: Visualization defaults
    CATEGORY_DATA: Data pipeline defaults
    CATEGORY_MEMORY: Memory system fallbacks
    CATEGORY_TRUNCATION: Text/item truncation limits
    CATEGORY_TURNS: Agent loop limits
    CATEGORY_DEPENDENCY: Optional dependency fallback
    CATEGORY_GENERIC: Generic or-operator fallbacks

## Usage

    from agent.fallback_registry import get_all_fallbacks, get_fallback, register_fallback

    # Lookup a fallback
    fb = get_fallback("model.fallback")
    if fb:
        print(f"Default: {fb['default_value']}")

    # Get all fallbacks
    all_fbs = get_all_fallbacks()

    # Register a new fallback (do this when adding new fallbacks)
    register_fallback({
        "name": "my.new_fallback",
        "description": "What it does",
        "category": CATEGORY_GENERIC,
        "default_value": 100,
        "trigger_condition": "When X is unavailable",
        "code_location": "agent/my_module.py:123",
        "is_legacy": False,
    })
"""

from __future__ import annotations

from typing import Any, Optional

CATEGORY_MODEL = "model"
CATEGORY_CONFIG = "config"
CATEGORY_TOKENIZER = "tokenizer"
CATEGORY_RENDERING = "rendering"
CATEGORY_DATA = "data"
CATEGORY_MEMORY = "memory"
CATEGORY_TRUNCATION = "truncation"
CATEGORY_TURNS = "turns"
CATEGORY_DEPENDENCY = "dependency"
CATEGORY_GENERIC = "generic"

_FALLBACKS: dict[str, dict[str, Any]] = {}


def register_fallback(fallback: dict[str, Any]) -> None:
    """Register a fallback mechanism.

    Required keys:
        - name: unique identifier (e.g., "model.fallback", "truncation.console_summary")
        - description: what this fallback does
        - category: one of CATEGORY_* constants
        - default_value: the fallback value
        - trigger_condition: when this fallback activates
        - code_location: file:line reference

    Optional keys:
        - is_legacy: True if this fallback should be removed (see Migration Policy)
    """
    name = fallback.get("name")
    if not name:
        raise ValueError("Fallback must have a 'name' key")
    if name in _FALLBACKS:
        raise ValueError(f"Fallback already registered: {name}")
    _FALLBACKS[name] = fallback


def get_fallback(name: str) -> Optional[dict[str, Any]]:
    """Get a fallback by name, or None if not found."""
    return _FALLBACKS.get(name)


def get_fallbacks_by_category(category: str) -> list[dict[str, Any]]:
    """Get all fallbacks in a category."""
    return [fb for fb in _FALLBACKS.values() if fb.get("category") == category]


def get_all_fallbacks() -> dict[str, dict[str, Any]]:
    """Get all registered fallbacks."""
    return _FALLBACKS.copy()


def is_legacy(name: str) -> bool:
    """Check if a fallback is marked as legacy (should be removed)."""
    fb = _FALLBACKS.get(name)
    return fb.get("is_legacy", False) if fb else False


# =============================================================================
# BUILT-IN FALLBACKS
# =============================================================================

# --- Model Fallbacks (LLM Quota/429 Handling) ---

register_fallback(
    {
        "name": "model.fallback_active",
        "description": "Session-level flag: when True, all agents use fallback model",
        "category": CATEGORY_MODEL,
        "default_value": False,
        "trigger_condition": "When any LLM returns 429/RESOURCE_EXHAUSTED",
        "code_location": "agent/model_fallback.py:13",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "model.fallback_model",
        "description": "The fallback model name used when quota is exhausted",
        "category": CATEGORY_MODEL,
        "default_value": None,
        "trigger_condition": "When fallback_active is True",
        "code_location": "agent/model_fallback.py:14",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "model.fallback.google",
        "description": "Fallback model for Google Gemini provider",
        "category": CATEGORY_MODEL,
        "default_value": "gemini-3-flash-preview",
        "trigger_condition": "When Google API returns 429",
        "code_location": "config.py:121",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "model.fallback.openai",
        "description": "Fallback model for OpenAI provider",
        "category": CATEGORY_MODEL,
        "default_value": "",
        "trigger_condition": "When OpenAI API returns 429",
        "code_location": "config.py:134",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "model.fallback.anthropic",
        "description": "Fallback model for Anthropic provider",
        "category": CATEGORY_MODEL,
        "default_value": "claude-haiku-3-5-20241022",
        "trigger_condition": "When Anthropic API returns 429",
        "code_location": "config.py:149",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "model.fallback.minimax",
        "description": "Fallback model for MiniMax provider",
        "category": CATEGORY_MODEL,
        "default_value": "MiniMax-M2.5-highspeed",
        "trigger_condition": "When MiniMax API returns 429",
        "code_location": "config.py:162",
        "is_legacy": False,
    }
)


# --- Config Fallbacks ---

register_fallback(
    {
        "name": "config.data_dir",
        "description": "Default data directory for sessions, logs, memory",
        "category": CATEGORY_CONFIG,
        "default_value": "~/.xhelio",
        "trigger_condition": "When XHELIO_DIR env or config not set",
        "code_location": "config.py:50",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "config.load_continue_on_error",
        "description": "Continue loading config if one file fails",
        "category": CATEGORY_CONFIG,
        "default_value": True,
        "trigger_condition": "When config file is malformed",
        "code_location": "config.py:26",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "session.history_default",
        "description": "Default empty history list",
        "category": CATEGORY_CONFIG,
        "default_value": [],
        "trigger_condition": "When session history is missing",
        "code_location": "agent/session.py:357",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "session.metadata_default",
        "description": "Default empty metadata dict",
        "category": CATEGORY_CONFIG,
        "default_value": {},
        "trigger_condition": "When session metadata is missing",
        "code_location": "agent/session.py:359",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "session.event_log_default",
        "description": "Default None for missing event log",
        "category": CATEGORY_CONFIG,
        "default_value": None,
        "trigger_condition": "When event log file doesn't exist",
        "code_location": "agent/session.py:367",
        "is_legacy": False,
    }
)


# --- Event Bus Fallback ---

register_fallback(
    {
        "name": "event_bus.fallback",
        "description": "Module-level EventBus for pre-session code",
        "category": CATEGORY_CONFIG,
        "default_value": None,
        "trigger_condition": "When get_event_bus() called before session creation",
        "code_location": "agent/event_bus.py:662-683",
        "is_legacy": False,
    }
)

# --- Tokenizer Fallback ---

register_fallback(
    {
        "name": "tokenizer.fallback",
        "description": "Heuristic token estimation when tiktoken unavailable",
        "category": CATEGORY_TOKENIZER,
        "default_value": True,
        "trigger_condition": "When tiktoken import fails",
        "code_location": "agent/token_counter.py:19-54",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "tokenizer.fallback_warned",
        "description": "Prevents duplicate fallback warnings",
        "category": CATEGORY_TOKENIZER,
        "default_value": False,
        "trigger_condition": "On first fallback warning",
        "code_location": "agent/token_counter.py:19",
        "is_legacy": False,
    }
)

# --- Rendering Defaults ---

register_fallback(
    {
        "name": "rendering.default_layout",
        "description": "Default Plotly layout dict",
        "category": CATEGORY_RENDERING,
        "default_value": {
            "width": 1100,
            "height": 600,
            "margin": {"l": 60, "r": 30, "t": 30, "b": 60},
        },
        "trigger_condition": "When no layout provided to renderer",
        "code_location": "rendering/plotly_renderer.py:28-36",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "rendering.default_width",
        "description": "Default Plotly figure width in pixels",
        "category": CATEGORY_RENDERING,
        "default_value": 1100,
        "trigger_condition": "When rendering single-column figure",
        "code_location": "rendering/plotly_renderer.py:36",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "rendering.panel_height",
        "description": "Height in pixels per subplot panel",
        "category": CATEGORY_RENDERING,
        "default_value": 300,
        "trigger_condition": "When calculating figure height for multi-panel plots",
        "code_location": "rendering/plotly_renderer.py:35",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "rendering.max_plot_points",
        "description": "Maximum points to plot (prevents browser freeze)",
        "category": CATEGORY_RENDERING,
        "default_value": 10000,
        "trigger_condition": "When config.get('max_plot_points') unavailable",
        "code_location": "rendering/plotly_renderer.py:39",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "rendering.webgl_threshold",
        "description": "Point count threshold to switch to WebGL (scattergl)",
        "category": CATEGORY_RENDERING,
        "default_value": 100000,
        "trigger_condition": "When trace has more points than this",
        "code_location": "rendering/plotly_renderer.py:62",
        "is_legacy": False,
    }
)

# --- Data Pipeline Defaults ---

register_fallback(
    {
        "name": "data.ops_library_max_entries",
        "description": "Maximum entries in operations library",
        "category": CATEGORY_DATA,
        "default_value": 50,
        "trigger_condition": "When config.get('ops_library_max_entries') unavailable",
        "code_location": "data_ops/ops_library.py:23",
        "is_legacy": False,
    }
)

# --- Memory Fallbacks ---

register_fallback(
    {
        "name": "memory.embedding_fallback",
        "description": "Tag-based search when embedding unavailable",
        "category": CATEGORY_MEMORY,
        "default_value": True,
        "trigger_condition": "When fastembed import fails or search errors",
        "code_location": "agent/memory.py:172,689",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "memory.default_enabled",
        "description": "Memory system enabled by default",
        "category": CATEGORY_MEMORY,
        "default_value": True,
        "trigger_condition": "When no explicit enable/disable set",
        "code_location": "agent/memory.py:825,984",
        "is_legacy": False,
    }
)

# --- Tool Catalog Defaults ---

register_fallback(
    {
        "name": "tools.default_categories",
        "description": "Default tool categories enabled for agents",
        "category": CATEGORY_CONFIG,
        "default_value": [
            "delegation",
            "discovery",
            "memory",
            "session",
            "data_ops",
            "visualization",
        ],
        "trigger_condition": "When no categories specified",
        "code_location": "(deleted) agent/tool_catalog.py:250",
        "is_legacy": True,
    }
)

register_fallback(
    {
        "name": "tools.default_extra",
        "description": "Default extra tools list (empty)",
        "category": CATEGORY_CONFIG,
        "default_value": [],
        "trigger_condition": "When no extra tools specified",
        "code_location": "(deleted) agent/tool_catalog.py:252",
        "is_legacy": True,
    }
)

# --- MiniMax API Fallback ---

register_fallback(
    {
        "name": "minimax.api_host",
        "description": "Default MiniMax API host URL",
        "category": CATEGORY_CONFIG,
        "default_value": "https://api.minimax.chat",
        "trigger_condition": "When config.get('minimax_api_host') unavailable",
        "code_location": "agent/minimax_mcp_client.py:21",
        "is_legacy": False,
    }
)

# --- Operations Log Suffix Fallback ---

register_fallback(
    {
        "name": "dataops.suffix_fallback",
        "description": "Producer lookup with dedup suffix stripping",
        "category": CATEGORY_DATA,
        "default_value": True,
        "trigger_condition": "When label has deduplication suffix (e.g., op_1)",
        "code_location": "data_ops/operations_log.py:40-81",
        "is_legacy": False,
    }
)

# --- Metadata Resource URL Fallback ---

register_fallback(
    {
        "name": "metadata.resource_url_fallback",
        "description": "Fallback URL construction for dataset resources",
        "category": CATEGORY_DATA,
        "default_value": True,
        "trigger_condition": "When primary resource URL lookup fails",
        "code_location": "knowledge/metadata_client.py:941",
        "is_legacy": False,
    }
)

# --- Pipeline Input Producer Fallback ---

register_fallback(
    {
        "name": "pipeline.input_producer_fallback",
        "description": "Use label_producer when input_producers missing",
        "category": CATEGORY_DATA,
        "default_value": True,
        "trigger_condition": "When input_producers absent in legacy pipelines",
        "code_location": "data_ops/pipeline.py:402-412",
        "is_legacy": False,
    }
)

# --- Data Source Priority Fallback ---

register_fallback(
    {
        "name": "knowledge.source_priority",
        "description": "CDAWeb first, PPI as fallback for data sources",
        "category": CATEGORY_DATA,
        "default_value": ["cdaweb", "ppi"],
        "trigger_condition": "When looking up data sources for a mission",
        "code_location": "knowledge/metadata_client.py:558,565",
        "is_legacy": False,
    }
)

# --- Observations Fallback ---

register_fallback(
    {
        "name": "observations.generic",
        "description": "Generic fallback for unclassified observations",
        "category": CATEGORY_GENERIC,
        "default_value": "generic",
        "trigger_condition": "When observation type cannot be classified",
        "code_location": "agent/observations.py:46",
        "is_legacy": False,
    }
)

# --- Viz Delegation Fallback ---

register_fallback(
    {
        "name": "viz.delegation_backend",
        "description": "Preferred viz backend (plotly or mpl)",
        "category": CATEGORY_RENDERING,
        "default_value": "delegate_to_viz",
        "trigger_condition": "When config.PREFER_VIZ_BACKEND not set",
        "code_location": "agent/core.py:200",
        "is_legacy": False,
    }
)

# --- Agent Registry Defaults ---

register_fallback(
    {
        "name": "registry.informed_defaults",
        "description": "Default informed tools per agent context",
        "category": CATEGORY_CONFIG,
        "default_value": [],
        "trigger_condition": "When no informed tools specified for context",
        "code_location": "agent/agent_registry.py:300-324",
        "is_legacy": False,
    }
)

# --- Truncation Limits (delegate to truncation.py) ---
# Note: Truncation limits are managed in agent/truncation.py DEFAULTS and ITEM_DEFAULTS
# This registry references them for completeness

register_fallback(
    {
        "name": "truncation.text_defaults",
        "description": "Text character count limits (40+ keys)",
        "category": CATEGORY_TRUNCATION,
        "default_value": "See agent/truncation.py DEFAULTS",
        "trigger_condition": "When trunc() called without explicit limit",
        "code_location": "agent/truncation.py:23-79",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "truncation.item_defaults",
        "description": "Item count limits (20+ keys)",
        "category": CATEGORY_TRUNCATION,
        "default_value": "See agent/truncation.py ITEM_DEFAULTS",
        "trigger_condition": "When trunc_items() called without explicit limit",
        "code_location": "agent/truncation.py:85-107",
        "is_legacy": False,
    }
)

# --- Turn Limits (delegate to turn_limits.py) ---
# Note: Turn limits are managed in agent/turn_limits.py DEFAULTS

register_fallback(
    {
        "name": "turns.default_limits",
        "description": "Agent loop limits (20+ keys)",
        "category": CATEGORY_TURNS,
        "default_value": "See agent/turn_limits.py DEFAULTS",
        "trigger_condition": "When get_limit() called without config override",
        "code_location": "agent/turn_limits.py:17-43",
        "is_legacy": False,
    }
)

# =============================================================================
# EUREKA FALLBACKS
# =============================================================================

register_fallback(
    {
        "name": "eureka.model",
        "description": "EurekaAgent model — falls back to config.SMART_MODEL",
        "category": CATEGORY_CONFIG,
        "default_value": None,
        "trigger_condition": "When eureka_model is null in config",
        "code_location": "agent/core.py:_ensure_eureka_agent",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "eureka.max_rounds",
        "description": "Maximum consecutive eureka-driven rounds before auto-pause",
        "category": CATEGORY_CONFIG,
        "default_value": 5,
        "trigger_condition": "When eureka_max_rounds is not set in config",
        "code_location": "agent/core.py:_maybe_extract_eurekas",
        "is_legacy": False,
    }
)

# =============================================================================
# CONTEXT COMPACTION FALLBACKS
# =============================================================================

register_fallback(
    {
        "name": "compaction.threshold",
        "description": "Trigger compaction when context reaches this fraction of the window",
        "category": CATEGORY_CONFIG,
        "default_value": 0.8,
        "trigger_condition": "When estimate_context_tokens / context_window >= threshold",
        "code_location": "agent/sub_agent.py:_check_and_compact, agent/core.py:_check_and_compact",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "compaction.keep_turns",
        "description": "Number of recent turns to keep intact during compaction",
        "category": CATEGORY_CONFIG,
        "default_value": 3,
        "trigger_condition": "When compacting older messages",
        "code_location": "agent/llm/anthropic_adapter.py:compact, agent/llm/openai_adapter.py:compact",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "compaction.context_registry",
        "description": "litellm community-maintained model context window registry",
        "category": CATEGORY_CONFIG,
        "default_value": "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
        "trigger_condition": "When get_context_limit() needs model context window size",
        "code_location": "agent/llm_utils.py:_fetch_litellm_registry",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "compaction.context_registry_cache",
        "description": "Stale litellm cache used when fetch fails",
        "category": CATEGORY_CONFIG,
        "default_value": "~/.xhelio/model_context_windows.json",
        "trigger_condition": "When litellm registry fetch fails and local cache exists",
        "code_location": "agent/llm_utils.py:_fetch_litellm_registry",
        "is_legacy": False,
    }
)


# =============================================================================
# GENERIC OR-FALLBACK PATTERNS
# =============================================================================
# These are common patterns found throughout the codebase. They are NOT
# individually registered unless they are critical paths. Instead, this section
# documents the pattern for awareness.
#
# Common patterns:
#   - response.text or ""
#   - event.data or {}
#   - tool_args.get("key", default)
#   - getattr(obj, "attr", default)
#
# Count: ~159 instances across codebase

# =============================================================================
# DEPENDENCY FALLBACKS
# =============================================================================

register_fallback(
    {
        "name": "dependency.tiktoken",
        "description": "Tiktoken tokenizer (optional dependency)",
        "category": CATEGORY_DEPENDENCY,
        "default_value": "fallback_heuristic",
        "trigger_condition": "When 'tiktoken' import fails",
        "code_location": "agent/token_counter.py:27",
        "is_legacy": False,
    }
)

register_fallback(
    {
        "name": "dependency.fastembed",
        "description": "Fastembed embeddings (optional dependency)",
        "category": CATEGORY_DEPENDENCY,
        "default_value": "tag_search",
        "trigger_condition": "When 'fastembed' import fails",
        "code_location": "agent/memory.py:172",
        "is_legacy": False,
    }
)

# --- Import Error Fallbacks (Bare except ImportError patterns) ---
# These are found in ~11 locations for optional dependencies

register_fallback(
    {
        "name": "dependency.optional_imports",
        "description": "Bare except ImportError patterns for optional deps",
        "category": CATEGORY_DEPENDENCY,
        "default_value": True,
        "trigger_condition": "When optional dependency not installed",
        "code_location": "scripts/loc_history.py:20, main.py:44, knowledge/*.py, etc.",
        "is_legacy": False,
    }
)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CATEGORY_MODEL",
    "CATEGORY_CONFIG",
    "CATEGORY_TOKENIZER",
    "CATEGORY_RENDERING",
    "CATEGORY_DATA",
    "CATEGORY_MEMORY",
    "CATEGORY_TRUNCATION",
    "CATEGORY_TURNS",
    "CATEGORY_DEPENDENCY",
    "CATEGORY_GENERIC",
    "register_fallback",
    "get_fallback",
    "get_fallbacks_by_category",
    "get_all_fallbacks",
    "is_legacy",
]
