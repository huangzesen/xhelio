import json
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Secret — stays in .env (per-provider env vars: GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, MINIMAX_API_KEY)

# User config — loaded from ~/.xhelio/config.json (primary)
# or project-root config.json (fallback).
CONFIG_PATH = Path.home() / ".xhelio" / "config.json"
_LOCAL_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
_user_config: dict = {}


def _ensure_default_preset(cfg: dict) -> bool:
    """Create a default preset from current provider/model config if none exists.

    Called from GET /config after module globals are initialized.
    Returns True if a preset was created.
    """
    wb = cfg.get("workbench")
    if isinstance(wb, dict) and wb.get("preset"):
        return False  # Already has an active preset

    # Build a preset from current resolve_agent_model for each agent type
    provider = cfg.get("llm_provider", "gemini")
    # Find display name from PROVIDERS list
    provider_name = provider.capitalize()
    for p in PROVIDERS:
        if p["id"] == provider:
            provider_name = p["name"]
            break

    agents: dict = {}
    for agent_info in AGENT_TYPES:
        agent_id = agent_info["id"]
        a_provider, a_model, a_base_url = resolve_agent_model(agent_id)
        entry: dict = {"provider": a_provider, "model": a_model}
        if a_base_url:
            entry["base_url"] = a_base_url
        agents[agent_id] = entry

    preset_slug = f"default-{provider}"
    preset = {
        "name": provider_name,
        "agents": agents,
        "capabilities": resolve_capabilities(),
    }

    presets = cfg.setdefault("presets", {})
    presets[preset_slug] = preset
    wb = cfg.setdefault("workbench", {})
    wb["preset"] = preset_slug
    wb.setdefault("agents", {})
    return True


def _migrate_inline_model_to_agent(cfg: dict) -> bool:
    """Migrate presets with inline_model field to agents.inline entry.

    Returns True if migration was performed.
    """
    presets = cfg.get("presets")
    if not isinstance(presets, dict):
        return False

    changed = False
    for _key, preset in presets.items():
        if not isinstance(preset, dict):
            continue
        inline_model = preset.pop("inline_model", None)
        if not inline_model:
            continue
        agents = preset.setdefault("agents", {})
        if "inline" not in agents:
            # Infer provider from orchestrator entry
            orch = agents.get("orchestrator", {})
            provider = orch.get("provider", cfg.get("llm_provider", "gemini"))
            agents["inline"] = {"provider": provider, "model": inline_model}
            changed = True

    return changed


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

    # Run migrations on loaded config
    needs_save = False
    needs_save |= _migrate_inline_model_to_agent(merged)

    if needs_save and CONFIG_PATH.exists():
        try:
            existing = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
        existing["presets"] = merged.get("presets", {})
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        tmp.rename(CONFIG_PATH)

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
LLM_PROVIDER = get(
    "llm_provider", "gemini"
)  # "gemini", "openai", "anthropic", "minimax"

# Single source of truth for all LLM providers.
PROVIDERS = [
    {"id": "gemini", "name": "Gemini", "env_key": "GOOGLE_API_KEY"},
    {"id": "openai", "name": "OpenAI", "env_key": "OPENAI_API_KEY", "supports_base_url": True},
    {"id": "anthropic", "name": "Anthropic", "env_key": "ANTHROPIC_API_KEY", "supports_base_url": True},
    {"id": "minimax", "name": "MiniMax", "env_key": "MINIMAX_API_KEY"},
    {"id": "grok", "name": "Grok", "env_key": "GROK_API_KEY"},
    {"id": "deepseek", "name": "DeepSeek", "env_key": "DEEPSEEK_API_KEY"},
    {"id": "qwen", "name": "Qwen", "env_key": "QWEN_API_KEY"},
    {"id": "kimi", "name": "Kimi", "env_key": "KIMI_API_KEY"},
    {"id": "glm", "name": "GLM", "env_key": "GLM_API_KEY"},
    {"id": "custom", "name": "CustomBot", "env_key": "CUSTOM_API_KEY"},
]

# Derived lookups
PROVIDER_IDS: list[str] = [p["id"] for p in PROVIDERS]
_PROVIDER_ENV_KEYS: dict[str, str] = {p["id"]: p["env_key"] for p in PROVIDERS}

# Agent type registry — single source of truth for all configurable agent types.
# Frontend fetches this via GET /agent-types to build the workbench UI.
AGENT_TYPES = [
    {"id": "orchestrator", "name": "Orchestrator", "icon": "🎯", "group": "brain",
     "description": "Routes requests to sub-agents"},
    {"id": "planner", "name": "Planner", "icon": "📋", "group": "brain",
     "description": "Multi-step plan generation"},
    {"id": "viz_plotly", "name": "Plotly Viz", "icon": "📊", "group": "visualization",
     "description": "Interactive Plotly visualizations"},
    {"id": "viz_mpl", "name": "Matplotlib", "icon": "📈", "group": "visualization",
     "description": "Static publication-quality plots"},
    {"id": "viz_jsx", "name": "JSX Viz", "icon": "⚛️", "group": "visualization",
     "description": "React component visualizations"},
    {"id": "data_ops", "name": "Data Ops", "icon": "🔧", "group": "data",
     "description": "Data transformation and computation"},
    {"id": "data_io", "name": "Data I/O", "icon": "📥", "group": "data",
     "description": "Structured data extraction and file import"},
    {"id": "envoy", "name": "Envoy", "icon": "🛰️", "group": "specialists",
     "description": "Per-mission data discovery and fetch"},
    {"id": "insight", "name": "Insight", "icon": "🔍", "group": "specialists",
     "description": "Multimodal plot analysis"},
    {"id": "eureka", "name": "Eureka", "icon": "💡", "group": "specialists",
     "description": "Automated discovery and insight"},
    {"id": "memory", "name": "Memory", "icon": "🧠", "group": "specialists",
     "description": "Long-term memory extraction"},
    {"id": "inline", "name": "Inline", "icon": "⚡", "group": "brain",
     "description": "Autocomplete, session titles, cheapest model"},
]

AGENT_TYPE_IDS = [a["id"] for a in AGENT_TYPES]

AGENT_GROUPS = [
    {"id": "brain", "name": "Brain", "order": 0},
    {"id": "visualization", "name": "Visualization", "order": 1},
    {"id": "data", "name": "Data", "order": 2},
    {"id": "specialists", "name": "Specialists", "order": 3},
]


def get_api_key(provider: str | None = None) -> str | None:
    """Return the API key for the given provider.

    Reads the environment variable mapped in PROVIDERS (e.g. GOOGLE_API_KEY
    for gemini, OPENAI_API_KEY for openai, etc.).
    """
    p = (provider or LLM_PROVIDER).lower()
    env_key = _PROVIDER_ENV_KEYS.get(p)
    if env_key:
        return os.getenv(env_key)
    return None


# ---- Per-provider defaults ---------------------------------------------------
# Imported from each provider's adapter module (DEFAULTS dict).
# Used as final fallback when neither the providers.<active>.key nor a
# top-level key is set in config.json.

def _build_provider_defaults() -> dict:
    """Build provider defaults by loading each provider's defaults.py directly.

    Uses importlib.util to load defaults.py files without triggering their
    package __init__.py (which would import adapter modules and their SDKs).
    """
    import importlib.util
    from pathlib import Path

    defaults = {}
    llm_dir = Path(__file__).parent / "agent" / "llm"
    for name in PROVIDER_IDS:
        defaults_file = llm_dir / name / "defaults.py"
        if not defaults_file.exists():
            continue
        spec = importlib.util.spec_from_file_location(
            f"_provider_defaults.{name}", defaults_file
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            defaults[name] = getattr(mod, "DEFAULTS", {})

    return defaults

_PROVIDER_DEFAULTS = _build_provider_defaults()


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


def resolve_agent_model(agent_type: str) -> tuple[str, str, str | None]:
    """Resolve provider, model, and base_url for a given agent type.

    Resolution order:
    1. workbench.agents[agent_type] (per-agent override)
    2. presets[workbench.preset].agents[agent_type] (active preset)
    3. Legacy agent_models[agent_type] ("provider/model" format)
    4. Tier-based defaults (LLM_PROVIDER + tier model)

    Returns:
        (provider, model, base_url) tuple
    """
    # 1. Workbench per-agent override
    wb = get("workbench", {})
    if isinstance(wb, dict):
        agent_cfg = wb.get("agents", {}).get(agent_type)
        if isinstance(agent_cfg, dict) and agent_cfg.get("provider") and agent_cfg.get("model"):
            return (
                agent_cfg["provider"],
                agent_cfg["model"],
                agent_cfg.get("base_url"),
            )

    # 2. Active preset
    if isinstance(wb, dict):
        preset_name = wb.get("preset")
        if preset_name:
            presets = get("presets", {})
            if isinstance(presets, dict):
                preset = presets.get(preset_name)
                if isinstance(preset, dict):
                    agent_cfg = preset.get("agents", {}).get(agent_type)
                    if isinstance(agent_cfg, dict) and agent_cfg.get("provider") and agent_cfg.get("model"):
                        return (
                            agent_cfg["provider"],
                            agent_cfg["model"],
                            agent_cfg.get("base_url"),
                        )

    # 3. Legacy agent_models override ("provider/model" format)
    agent_models = get("agent_models", {})
    if isinstance(agent_models, dict):
        value = agent_models.get(agent_type)
        if value and isinstance(value, str):
            if "/" not in value:
                raise ValueError(
                    f"agent_models[{agent_type}] = {value!r} — expected 'provider/model' format"
                )
            provider, model = value.split("/", 1)
            return (provider, model, None)

    # 4. Tier-based defaults
    _tier_map = {
        "orchestrator": SMART_MODEL,
        "planner": PLANNER_MODEL,
        "insight": INSIGHT_MODEL,
        "inline": INLINE_MODEL,
    }
    model = _tier_map.get(agent_type, SUB_AGENT_MODEL)
    return (LLM_PROVIDER, model, None)


def resolve_capabilities() -> dict:
    """Resolve web_search_provider and vision_provider from workbench config.

    For each capability, checks workbench.capabilities first, then falls back
    to provider-level config. "own" resolves to the orchestrator's actual
    provider (from workbench, not the global LLM_PROVIDER).
    """
    # Start with provider-level defaults
    result = {
        "web_search_provider": _provider_get("web_search_provider"),
        "vision_provider": _provider_get("vision_provider"),
    }

    # Override with workbench capabilities (per-capability, not all-or-nothing)
    wb = get("workbench", {})
    if isinstance(wb, dict):
        caps = wb.get("capabilities", {})
        if isinstance(caps, dict):
            for cap_key, result_key in [
                ("web_search", "web_search_provider"),
                ("vision", "vision_provider"),
            ]:
                val = caps.get(cap_key)
                if val is None:
                    continue  # Not set — keep provider-level default
                if val == "own":
                    # Resolve to orchestrator's actual provider
                    orch_provider, _, _ = resolve_agent_model("orchestrator")
                    val = orch_provider
                result[result_key] = val if val != "disabled" else None

    return result


def _migrate_combos_to_presets(cfg: dict) -> bool:
    """Migrate old combos config to workbench presets format.

    Returns True if migration was performed and config was modified.
    """
    if "workbench" in cfg:
        return False  # Already migrated

    combos = cfg.get("combos")
    has_combos = isinstance(combos, dict) and combos.get("saved")
    has_agent_models = isinstance(cfg.get("agent_models"), dict) and cfg.get("agent_models")

    if not has_combos and not has_agent_models:
        return False

    # Convert each combo to a preset
    presets = {}
    for key, combo in (combos.get("saved", {}).items() if has_combos else []):
        if not isinstance(combo, dict):
            continue
        provider = combo.get("provider", "gemini")
        models = combo.get("models", {})

        # Map old model tiers to per-agent configs
        agents = {}
        tier_to_agents = {
            "model": ["orchestrator"],
            "sub_agent_model": ["viz_plotly", "viz_mpl", "viz_jsx", "data_ops", "data_io", "envoy"],
            "insight_model": ["insight", "eureka"],
            "inline_model": ["inline", "memory"],
            "planner_model": ["planner"],
        }
        for tier, agent_types in tier_to_agents.items():
            model_name = models.get(tier) or models.get("model", "")
            for at in agent_types:
                agents[at] = {
                    "provider": provider,
                    "model": model_name,
                    "base_url": combo.get("base_url"),
                }

        presets[key] = {
            "name": combo.get("name", key),
            "agents": agents,
            "capabilities": {
                "web_search": combo.get("web_search_provider"),
                "vision": combo.get("vision_provider"),
            },
        }

    # Set up workbench from active combo
    active = combos.get("active") if has_combos else None
    cfg["presets"] = presets
    cfg["workbench"] = {
        "preset": active if active in presets else None,
        "agents": {},
        "capabilities": {
            "web_search": None,
            "vision": None,
        },
    }

    # Also migrate agent_models overrides
    agent_models = cfg.get("agent_models", {})
    if isinstance(agent_models, dict):
        for agent_type, value in agent_models.items():
            if isinstance(value, str) and "/" in value:
                p, m = value.split("/", 1)
                cfg["workbench"]["agents"][agent_type] = {
                    "provider": p,
                    "model": m,
                    "base_url": None,
                }

    return True


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
INSIGHT_FEEDBACK = get("reasoning.insight_feedback", False)
INSIGHT_FEEDBACK_MAX_ITERS = get("reasoning.insight_feedback_max_iterations", 2)
PIPELINE_CONFIRMATION = get("reasoning.pipeline_confirmation", True)

# ---- Gemini-specific settings ------------------------------------------------
# Thinking levels for Gemini 3+ models. Ignored for Gemini < 3.
# "model" = orchestrator + planner (smart tier), "sub_agent" = mission/viz agents.
# Values: "off", "low", "high".
GEMINI_THINKING_MODEL = _provider_get("thinking_model", "high")
GEMINI_THINKING_SUB_AGENT = _provider_get("thinking_sub_agent", "high")
GEMINI_THINKING_INSIGHT = _provider_get("thinking_insight", "low")


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
    "memory_reload_interval": "Hot reload: restart chat session with fresh LTM every N rounds. Set to 0 to disable.",
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
    # Gemini Interactions API
    "use_interactions_api": "Use Gemini Interactions API for server-side conversation state. Eliminates quadratic context growth. Falls back to Chat API if interactions fail.",
    "use_tool_store": "Browse-and-load tool pattern. Model discovers tools on first turn and loads what it needs. Reduces per-call tool token cost. Only effective with Interactions API.",
    # Truncation
    "truncation": "Override text character limits for truncation. Keys are named limits (e.g. 'console.summary', 'history.error'). Values are integers; 0 means no truncation. See agent/truncation.py for all limit names.",
    "truncation_items": "Override item count limits for list truncation. Keys are named limits (e.g. 'items.tool_args', 'items.columns'). Values are integers; 0 means no truncation. See agent/truncation.py for all limit names.",
    # Provider-specific
    "providers.gemini.rate_limit_interval": "Minimum seconds between API calls to Gemini. Default: 0 (disabled).",
    "providers.openai.rate_limit_interval": "Minimum seconds between API calls to OpenAI. Default: 0 (disabled).",
    "providers.anthropic.rate_limit_interval": "Minimum seconds between API calls to Anthropic. Default: 0 (disabled).",
    "providers.minimax.rate_limit_interval": "Minimum seconds between API calls to MiniMax. Prevents 500 errors from rate limiting. Default: 2.0s.",
}


def list_all_api_keys() -> list[dict]:
    """List all API key entries from .env with masked values."""
    from dotenv import dotenv_values

    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return []

    env_vars = dotenv_values(str(env_path))
    keys = []

    # Known provider mapping (reverse of _PROVIDER_ENV_KEYS)
    env_to_provider = {v: k for k, v in _PROVIDER_ENV_KEYS.items()}

    for name, value in env_vars.items():
        if not name.endswith("_KEY"):
            continue
        configured = bool(value and value.strip())
        masked = f"...{value.strip()[-4:]}" if configured and len(value.strip()) >= 4 else ("...****" if configured else None)
        keys.append({
            "name": name,
            "provider": env_to_provider.get(name),
            "masked": masked,
            "configured": configured,
        })

    return keys


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
        PLANNER_MODEL
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
        INSIGHT_FEEDBACK, \
        INSIGHT_FEEDBACK_MAX_ITERS
    global PIPELINE_CONFIRMATION
    global GEMINI_THINKING_MODEL, GEMINI_THINKING_SUB_AGENT, GEMINI_THINKING_INSIGHT

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
    DATA_BACKEND = get("data_backend", "cdf")
    CATALOG_SEARCH_METHOD = get("catalog_search_method", "semantic")
    PARALLEL_FETCH = get("parallel_fetch", True)
    PARALLEL_MAX_WORKERS = get("parallel_max_workers", 4)
    MAX_PLOT_POINTS = get("max_plot_points", 10_000)
    PREFER_VIZ_BACKEND = get("prefer_viz_backend", "matplotlib")
    OBSERVATION_SUMMARIES = get("reasoning.observation_summaries", True)
    SELF_REFLECTION = get("reasoning.self_reflection", True)
    INSIGHT_FEEDBACK = get("reasoning.insight_feedback", False)
    INSIGHT_FEEDBACK_MAX_ITERS = get("reasoning.insight_feedback_max_iterations", 2)
    PIPELINE_CONFIRMATION = get("reasoning.pipeline_confirmation", True)
    GEMINI_THINKING_MODEL = _provider_get("thinking_model", "high")
    GEMINI_THINKING_SUB_AGENT = _provider_get("thinking_sub_agent", "high")
    GEMINI_THINKING_INSIGHT = _provider_get("thinking_insight", "low")

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


# =============================================================================
# Registry protocol adapter
# =============================================================================


class _ProviderRegistryAdapter:
    name = "llm.providers"
    description = "Supported LLM providers with API key env vars"

    def get(self, key: str):
        for p in PROVIDERS:
            if p["id"] == key:
                return p
        return None

    def list_all(self) -> dict:
        return {p["id"]: p for p in PROVIDERS}


PROVIDER_REGISTRY = _ProviderRegistryAdapter()
from agent.registry_protocol import register_registry  # noqa: E402
register_registry(PROVIDER_REGISTRY)
