"""VIZ_BACKENDS configuration — backend name to agent metadata mapping."""

from .agent_registry import (
    VIZ_PLOTLY_TOOLS, VIZ_MPL_TOOLS, VIZ_JSX_TOOLS,
    CTX_VIZ_PLOTLY, CTX_VIZ_MPL, CTX_VIZ_JSX,
)

VIZ_BACKENDS: dict[str, dict] = {
    "plotly": {
        "agent_id": "VizAgent[Plotly]",
        "agent_type": "viz_plotly",
        "agent_ctx": CTX_VIZ_PLOTLY,
        "prompt_dir": "viz_plotly",
        "tools": VIZ_PLOTLY_TOOLS,
        "needs_session_dir": False,
    },
    "mpl": {
        "agent_id": "VizAgent[Mpl]",
        "agent_type": "viz_mpl",
        "agent_ctx": CTX_VIZ_MPL,
        "prompt_dir": "viz_mpl",
        "tools": VIZ_MPL_TOOLS,
        "needs_session_dir": True,
        "llm_retry_timeout": 60.0,
        "llm_max_retries": 2,
        "llm_reset_threshold": 2,
    },
    "jsx": {
        "agent_id": "VizAgent[JSX]",
        "agent_type": "viz_jsx",
        "agent_ctx": CTX_VIZ_JSX,
        "prompt_dir": "viz_jsx",
        "tools": VIZ_JSX_TOOLS,
        "needs_session_dir": False,
    },
}

# Alias: the LLM-facing tool schema uses "matplotlib" as the enum value
VIZ_BACKENDS["matplotlib"] = VIZ_BACKENDS["mpl"]
