"""
MCP server entry point for XHelio.

Exposes the OrchestratorAgent as MCP tools over stdio transport,
allowing any MCP-compatible client (Claude Desktop, Claude Code, Cursor, etc.)
to use XHelio as a data/visualization specialist.

Usage:
    python mcp_server.py              # Start MCP server (stdio)
    python mcp_server.py -v           # With verbose logging
    python mcp_server.py -m MODEL     # Override LLM model
"""

import argparse
import sys
import threading

from mcp.server.fastmcp import FastMCP, Image

# ---------------------------------------------------------------------------
# CLI args (parsed at import time so FastMCP sees them before run())
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description="xhelio MCP server")
_parser.add_argument("-m", "--model", default=None, help="Override LLM model")
_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
_args, _ = _parser.parse_known_args()

_model: str | None = _args.model
_verbose: bool = _args.verbose

# ---------------------------------------------------------------------------
# Lazy singleton agent
# ---------------------------------------------------------------------------
_agent = None
_agent_lock = threading.Lock()


def _get_agent():
    """Return the OrchestratorAgent singleton, creating it on first call."""
    global _agent
    if _agent is not None:
        return _agent
    with _agent_lock:
        if _agent is not None:
            return _agent
        from agent.core import create_agent
        _agent = create_agent(verbose=_verbose, model=_model)
        _agent.web_mode = True  # suppress auto-opening exported files
        return _agent


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "xhelio",
    instructions=(
        "XHelio is an AI-powered natural language interface for heliophysics "
        "and scientific data. Use the 'chat' tool to send natural language "
        "requests about scientific data visualization and analysis. The agent "
        "supports 52 missions from NASA CDAWeb including PSP, Solar "
        "Orbiter, ACE, Wind, MMS, THEMIS, and more. It can fetch data, compute "
        "derived quantities, and render interactive Plotly plots. When a plot is "
        "produced, it will be returned as a PNG image alongside the text response."
    ),
)


@mcp.tool()
def chat(message: str) -> list:
    """Send a natural language message to XHelio and get a response.

    The agent can search mission datasets, fetch data, compute derived
    quantities (magnitude, smoothing, derivatives, spectrograms, etc.), and
    render Plotly visualizations. When a plot is produced, it is returned as
    a PNG image alongside the text response.

    Examples:
        - "What missions do you support?"
        - "Show me ACE magnetic field data for last week"
        - "Compare PSP and Solar Orbiter proton density for January 2024"
        - "Compute the magnitude of the magnetic field and plot it"
        - "Zoom in to January 10-15"
        - "Export the plot as PDF"
    """
    agent = _get_agent()
    try:
        response = agent.process_message(message)
    except Exception as e:
        return [f"Error processing message: {e}"]

    result = [response]

    # Check for a plot figure and include as PNG if available
    try:
        fig = agent.get_plotly_figure()
        if fig is not None and fig.data:
            png_bytes = fig.to_image(format="png", width=1100)
            result.append(Image(data=png_bytes, format="png"))
    except Exception:
        pass  # image export failure is non-fatal

    return result


@mcp.tool()
def reset_session() -> str:
    """Reset the XHelio session.

    Clears conversation history, all in-memory data, and the current plot.
    Use this to start a fresh analysis session.
    """
    agent = _get_agent()
    agent.reset()
    try:
        from data_ops.store import get_store
        get_store().clear()
    except Exception:
        pass
    return "Session reset. Conversation history and data store cleared."


@mcp.tool()
def get_status() -> str:
    """Get the current status of the XHelio session.

    Returns the LLM model name, token usage, number of data entries
    in memory, and active plan status (if any).
    """
    agent = _get_agent()

    lines = []

    # Model
    lines.append(f"Model: {agent.model_name}")

    # Token usage
    usage = agent.get_token_usage()
    cached = usage.get('cached_tokens', 0)
    cache_str = f" / {cached} cached" if cached else ""
    lines.append(
        f"Tokens: {usage['input_tokens']} in / {usage['output_tokens']} out"
        f" / {usage['thinking_tokens']} thinking{cache_str} ({usage['api_calls']} API calls)"
    )

    # Data store
    try:
        from data_ops.store import get_store
        entries = get_store().list_entries()
        if entries:
            lines.append(f"Data entries: {len(entries)}")
            for entry in entries:
                label = entry.get("label", "(no label)")
                points = entry.get("rows", "(unknown)")
                lines.append(f"  - {label} ({points} points)")
        else:
            lines.append("Data entries: 0")
    except Exception:
        lines.append("Data entries: unavailable")

    # Plan status
    plan_status = agent.get_plan_status()
    if plan_status:
        lines.append(f"Plan: {plan_status}")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
