"""SPICE envoy kind — tools populated dynamically from heliospice MCP server."""

# Starts empty; populated at runtime by add_tools()
TOOLS: list[dict] = []
HANDLERS: dict = {}

GLOBAL_TOOLS: list[str] = [
    "ask_clarification",
    "manage_session_assets",
    "list_fetched_data",
    "review_memory",
    "events",
]
