"""Agent layer for natural language heliophysics data visualization.

Lazy imports to prevent circular dependency:
knowledge.* → agent.event_bus → agent.__init__ → agent.core → knowledge.*
"""


def __getattr__(name: str):
    if name in ("OrchestratorAgent", "create_agent"):
        from .core import OrchestratorAgent, create_agent
        return OrchestratorAgent if name == "OrchestratorAgent" else create_agent
    if name in ("TOOLS", "get_tool_schemas"):
        from .tools import TOOLS, get_tool_schemas
        return TOOLS if name == "TOOLS" else get_tool_schemas
    if name == "get_system_prompt":
        from .prompts import get_system_prompt
        return get_system_prompt
    raise AttributeError(f"module 'agent' has no attribute {name!r}")
