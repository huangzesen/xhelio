"""
Tool registry for JSX/Recharts visualization operations.

Describes the JSX visualization tools as structured data, mirroring
rendering/registry.py for Plotly. The actual tool schemas for the LLM
are defined in agent/tools.py; this registry is for internal reference
and validation.
"""

TOOLS = [
    {
        "name": "generate_jsx_component",
        "description": "Generate and compile a React/Recharts JSX component for interactive visualization.",
        "parameters": [
            {"name": "code", "type": "string", "required": True,
             "description": "JSX/TSX source code for a React component using Recharts. "
                            "Must export default a component. Use useData(label) to access data."},
            {"name": "description", "type": "string", "required": False,
             "description": "Short description of what the component shows (for the user)."},
        ],
    },
    {
        "name": "manage_jsx_output",
        "description": "Manage JSX component outputs: list, view source, recompile, or delete.",
        "parameters": [
            {"name": "action", "type": "string", "required": True,
             "enum": ["list", "get_source", "recompile", "delete"],
             "description": "Action to perform on JSX outputs."},
            {"name": "script_id", "type": "string", "required": False,
             "description": "The script ID (required for get_source, recompile, delete)."},
        ],
    },
]

# Build lookup dict for fast access
_TOOL_MAP = {t["name"]: t for t in TOOLS}
