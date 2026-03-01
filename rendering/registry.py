"""
Tool registry for visualization operations.

Describes the visualization tools as structured data.
The VizAgent sub-agents use this registry to understand what
operations are available and validate arguments before dispatching to
PlotlyRenderer.

Adding a new capability:
    1. Add or update an entry in TOOLS below
    2. Implement the method in plotly_renderer.py
    3. Add dispatch logic in agent/core.py _execute_tool()
"""

TOOLS = [
    {
        "name": "render_plotly_json",
        "description": "Create or update the plot by providing a Plotly figure JSON with data_label placeholders. The system fills in actual data arrays from memory.",
        "parameters": [
            {"name": "figure_json", "type": "object", "required": True,
             "description": "Plotly figure dict with 'data' (array of trace stubs, "
                            "each with 'data_label' and standard Plotly trace properties) "
                            "and 'layout' (standard Plotly layout dict). "
                            "Each trace's 'data_label' fills x from the DataFrame index "
                            "(timestamps for timeseries entries where is_timeseries is true, "
                            "raw index values for general data where is_timeseries is false) "
                            "and y from column values. "
                            "Multi-panel: define yaxis, yaxis2 with domains."},
        ],
    },
    {
        "name": "manage_plot",
        "description": "Imperative operations on the current figure: export, reset, get state.",
        "parameters": [
            {"name": "action", "type": "string", "required": True,
             "enum": ["reset", "get_state", "export"],
             "description": "Action to perform"},
            {"name": "filename", "type": "string", "required": False,
             "description": "Output filename for export action"},
            {"name": "format", "type": "string", "required": False, "default": "png",
             "enum": ["png", "pdf"],
             "description": "Export format (default: png)"},
        ],
    },
]

# Build lookup dict for fast access
_TOOL_MAP = {t["name"]: t for t in TOOLS}



