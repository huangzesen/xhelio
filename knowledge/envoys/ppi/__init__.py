"""PPI envoy kind — tools for PDS Planetary Plasma Interactions data."""

from .handlers import handle_fetch_data_ppi
from knowledge.envoys.cdaweb.handlers import handle_browse_parameters

TOOLS: list[dict] = [
    {
        "name": "browse_parameters",
        "description": """Browse all parameters for one or more PDS datasets. Returns the full parameter metadata
(name, type, units, description, size, fill) from the local metadata cache.

Use this to discover what variables a dataset contains before calling fetch_data_ppi.
Accepts a single dataset_id or multiple dataset_ids in one call.""",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "Single dataset ID (e.g., 'pds3:JNO-J-3-FGM-CAL-V1.0:DATA')",
                },
                "dataset_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Multiple dataset IDs to query at once",
                },
            },
        },
    },
    {
        "name": "fetch_data_ppi",
        "description": """Download and load PDS PPI archive data into memory.
Downloads data from the NASA PDS Planetary Plasma Interactions archive for the requested time range.
Supports PDS4 URN IDs (e.g., 'urn:nasa:pds:cassini-mag-cal:data-1sec-krtp') and
PDS3 IDs (e.g., 'pds3:JNO-J-3-FGM-CAL-V1.0:DATA').
The data is stored with a label like 'pds3:JNO-J-3-FGM-CAL-V1.0:DATA.BX PLANETOCENTRIC'.""",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "PDS dataset ID — PDS4 URN (e.g., 'urn:nasa:pds:cassini-mag-cal:data-1sec-krtp') or PDS3 (e.g., 'pds3:JNO-J-3-FGM-CAL-V1.0:DATA')",
                },
                "parameter_id": {
                    "type": "string",
                    "description": "Parameter name (e.g., 'BR', 'BX PLANETOCENTRIC')",
                },
                "time_start": {
                    "type": "string",
                    "description": "Start time in ISO 8601 format (e.g., '2024-01-15')",
                },
                "time_end": {
                    "type": "string",
                    "description": "End time in ISO 8601 format (e.g., '2024-01-20')",
                },
            },
            "required": ["dataset_id", "parameter_id", "time_start", "time_end"],
        },
    },
]

HANDLERS: dict = {
    "browse_parameters": handle_browse_parameters,
    "fetch_data_ppi": handle_fetch_data_ppi,
}

GLOBAL_TOOLS: list[str] = [
    "ask_clarification",
    "manage_session_assets",
    "list_fetched_data",
    "review_memory",
    "events",
]
