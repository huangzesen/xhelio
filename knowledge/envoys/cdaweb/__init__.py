"""CDAWeb envoy kind — tools for CDAWeb data discovery and fetch."""

from .handlers import handle_fetch_data_cdaweb, handle_browse_parameters

TOOLS: list[dict] = [
    {
        "name": "browse_parameters",
        "description": """Browse all parameters for one or more datasets. Returns the full parameter metadata
(name, type, units, description, size, fill) from the local metadata cache.

Use this to discover what variables a dataset contains before calling fetch_data_cdaweb.
Accepts a single dataset_id or multiple dataset_ids in one call.""",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "Single dataset ID (e.g., 'PSP_FLD_L2_MAG_RTN_1MIN')",
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
        "name": "fetch_data_cdaweb",
        "description": """Download and load CDAWeb timeseries data into memory.
Downloads CDF files from NASA CDAWeb for the requested time range, then stores the data in memory.
The data is stored with a label like 'AC_H2_MFI.BGSEc' for later reference.
Returns a unique data ID (e.g., 'a3f7c2e1_1') in the response.""",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "CDAWeb dataset ID (e.g., 'AC_H2_MFI', 'PSP_FLD_L2_MAG_RTN_1MIN')",
                },
                "parameter_id": {
                    "type": "string",
                    "description": "Parameter name (e.g., 'BGSEc', 'Magnitude')",
                },
                "time_start": {
                    "type": "string",
                    "description": "Start time in ISO 8601 format (e.g., '2024-01-15')",
                },
                "time_end": {
                    "type": "string",
                    "description": "End time in ISO 8601 format (e.g., '2024-01-20')",
                },
                "force_large_download": {
                    "type": "boolean",
                    "description": "Set to true to override the 1 GB download safety limit.",
                },
            },
            "required": ["dataset_id", "parameter_id", "time_start", "time_end"],
        },
    },
]

HANDLERS: dict = {
    "browse_parameters": handle_browse_parameters,
    "fetch_data_cdaweb": handle_fetch_data_cdaweb,
}

GLOBAL_TOOLS: list[str] = [
    "ask_clarification",
    "manage_session_assets",
    "list_fetched_data",
    "review_memory",
    "events",
]
