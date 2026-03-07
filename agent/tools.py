"""
Tool definitions for Gemini function calling.

Each tool schema defines what the LLM can call and what parameters it needs.
Tools are executed by the agent core based on LLM decisions.

Tool access per agent is controlled by explicit name lists in agent_registry.py.
"""

import config

TOOLS = [
    {
        "name": "search_datasets",
        "description": """Search for mission datasets by keyword. Use this when:
- User mentions a mission (Parker, ACE, Solar Orbiter, OMNI, Wind, DSCOVR, MMS, STEREO, Cassini, Voyager, Juno, MAVEN, Galileo)
- User mentions a data type (magnetic field, solar wind, plasma, density)
- User asks what data is available

Covers both CDAWeb and PDS PPI datasets. Returns matching mission, instrument, and dataset information.
Each dataset includes start_date and stop_date showing the available time coverage.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'parker magnetic', 'ACE solar wind', 'omni')",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_parameters",
        "description": """List plottable parameters for a specific dataset. Use this after search_datasets to find what parameters can be plotted.

Reads from local metadata cache. Falls back to downloading a Master CDF skeleton from CDAWeb if not cached locally.
Returns list of 1D numeric parameters with names, units, and descriptions.""",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "CDAWeb dataset ID (e.g., 'PSP_FLD_L2_MAG_RTN_1MIN')",
                }
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "browse_datasets",
        "description": """Browse all available science datasets for a mission. Returns TIME COVERAGE
(start_date, stop_date) for every dataset — use this to verify data availability before fetching.

Also use when:
- User asks "what datasets are available?" or "what else can I plot?"
- You need to find a dataset not in the recommended list
- User asks about a specific instrument or data type you don't have in your prompt

Returns a filtered list excluding calibration/housekeeping/ephemeris data.
Each entry has: id, description, start_date, stop_date, parameter_count, instrument.""",
        "parameters": {
            "type": "object",
            "properties": {
                "mission_id": {
                    "type": "string",
                    "description": "Mission ID (e.g., 'PSP', 'ACE', 'SolO', 'OMNI', 'WIND', 'DSCOVR', 'MMS', 'STEREO_A')",
                }
            },
            "required": ["mission_id"],
        },
    },
    {
        "name": "list_missions",
        "description": """List all missions with cached metadata. Use this when:
- User asks "what missions are available?"
- You need to see which missions have local data before browsing datasets
- You want a quick overview of the data catalog

Returns mission IDs and dataset counts. No parameters required.""",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_dataset_docs",
        "description": """Fetch CDAWeb documentation for a dataset. Use this when:
- User asks about coordinate systems, calibration, or data quality
- User asks who the PI or data contact is
- User asks what a parameter measures or how it was derived
- User asks for full parameter list with descriptions and units
- You need domain context to interpret or explain data
Returns: contact info, resource URL, documentation text, full parameter list (name, type, units, description, size), time range, and validation status.""",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "CDAWeb dataset ID (e.g., 'AC_H2_MFI')",
                }
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "ask_clarification",
        "description": """Ask the user a clarifying question when the request is ambiguous or the user's intent is unclear. Use this when:
- Multiple datasets could match the request
- Time range is not specified
- Parameter choice is unclear
- You need more information to proceed
- The user expresses dissatisfaction or criticism but doesn't specify what action to take
- The user corrects you but the desired fix is ambiguous

Do NOT guess when the user is unhappy — ask what they want instead.""",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The clarifying question to ask the user",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of choices to present (keep to 3-4 options)",
                },
                "context": {
                    "type": "string",
                    "description": "Brief explanation of why you need this information",
                },
            },
            "required": ["question"],
        },
    },
    # --- Data Operations Tools ---
    {
        "name": "fetch_data",
        "description": """Download and load timeseries data from CDAWeb or PDS archives into memory.
Downloads CDF files from CDAWeb or PDS PPI archives for the requested time range (requires network), then stores the data in memory for Python-side operations (magnitude, averages, differences, etc.) or plotting.
Supports CDAWeb datasets (e.g., 'AC_H2_MFI') and PDS PPI datasets (URN IDs like 'urn:nasa:pds:cassini-mag-cal:data-1sec-krtp').

The data is stored in memory with a label like 'AC_H2_MFI.BGSEc' for later reference by compute and plot tools.
Returns a unique data ID (e.g., 'a3f7c2e1_1') in the response. Use this ID in all subsequent tool calls to reference this data.""",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "Dataset ID — CDAWeb (e.g., 'AC_H2_MFI') or PDS URN (e.g., 'urn:nasa:pds:cassini-mag-cal:data-1sec-krtp')",
                },
                "parameter_id": {
                    "type": "string",
                    "description": "Parameter name (e.g., 'BGSEc', 'Magnitude', 'BR')",
                },
                "time_start": {
                    "type": "string",
                    "description": "Start time in ISO 8601 format (e.g., '2024-01-15' or '2024-01-15T06:00:00'). Resolve relative expressions like 'last week' to actual dates yourself using today's date.",
                },
                "time_end": {
                    "type": "string",
                    "description": "End time in ISO 8601 format (e.g., '2024-01-20' or '2024-01-15T18:00:00'). Resolve relative expressions like 'last week' to actual dates yourself using today's date.",
                },
                "force_large_download": {
                    "type": "boolean",
                    "description": "Set to true to override the 1 GB download safety limit. Only use when the user explicitly confirms a large download.",
                },
            },
            "required": ["dataset_id", "parameter_id", "time_start", "time_end"],
        },
    },
    {
        "name": "list_fetched_data",
        "description": "Show all timeseries currently held in memory. Returns data IDs, labels, shapes, units, and time ranges. Use the 'id' field to reference data in other tools.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "plan_check",
        "description": "Load and review a saved plan from filesystem. Use this after receiving a planning complete message to retrieve the plan details.",
        "parameters": {
            "type": "object",
            "properties": {
                "plan_file": {
                    "type": "string",
                    "description": "Path to the plan JSON file (provided by planner)."
                }
            },
            "required": ["plan_file"]
        },
    },
    {
        "name": "custom_operation",
        "description": """Apply a pandas/numpy/xarray/scipy/pywt operation to in-memory data. This is the universal compute tool — use it for ALL data transformations after fetching data with fetch_data.

The code must:
- Assign the result to `result` (DataFrame/Series with DatetimeIndex, or xarray DataArray with 'time' dim). When `force_timeseries` is false, result can have any index type.
- Use only sandbox variables, `pd` (pandas), `np` (numpy), `xr` (xarray), `scipy` (full scipy), `pywt` (PyWavelets), and optional: `numba`, `sklearn`, `statsmodels`, `astropy`, `lmfit`, `sympy`, `mpl_cm` (matplotlib colormaps) — no imports, no file I/O

Each source becomes a named variable in the sandbox:
- 2D data (DataFrame): `df_<SUFFIX>` where SUFFIX is the part after the last '.' (e.g., 'DATASET.BR' → df_BR)
- 3D+ data (xarray DataArray): `da_<SUFFIX>` (e.g., 'DATASET.EFLUX_VS_PA_E' → da_EFLUX_VS_PA_E)
- If label has no '.', the full label is used as suffix
- The first DataFrame source is also aliased as `df` for backward compatibility
- When multiple sources have the same suffix, ID-based disambiguation is added (e.g., df_BR_1, df_BR_2)

The result can be a DataFrame (with DatetimeIndex) OR an xarray DataArray (with 'time' dim). DataArray results are stored as-is — useful for intermediate xarray→xarray operations or for 2D spectrograms.

Set `force_timeseries: false` when the operation produces non-timeseries output from timeseries input — e.g., power spectral density (Welch), histograms, correlation matrices, statistical summaries. This skips the DatetimeIndex enforcement so results with frequency/bin/category indices are accepted.

xarray operations (DataArray sources — check storage_type in fetch_data response):
- Slice 3D to 2D: `result = da_EFLUX_VS_PA_E.isel(dim1=0)`
- Average over a dim: `result = da_EFLUX_VS_PA_E.mean(dim='dim1')`
- IMPORTANT: When producing a DataFrame for heatmap/spectrogram plotting, use meaningful column names
  (e.g., pitch angle values, energy bins) — NOT generic indices ('0', '1', '2').
- For log-scale spectrograms: apply np.log10 here — the viz agent CANNOT apply log scaling.

Use `search_function_docs` and `get_function_docs` to look up unfamiliar scipy/pywt APIs before writing code.

Do NOT call this tool when the request cannot be expressed as a pandas/numpy operation (e.g., "email me the data", "upload to server"). Instead, explain to the user what is and isn't possible.""",
        "parameters": {
            "type": "object",
            "properties": {
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Data IDs (from list_fetched_data) of source timeseries in memory. Each becomes a sandbox variable: df_<SUFFIX> (DataFrame) or da_<SUFFIX> (xarray DataArray) where SUFFIX is derived from the source label's suffix. First DataFrame also available as 'df'. For single-source ops, pass one-element array.",
                },
                "source_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "DEPRECATED: Use source_ids instead. Labels of source timeseries in memory (for backward compatibility).",
                },
                "code": {
                    "type": "string",
                    "description": "Python code using df/da_ variables, pd (pandas), np (numpy), xr (xarray), scipy (full scipy), pywt (PyWavelets), and optional: numba, sklearn, statsmodels, astropy, lmfit, sympy, mpl_cm. Must assign to 'result'.",
                },
                "output_label": {
                    "type": "string",
                    "description": "Label for the result (e.g., 'B_normalized', 'B_clipped')",
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of the operation",
                },
                "units": {
                    "type": "string",
                    "description": "Physical units of the result (e.g., 'nT', 'km/s', 'nT/s', 'cm^-3'). If omitted, inherits from source. Set explicitly when the operation changes dimensions (e.g., derivative adds '/s', multiply changes units, normalize produces dimensionless '').",
                },
                "force_timeseries": {
                    "type": "boolean",
                    "description": "When true (default), results from timeseries sources must have a DatetimeIndex. Set to false for operations that produce non-timeseries output from timeseries input (e.g., PSD/Welch, histograms, correlation matrices).",
                },
            },
            "required": ["source_ids", "code", "output_label", "description"],
        },
    },
    {
        "name": "store_dataframe",
        "description": """Create a new DataFrame from scratch and store it in memory. Use this when:
- You have text data (event lists, search results, catalogs) that should become a plottable dataset
- The user wants to manually define data points (e.g., from a table in a paper or website)
- You need to create a dataset that doesn't come from CDAWeb

The code must:
- Use only `pd` (pandas) and `np` (numpy) — no imports, no file I/O, no `df` variable
- Assign the result to `result` (must be a DataFrame or Series with DatetimeIndex)
- Create a DatetimeIndex from dates using pd.to_datetime() and .set_index()

Examples:
- Event catalog:
  ```
  dates = ['2024-01-01', '2024-02-15', '2024-05-10']
  values = [5.2, 7.8, 6.1]
  result = pd.DataFrame({'x_class_flux': values}, index=pd.to_datetime(dates))
  ```
- Numeric timeseries:
  ```
  result = pd.DataFrame({'value': [1.0, 2.5, 3.0]}, index=pd.date_range('2024-01-01', periods=3, freq='D'))
  ```
- Event catalog with string columns:
  ```
  dates = pd.to_datetime(['2024-01-10', '2024-03-22'])
  result = pd.DataFrame({'class': ['X1.5', 'X2.1'], 'region': ['AR3555', 'AR3590']}, index=dates)
  ```""",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code using pd (pandas) and np (numpy) that constructs data and assigns to 'result'. Must produce a DataFrame with DatetimeIndex.",
                },
                "output_label": {
                    "type": "string",
                    "description": "Label for the stored dataset (e.g., 'xclass_flares_2024', 'cme_catalog')",
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of the dataset",
                },
                "units": {
                    "type": "string",
                    "description": "Optional units for the data columns (e.g., 'W/m²', 'km/s')",
                },
            },
            "required": ["code", "output_label", "description"],
        },
    },
    # --- Function Documentation Tools ---
    {
        "name": "search_function_docs",
        "description": """Search the scientific computing function catalog by keyword. Use this to find functions for signal processing, spectral analysis, filtering, interpolation, wavelets, statistics, etc.

Returns function names, sandbox call syntax, and one-line summaries.

Cataloged libraries: scipy.signal, scipy.fft, scipy.interpolate, scipy.stats, scipy.integrate, pywt (PyWavelets).""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keyword (e.g., 'bandpass filter', 'spectrogram', 'wavelet', 'interpolate')",
                },
                "package": {
                    "type": "string",
                    "enum": [
                        "scipy.signal",
                        "scipy.fft",
                        "scipy.interpolate",
                        "scipy.stats",
                        "scipy.integrate",
                        "pywt",
                    ],
                    "description": "Optional: restrict search to a specific package",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_function_docs",
        "description": """Get the full docstring and signature for a specific function. Use this after search_function_docs to understand function parameters, return values, and usage examples before writing code.""",
        "parameters": {
            "type": "object",
            "properties": {
                "package": {
                    "type": "string",
                    "description": "Package path (e.g., 'scipy.signal', 'pywt')",
                },
                "function_name": {
                    "type": "string",
                    "description": "Function name (e.g., 'butter', 'cwt', 'spectrogram')",
                },
            },
            "required": ["package", "function_name"],
        },
    },
    # --- Describe & Export Tools ---
    {
        "name": "describe_data",
        "description": """Get statistical summary of an in-memory timeseries. Use this when:
- User asks "what does the data look like?" or "summarize the data"
- You want to understand the data before deciding what operations to apply
- User asks about min, max, average, or data quality

Returns statistics (min, max, mean, std, percentiles, NaN count) and the LLM can narrate findings.
Optionally filter to a time sub-range (time_start/time_end) to inspect a specific region.""",
        "parameters": {
            "type": "object",
            "properties": {
                "data_id": {
                    "type": "string",
                    "description": "Data ID (from list_fetched_data) of the data in memory. Use this instead of label for precise reference.",
                },
                "label": {
                    "type": "string",
                    "description": "Label of the data in memory (e.g., 'AC_H2_MFI.BGSEc'). For backward compatibility — prefer data_id for precise reference.",
                },
                "time_start": {
                    "type": "string",
                    "description": "Optional start time (ISO 8601, no 'Z' suffix — e.g. '2024-01-15T00:00:00') to filter data before computing stats. Omit for full range.",
                },
                "time_end": {
                    "type": "string",
                    "description": "Optional end time (ISO 8601, no 'Z' suffix — e.g. '2024-01-20T00:00:00') to filter data before computing stats. Omit for full range.",
                },
            },
            "required": ["data_id"],
        },
    },
    {
        "name": "preview_data",
        "description": """Preview actual values (first/last N rows) of an in-memory timeseries. Use this when:
- You need to see actual data values to diagnose an issue
- User asks "show me the data" or "what values are in there?"
- A plot looks wrong and you want to check the underlying data
- You want to verify a computation produced correct results

Returns timestamps and values for the requested rows. Use describe_data for statistics instead.
Optionally filter to a time sub-range (time_start/time_end) to inspect a specific region.

Use position='sampled' to get evenly-spaced rows across the full range (~20 rows).
This is useful for spotting trends, gaps, or anomalies without reading the full dataset.""",
        "parameters": {
            "type": "object",
            "properties": {
                "data_id": {
                    "type": "string",
                    "description": "Data ID (from list_fetched_data) of the data in memory. Use this instead of label for precise reference.",
                },
                "label": {
                    "type": "string",
                    "description": "Label of the data in memory (e.g., 'AC_H2_MFI.BGSEc'). For backward compatibility — prefer data_id for precise reference.",
                },
                "n_rows": {
                    "type": "integer",
                    "description": "Number of rows to show from each end (default: 5, max: 50)",
                },
                "position": {
                    "type": "string",
                    "enum": ["head", "tail", "both", "sampled"],
                    "description": "Which rows to show: 'head' (first N), 'tail' (last N), 'both' (default), or 'sampled' (evenly spaced across full range)",
                },
                "stride": {
                    "type": "integer",
                    "description": "Step size for 'sampled' mode (e.g., 10 = every 10th row). Default: auto (~20 rows). Ignored if position is not 'sampled'.",
                },
                "time_start": {
                    "type": "string",
                    "description": "Optional start time (ISO 8601, no 'Z' suffix — e.g. '2024-01-15T00:00:00') to filter data before previewing. Omit for full range.",
                },
                "time_end": {
                    "type": "string",
                    "description": "Optional end time (ISO 8601, no 'Z' suffix — e.g. '2024-01-20T00:00:00') to filter data before previewing. Omit for full range.",
                },
            },
            "required": ["data_id"],
        },
    },
    {
        "name": "save_data",
        "description": """Export an in-memory timeseries to a CSV file.

ONLY use this when the user explicitly asks to save, export, or download data.
Do NOT use this proactively after computations — data stays in memory for plotting.

The CSV file has a datetime column (ISO 8601 UTC) followed by data columns.
If no filename is given, one is auto-generated from the label.""",
        "parameters": {
            "type": "object",
            "properties": {
                "data_id": {
                    "type": "string",
                    "description": "Data ID (from list_fetched_data) of the data to export. Use this instead of label for precise reference.",
                },
                "label": {
                    "type": "string",
                    "description": "Label of the data in memory to export. For backward compatibility — prefer data_id for precise reference.",
                },
                "filename": {
                    "type": "string",
                    "description": "Output filename (e.g., 'ace_mag.csv'). '.csv' is appended if missing. Default: auto-generated from label.",
                },
            },
            "required": ["data_id"],
        },
    },
    {
        "name": "merge_datasets",
        "description": """Merge multiple time ranges of the same data product into one dataset.

Only works for entries with the same hash prefix (same data product from the same source).
Use list_fetched_data to find datasets with the same label but different time ranges.
The merged result combines all time ranges, sorts by time, and removes duplicates.""",
        "parameters": {
            "type": "object",
            "properties": {
                "data_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of data IDs (from list_fetched_data) to merge. Must all be the same data product.",
                },
            },
            "required": ["data_ids"],
        },
    },
    # --- Visualization ---
    {
        "name": "render_plotly_json",
        "description": """Create or update the plot by providing a Plotly figure JSON.

You generate a standard Plotly figure dict with `data` (array of traces) and `layout`.
Instead of providing actual data arrays (x, y, z), put a `data_label` field in each
trace dict. The system resolves each label to real data from memory and fills in x/y/z.

## Trace stubs

Each trace in `data` needs:
- `data_label` (string, required): label of the data in memory (from list_fetched_data)
- `type` (string): Plotly trace type — "scatter" (default), "heatmap", "bar", etc.
- All other Plotly trace properties are passed through as-is (mode, line, marker, etc.)
- `xaxis` and `yaxis` (strings): axis references like "x", "x2", "y", "y2" for multi-panel
- Vector data (3-column) is auto-decomposed into (x), (y), (z) component traces

## Layout

Standard Plotly layout dict. For multi-panel plots, define multiple yaxes with domains:
- `yaxis`: {"domain": [0.55, 1], "title": {"text": "nT"}}
- `yaxis2`: {"domain": [0, 0.45], "title": {"text": "Hz"}}
- `xaxis`: {"domain": [0, 1], "anchor": "y"}
- `xaxis2`: {"domain": [0, 1], "anchor": "y2", "matches": "x"}

Shapes, annotations, and all standard Plotly layout properties work directly.

## Automatic processing

The system automatically handles:
- DatetimeIndex → ISO 8601 strings for x-axis
- Vector data (n,3) → 3 separate component traces with color assignment
- Large datasets (>5000 pts) → min-max downsampling
- Very large datasets (>100K pts) → WebGL (scattergl)
- NaN values → None (Plotly requirement)
- Heatmap colorbar positioning from yaxis domain

## Example: single panel

```json
{"data": [{"type": "scatter", "data_label": "ACE_Bmag", "mode": "lines", "line": {"color": "red"}}],
 "layout": {"title": {"text": "ACE Magnetic Field"}, "yaxis": {"title": {"text": "nT"}}}}
```

## Example: two panels

```json
{"data": [
    {"type": "scatter", "data_label": "ACE_Bmag", "xaxis": "x", "yaxis": "y"},
    {"type": "scatter", "data_label": "ACE_density", "xaxis": "x2", "yaxis": "y2"}
  ],
 "layout": {
    "xaxis":  {"domain": [0, 1], "anchor": "y"},
    "xaxis2": {"domain": [0, 1], "anchor": "y2", "matches": "x"},
    "yaxis":  {"domain": [0.55, 1], "anchor": "x", "title": {"text": "B (nT)"}},
    "yaxis2": {"domain": [0, 0.45], "anchor": "x2", "title": {"text": "n (cm⁻³)"}}
  }}
```""",
        "parameters": {
            "type": "object",
            "properties": {
                "figure_json": {
                    "type": "object",
                    "description": "Plotly figure dict with 'data' (array of trace stubs with data_label) and 'layout'.",
                }
            },
            "required": ["figure_json"],
        },
    },
    {
        "name": "manage_plot",
        "description": """Imperative operations on the current figure: export, reset, get state.
Use action parameter to select the operation.""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["reset", "get_state", "export"],
                    "description": "Action to perform",
                },
                "filename": {
                    "type": "string",
                    "description": "Output filename for export action",
                },
                "format": {
                    "type": "string",
                    "enum": ["png", "pdf"],
                    "description": "Export format: 'png' (default) or 'pdf'",
                },
            },
            "required": ["action"],
        },
    },
    # --- Session Assets ---
    {
        "name": "get_session_assets",
        "description": "Get a snapshot of all session assets and their status: "
        "plot (active/restorable/none), data entries (loaded vs deferred), "
        "and operation count. Orchestrator and planner see this automatically; "
        "other agents can call this for on-demand status.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "restore_plot",
        "description": "Restore a deferred plot from a resumed session. "
        "When session context shows the plot as 'restorable', "
        "call this before delegate_to_insight or other plot-dependent tools. "
        "No-op if already active.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    # --- Full Catalog Search ---
    {
        "name": "search_full_catalog",
        "description": """Search the locally cached dataset catalog (CDAWeb + PPI, 2500+ datasets) by keyword. Cache is refreshed from CDAWeb REST API every 24 hours. Use this when:
- User asks about a mission or instrument NOT in the supported missions table
- User wants to browse broadly across all available data (e.g., "what magnetospheric data is available?")
- User asks about a mission you don't have a specialist agent for
- User wants to search by physical quantity across all missions (e.g., "proton density datasets")

Covers both CDAWeb and PDS PPI datasets. Returns matching dataset IDs and titles.

Datasets found here can be fetched directly with `fetch_data` and plotted via `the active visualization agent` — no mission agent needed.

Do NOT use this for missions already in the routing table (PSP, ACE, etc.) — use delegate_to_envoy instead.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms (mission name, instrument, physical quantity, e.g., 'cluster magnetic field', 'voyager 2', 'proton density')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default 20)",
                },
            },
            "required": ["query"],
        },
    },
    # --- Web Search (orchestrator + planner only) ---
    {
        "name": "web_search",
        "description": """Search the web for real-world context. Use this when:
- User asks about solar events, flares, CMEs, geomagnetic storms, or space weather
- User asks what happened during a specific time period
- User wants scientific context or explanations of heliophysics phenomena
- User asks for an ICME list, event catalog, or recent news

Use for contextual knowledge only. For mission datasets, use `search_datasets` and mission agents instead.

Returns grounded text with source URLs.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (e.g., 'major solar storms January 2024', 'ICME list 2024', 'X-class flare events')",
                }
            },
            "required": ["query"],
        },
    },
    # --- Document Reading ---
    {
        "name": "read_document",
        "description": """Read a PDF or image file and extract its text content using Gemini vision.
Supported formats: PDF (.pdf), PNG (.png), JPEG (.jpg, .jpeg), GIF (.gif), WebP (.webp), BMP (.bmp), TIFF (.tiff).
Use this when:
- User uploads or references a PDF or image file
- User wants to read, summarize, or extract content from a document
- User asks questions about a document's contents
The extracted text is saved to the agent data directory (default ~/.xhelio/documents/) for persistence across sessions.
Returns the extracted text content and the saved file path.""",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to read",
                },
                "prompt": {
                    "type": "string",
                    "description": "Optional prompt for targeted extraction (e.g., 'extract the data table', 'list all dates and values'). If not provided, a default extraction prompt is used.",
                },
            },
            "required": ["file_path"],
        },
    },
    # --- Memory review ---
    {
        "name": "review_memory",
        "description": "Rate how useful an injected operational memory was for this task. After completing your main task, pick at most 4 memories worth commenting on from the Operational Knowledge section.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The memory ID (from the [ID] prefix in the Operational Knowledge section)",
                },
                "stars": {
                    "type": "integer",
                    "description": "1-5 rating: 5=directly prevented mistake, 4=useful context, 3=relevant but no impact, 2=irrelevant, 1=misleading",
                },
                "rating": {
                    "type": "string",
                    "description": "Why this star count (e.g., 'Caught the NaN issue before fetch')",
                },
                "criticism": {
                    "type": "string",
                    "description": "What's wrong or could be better about the memory (e.g., 'Too vague — doesn't say which datasets')",
                },
                "suggestion": {
                    "type": "string",
                    "description": "How to improve the memory's content or scope (e.g., 'Add dataset IDs (AC_H2_MFI)')",
                },
                "comment": {
                    "type": "string",
                    "description": "Any extra observation (e.g., 'Would have been useless for a different mission')",
                },
            },
            "required": ["memory_id", "stars", "rating", "criticism", "suggestion", "comment"],
        },
    },
    # --- Routing ---
    {
        "name": "delegate_to_envoy",
        "description": """Delegate a data request to a mission-specific specialist agent. Use this when:
- The user asks about a specific mission's data (e.g., "show me ACE magnetic field data")
- The user wants to fetch, compute, or describe data from a specific mission
- You need mission-specific knowledge (dataset IDs, parameter names, analysis patterns)

Do NOT delegate:
- Visualization requests (plotting, zoom, render changes) — use the active visualization agent
- Requests to plot already-loaded data — use the active visualization agent
- General questions about capabilities

You can call delegate_to_envoy multiple times with the same mission_id in parallel — if the
primary agent is busy, an ephemeral overflow agent handles the request concurrently. However,
combining related requests into one call is often more efficient because the envoy agent has
full context of all sub-tasks.

The specialist will search datasets, fetch data, run computations, and report back what was done. You then decide whether to visualize the results.""",
        "parameters": {
            "type": "object",
            "properties": {
                "mission_id": {
                    "type": "string",
                    "description": "Spacecraft mission ID from the supported missions table (e.g., 'PSP', 'ACE', 'SolO', 'OMNI', 'WIND', 'DSCOVR', 'MMS', 'STEREO_A').",
                },
                "request": {
                    "type": "string",
                    "description": "The data request to send to the specialist (e.g., 'fetch magnetic field data for last week')",
                },
                "wait": {
                    "type": "boolean",
                    "description": "If true, wait for result. If false, fire-and-forget (returns null immediately). Default true.",
                    "default": True
                },
            },
            "required": ["mission_id", "request"],
        },
    },
    {
        "name": "delegate_to_viz",
        "description": """Delegate a visualization request to the visualization specialist agent. Use this when:
- The user asks to plot, display, or visualize data
- The user wants to change plot appearance (render type, colors, axis labels, title, log scale)
- The user wants to zoom, set time range, or resize the canvas
- The user wants publication-quality static figures (matplotlib backend)
- The user wants rich interactive dashboards with multiple linked charts (JSX backend)

Backend selection (backend parameter — check the default value in the schema):
- "plotly": Interactive scientific plots with zoom, pan, WebGL support
- "matplotlib": Publication-quality static figures, histograms, polar plots, 3D surfaces, complex layouts
- "jsx": React/Recharts dashboards with multiple linked charts

Export requests (PNG/PDF) are handled automatically when delegated to Plotly backend — no special handling needed.

Do NOT delegate:
- Data requests (fetch, compute, describe) — use delegate_to_envoy
- Dataset search or parameter listing — handle directly

The specialist has access to all visualization methods and can see what data is in memory.""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The visualization request (e.g., 'plot ACE_Bmag and PSP_Bmag together', 'switch to scatter plot', 'set log scale on y-axis', 'create a histogram')",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context about what data is available or what was just done",
                },
                "backend": {
                    "type": "string",
                    "enum": ["plotly", "matplotlib", "jsx"],
                    "description": "Visualization backend to use. Defaults to the configured PREFER_VIZ_BACKEND setting if not specified.",
                },
                "wait": {
                    "type": "boolean",
                    "description": "If true, wait for result. If false, fire-and-forget (returns null immediately). Default true.",
                    "default": True
                },
            },
            "required": ["request"],
        },
    },
    # --- Matplotlib Visualization Tools ---
    {
        "name": "generate_mpl_script",
        "description": """Generate and execute a matplotlib script to create a visualization.

Write a standard matplotlib script. The following are ALREADY available — do NOT import them:
- `plt` (matplotlib.pyplot)
- `np` (numpy)
- `pd` (pandas)

Helper functions available:
- `load_data(label)` → pd.DataFrame — Load data by label from memory
- `load_meta(label)` → dict — Load metadata (units, description, etc.)
- `available_labels()` → list[str] — List all available data labels

IMPORTANT:
- Do NOT call plt.show() — it will fail in headless mode
- Do NOT call plt.savefig() — it is called automatically after your script
- You MAY import additional matplotlib submodules (mpl_toolkits.mplot3d, etc.)
- You MAY import scipy for signal processing
- You MAY use print() for debugging — stdout is captured
- The output is always a PNG image

Example:
```python
# Load data
df = load_data("AC_H2_MFI.Magnitude")
meta = load_meta("AC_H2_MFI.Magnitude")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df.iloc[:, 0], linewidth=0.5)
ax.set_xlabel("Time")
ax.set_ylabel(f"Magnitude ({meta.get('units', '')})")
ax.set_title("ACE Magnetic Field Magnitude")
fig.autofmt_xdate()
```""",
        "parameters": {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "The matplotlib Python script to execute",
                },
                "description": {
                    "type": "string",
                    "description": "Short description of what the plot shows (for the user)",
                },
            },
            "required": ["script"],
        },
    },
    {
        "name": "manage_mpl_output",
        "description": """Manage matplotlib outputs and scripts for the current session.

Actions:
- list: List all MPL outputs (script_id, description, timestamp, has_output)
- get_script: Get the full Python script for a given script_id
- rerun: Re-execute a previously saved script
- delete: Delete a script and its output""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "get_script", "rerun", "delete"],
                    "description": "The action to perform",
                },
                "script_id": {
                    "type": "string",
                    "description": "The script ID (required for get_script, rerun, delete)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "generate_jsx_component",
        "description": """Generate and compile a React/Recharts JSX component for interactive visualization.

Write a React component using Recharts. The following are available:
- All Recharts components (LineChart, BarChart, AreaChart, ScatterChart, ComposedChart,
  PieChart, RadarChart, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Legend, etc.)
- React hooks (useState, useEffect, useMemo, useCallback, useRef)

Data access hooks (pre-injected — do NOT import these):
- `useData(label)` → array of row objects with `_time` (ISO string) or `_index` plus column values
- `useAllLabels()` → string array of all available data labels

CRITICAL RULES:
- You MUST `export default` your component
- Only import from 'react' and 'recharts'
- Do NOT use fetch(), eval(), window.location, or any browser APIs
- Do NOT use innerHTML, document.write, or DOM manipulation
- Use ResponsiveContainer for responsive sizing
- Handle empty data gracefully (check array length)

Example:
```tsx
import React, { useMemo } from "react";
import {
  ResponsiveContainer, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend
} from "recharts";

const Dashboard = () => {
  const data = useData("AC_H2_MFI.Magnitude");

  const chartData = useMemo(() =>
    data.map(d => ({
      time: new Date(d._time).toLocaleTimeString(),
      value: d["Magnitude"],
    })),
    [data]
  );

  if (!chartData.length) return <div>No data available</div>;

  return (
    <div style={{ width: "100%", height: 400 }}>
      <ResponsiveContainer>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="value" stroke="#8884d8" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default Dashboard;
```""",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The JSX/TSX source code for the React/Recharts component",
                },
                "description": {
                    "type": "string",
                    "description": "Short description of what the component shows (for the user)",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "manage_jsx_output",
        "description": """Manage JSX component outputs for the current session.

Actions:
- list: List all JSX outputs (script_id, description, timestamp, has_output)
- get_source: Get the original TSX source for a given script_id
- recompile: Re-compile a previously saved component
- delete: Delete a component and its outputs""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "get_source", "recompile", "delete"],
                    "description": "The action to perform",
                },
                "script_id": {
                    "type": "string",
                    "description": "The script ID (required for get_source, recompile, delete)",
                },
            },
            "required": ["action"],
        },
    },
    # --- Data Operations Tools ---
    {
        "name": "delegate_to_data_ops",
        "description": """Delegate data transformation or analysis to the DataOps specialist agent. Use this when:
- The user wants to compute derived quantities (magnitude, smoothing, resampling, derivatives, etc.)
- The user wants statistical summaries (describe data)

Do NOT delegate:
- Data fetching (use delegate_to_envoy — fetching requires mission-specific knowledge)
- Visualization requests (use the active visualization agent)
- Creating datasets from text/search results (use delegate_to_data_io)
- Dataset search or parameter listing (handle directly or use delegate_to_envoy)
- Data export to CSV — only do this when explicitly requested by the user

Multiple concurrent data_ops delegations are supported — each request goes to a separate agent when the primary is busy.

The DataOps agent can see all data currently in memory via list_fetched_data.""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "What to compute/analyze (e.g., 'compute magnitude of AC_H2_MFI.BGSEc', 'describe ACE_Bmag')",
                },
                "context": {
                    "type": "string",
                    "description": "Optional: available labels, prior results, or other context",
                },
                "wait": {
                    "type": "boolean",
                    "description": "If true, wait for result. If false, fire-and-forget (returns null immediately). Default true.",
                    "default": True
                },
            },
            "required": ["request"],
        },
    },
    {
        "name": "load_file",
        "description": """Load a local data file (CSV, JSON, Parquet, Excel) into the data store.

Use this when:
- The user provides a file path to tabular data
- A collaborator sent a data file that needs to be loaded for analysis
- Importing external datasets to combine with archive data

Supports: CSV, TSV, JSON (records or table format), Parquet, Excel (.xlsx/.xls).
Auto-detects datetime index columns for timeseries data.""",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the data file",
                },
                "output_label": {
                    "type": "string",
                    "description": "Label for the loaded dataset (e.g., 'psp_fitted_velocities_e22')",
                },
                "time_column": {
                    "type": "string",
                    "description": "Column name to use as datetime index. Auto-detected if omitted.",
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of the data",
                },
                "units": {
                    "type": "string",
                    "description": "Units string (e.g., 'km/s', 'nT')",
                },
            },
            "required": ["file_path", "output_label"],
        },
    },
    {
        "name": "delegate_to_data_io",
        "description": """Delegate data I/O tasks to the DataIO specialist agent. Use this when:
- The user wants to load a local file (CSV, JSON, Parquet, Excel) into the data store
- The user wants to turn unstructured text into a plottable dataset (event lists, search results, catalogs)
- The user wants to extract data tables from a document (PDF or image)
- You have Google Search results with dates and values that should become a DataFrame
- The user says "load this file", "create a dataset from..." or "make a timeline of..."

Do NOT delegate:
- Data fetching from CDAWeb (use delegate_to_envoy)
- Data transformations on existing in-memory data (use delegate_to_data_ops)
- Visualization requests (use the active visualization agent)

The DataIO agent can load files (load_file), read documents (read_document), create DataFrames (store_dataframe), and see what data is in memory (list_fetched_data).""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "What to load/extract and store (e.g., 'Load /data/psp_fits.csv as psp_fitted_velocities', 'Create a DataFrame from these X-class flares: [dates and values]. Label it xclass_flares_2024.')",
                },
                "context": {
                    "type": "string",
                    "description": "Optional: source text, search results, or file path to extract data from",
                },
                "wait": {
                    "type": "boolean",
                    "description": "If true, wait for result. If false, fire-and-forget (returns null immediately). Default true.",
                    "default": True
                },
            },
            "required": ["request"],
        },
    },
    # --- SPICE Ephemeris Tools ---
    # NOTE: SPICE tools are discovered dynamically from the heliospice MCP
    # server at startup and added to both orchestrator and envoy call sets.
    {
        "name": "delegate_to_insight",
        "description": """Delegate a plot analysis request to the Insight specialist. The Insight agent receives a high-resolution PNG of the current figure along with data context (labels, units, time ranges).

Requires an active plot. If the plot is restorable (resumed session), call restore_plot first.

The `request` parameter controls what the Insight agent focuses on — phrase it as a specific question or task:
- Scientific interpretation: "Analyze this figure and identify solar wind features and ICME signatures"
- Quality check: "Check this figure for issues — wrong labels, missing data, artifacts, scale problems"
- Specific question: "What's happening around January 15 in the magnetic field panel?"
- Data validation: "Do the value ranges look physically reasonable? Are there suspicious gaps or spikes?"

Do NOT use for plot modifications (zoom, restyle, add traces) — use the active visualization agent for those.""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "Specific question or task for the Insight agent — be precise about what to analyze, check, or interpret",
                },
                "context": {
                    "type": "string",
                    "description": "Optional additional context about what was plotted or what to focus on",
                },
                "wait": {
                    "type": "boolean",
                    "description": "If true, wait for result. If false, fire-and-forget (returns null immediately). Default true.",
                    "default": True
                },
            },
            "required": ["request"],
        },
    },
    # --- request_planning ---
    {
        "name": "request_planning",
        "description": """Research data availability and produce a structured plan for data requests.

The planner searches datasets, verifies time coverage, and returns a plan with tasks.
YOU must then execute the plan by calling delegation tools for each task:
- Fetch tasks (mission ID like PSP, ACE) → delegate_to_envoy
- Compute tasks (__data_ops__) → delegate_to_data_ops
- Visualization tasks (__visualization__) → delegate_to_viz

Execute fetch tasks IN PARALLEL (multiple delegate_to_envoy calls in one response).
Execute compute and viz tasks AFTER fetches complete.

Use this as your FIRST action for any request involving fetching mission data.
Skip this only for: answering questions, modifying existing figures, follow-up
operations on already-loaded data.""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The full user request to plan",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this request needs planning",
                },
                "time_start": {
                    "type": "string",
                    "description": "Start time in ISO 8601 format. Resolve relative expressions yourself.",
                },
                "time_end": {
                    "type": "string",
                    "description": "End time in ISO 8601 format. Resolve relative expressions yourself.",
                },
            },
            "required": ["request", "reasoning", "time_start", "time_end"],
        },
    },
    # ── Pipeline tool (consolidated) ─────────────────────────────────────
    {
        "name": "pipeline",
        "description": """Unified tool for pipeline operations: inspection, modification, execution, and saved pipelines.

Actions:
- "info": Inspect the pipeline DAG. Use node_id for full detail, list_library=true for ops library.
- "modify": Modify the pipeline DAG (update_params, remove, insert_after, apply_library_op, save_to_library). Uses sub_action parameter.
- "execute": Re-execute stale/pending nodes. Use use_cache to control data fetching.
- "save": Save the session pipeline as a reusable saved pipeline.
- "run": Run a saved pipeline with a new time range, or list available pipelines.
- "search": Search saved pipelines by query, mission, dataset, or tags.

Use this when:
- User asks about the pipeline, workflow, or what operations were performed
- User wants to modify, execute, save, run, or search pipelines""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["info", "modify", "execute", "save", "run", "search"],
                    "description": "The pipeline action to perform",
                },
                # For action="info"
                "node_id": {
                    "type": "string",
                    "description": "For info action: show full detail for this node.",
                },
                "list_library": {
                    "type": "boolean",
                    "description": "For info action: if true, list saved operations from the reusable ops library.",
                },
                # For action="modify"
                "sub_action": {
                    "type": "string",
                    "enum": [
                        "update_params",
                        "remove",
                        "insert_after",
                        "apply_library_op",
                        "save_to_library",
                    ],
                    "description": "For modify action: the mutation to perform.",
                },
                "params": {
                    "type": "object",
                    "description": "For modify action (update_params): new parameter values to merge. For insert_after: parameters for the new node.",
                },
                "after_id": {
                    "type": "string",
                    "description": "For modify action (insert_after): the node ID to insert after.",
                },
                "tool": {
                    "type": "string",
                    "description": "For modify action (insert_after): the tool type of the new node.",
                },
                "output_label": {
                    "type": "string",
                    "description": "For modify action (insert_after): the output label for the new node.",
                },
                "library_entry_id": {
                    "type": "string",
                    "description": "For modify action (apply_library_op): the library entry ID to apply.",
                },
                # For action="execute"
                "use_cache": {
                    "type": "boolean",
                    "description": "For execute action: if true (default), use cached data. If false, re-fetch from remote.",
                },
                # For action="save"
                "name": {
                    "type": "string",
                    "description": "For save action: human-readable name for the pipeline.",
                },
                "description": {
                    "type": "string",
                    "description": "For save action: description of what the pipeline produces.",
                },
                "render_op_id": {
                    "type": "string",
                    "description": "For save action: optional render op ID to extract pipeline from.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "For save action: optional tags for categorization.",
                },
                # For action="run"
                "pipeline_id": {
                    "type": "string",
                    "description": "For run action: pipeline ID to execute.",
                },
                "time_start": {
                    "type": "string",
                    "description": "For run action: start time in ISO 8601 format.",
                },
                "time_end": {
                    "type": "string",
                    "description": "For run action: end time in ISO 8601 format.",
                },
                "list_pipelines": {
                    "type": "boolean",
                    "description": "For run action: if true, list all available pipelines.",
                },
                # For action="search"
                "query": {
                    "type": "string",
                    "description": "For search action: natural language search query.",
                },
                "mission": {
                    "type": "string",
                    "description": "For search action: optional mission filter.",
                },
                "dataset": {
                    "type": "string",
                    "description": "For search action: optional dataset substring filter.",
                },
                "limit": {
                    "type": "integer",
                    "description": "For search action: max results (default 10).",
                },
            },
            "required": ["action"],
        },
    },
    # --- Event feed (pull-based session context) ---
    {
        "name": "events",
        "description": """Unified tool for session event operations.

Actions:
- "check": Check for session events since your last check. Returns summaries of what happened — data fetched, computations, plots, errors, delegations. Call at the start of your work to see prior session context. First call returns ALL relevant events; subsequent calls return only new ones (no duplicates).
- "details": Get full details for specific events by ID. Use after checking events when you need exact tool arguments, result data, or error details.""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["check", "details"],
                    "description": "The event action to perform",
                },
                # For action="check"
                "max_events": {
                    "type": "integer",
                    "description": "For check action: max events to return (default 50, max 200).",
                },
                "event_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "For check action: optional filter for event types (data_fetched, data_computed, render_executed, tool_error, fetch_error, delegation, delegation_done, user_message, agent_response, sub_agent_tool, sub_agent_error).",
                },
                # For action="details"
                "event_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "For details action: event IDs to get details for (e.g., ['evt_0012', 'evt_0015']).",
                },
            },
            "required": ["action"],
        },
    },
    # --- Event feed admin (orchestrator/planner only) ---
    {
        "name": "events_admin",
        "description": """Extended event feed with cross-agent visibility.

Actions:
- "check": Same as events — check for session events since your last check. Returns summaries of what happened (data fetched, computations, plots, errors, delegations). First call returns ALL relevant events; subsequent calls return only new ones.
- "details": Same as events — get full details for specific events by ID.
- "peek": See a sub-agent's internal events (tool calls, errors, thinking) that are invisible to your normal feed. Use when a delegation failed and you need to understand why. Stateless — does NOT advance your cursor or interfere with check.
- "peek_details": Get full details for events found via peek, by ID.""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["check", "details", "peek", "peek_details"],
                    "description": "The event action to perform",
                },
                # For action="check"
                "max_events": {
                    "type": "integer",
                    "description": "For check/peek: max events to return (default 50, max 200).",
                },
                "event_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "For check/peek: optional filter for event types (data_fetched, data_computed, render_executed, tool_error, fetch_error, delegation, delegation_done, user_message, agent_response, sub_agent_tool, sub_agent_error, tool_call, tool_result, thinking, high_nan, recovery, debug).",
                },
                # For action="details" / "peek_details"
                "event_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "For details/peek_details: event IDs to get details for (e.g., ['evt_0012', 'evt_0015']).",
                },
                # For action="peek"
                "agent": {
                    "type": "string",
                    "description": "For peek: agent name filter (viz_plotly, dataops, envoy:PSP, envoy, planner, etc.). Prefix match for envoy agents — 'envoy' matches 'envoy:PSP', 'envoy:ACE'.",
                },
                "since_seconds": {
                    "type": "number",
                    "description": "For peek: only return events from the last N seconds.",
                },
            },
            "required": ["action"],
        },
    },
    # --- Control center (turnless orchestrator) ---
    {
        "name": "list_active_work",
        "description": (
            "List all currently running background work units (sub-agent delegations, "
            "planner tasks, etc.). Returns: id, kind, agent_type, agent_name, task_summary, "
            "request (the original prompt), elapsed time, and started_at timestamp. "
            "Use this to understand what is in flight before deciding "
            "whether to cancel, wait, or launch new work."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "cancel_work",
        "description": (
            "Cancel one or more running work units. Provide either a specific unit_id "
            "(from list_active_work), an agent_type to cancel all of that type, or set "
            "cancel_all to true to cancel everything. Cancelled work stops as soon as "
            "possible (after the current atomic operation completes)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "unit_id": {
                    "type": "string",
                    "description": "Specific work unit ID to cancel (from list_active_work).",
                },
                "agent_type": {
                    "type": "string",
                    "description": (
                        "Cancel all units of this agent type. "
                        "One of: mission, data_ops, data_extraction, viz, planner."
                    ),
                },
                "cancel_all": {
                    "type": "boolean",
                    "description": "If true, cancel ALL running work units.",
                },
            },
            "required": [],
        },
    },
]


def _build_full_tool_reference() -> str:
    """Build a formatted reference of all tools with full descriptions."""
    lines = []
    for t in TOOLS:
        lines.append(f"### {t['name']}")
        lines.append(t["description"])
        lines.append("")
    return "\n".join(lines)


FULL_TOOL_REFERENCE = _build_full_tool_reference()


COMMENTARY_PROPERTY = {
    "commentary": {
        "type": "string",
        "description": (
            "Brief active-voice sentence describing what you are doing and why. "
            "One sentence preferred, two max. Shown to the user in the chat. "
            "Examples: 'Searching for ACE magnetic field datasets', "
            "'Computing the magnitude of the magnetic field vector', "
            "'Plotting proton density and speed on a two-panel figure'"
        ),
    }
}


def _inject_commentary(schema: dict) -> dict:
    """Inject the commentary property into a tool schema.

    Returns a shallow-copied schema with commentary added as a required
    string parameter.
    """
    schema = dict(schema)  # shallow copy
    params = dict(schema["parameters"])
    props = dict(params.get("properties", {}))
    props.update(COMMENTARY_PROPERTY)
    params["properties"] = props
    req = list(params.get("required", []))
    if "commentary" not in req:
        req.append("commentary")
    params["required"] = req
    schema["parameters"] = params
    return schema


def _inject_viz_backend_default(schema: dict) -> dict:
    """Set the default value on delegate_to_viz's backend parameter."""
    if schema.get("name") != "delegate_to_viz":
        return schema
    backend = config.PREFER_VIZ_BACKEND
    schema = dict(schema)  # shallow copy
    params = dict(schema["parameters"])
    props = dict(params.get("properties", {}))
    if "backend" in props:
        props["backend"] = dict(props["backend"], default=backend)
    params["properties"] = props
    schema["parameters"] = params
    return schema


def get_tool_schemas(names: list[str] | None = None) -> list[dict]:
    """Return tool schemas for LLM function calling.

    Every schema is augmented with a required ``commentary`` parameter
    that the LLM must fill with a brief active-voice sentence describing
    what it is doing.  The commentary is popped before tool execution and
    emitted as a TEXT_DELTA event so the user sees it in the chat stream.

    Args:
        names: Optional list of tool names to include.
            If None, returns all tools.

    Returns:
        List of tool schema dicts.
    """
    base = TOOLS if names is None else [t for t in TOOLS if t["name"] in set(names)]
    base = [_inject_viz_backend_default(t) for t in base]
    return [_inject_commentary(t) for t in base]


def get_function_schemas(names: list[str] | None = None) -> "list[FunctionSchema]":
    """Return tool schemas as ``FunctionSchema`` objects ready for LLM adapters.

    Convenience wrapper around :func:`get_tool_schemas` that eliminates the
    boilerplate ``FunctionSchema(name=..., description=..., parameters=...)``
    list comprehension repeated across every agent file.

    Args:
        names: Optional list of tool names to include.
            If None, returns all tools.
    """
    from .llm.base import FunctionSchema

    return [
        FunctionSchema(
            name=ts["name"],
            description=ts["description"],
            parameters=ts["parameters"],
        )
        for ts in get_tool_schemas(names=names)
    ]


def register_dynamic_tools(tools: list[dict]) -> None:
    """Register dynamically discovered tools (e.g. from MCP server).

    Appends the tool schemas to TOOLS and rebuilds FULL_TOOL_REFERENCE.
    Skips tools that are already registered (by name).

    Args:
        tools: List of tool schema dicts with 'name', 'description',
               and 'parameters' keys.
    """
    global FULL_TOOL_REFERENCE

    existing = {t["name"] for t in TOOLS}
    added = []
    for t in tools:
        if t["name"] not in existing:
            TOOLS.append(t)
            added.append(t["name"])

    if added:
        FULL_TOOL_REFERENCE = _build_full_tool_reference()
