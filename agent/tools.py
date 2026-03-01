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

Covers both CDAWeb and PDS PPI datasets. Returns matching mission, instrument, and dataset information.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'parker magnetic', 'ACE solar wind', 'omni')"
                }
            },
            "required": ["query"]
        }
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
                    "description": "CDAWeb dataset ID (e.g., 'PSP_FLD_L2_MAG_RTN_1MIN')"
                }
            },
            "required": ["dataset_id"]
        }
    },
    {
        "name": "get_data_availability",
        "description": """Check the available time range for a dataset. Reads from local metadata cache (may download Master CDF if not cached). Use this to:
- Verify data exists for a requested time range before fetching or plotting
- Tell the user how far back data goes or when it was last updated
- Diagnose "no data" errors by checking if the time range is valid

Returns the earliest and latest available dates for the dataset.""",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "CDAWeb dataset ID (e.g., 'AC_H2_MFI', 'PSP_FLD_L2_MAG_RTN_1MIN')"
                }
            },
            "required": ["dataset_id"]
        }
    },
    {
        "name": "browse_datasets",
        "description": """Browse all available science datasets for a mission. Use this when:
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
                    "description": "Mission ID (e.g., 'PSP', 'ACE', 'SolO', 'OMNI', 'WIND', 'DSCOVR', 'MMS', 'STEREO_A')"
                }
            },
            "required": ["mission_id"]
        }
    },
    {
        "name": "list_missions",
        "description": """List all missions with cached metadata. Use this when:
- User asks "what missions are available?"
- You need to see which missions have local data before browsing datasets
- You want a quick overview of the data catalog

Returns mission IDs and dataset counts. No parameters required.""",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_dataset_docs",
        "description": """Fetch CDAWeb documentation for a dataset from the online CDAWeb Notes pages (requires network). Use this when:
- User asks about coordinate systems, calibration, or data quality
- User asks who the PI or data contact is
- User asks what a parameter measures or how it was derived
- You need domain context to interpret or explain data
Fetches documentation from CDAWeb's online Notes pages and REST API. Returns instrument descriptions, variable definitions, coordinate info, and PI contact.""",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "CDAWeb dataset ID (e.g., 'AC_H2_MFI')"
                }
            },
            "required": ["dataset_id"]
        }
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
                    "description": "The clarifying question to ask the user"
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of choices to present (keep to 3-4 options)"
                },
                "context": {
                    "type": "string",
                    "description": "Brief explanation of why you need this information"
                }
            },
            "required": ["question"]
        }
    },

    # --- Data Operations Tools ---
    {
        "name": "fetch_data",
        "description": """Download and load timeseries data from CDAWeb or PDS archives into memory.
Downloads CDF files from CDAWeb or PDS PPI archives for the requested time range (requires network), then stores the data in memory for Python-side operations (magnitude, averages, differences, etc.) or plotting.
Supports CDAWeb datasets (e.g., 'AC_H2_MFI') and PDS PPI datasets (URN IDs like 'urn:nasa:pds:cassini-mag-cal:data-1sec-krtp').

The data is stored in memory with a label like 'AC_H2_MFI.BGSEc' for later reference by compute and plot tools.""",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "Dataset ID — CDAWeb (e.g., 'AC_H2_MFI') or PDS URN (e.g., 'urn:nasa:pds:cassini-mag-cal:data-1sec-krtp')"
                },
                "parameter_id": {
                    "type": "string",
                    "description": "Parameter name (e.g., 'BGSEc', 'Magnitude', 'BR')"
                },
                "time_start": {
                    "type": "string",
                    "description": "Start time in ISO 8601 format (e.g., '2024-01-15' or '2024-01-15T06:00:00'). Resolve relative expressions like 'last week' to actual dates yourself using today's date."
                },
                "time_end": {
                    "type": "string",
                    "description": "End time in ISO 8601 format (e.g., '2024-01-20' or '2024-01-15T18:00:00'). Resolve relative expressions like 'last week' to actual dates yourself using today's date."
                },
                "force_large_download": {
                    "type": "boolean",
                    "description": "Set to true to override the 1 GB download safety limit. Only use when the user explicitly confirms a large download."
                }
            },
            "required": ["dataset_id", "parameter_id", "time_start", "time_end"]
        }
    },
    {
        "name": "list_fetched_data",
        "description": "Show all timeseries currently held in memory. Returns labels, shapes, units, and time ranges.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "custom_operation",
        "description": """Apply a pandas/numpy/xarray/scipy/pywt operation to in-memory data. This is the universal compute tool — use it for ALL data transformations after fetching data with fetch_data.

The code must:
- Assign the result to `result` (DataFrame/Series with DatetimeIndex, or xarray DataArray with 'time' dim)
- Use only sandbox variables, `pd` (pandas), `np` (numpy), `xr` (xarray), `scipy` (full scipy), and `pywt` (PyWavelets) — no imports, no file I/O

Each source label becomes a named variable in the sandbox:
- 2D data (DataFrame): `df_<SUFFIX>` where SUFFIX is the part after the last '.' (e.g., 'DATASET.BR' → df_BR)
- 3D+ data (xarray DataArray): `da_<SUFFIX>` (e.g., 'DATASET.EFLUX_VS_PA_E' → da_EFLUX_VS_PA_E)
- If label has no '.', the full label is used as suffix
- The first DataFrame source is also aliased as `df` for backward compatibility

The result can be a DataFrame (with DatetimeIndex) OR an xarray DataArray (with 'time' dim). DataArray results are stored as-is — useful for intermediate xarray→xarray operations or for 2D spectrograms.

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
                "source_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels of source timeseries in memory. Each becomes a sandbox variable: df_<SUFFIX> (DataFrame) or da_<SUFFIX> (xarray DataArray) where SUFFIX is the part after the last '.'. First DataFrame also available as 'df'. For single-source ops, pass one-element array."
                },
                "code": {
                    "type": "string",
                    "description": "Python code using df/da_ variables, pd (pandas), np (numpy), xr (xarray), scipy (full scipy), pywt (PyWavelets). Must assign to 'result'."
                },
                "output_label": {
                    "type": "string",
                    "description": "Label for the result (e.g., 'B_normalized', 'B_clipped')"
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of the operation"
                },
                "units": {
                    "type": "string",
                    "description": "Physical units of the result (e.g., 'nT', 'km/s', 'nT/s', 'cm^-3'). If omitted, inherits from source. Set explicitly when the operation changes dimensions (e.g., derivative adds '/s', multiply changes units, normalize produces dimensionless '')."
                }
            },
            "required": ["source_labels", "code", "output_label", "description"]
        }
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
                    "description": "Python code using pd (pandas) and np (numpy) that constructs data and assigns to 'result'. Must produce a DataFrame with DatetimeIndex."
                },
                "output_label": {
                    "type": "string",
                    "description": "Label for the stored dataset (e.g., 'xclass_flares_2024', 'cme_catalog')"
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of the dataset"
                },
                "units": {
                    "type": "string",
                    "description": "Optional units for the data columns (e.g., 'W/m²', 'km/s')"
                }
            },
            "required": ["code", "output_label", "description"]
        }
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
                    "description": "Search keyword (e.g., 'bandpass filter', 'spectrogram', 'wavelet', 'interpolate')"
                },
                "package": {
                    "type": "string",
                    "enum": ["scipy.signal", "scipy.fft", "scipy.interpolate", "scipy.stats", "scipy.integrate", "pywt"],
                    "description": "Optional: restrict search to a specific package"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_function_docs",
        "description": """Get the full docstring and signature for a specific function. Use this after search_function_docs to understand function parameters, return values, and usage examples before writing code.""",
        "parameters": {
            "type": "object",
            "properties": {
                "package": {
                    "type": "string",
                    "description": "Package path (e.g., 'scipy.signal', 'pywt')"
                },
                "function_name": {
                    "type": "string",
                    "description": "Function name (e.g., 'butter', 'cwt', 'spectrogram')"
                }
            },
            "required": ["package", "function_name"]
        }
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
                "label": {
                    "type": "string",
                    "description": "Label of the data in memory (e.g., 'AC_H2_MFI.BGSEc')"
                },
                "time_start": {
                    "type": "string",
                    "description": "Optional start time (ISO 8601, no 'Z' suffix — e.g. '2024-01-15T00:00:00') to filter data before computing stats. Omit for full range."
                },
                "time_end": {
                    "type": "string",
                    "description": "Optional end time (ISO 8601, no 'Z' suffix — e.g. '2024-01-20T00:00:00') to filter data before computing stats. Omit for full range."
                }
            },
            "required": ["label"]
        }
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
                "label": {
                    "type": "string",
                    "description": "Label of the data in memory (e.g., 'AC_H2_MFI.BGSEc')"
                },
                "n_rows": {
                    "type": "integer",
                    "description": "Number of rows to show from each end (default: 5, max: 50)"
                },
                "position": {
                    "type": "string",
                    "enum": ["head", "tail", "both", "sampled"],
                    "description": "Which rows to show: 'head' (first N), 'tail' (last N), 'both' (default), or 'sampled' (evenly spaced across full range)"
                },
                "stride": {
                    "type": "integer",
                    "description": "Step size for 'sampled' mode (e.g., 10 = every 10th row). Default: auto (~20 rows). Ignored if position is not 'sampled'."
                },
                "time_start": {
                    "type": "string",
                    "description": "Optional start time (ISO 8601, no 'Z' suffix — e.g. '2024-01-15T00:00:00') to filter data before previewing. Omit for full range."
                },
                "time_end": {
                    "type": "string",
                    "description": "Optional end time (ISO 8601, no 'Z' suffix — e.g. '2024-01-20T00:00:00') to filter data before previewing. Omit for full range."
                }
            },
            "required": ["label"]
        }
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
                "label": {
                    "type": "string",
                    "description": "Label of the data in memory to export"
                },
                "filename": {
                    "type": "string",
                    "description": "Output filename (e.g., 'ace_mag.csv'). '.csv' is appended if missing. Default: auto-generated from label."
                }
            },
            "required": ["label"]
        }
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
                    "description": "Plotly figure dict with 'data' (array of trace stubs with data_label) and 'layout'."
                }
            },
            "required": ["figure_json"]
        }
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
                    "description": "Action to perform"
                },
                "filename": {
                    "type": "string",
                    "description": "Output filename for export action"
                },
                "format": {
                    "type": "string",
                    "enum": ["png", "pdf"],
                    "description": "Export format: 'png' (default) or 'pdf'"
                }
            },
            "required": ["action"]
        }
    },

    # --- Session Assets ---
    {
        "name": "get_session_assets",
        "description": "Get a snapshot of all session assets and their status: "
                       "plot (active/restorable/none), data entries (loaded vs deferred), "
                       "and operation count. Orchestrator and planner see this automatically; "
                       "other agents can call this for on-demand status.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "restore_plot",
        "description": "Restore a deferred plot from a resumed session. "
                       "When session context shows the plot as 'restorable', "
                       "call this before delegate_to_insight or other plot-dependent tools. "
                       "No-op if already active.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
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

Do NOT use this for missions already in the routing table (PSP, ACE, etc.) — use delegate_to_mission instead.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms (mission name, instrument, physical quantity, e.g., 'cluster magnetic field', 'voyager 2', 'proton density')"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default 20)"
                }
            },
            "required": ["query"]
        }
    },

    # --- Web Search (orchestrator + planner only) ---
    {
        "name": "google_search",
        "description": """Search the web using Google Search for real-world context. Use this when:
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
                    "description": "The search query (e.g., 'major solar storms January 2024', 'ICME list 2024', 'X-class flare events')"
                }
            },
            "required": ["query"]
        }
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
                    "description": "Absolute path to the file to read"
                },
                "prompt": {
                    "type": "string",
                    "description": "Optional prompt for targeted extraction (e.g., 'extract the data table', 'list all dates and values'). If not provided, a default extraction prompt is used."
                }
            },
            "required": ["file_path"]
        }
    },

    # --- Memory ---
    {
        "name": "recall_memories",
        "description": """Search or browse archived memories from past sessions. Use when:
- The user references something from a previous session ("last time", "before", "we did X")
- You need context about past analyses, preferences, or lessons learned
- The user asks what they've done before or what data they've looked at

Returns a list of archived memory entries with type, scopes, content, and date.
Uses tag-based search for better relevance than simple keyword matching.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keyword (e.g., 'ACE magnetic', 'smoothing'). Leave empty to list recent entries."
                },
                "type": {
                    "type": "string",
                    "enum": ["preference", "summary", "pitfall", "reflection"],
                    "description": "Optional: filter by memory type"
                },
                "scope": {
                    "type": "string",
                    "description": "Optional: filter by scope — returns memories that include this scope (e.g., 'generic', 'mission:PSP', 'visualization', 'data_ops')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max entries to return (default 20)"
                }
            },
            "required": []
        }
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
                    "description": "The memory ID (from the [ID] prefix in the Operational Knowledge section)"
                },
                "stars": {
                    "type": "integer",
                    "description": "1-5 rating: 5=directly prevented mistake, 4=useful context, 3=relevant but no impact, 2=irrelevant, 1=misleading"
                },
                "comment": {
                    "type": "string",
                    "description": "Structured feedback with ALL four labeled sections on separate lines (never skip any):\n(1) Rating: why this star count\n(2) Criticism: what's wrong or could be better\n(3) Suggestion: how to improve the memory's content or scope\n(4) Comment: any extra observation\nExample:\n(1) Rating: Caught the NaN issue before fetch\n(2) Criticism: Too vague — doesn't say which datasets\n(3) Suggestion: Add dataset IDs (AC_H2_MFI)\n(4) Comment: Would have been useless for a different mission"
                }
            },
            "required": ["memory_id", "stars", "comment"]
        }
    },

    # --- Routing ---
    {
        "name": "delegate_to_mission",
        "description": """Delegate a data request to a mission-specific specialist agent. Use this when:
- The user asks about a specific mission's data (e.g., "show me ACE magnetic field data")
- The user wants to fetch, compute, or describe data from a specific mission
- You need mission-specific knowledge (dataset IDs, parameter names, analysis patterns)

Do NOT delegate:
- Visualization requests (plotting, zoom, render changes) — use the active visualization agent
- Requests to plot already-loaded data — use the active visualization agent
- General questions about capabilities

You can call delegate_to_mission multiple times with the same mission_id in parallel — if the
primary agent is busy, an ephemeral overflow agent handles the request concurrently. However,
combining related requests into one call is often more efficient because the mission agent has
full context of all sub-tasks.

The specialist will search datasets, fetch data, run computations, and report back what was done. You then decide whether to visualize the results.""",
        "parameters": {
            "type": "object",
            "properties": {
                "mission_id": {
                    "type": "string",
                    "description": "Spacecraft mission ID from the supported missions table (e.g., 'PSP', 'ACE', 'SolO', 'OMNI', 'WIND', 'DSCOVR', 'MMS', 'STEREO_A', 'SPICE'). Use 'SPICE' ONLY for ephemeris requests (position, velocity, trajectory, distance, coordinate transforms). SPICE provides NO science data (magnetic field, plasma, particles, etc.) — use the appropriate mission agent for science data."
                },
                "request": {
                    "type": "string",
                    "description": "The data request to send to the specialist (e.g., 'fetch magnetic field data for last week')"
                }
            },
            "required": ["mission_id", "request"]
        }
    },
    {
        "name": "delegate_to_viz_plotly",
        "description": """Delegate a visualization request to the Plotly visualization specialist agent. Use this when:
- The user asks to plot, display, or visualize data
- The user wants to change plot appearance (render type, colors, axis labels, title, log scale)
- The user wants to zoom, set time range, or resize the canvas

Export requests (PNG/PDF) are handled automatically when delegated here — no special handling needed.

Do NOT delegate:
- Data requests (fetch, compute, describe) — use delegate_to_mission
- Dataset search or parameter listing — handle directly

The specialist has access to all visualization methods and can see what data is in memory.""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The visualization request (e.g., 'plot ACE_Bmag and PSP_Bmag together', 'switch to scatter plot', 'set log scale on y-axis')"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context about what data is available or what was just done"
                }
            },
            "required": ["request"]
        }
    },
    {
        "name": "delegate_to_viz_mpl",
        "description": """Delegate a visualization request to the matplotlib visualization specialist agent. Use this when:
- The user asks to plot, display, or visualize data
- The user wants to change plot appearance
- The user wants publication-quality static figures
- The request is for a non-timeseries plot type: histogram, polar, 3D surface, contour, scatter matrix, violin, box plot, pie chart, or plots with insets
- The user wants explicit script control or custom matplotlib code
- Complex multi-axis layouts that don't fit Plotly's subplot model

The MPL agent generates and executes matplotlib Python scripts in a subprocess sandbox.
Scripts can access all data in memory via load_data(label), load_meta(label), and
available_labels() helper functions.""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The visualization request (e.g., 'create a histogram of field magnitudes', 'make a polar plot of solar wind direction')"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context about what data is available or what was just done"
                }
            },
            "required": ["request"]
        }
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
                    "description": "The matplotlib Python script to execute"
                },
                "description": {
                    "type": "string",
                    "description": "Short description of what the plot shows (for the user)"
                }
            },
            "required": ["script"]
        }
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
                    "description": "The action to perform"
                },
                "script_id": {
                    "type": "string",
                    "description": "The script ID (required for get_script, rerun, delete)"
                }
            },
            "required": ["action"]
        }
    },
    # --- JSX/Recharts Visualization Tools ---

    {
        "name": "delegate_to_viz_jsx",
        "description": """Delegate a visualization request to the JSX/Recharts visualization specialist agent. Use this when:
- The user asks to plot, display, or visualize data
- The user wants rich interactive dashboards with multiple linked charts
- The user wants custom React components or Recharts-based visualization
- The user wants multiple linked charts in a single view
- The user wants custom interactive components beyond what Plotly offers

The JSX agent writes React/Recharts components that compile to interactive browser widgets.
Components access data via useData(label) hooks.

Do NOT delegate:
- Data requests (fetch, compute, describe) — use delegate_to_mission
- Dataset search or parameter listing — handle directly""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The visualization request (e.g., 'create a Recharts dashboard of solar wind data', 'build an interactive multi-chart view')"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context about what data is available or what was just done"
                }
            },
            "required": ["request"]
        }
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
                    "description": "The JSX/TSX source code for the React/Recharts component"
                },
                "description": {
                    "type": "string",
                    "description": "Short description of what the component shows (for the user)"
                }
            },
            "required": ["code"]
        }
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
                    "description": "The action to perform"
                },
                "script_id": {
                    "type": "string",
                    "description": "The script ID (required for get_source, recompile, delete)"
                }
            },
            "required": ["action"]
        }
    },

    # --- Data Operations Tools ---

    {
        "name": "delegate_to_data_ops",
        "description": """Delegate data transformation or analysis to the DataOps specialist agent. Use this when:
- The user wants to compute derived quantities (magnitude, smoothing, resampling, derivatives, etc.)
- The user wants statistical summaries (describe data)

Do NOT delegate:
- Data fetching (use delegate_to_mission — fetching requires mission-specific knowledge)
- Visualization requests (use the active visualization agent)
- Creating datasets from text/search results (use delegate_to_data_extraction)
- Dataset search or parameter listing (handle directly or use delegate_to_mission)
- Data export to CSV — only do this when explicitly requested by the user

Multiple concurrent data_ops delegations are supported — each request goes to a separate actor when the primary is busy.

The DataOps agent can see all data currently in memory via list_fetched_data.""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "What to compute/analyze (e.g., 'compute magnitude of AC_H2_MFI.BGSEc', 'describe ACE_Bmag')"
                },
                "context": {
                    "type": "string",
                    "description": "Optional: available labels, prior results, or other context"
                }
            },
            "required": ["request"]
        }
    },
    {
        "name": "delegate_to_data_extraction",
        "description": """Delegate text-to-DataFrame conversion to the DataExtraction specialist agent. Use this when:
- The user wants to turn unstructured text into a plottable dataset (event lists, search results, catalogs)
- The user wants to extract data tables from a document (PDF or image)
- You have Google Search results with dates and values that should become a DataFrame
- The user says "create a dataset from..." or "make a timeline of..."

Do NOT delegate:
- Data fetching from CDAWeb (use delegate_to_mission)
- Data transformations on existing in-memory data (use delegate_to_data_ops)
- Visualization requests (use the active visualization agent)

The DataExtraction agent can read documents (read_document), create DataFrames (store_dataframe), and see what data is in memory (list_fetched_data).""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "What to extract and store (e.g., 'Create a DataFrame from these X-class flares: [dates and values]. Label it xclass_flares_2024.')"
                },
                "context": {
                    "type": "string",
                    "description": "Optional: source text, search results, or file path to extract data from"
                }
            },
            "required": ["request"]
        }
    },

    # --- SPICE Ephemeris Tools ---
    # NOTE: SPICE tools are discovered dynamically from the heliospice MCP
    # server at runtime via register_dynamic_tools(). No hardcoded schemas.

    {

        "name": "delegate_to_insight",
        "description": """Delegate a plot analysis request to the Insight specialist. Use when the user asks to analyze, interpret, or describe the current plot, or asks about features, anomalies, or patterns visible in the data.

Requires an active plot. The Insight agent receives a high-resolution PNG of the current figure along with data context (labels, units, time ranges) and returns a scientific interpretation.

Examples of when to use:
- "What do you see in this plot?"
- "Are there any anomalies?"
- "Describe the features in this data"
- "What's happening around January 15?"
- "Interpret this magnetic field signature"
- "Analyze this plot"

Do NOT use for plot modifications (zoom, restyle, add traces) — use the active visualization agent for those.""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "What to analyze or the user's question about the plot"
                },
                "context": {
                    "type": "string",
                    "description": "Optional additional context about what was plotted or what to focus on"
                }
            },
            "required": ["request"]
        }
    },

    # --- request_planning ---
    {

        "name": "request_planning",
        "description": """Activate the planning system for data requests. Use this as your FIRST action
for any request that involves fetching, computing, or plotting data — even single-mission requests.
The planner resolves time ranges, discovers datasets, and coordinates execution.

Only skip this for genuinely simple requests:
- Answering questions about available data or missions
- Modifying an existing figure (zoom, colors, titles)
- Follow-up operations on already-loaded data""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The full user request to plan and execute"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this request needs multi-step planning"
                },
                "time_start": {
                    "type": "string",
                    "description": "Start time in ISO 8601 format (e.g. '2012-05-01' or '1979-07-01T00:00:00'). Resolve relative expressions to actual dates yourself."
                },
                "time_end": {
                    "type": "string",
                    "description": "End time in ISO 8601 format (e.g. '2013-01-31' or '1979-07-15T00:00:00'). Resolve relative expressions to actual dates yourself."
                }
            },
            "required": ["request", "reasoning", "time_start", "time_end"]
        }
    },

    # ── Pipeline tools ────────────────────────────────────────────────
    {

        "name": "get_pipeline_info",
        "description": """Inspect the current data pipeline DAG.

Modes (pick one):
- No params → compact summary of all nodes, edges, and staleness
- node_id → full detail for one node (complete code, description, params, parents/children, ops library match)
- list_library=true → list saved operations from the reusable ops library

Use this when:
- User asks about the pipeline, workflow, or what operations were performed
- You need to see the full code of a compute node
- You want to browse the ops library for reusable operations""",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Show full detail for this node (complete code, params, connections, library match)."
                },
                "list_library": {
                    "type": "boolean",
                    "description": "If true, list saved operations from the reusable ops library."
                }
            },
            "required": []
        }
    },
    {

        "name": "modify_pipeline_node",
        "description": """Modify the pipeline DAG: update parameters, remove a node, insert a new node, or integrate with the ops library.

Actions:
- "update_params": Change a node's parameters (e.g., time range, code). Marks the node and all downstream nodes stale.
- "remove": Remove a node from the DAG. Reports orphaned output labels.
- "insert_after": Insert a new node after an existing one, rewiring downstream consumers.
- "apply_library_op": Replace a compute node's code with a saved library entry. Requires node_id + library_entry_id. Marks stale.
- "save_to_library": Save a compute node's code to the ops library for reuse. Requires node_id.

After modification, stale nodes can be re-executed with execute_pipeline.""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["update_params", "remove", "insert_after", "apply_library_op", "save_to_library"],
                    "description": "The mutation action to perform"
                },
                "node_id": {
                    "type": "string",
                    "description": "The node ID to modify (e.g., 'op_001'). Required for update_params, remove, apply_library_op, save_to_library."
                },
                "params": {
                    "type": "object",
                    "description": "For update_params: new parameter values to merge. For insert_after: parameters for the new node."
                },
                "after_id": {
                    "type": "string",
                    "description": "For insert_after: the node ID to insert after."
                },
                "tool": {
                    "type": "string",
                    "description": "For insert_after: the tool type of the new node (e.g., 'custom_operation')."
                },
                "output_label": {
                    "type": "string",
                    "description": "For insert_after: the output label for the new node."
                },
                "library_entry_id": {
                    "type": "string",
                    "description": "For apply_library_op: the 8-char hex ID of the library entry to apply."
                }
            },
            "required": ["action"]
        }
    },
    {

        "name": "execute_pipeline",
        "description": """Re-execute stale and pending nodes in the pipeline DAG.

Only runs nodes that have been marked stale (by modify_pipeline_node) or pending (newly inserted). Uses backdating: if a node's output is unchanged after re-execution, all its downstream nodes are skipped.

Use this after modify_pipeline_node to apply changes. The pipeline updates the DataStore with new data and can update the plot.

Set use_cache=true (default) to use cached fetch data when available, or false to re-fetch from remote servers.""",
        "parameters": {
            "type": "object",
            "properties": {
                "use_cache": {
                    "type": "boolean",
                    "description": "If true (default), use cached data for fetch operations. If false, re-fetch from remote servers."
                }
            },
            "required": []
        }
    },

    # ── Saved Pipeline tools ─────────────────────────────────────────
    {

        "name": "save_pipeline",
        "description": """Save the current session's pipeline as a reusable saved pipeline.

Extracts the data pipeline (fetch → compute → plot) from the current session and saves it so it can be re-run with any time range — no LLM needed.

Saved pipelines are tied to specific datasets (e.g., AC_H2_MFI.BGSEc) and only parameterize the time range. Use this when:
- The user says "save this workflow" or "save this pipeline"
- The user wants to reuse the current analysis for different time periods
- After completing a multi-step analysis that could be replayed

If render_op_id is provided, only the pipeline for that specific render is extracted. Otherwise, the full session pipeline is used.""",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Human-readable name for the pipeline (e.g., 'ACE Solar Wind Overview')"
                },
                "description": {
                    "type": "string",
                    "description": "Description of what the pipeline produces"
                },
                "render_op_id": {
                    "type": "string",
                    "description": "Optional: extract only the pipeline for this specific render operation (e.g., 'op_005'). If omitted, uses the full pipeline."
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorization (e.g., ['ace', 'solar-wind', 'magnetic-field'])"
                }
            },
            "required": ["name"]
        }
    },
    {

        "name": "run_pipeline",
        "description": """Run a saved pipeline with a new time range, or list available pipelines.

Modes:
- list_pipelines=true → returns all saved pipelines with names, datasets, and step counts
- pipeline_id + time_start + time_end → executes the pipeline for the given time range

Saved pipelines replay the exact same fetch/compute/plot workflow as the original session, just with different dates. No LLM calls needed — instant replay.

Use this when:
- The user says "run the pipeline" or "apply the pipeline to January 2025"
- The user asks "what pipelines do I have?"
- The user wants to reproduce an analysis for a different time window""",
        "parameters": {
            "type": "object",
            "properties": {
                "pipeline_id": {
                    "type": "string",
                    "description": "Pipeline ID to execute (e.g., 'pl_a1b2c3d4')"
                },
                "time_start": {
                    "type": "string",
                    "description": "Start time in ISO 8601 format (e.g., '2024-01-01')"
                },
                "time_end": {
                    "type": "string",
                    "description": "End time in ISO 8601 format (e.g., '2024-01-07')"
                },
                "list_pipelines": {
                    "type": "boolean",
                    "description": "If true, list all available saved pipelines instead of executing one"
                }
            },
            "required": []
        }
    },

    {

        "name": "search_pipelines",
        "description": """Search saved pipelines by query, dataset, mission, or tags.

Use this to find reusable workflows that can be replayed with run_pipeline.

Returns matching pipelines with names, datasets, missions, step counts, and tags.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query (e.g., 'ACE solar wind overview', 'magnetic field comparison')"
                },
                "mission": {
                    "type": "string",
                    "description": "Optional mission filter (e.g., 'ACE', 'PSP')"
                },
                "dataset": {
                    "type": "string",
                    "description": "Optional dataset substring filter (e.g., 'AC_H2_MFI')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 10)"
                }
            },
            "required": []
        }
    },

    # --- Event feed (pull-based session context) ---
    {
        "name": "check_events",
        "description": (
            "Check for session events since your last check. Returns summaries of "
            "what happened — data fetched, computations, plots, errors, delegations. "
            "Call at the start of your work to see prior session context. "
            "First call returns ALL relevant events; subsequent calls return only new ones (no duplicates).\n"
            "If the response exceeds the token quota, you'll get a warning with "
            "quota_exceeded=true. Re-call with compact=true to get an LLM-compacted summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "max_events": {
                    "type": "integer",
                    "description": "Max events to return (default 50, max 200).",
                },
                "event_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional filter: only return these event types. "
                        "Common types: data_fetched, data_computed, render_executed, "
                        "tool_error, fetch_error, delegation, delegation_done, "
                        "user_message, agent_response, sub_agent_tool, sub_agent_error. "
                        "If omitted, all events tagged for your context are returned."
                    ),
                },
                "compact": {
                    "type": "boolean",
                    "description": (
                        "If true, compact the event history via LLM summarization when "
                        "it exceeds the token quota. Use this after receiving a "
                        "quota_exceeded warning to get a shorter summary."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_event_details",
        "description": (
            "Get full details for specific events by ID. Use after check_events "
            "when you need exact tool arguments, result data, or error details.\n"
            "If the response exceeds the token quota, you'll get a warning with "
            "quota_exceeded=true. Re-call with compact=true to get an LLM-compacted summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "event_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Event IDs from check_events (e.g., ['evt_0012', 'evt_0015'])",
                },
                "compact": {
                    "type": "boolean",
                    "description": (
                        "If true, compact the event details via LLM summarization when "
                        "it exceeds the token quota. Use this after receiving a "
                        "quota_exceeded warning."
                    ),
                },
            },
            "required": ["event_ids"],
        },
    },

    # --- Self-curation: manage tool log subscriptions ---
    {
        "name": "manage_tool_logs",
        "description": (
            "View, subscribe to, or unsubscribe from other tools' execution logs. "
            "Three actions:\n"
            "- view: Retrieve recent logs for specified tools (on-demand, no permanent change)\n"
            "- add: Subscribe to a tool's logs so they automatically appear in your session history going forward\n"
            "- drop: Unsubscribe from a tool's logs (cannot drop tools you can call)\n"
            "You MUST provide reasoning for every action."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["view", "add", "drop"],
                    "description": "view = fetch recent logs; add = subscribe to future logs; drop = unsubscribe",
                },
                "tool_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tool name(s) to act on. For add/drop, typically one tool. For view, can be multiple.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Mandatory explanation for why you're performing this action",
                },
                "max_events": {
                    "type": "integer",
                    "description": "For action=view only: maximum number of events to return (default 20)",
                },
            },
            "required": ["action", "tool_names", "reasoning"],
        },
    },

    # --- Control center (turnless orchestrator) ---
    {
        "name": "list_active_work",
        "description": (
            "List all currently running background work units (sub-agent delegations, "
            "planner tasks, etc.). Returns their IDs, agent types, task summaries, and "
            "elapsed time. Use this to understand what is in flight before deciding "
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

    # ── Meta-tools (tool store: browse + load) ──
    {
        "name": "browse_tools",
        "description": (
            "List available tool categories and what each tool does. "
            "Each tool is annotated with your access level: "
            "\"call\" (you can invoke it), \"informed\" (you see its event logs), "
            "or \"available\" (exists in the system). "
            "Call this first to discover all available tools, then use "
            "load_tools to activate what you need."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Optional: filter to a specific category "
                        "(e.g., 'delegation', 'discovery', 'data_ops', "
                        "'visualization', 'pipeline', 'spice')"
                    ),
                },
            },
        },
    },
    {
        "name": "load_tools",
        "description": (
            "Load tools into the active session by category or name. "
            "Only tools marked \"call\" in browse_tools can be loaded. "
            "Loaded tools persist for the entire request — no need to "
            "reload. You can load more tools later if needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Category names (e.g., 'delegation', 'discovery') "
                        "or individual tool names (e.g., 'fetch_data'). "
                        "Categories load all tools in that group."
                    ),
                },
            },
            "required": ["tools"],
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


# ---------------------------------------------------------------------------
# Dynamic SPICE tool registration (populated from MCP discovery)
# ---------------------------------------------------------------------------

_spice_tool_names: set[str] = set()


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
            _spice_tool_names.add(t["name"])
            added.append(t["name"])

    if added:
        FULL_TOOL_REFERENCE = _build_full_tool_reference()


def get_spice_tool_names() -> set[str]:
    """Return the set of tool names that were registered from the MCP server."""
    return set(_spice_tool_names)
