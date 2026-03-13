"""
Tool definitions for Gemini function calling.

Each tool schema defines what the LLM can call and what parameters it needs.
Tools are executed by the agent core based on LLM decisions.

Tool access per agent is controlled by explicit name lists in agent_registry.py.
"""

import config

TOOLS = [
    {
        "name": "xhelio__envoy_query",
        "description": """Query envoy capabilities. Navigate the envoy tree layer by layer, or search across envoys using regex.

Without arguments: lists all available envoys with summaries.
With envoy only: shows that envoy's top-level structure (instruments, functions, etc.).
With envoy + path: drills into the envoy's JSON tree using dot-separated path (e.g., "instruments.FIELDS/MAG.datasets.PSP_FLD_L2_MAG_RTN_1MIN").
With search: finds matching entries by regex across all string values in envoy trees.

Examples:
  envoy_query()                                           → list all envoys
  envoy_query(envoy="MY_ENVOY")                           → envoy details
  envoy_query(search="(?i)trajectory")                    → find entries matching regex""",
        "parameters": {
            "type": "object",
            "properties": {
                "envoy": {
                    "type": "string",
                    "description": "Envoy ID. Omit to list all envoys.",
                },
                "path": {
                    "type": "string",
                    "description": "Dot-separated path into the envoy's JSON tree. Mirrors the JSON structure exactly. Use envoy_query(envoy=X) first to see available paths.",
                },
                "search": {
                    "type": "string",
                    "description": "Regex pattern to search across all string values. Returns matching envoy + path pairs. Can combine with envoy to scope the search.",
                },
            },
            "required": [],
        },
        "category": "discovery",
    },
    {
        "name": "xhelio__manage_envoy",
        "description": """Create or remove envoy kinds at runtime.

Use 'create' to generate a new envoy kind from a Python package or MCP server.
Use 'remove' to delete a runtime-created envoy kind.

The orchestrator should research the package first (via run_code), discuss tool
design with the user, then call this with finalized tool definitions.""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "remove"],
                    "description": "Action to perform",
                },
                "kind": {
                    "type": "string",
                    "description": "Kind name (lowercase, e.g., 'pfss')",
                },
                "envoy_id": {
                    "type": "string",
                    "description": "Envoy ID (uppercase, e.g., 'PFSS')",
                },
                "source": {
                    "type": "string",
                    "description": "Package name or MCP server name (for create)",
                },
                "source_type": {
                    "type": "string",
                    "enum": ["package", "mcp"],
                    "description": "Source type (for create)",
                },
                "tools": {
                    "type": "array",
                    "description": "Tool definitions (for create). Each item: {name, description, parameters, handler_code}. "
                        "handler_code is a Python snippet defining a function with the SAME name as 'name' "
                        "(e.g. name='get_today' → 'def get_today(...):'). The function takes keyword args matching the parameters schema.",
                    "items": {"type": "object"},
                },
            },
            "required": ["action", "kind", "envoy_id"],
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
    {
        "name": "xhelio__ask_user_permission",
        "description": """Ask the user for explicit permission before taking a potentially dangerous or irreversible action.
This tool BLOCKS until the user responds with approve or deny.

Use this before:
- Installing Python packages (pip install)
- Modifying the sandbox configuration (adding packages to the computation environment)
- Any action that writes to disk or modifies system state beyond normal session data

Present clear descriptions of what will happen and why. The user sees the exact command.""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Short action name (e.g., 'install_package', 'modify_sandbox')",
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of what will happen and why",
                },
                "command": {
                    "type": "string",
                    "description": "The exact command or operation to be executed (shown to user for review)",
                },
            },
            "required": ["action", "description", "command"],
        },
    },
    {
        "name": "xhelio__manage_sandbox_packages",
        "description": """Manage packages in the computation sandbox.

- action="list": Show all currently available packages in the sandbox.
- action="install": Install a new package via pip, verify the import, and register it in the sandbox. Requires pip_name, import_path, sandbox_alias, and description.
- action="add": Add an already-installed package to the sandbox. Requires import_path, sandbox_alias, and description.

BEFORE calling action="install":
1. Use web_search to research the correct pip package name and import path
2. Verify the package name, import path, and sandbox alias

After successful install/add, the package becomes available in run_code.""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "install", "add"],
                    "description": "Action to perform",
                },
                "pip_name": {
                    "type": "string",
                    "description": "Package name for pip install (action='install' only, e.g., 'scikit-learn')",
                },
                "import_path": {
                    "type": "string",
                    "description": "Python import path (action='install' and 'add', e.g., 'sklearn')",
                },
                "sandbox_alias": {
                    "type": "string",
                    "description": "Alias in the sandbox namespace (action='install' and 'add', e.g., 'sklearn')",
                },
                "description": {
                    "type": "string",
                    "description": "What this package does and why it is needed (action='install' and 'add')",
                },
                "catalog_submodules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Submodules to catalog for function search (e.g., ['sklearn.decomposition'])",
                },
            },
            "required": ["action"],
        },
    },
    # --- Data Operations Tools ---
    {
        "name": "xhelio__assets",
        "description": "Read-only overview of all session assets with enriched metadata. "
        "Returns summary counts, data shapes, file sizes, figure types, and lineage. "
        "For mutations use manage_data, manage_files, or manage_figure.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list"],
                    "description": "Action to perform. Only 'list' is supported.",
                },
                "kind": {
                    "type": "string",
                    "enum": ["data", "file", "figure"],
                    "description": "Filter by asset kind. Omit for all.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "plan",
        "description": """Manage a multi-step execution plan for complex requests.

Actions:
- "create": Save a new plan. Provide tasks (array), summary, and reasoning.
- "update": Update a step's status. Provide step (0-based index) and status ("completed"/"failed"/"skipped") or note.
- "check": Return the current plan.
- "drop": Discard the current plan.

Use this when a request involves multiple data fetches, computations, or visualizations.
Create the plan first, then execute steps via delegation tools, updating each step as you go.""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "update", "check", "drop"],
                    "description": "The plan action to perform.",
                },
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "instruction": {"type": "string"},
                            "mission": {"type": "string", "description": "Envoy ID, '__visualization__', '__data_ops__', '__data_io__', or null"},
                        },
                        "required": ["description", "instruction"],
                    },
                    "description": "Task list (action='create' only).",
                },
                "summary": {
                    "type": "string",
                    "description": "Brief summary of the plan (action='create' only).",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why this plan was chosen (action='create' only).",
                },
                "step": {
                    "type": "integer",
                    "description": "0-based step index (action='update' only).",
                },
                "status": {
                    "type": "string",
                    "enum": ["completed", "failed", "skipped"],
                    "description": "New status for the step (action='update' only).",
                },
                "note": {
                    "type": "string",
                    "description": "Optional note to attach to the step (action='update' only).",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "xhelio__run_code",
        "description": """Execute Python code in a sandboxed environment. This is the universal compute tool — use it for ALL data transformations, analysis, and computation.

The sandbox directory persists between calls within a session — downloaded files remain available in subsequent calls. Previously staged input files and other artifacts from prior calls may also exist.

The sandbox allows:
- Full imports: pandas, numpy, xarray, scipy, pywt, sklearn, statsmodels, astropy, etc.
- Network access: requests, urllib, http, aiohttp — for downloading remote data files
- File I/O within the sandbox directory (read/write parquet, CSV, JSON, text files)
- Print output (captured and returned)

The sandbox blocks:
- subprocess, ctypes, multiprocessing (no process spawning)
- eval(), exec(), compile() (no dynamic code execution)

**Inputs from the data store:** Labels listed in `inputs` are staged as files in the sandbox:
- DataFrames → `<label>.parquet` (read with `pd.read_parquet('<label>.parquet')`)
- xarray DataArrays → `<label>.nc` (read with `xr.open_dataarray('<label>.nc')`)
- Dicts → `<label>.json`
- Strings → `<label>.txt`
- Bytes → `<label>.bin`
- File assets (IDs starting with `file_`) → staged under their original filename

**Saving to the data store:** Use the `outputs` parameter to map store labels to variable names. After execution, each named variable is read from the namespace and stored. Supports DataFrames (parquet), xarray objects (netCDF), and JSON-serializable types.

Example: `outputs={"Bmag": "magnitude", "Bangle": "angle"}` stores the `magnitude` variable as "Bmag" and `angle` as "Bangle".

**Download → Register → Process workflow:**
1. Use `run_code` to download a file (it persists in the sandbox)
2. Register it via `manage_files(action="register")` for DAG tracking and session replay
3. Use `run_code` with `inputs=["file_<id>"]` to process the registered file

For standalone computations (no input data), omit `inputs`. For fire-and-forget execution (just print output), omit `outputs`.""",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute.",
                },
                "inputs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Store labels/IDs or file asset IDs (prefix 'file_') to stage in the sandbox. Data entries are staged as typed files. File assets are staged under their original filename.",
                },
                "outputs": {
                    "type": "object",
                    "description": "Mapping of {store_label: variable_name}. After execution, each variable_name is read from the namespace and stored under store_label. Supports DataFrames, xarray objects, and JSON-serializable types.",
                    "additionalProperties": {"type": "string"},
                },
                "description": {
                    "type": "string",
                    "description": "What this code does (human-readable).",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds. Default 30. Use 120-300 for network downloads.",
                },
            },
            "required": ["code", "description"],
        },
    },
    # --- Function Documentation Tool ---
    {
        "name": "xhelio__function_docs",
        "description": """Look up scientific computing function documentation.

Actions:
  search — keyword search across the function catalog. Returns function names, sandbox call syntax, and one-line summaries.
  get    — full docstring and signature for a specific function. Use after search to understand parameters, return values, and usage examples before writing code.

Cataloged libraries: scipy.signal, scipy.fft, scipy.interpolate, scipy.stats, scipy.integrate, pywt (PyWavelets).""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "get"],
                    "description": "Action to perform",
                },
                "query": {
                    "type": "string",
                    "description": "(search) Search keyword (e.g., 'bandpass filter', 'spectrogram', 'wavelet', 'interpolate')",
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
                    "description": "(search) Restrict search to a specific package. (get) Package path — required.",
                },
                "function_name": {
                    "type": "string",
                    "description": "(get) Function name (e.g., 'butter', 'cwt', 'spectrogram')",
                },
            },
            "required": ["action"],
        },
    },
    # --- Describe & Export Tools ---
    {
        "name": "xhelio__manage_data",
        "description": """Inspect, transform, and manage in-memory data.

Actions:
  describe — Statistical summary (min, max, mean, std, percentiles, NaN count, cadence).
             Use when you need to understand data before operations or the user asks about data quality.
  preview  — Show actual values (first/last/sampled rows). Use to diagnose issues or verify computations.
  merge    — Merge multiple time ranges of the same data product into one dataset.
  save     — Export to CSV (DataFrames) or NetCDF (xarray). Only when user explicitly asks.
  delete   — Remove a dataset from memory. Only when user explicitly asks.""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["describe", "preview", "merge", "save", "delete"],
                    "description": "Action to perform.",
                },
                "data_id": {
                    "type": "string",
                    "description": "Data ID (from assets). Required for describe, preview, save, delete.",
                },
                "label": {
                    "type": "string",
                    "description": "Label of the data in memory. For backward compatibility — prefer data_id.",
                },
                "time_start": {
                    "type": "string",
                    "description": "(describe, preview) Optional start time (ISO 8601, no 'Z' suffix) to filter data. Omit for full range.",
                },
                "time_end": {
                    "type": "string",
                    "description": "(describe, preview) Optional end time (ISO 8601, no 'Z' suffix) to filter data. Omit for full range.",
                },
                "n_rows": {
                    "type": "integer",
                    "description": "(preview) Number of rows to show from each end (default: 5, max: 50).",
                },
                "position": {
                    "type": "string",
                    "enum": ["head", "tail", "both", "sampled"],
                    "description": "(preview) Which rows: 'head', 'tail', 'both' (default), or 'sampled' (evenly spaced).",
                },
                "stride": {
                    "type": "integer",
                    "description": "(preview) Step size for 'sampled' mode. Default: auto (~20 rows).",
                },
                "data_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "(merge) List of data IDs to merge. Must all be the same data product.",
                },
                "filename": {
                    "type": "string",
                    "description": "(save) Output filename. Auto-generated from label if omitted.",
                },
            },
            "required": ["action"],
        },
    },
    # --- Visualization ---
    {
        "name": "xhelio__render_plotly_json",
        "description": """Create or update the plot by providing a Plotly figure JSON.

You generate a standard Plotly figure dict with `data` (array of traces) and `layout`.
Instead of providing actual data arrays (x, y, z), put a `data_label` field in each
trace dict. The system resolves each label to real data from memory and fills in x/y/z.

## Trace stubs

Each trace in `data` needs:
- `data_label` (string, required): label of the data in memory (from assets)
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
{"data": [{"type": "scatter", "data_label": "Bmag", "mode": "lines", "line": {"color": "red"}}],
 "layout": {"title": {"text": "Magnetic Field"}, "yaxis": {"title": {"text": "nT"}}}}
```

## Example: two panels

```json
{"data": [
    {"type": "scatter", "data_label": "Bmag", "xaxis": "x", "yaxis": "y"},
    {"type": "scatter", "data_label": "density", "xaxis": "x2", "yaxis": "y2"}
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
        "name": "xhelio__manage_plot",
        "description": """Imperative operations on the current Plotly figure: reset or get state.
Use action parameter to select the operation. For export, use manage_figure(action="export").""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["reset", "get_state"],
                    "description": "Action to perform",
                },
            },
            "required": ["action"],
        },
    },
    # --- Document Reading ---
    {
        "name": "xhelio__read_document",
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
        "name": "xhelio__review_memory",
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
        "description": """Delegate a data request to a specialist envoy agent. Use this when:
- The user needs data or capabilities provided by a registered envoy
- You need envoy-specific knowledge and tools

Envoys are registered dynamically via manage_envoy. If no envoys are available, use manage_envoy to create one first.

Do NOT delegate:
- Visualization requests (plotting, zoom, render changes) — use the active visualization agent
- Requests to plot already-loaded data — use the active visualization agent
- General questions about capabilities

You can call delegate_to_envoy multiple times with the same envoy in parallel — if the
primary agent is busy, an ephemeral overflow agent handles the request concurrently. However,
combining related requests into one call is often more efficient because the envoy agent has
full context of all sub-tasks.

The specialist will search datasets, fetch data, run computations, and report back what was done. You then decide whether to visualize the results.""",
        "parameters": {
            "type": "object",
            "properties": {
                "envoy": {
                    "type": "string",
                    "description": "Envoy ID. Must be a registered envoy.",
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
            "required": ["envoy", "request"],
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
                    "description": "The visualization request (e.g., 'plot distance over time', 'switch to scatter plot', 'set log scale on y-axis', 'create a histogram')",
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
        "name": "xhelio__generate_mpl_script",
        "description": """Generate and execute a matplotlib script to create a visualization.

Write a standard matplotlib script. Inside the script, these are PRE-IMPORTED (do NOT import them):
- `plt` (matplotlib.pyplot), `np` (numpy), `pd` (pandas)

Inside the script, these helper functions are available (NOT separate tools — only usable within this script):
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
df = load_data("AC_H2_MFI.Magnitude")
meta = load_meta("DATASET.Magnitude")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df.iloc[:, 0], linewidth=0.5)
ax.set_xlabel("Time")
ax.set_ylabel(f"Magnitude ({meta.get('units', '')})")
ax.set_title("Magnetic Field Magnitude")
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
        "name": "xhelio__manage_mpl_output",
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
        "name": "xhelio__generate_jsx_component",
        "description": """Generate and compile a React/Recharts JSX component for interactive visualization.

Write a React component using Recharts. Inside the component, the following are available:
- All Recharts components (LineChart, BarChart, AreaChart, ScatterChart, ComposedChart,
  PieChart, RadarChart, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Legend, etc.)
- React hooks (useState, useEffect, useMemo, useCallback, useRef)

Inside the component, these data hooks are available (NOT separate tools — only usable within component code):
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
        "name": "xhelio__manage_jsx_output",
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

The DataOps agent can see all data currently in memory via assets.""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "What to compute/analyze (e.g., 'compute magnitude of DATASET.VEC', 'describe label_name')",
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
        "name": "xhelio__manage_files",
        "description": """Manage file assets: register, list, inspect, or delete.

Actions:
  list — List registered file assets.
  register — Register a file and copy to session storage. Sandbox files are moved; external files are copied. Returns asset_id and session_path.
  info — File metadata: size, type, path, linked data entries.
  delete — Remove file registration and session copy.""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "register", "info", "delete"],
                    "description": "The operation to perform.",
                },
                "asset_id": {
                    "type": "string",
                    "description": "File asset ID (for info, delete).",
                },
                "file_path": {
                    "type": "string",
                    "description": "Absolute file path to register (register only).",
                },
                "name": {
                    "type": "string",
                    "description": "Human-readable name. Defaults to filename.",
                },
                "source_url": {
                    "type": "string",
                    "description": "URL the file was downloaded from (register only). Used for ID generation and provenance.",
                },
            },
            "required": ["action"],
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
- Data fetching via envoys (use delegate_to_envoy)
- Data transformations on existing in-memory data (use delegate_to_data_ops)
- Visualization requests (use the active visualization agent)

The DataIO agent can load files (load_file), read documents (read_document), create DataFrames (run_code with outputs), and see what data is in memory (assets).""",
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
    # ── Pipeline tool (consolidated) ─────────────────────────────────────
    {
        "name": "xhelio__pipeline",
        "description": """Inspect the data pipeline DAG or replay a past session's pipeline.

Actions:
- "info": Inspect the pipeline DAG. Use node_id for a single node's detail, list_library=true for the reusable ops library.
- "replay": Replay a past session's pipeline into the current session. Requires session_id. Optionally specify op_id to replay a subgraph, or time_range to override fetch parameters.

Use this when:
- User asks about the pipeline, workflow, or what operations were performed
- User wants to re-run a pipeline from a previous session""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["info", "replay"],
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
                # For action="replay"
                "session_id": {
                    "type": "string",
                    "description": "For replay action: the session ID whose pipeline to replay.",
                },
                "op_id": {
                    "type": "string",
                    "description": "For replay action: optional node ID to replay only its subgraph. If omitted, replays all leaf pipelines.",
                },
                "time_range": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "description": "Start time in ISO 8601 format."},
                        "end": {"type": "string", "description": "End time in ISO 8601 format."},
                    },
                    "description": "For replay action: optional time range override for fetch nodes.",
                },
            },
            "required": ["action"],
        },
    },
    # --- Event feed (pull-based session context) ---
    {
        "name": "xhelio__events",
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
    # --- Event feed admin (orchestrator only) ---
    {
        "name": "xhelio__events_admin",
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
                    "description": "For peek: agent name filter (viz_plotly, dataops, envoy, etc.). Prefix match for envoy agents — 'envoy' matches all envoy agents.",
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
        "name": "manage_workers",
        "description": (
            "Manage running background work units (sub-agent delegations). "
            "action='list' (default): list all running units with id, kind, agent_type, "
            "agent_name, task_summary, request, elapsed time, and started_at. "
            "action='cancel': cancel work by unit_id, agent_type, or cancel_all."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "cancel"],
                    "description": "Action to perform. Default: 'list'.",
                },
                "unit_id": {
                    "type": "string",
                    "description": "For cancel: specific work unit ID to cancel.",
                },
                "agent_type": {
                    "type": "string",
                    "description": (
                        "For cancel: cancel all units of this agent type. "
                        "One of: mission, data_ops, data_extraction, viz."
                    ),
                },
                "cancel_all": {
                    "type": "boolean",
                    "description": "For cancel: if true, cancel ALL running work units.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "xhelio__manage_figure",
        "description": """Manage figure assets: list, display, save from URL, export, delete, or restore.

Actions:
  list — List registered figures. Optional figure_kind filter.
  show — Re-display a figure in chat by asset_id.
  save_from_url — Download an external image and register as figure asset.
  export — Export a figure to file (PNG/PDF).
  delete — Remove a figure from the registry.
  restore — Restore deferred Plotly figure from resumed session.""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "show", "save_from_url", "export", "delete", "restore"],
                    "description": "The operation to perform.",
                },
                "asset_id": {
                    "type": "string",
                    "description": "Figure asset ID (for show, export, delete).",
                },
                "figure_kind": {
                    "type": "string",
                    "enum": ["plotly", "mpl", "jsx", "image"],
                    "description": "Filter figures by rendering backend (list only).",
                },
                "url": {
                    "type": "string",
                    "description": "Image URL to download (save_from_url only).",
                },
                "name": {
                    "type": "string",
                    "description": "Human-readable name (save_from_url only).",
                },
                "format": {
                    "type": "string",
                    "enum": ["png", "pdf"],
                    "description": "Export format (export only). Default: png.",
                },
            },
            "required": ["action"],
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
            "Examples: 'Searching for magnetic field datasets', "
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


def _inject_envoy_list(schema: dict) -> dict:
    """Inject the current envoy list into delegate_to_envoy's schema."""
    if schema.get("name") != "delegate_to_envoy":
        return schema
    from agent.envoy_kinds.registry import MISSION_KINDS
    envoy_ids = sorted(MISSION_KINDS.keys())

    schema = dict(schema)  # shallow copy
    params = dict(schema["parameters"])
    props = dict(params.get("properties", {}))

    if envoy_ids:
        # Set enum so LLM can only pick valid envoy IDs
        props["envoy"] = dict(props["envoy"], enum=envoy_ids)
        envoy_list = ", ".join(envoy_ids)
        schema["description"] = schema["description"].replace(
            "Envoys are registered dynamically via manage_envoy. "
            "If no envoys are available, use manage_envoy to create one first.",
            f"Available envoys: {envoy_list}",
        )
    # else: keep the "no envoys" guidance as-is

    params["properties"] = props
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
    base = [_inject_envoy_list(t) for t in base]
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



def get_tool_schemas_for_agent(
    names: list[str], agent_ctx: str
) -> list[dict]:
    """Return tool schemas filtered by agent permissions.

    For tools with action-level permissions, the ``enum`` list is filtered
    to only show actions the agent is allowed to use.
    """
    import copy
    from agent.agent_registry import AGENT_PERMISSIONS

    schemas = get_tool_schemas(names=names)
    agent_perms = AGENT_PERMISSIONS.get(agent_ctx, {})
    if not agent_perms:
        return copy.deepcopy(schemas)

    filtered = []
    for schema in schemas:
        tool_name = schema["name"]
        allowed_actions = agent_perms.get(tool_name)
        if allowed_actions is None:
            filtered.append(copy.deepcopy(schema))
            continue

        schema = copy.deepcopy(schema)
        props = schema.get("parameters", {}).get("properties", {})
        action_prop = props.get("action")
        if action_prop and "enum" in action_prop:
            action_prop["enum"] = [
                a for a in action_prop["enum"] if a in allowed_actions
            ]
        filtered.append(schema)
    return filtered


def get_function_schemas_for_agent(
    names: list[str], agent_ctx: str
) -> "list[FunctionSchema]":
    """Return tool schemas as FunctionSchema objects, filtered by agent permissions."""
    from .llm.base import FunctionSchema

    return [
        FunctionSchema(
            name=ts["name"],
            description=ts["description"],
            parameters=ts["parameters"],
        )
        for ts in get_tool_schemas_for_agent(names=names, agent_ctx=agent_ctx)
    ]


# =============================================================================
# Registry protocol adapter
# =============================================================================


class _ToolSchemaRegistryAdapter:
    name = "tools.schemas"
    description = "LLM function-calling JSON schemas for all tools"

    def get(self, key: str):
        for t in TOOLS:
            if t["name"] == key:
                return t
        return None

    def list_all(self) -> dict:
        return {t["name"]: t for t in TOOLS}


TOOL_SCHEMA_REGISTRY = _ToolSchemaRegistryAdapter()
from agent.registry_protocol import register_registry  # noqa: E402
register_registry(TOOL_SCHEMA_REGISTRY)
