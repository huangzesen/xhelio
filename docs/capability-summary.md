# Capability Summary

Current state of the XHelio project as of February 2026.

## What It Does

An AI agent that lets users explore and visualize spacecraft/heliophysics data through natural language. Users type requests like "Show me ACE magnetic field data for last week" and the agent searches datasets, fetches data, computes derived quantities, and renders interactive Plotly plots — all through conversation.

## Architecture

```
User input
  |
  v
api_server.py + frontend/  (FastAPI backend + React SPA)
  |  - FastAPI REST + SSE endpoints (71 total)
  |  - React 19 frontend: 5 pages (Chat, Data Tools, Pipeline, Gallery, Settings)
  |  - Multi-session management (up to 10 concurrent sessions)
  |  - SSE streaming for agent responses
  |  - Catalog browsing, data fetch/preview, config, memory CRUD, pipeline/replay endpoints
  |
  v
main.py  (readline CLI, --verbose/--model flags, token usage on exit)
  |  - Commands: quit, reset, status, retry, cancel, errors, sessions, capabilities, help
  |  - Flags: --continue/-c (resume latest), --session/-s ID (resume specific)
  |  - Flags: --refresh (update time ranges), --refresh-full (rebuild all),
  |           --refresh-all (rebuild all missions from CDAWeb)
  |  - Single-command mode: python main.py "request"
  |  - Auto-saves session every turn; checks for incomplete plans on startup
  |  - Mission data menu on startup (interactive refresh prompt)
  |
  v
mcp_server.py  (MCP server over stdio, for Claude Desktop / Claude Code / Cursor)
  |  - Wraps same OrchestratorAgent as main.py / api_server.py
  |  - Tools: chat (text + PNG image), reset_session, get_status
  |  - Flags: --model, --verbose
  |  - Lazy agent init, web_mode=True (suppresses auto-open)
  |
  v
heliospice-mcp  (Standalone SPICE ephemeris MCP server, from heliospice package)
  |  - No LLM needed — lightweight SPICE wrapper
  |  - Tools: get_spacecraft_position, get_spacecraft_trajectory,
  |           get_spacecraft_velocity, compute_distance, transform_coordinates,
  |           list_spice_missions, list_coordinate_frames, manage_kernels
  |  - Flags: --verbose
  |  - Auto-downloads NAIF kernels on first use
  |  - Installed via: pip install heliospice[mcp]
  |
  v
agent/core.py  OrchestratorAgent  (LLM-driven orchestrator)
  |  - Routes: fetch -> mission agents, compute -> DataOps agent, viz -> visualization agent, analysis -> insight agent
  |  - Complex multi-mission requests -> planner -> sub-agents
  |  - Token usage tracking (input/output/thinking/api_calls, includes all sub-agents)
  |  - Six model tiers: smart (orchestrator + planner), sub-agent (mission/viz/data), insight, inline (follow-ups, autocomplete), planner, fallback
  |  - Configurable via ~/.xhelio/config.json (model / sub_agent_model / insight_model / inline_model / planner_model / fallback_model keys)
  |  - Thinking levels: HIGH (orchestrator + planner), LOW (all sub-agents), OFF (inline)
  |
  +---> agent/viz_plotly_agent.py     Visualization sub-agent
  |       VizPlotlyActor             Focused Gemini session for all visualization
  |       render_plotly_json()       Create/update plots via Plotly figure JSON with data_label placeholders
  |       list_fetched_data()        Discover available data in memory
  |       Inbox queue + main loop    Persistent actor with dedicated thread
  |                                  System prompt with Plotly JSON examples
  |
  +---> agent/insight_agent.py     Insight sub-agent (multimodal plot analysis)
  |       InsightActor              Receives rendered PNG + data context
  |       analyze_plot()            Single LLM call with vision (no tool loop)
  |                                  Returns scientific interpretation
  |
  +---> agent/data_ops_agent.py   DataOps sub-agent (compute/describe/export tools)
  |       DataOpsActor            Focused Gemini session for data transformations
  |       Inbox queue + main loop Persistent actor with dedicated thread
  |                               System prompt with computation patterns + code guidelines
  |                               No fetch or plot tools — operates on in-memory data
  |
  +---> agent/data_extraction_agent.py  DataExtraction sub-agent (text-to-DataFrame)
  |       DataExtractionActor     Focused Gemini session for unstructured-to-structured conversion
  |       Inbox queue + main loop Persistent actor with dedicated thread
  |                               Tools: store_dataframe, read_document, ask_clarification
  |                               Turns search results, documents, event lists into DataFrames
  |
  +---> agent/mission_agent.py    Mission sub-agents (fetch-only tools)
  |       MissionActor            Focused Gemini session per spacecraft mission
  |       Inbox queue + main loop Persistent actor with dedicated thread
  |                               Two-mode task prompt: candidate inspection vs direct fetch
  |                               Rich system prompt with recommended datasets + analysis patterns
  |                               No compute or plot tools — reports fetched labels to orchestrator
  |
  +---> agent/prompts.py           Prompt formatting + tool result formatters
  |       get_system_prompt()      Dynamic system prompt with {today} date
  |
  +---> agent/planner.py          Task planning
  |       is_complex_request()    Regex heuristics for complexity detection
  |       PlannerAgent            Chat-based planner with plan-execute-replan loop
  |                               Emits task batches, observes results, adapts plan
  |                               Uses structured JSON output (no tool calling)
  |                               Model: PLANNER_MODEL (defaults to SMART_MODEL)
  |
  +---> agent/session.py           Session persistence
  |       SessionManager          Save/load chat history + DataStore to ~/.xhelio/sessions/
  |                                Auto-save every turn, --continue/--session CLI flags
  |
  +---> agent/memory.py            Long-term memory (cross-session)
  |       MemoryStore              Persist preferences + summaries + pitfalls + reflections to ~/.xhelio/memory.json
  |       Memory                   Dataclass with confidence, tags, access tracking, scoped types
  |       generate_tags()          NLP-based keyword extraction for tag-based search/dedup
  |                                Schema v2: tag-based search, scoped memory injection, access tracking
  |
  +---> agent/memory_agent.py     MemoryAgent (periodic extraction + consolidation)
  |       MemoryAgent              Analyzes curated EventBus events via single LLM call
  |                                Periodic: async on daemon thread every N turns (default 2), incremental event slicing
  |                                Single chronological timeline: conversation + ops + routing interleaved
  |                                Two-phase consolidation: rule-based pre-filter + per-group LLM merge
  |
  +---> agent/tasks.py            Task management
  |       Task, TaskPlan          Data structures (mission, depends_on fields)
  |       TaskStore               JSON persistence to ~/.xhelio/tasks/
  |
  +---> knowledge/                Dataset discovery + prompt generation
  |       function_catalog.py      Auto-generated searchable catalog of scipy/pywt functions (search + docstring retrieval)
  |       missions/cdaweb/*.json   Per-mission JSON files (54 CDAWeb, auto-generated from CDAS REST)
  |       missions/ppi/*.json      Per-mission JSON files (17 PPI, auto-generated from Metadex)
  |       mission_loader.py        Lazy-loading cache, routing table, dataset access (deep-merges CDAWeb + PPI)
  |       mission_prefixes.py      Shared CDAWeb dataset ID prefix map (40+ missions)
  |       cdaweb_metadata.py       CDAWeb REST API client — InstrumentType-based grouping
  |       catalog_search.py        Full dataset catalog fetch/cache/search (CDAWeb + PPI)
  |       catalog.py               Thin routing layer (loads from JSON, backward-compat SPACECRAFT dict)
  |       prompt_builder.py        Two-layer system prompt (slim base + optional full catalog for caching) + rich mission/visualization prompts
  |       metadata_client.py       Dataset metadata (3-layer cache: memory → file → Master CDF)
  |       master_cdf.py            Master CDF skeleton download + parameter metadata extraction
  |       startup.py               Mission data startup: status check, interactive refresh menu, CLI flag resolution
  |       bootstrap.py             Mission JSON auto-generation from CDAS REST + Master CDF
  |
  +---> data_ops/                 Python-side data pipeline (pandas-backed)
  |       fetch.py                  Data fetching via CDF backend
  |       fetch_cdf.py              CDF data fetching + Master CDF-based variable listing
  |       store.py                  In-memory DataStore singleton (label -> DataEntry w/ DataFrame)
  |       custom_ops.py             AST-validated sandboxed executor for LLM-generated pandas/numpy code
  |       pipeline.py               Pipeline DAG (live session) + SavedPipeline (extracted replayable) + is_vanilla/appropriation_fingerprint
  |
  +---> agent/pipeline_store.py  PipelineStore — searchable index of non-trivial saved pipelines
  |       PipelineEntry             Metadata record (name, tags, datasets, missions, family_id, variant_ids)
  |       register()                Vanilla filter + family-based dedup (same data logic → one entry)
  |       search()                  Embedding-based or tag-based search with mission/dataset pre-filtering
  |
  +---> rendering/                Plotly-based visualization engine
  |       plotly_renderer.py        Fills data_label placeholders in LLM-generated Plotly JSON, multi-panel, PNG/PDF export via kaleido
  |       registry.py               Tool registry (2 declarative tools) — single source of truth for viz capabilities
  |
  +---> heliospice (external package, pip install heliospice[mcp])
  |       missions.py               NAIF ID registry + fuzzy mission name resolution
  |       kernel_manager.py          Thread-safe kernel download, cache, load/unload (singleton)
  |       ephemeris.py               get_position, get_trajectory, get_state (SpiceyPy under lock)
  |       frames.py                  Coordinate frame transforms (SPICE pxform + manual RTN)
  |       server.py                  MCP server (heliospice-mcp CLI)
  |
  +---> scripts/                  Tooling
          generate_mission_data.py  Auto-populate JSON from CDAS REST + Master CDF
          fetch_metadata_cache.py   Download metadata cache (Master CDF)
          agent_server.py           TCP socket server for multi-turn agent testing
          run_agent_tests.py        Integration test suite (6 scenarios)
          regression_test_20260207.py  Regression tests from 2026-02-07 session
          stress_test.py            Stress testing
```

## Tools (46 tool schemas)

### Dataset Discovery
| Tool | Purpose |
|------|---------|
| `search_datasets` | Keyword search across spacecraft/instruments (local catalog) |
| `browse_datasets` | Browse all science datasets for a mission (filtered by calibration exclusion lists) |
| `list_parameters` | List plottable parameters for a dataset (Master CDF / local cache) |
| `get_data_availability` | Check available time range for a dataset (local cache / CDAS REST) |
| `get_dataset_docs` | Fetch CDAWeb documentation for a dataset (instrument info, coordinates, PI contact) |
| `list_missions` | List all known spacecraft missions with capabilities |
| `search_full_catalog` | Search full CDAWeb catalog (2000+ datasets, CDAS REST primary) by keyword |
| `google_search` | Web search — Gemini uses built-in Google Search grounding; non-Gemini providers use Tavily fallback |

### Visualization
| Tool | Purpose |
|------|---------|
| `render_plotly_json` | Create or update the plot by providing a Plotly figure JSON with `data_label` placeholders. The system fills in actual data arrays from memory. |
| `manage_plot` | Imperative operations on the current figure: export (PNG/PDF), reset, zoom/time range, get state |

The viz agent uses `render_plotly_json` and `manage_plot` for all visualization operations. The LLM provides a Plotly figure JSON with `data_label` placeholders in trace stubs, and the system resolves these against in-memory data. The tool registry (`rendering/registry.py`) describes both tools with their parameters and examples.

### Data Operations (fetch -> custom_operation -> plot)
| Tool | Purpose |
|------|---------|
| `fetch_data` | Pull data into memory via CDF download (label: `DATASET.PARAM`) |
| `list_fetched_data` | Show all in-memory timeseries |
| `custom_operation` | LLM-generated pandas/numpy/scipy/pywt code (AST-validated, sandboxed) — handles magnitude, arithmetic, smoothing, resampling, derivatives, filtering, spectrograms, wavelets, and any other transformation |
| `describe_data` | Statistical summary of in-memory data (min/max/mean/std/percentiles/NaN) |
| `preview_data` | Preview actual values (first/last N rows) of in-memory timeseries for debugging or inspection |
| `save_data` | Export in-memory timeseries to CSV file |

### Data Extraction
| Tool | Purpose |
|------|---------|
| `store_dataframe` | Create a new DataFrame from scratch and store it in memory (event lists, catalogs, search results, manual data) |

### Function Documentation
| Tool | Purpose |
|------|---------|
| `search_function_docs` | Search scientific computing function catalog by keyword (scipy.signal, scipy.fft, scipy.interpolate, scipy.stats, scipy.integrate, pywt) |
| `get_function_docs` | Get full docstring and signature for a specific function |

### Document Reading
| Tool | Purpose |
|------|---------|
| `read_document` | Read PDF and image files using Gemini vision (extracts text, tables, charts) |

### Memory
| Tool | Purpose |
|------|---------|
| `recall_memories` | Search or browse archived memories from past sessions (preferences, summaries, pitfalls, reflections). Supports optional type and scope filters. |
| `review_memory` | Rate how useful an injected operational memory was (1-5 stars + comment). Called by sub-agents after completing their main task. |

### SPICE Ephemeris
| Tool | Purpose |
|------|---------|
| `get_spacecraft_position` | Get position of a spacecraft at a single time (x,y,z km + distance AU + light time) |
| `get_spacecraft_trajectory` | Compute trajectory over time range (stored as DataEntry for plotting) |
| `get_spacecraft_velocity` | Compute velocity over time range (stored as DataEntry for plotting) |
| `compute_distance` | Compute distance between two bodies over a time range |
| `transform_coordinates` | Transform 3D vector between frames (J2000, ECLIPJ2000, RTN, etc.) |
| `list_spice_missions` | List all supported SPICE missions with NAIF IDs |
| `list_coordinate_frames` | List available coordinate frames with descriptions |

### Conversation
| Tool | Purpose |
|------|---------|
| `ask_clarification` | Ask user when request is ambiguous |

### Routing
| Tool | Purpose |
|------|---------|
| `delegate_to_mission` | LLM-driven delegation to a mission specialist sub-agent |
| `delegate_to_data_ops` | LLM-driven delegation to the data ops specialist sub-agent |
| `delegate_to_data_extraction` | LLM-driven delegation to the data extraction specialist sub-agent |
| `delegate_to_viz_plotly` | LLM-driven delegation to the visualization sub-agent |
| `delegate_to_insight` | LLM-driven delegation to the insight sub-agent for multimodal plot analysis |
| `request_planning` | Activate multi-step planning system for complex requests (orchestrator can trigger dynamically) |

### Pipeline (Live DAG)
| Tool | Purpose |
|------|---------|
| `get_pipeline_info` | Inspect the current DAG: compact summary (default), full node detail with code (`node_id`), or browse ops library (`list_library`). Node detail includes ops library match detection. |
| `modify_pipeline_node` | Mutate the DAG: `update_params`, `remove`, `insert_after`, `apply_library_op` (reuse saved code), `save_to_library` (curate code). Marks affected nodes stale. |
| `execute_pipeline` | Re-run stale/pending nodes with backdating (skip descendants if output unchanged). Optional `use_cache` flag. |

The pipeline DAG (`data_ops/pipeline.py`) is constructed on-demand from the `OperationsLog` and cached on the orchestrator. It provides an abstract operation layer the agent can inspect, modify, and re-execute — enabling workflows like "change the time range and re-run everything downstream" or "insert a smoothing step before the plot." Asset-centric design (nodes = data artifacts), lazy staleness (mutations mark stale but don't recompute until triggered), and backdating (unchanged outputs skip descendants). The ops library integration (`data_ops/ops_library.py`) allows saving compute node code for reuse and applying library entries to pipeline nodes.

### Pipeline (Saved / Replayable)
| Tool | Purpose |
|------|---------|
| `save_pipeline` | Extract a replayable pipeline from the current session's operations log. Saves to `~/.xhelio/pipelines/` and registers in PipelineStore. |
| `run_pipeline` | Replay a saved pipeline with a new time range. Loads by `pl_` ID from disk — works regardless of store registration. |
| `search_pipelines` | Search saved pipelines by query text, mission, or dataset. Delegates to PipelineStore semantic search. |

**Two-tier pipeline system:**

1. **Session replay DAG** (raw) — the chronological operations log (`operations.json`). Every tool call the user made during a conversation. Intertwined, messy, contains failures and exploratory dead-ends.

2. **SavedPipeline** (extracted) — a clean, replayable DAG extracted from the session replay. Steps are classified into **appropriation** (fetch + transform, chainable) and **presentation** (render, terminal). Each pipeline is parameterized by time range and can be replayed without an LLM.

**PipelineStore** (`agent/pipeline_store.py`) manages a searchable metadata index on top of saved pipeline files:

- **Vanilla filter**: Simple pipelines (<3 fetches, no transforms) are still saved to disk but not registered in the store — they won't appear in search results or context injection. These are trivial fetch-and-render workflows that add noise.
- **Family-based dedup**: Pipelines with identical appropriation phases (same fetches and transforms, different visualizations) are grouped under a single "family" entry using SHA-256 fingerprinting. The fingerprint uses canonical position-based step IDs so different session-specific `s001`-style IDs produce the same hash for identical structure. Each family entry tracks all variant `pl_` IDs.
- **Schema v2**: `PipelineEntry` includes `family_id` (64-char SHA-256 hex) and `variant_ids` (list of all `pl_` IDs sharing the family). Migration from v1 is automatic.

### Event Feed (Pull-Based Session Context)
| Tool | Purpose |
|------|---------|
| `check_events` | Check for session events since last check. Returns summaries of data fetches, computations, plots, errors, delegations. Cursor-based — first call returns all relevant events, subsequent calls return only new ones (no duplicates). When response exceeds token quota, returns `quota_exceeded=true` warning; re-call with `compact=true` for LLM-compacted summary. Available to all agents. |
| `get_event_details` | Get full details for specific events by ID. Use after check_events for exact tool arguments, result data, or error details. Supports `compact=true` for LLM compaction when over quota. |

## Sub-Agent Architecture (9 agents)

### OrchestratorAgent (agent/core.py)
- Sees tools: discovery, conversation, routing, document, spice, pipeline + `list_fetched_data` extra
- Routes: data fetching -> MissionActor, computation -> DataOpsActor, text-to-data -> DataExtractionActor, visualization -> VizPlotlyActor, plot analysis -> InsightActor
- Handles multi-step plans with mission-tagged task dispatch (`__data_ops__`, `__data_extraction__`, `__visualization__`)

### MissionActor (agent/mission_agent.py)
- Sees tools: discovery, data_ops_fetch, conversation + `list_fetched_data` extra
- One agent per spacecraft, cached per session
- Rich system prompt with recommended datasets and analysis patterns
- **Two-mode operation**: when planner provides `candidate_datasets`, inspects candidates via `list_parameters` and selects best dataset/parameters autonomously; otherwise executes exact instructions directly
- Handles all-NaN fallback: skips empty parameters, tries next candidate dataset
- No compute tools — reports fetched data labels to orchestrator
- See `docs/planning-workflow.md` for detailed flow

### DataOpsActor (agent/data_ops_agent.py)
- Sees tools: data_ops_compute (`custom_operation`, `describe_data`, `save_data`), function_docs (`search_function_docs`, `get_function_docs`), conversation + `list_fetched_data` extra
- **Two-phase compute**: Think phase (explore data + research function APIs) then Execute phase (write code with enriched context)
- Think phase uses ephemeral chat with function_docs + data inspection tools (same pattern as PlannerAgent discovery)
- Sandbox includes full `scipy` and `pywt` (PyWavelets) for signal processing, wavelets, filtering, interpolation, etc.
- Function documentation catalog auto-generated from scipy submodules and pywt docstrings
- Singleton, cached per session
- System prompt with computation patterns and code guidelines
- No fetch tools — operates on already-fetched data in memory

### DataExtractionActor (agent/data_extraction_agent.py)
- Sees tools: data_extraction (`store_dataframe`), document (`read_document`), conversation (`ask_clarification`) + `list_fetched_data` extra
- Singleton, cached per session
- System prompt with extraction patterns, DataFrame creation guidelines, and document reading workflow
- Turns unstructured text (search results, document tables, event catalogs) into structured DataFrames
- No fetch, compute, or plot tools — creates data from text only

### VizPlotlyActor (agent/viz_plotly_agent.py)
- Sees tools: `render_plotly_json` + `manage_plot` + `list_fetched_data` (3 tools total)
- `render_plotly_json`: LLM provides Plotly figure JSON with `data_label` placeholders; system fills in actual data
- The viz agent owns all visualization: `render_plotly_json` + `manage_plot` + `list_fetched_data`

### InsightActor (agent/insight_agent.py)
- No tools — uses single multimodal LLM call (vision) via `analyze_plot()`
- Receives rendered PNG of the current plot + data context (store entries, trace labels, units, time ranges)
- Returns structured scientific analysis: overview, notable features, data quality, coordinate system awareness, interpretation, suggestions
- System prompt from `build_insight_prompt()` in `knowledge/prompt_builder.py`
- Singleton, cached per session
- Triggered via `delegate_to_insight` routing tool (requires an active plot)
- Image exported at `scale=2` (higher than thumbnails) so LLM can read axis labels
- **Automatic Figure Feedback** (`review_figure()`): opt-in via `reasoning.insight_feedback: true` in config. After every successful `render_plotly_json`, exports the figure to PNG, gathers user request + data context + conversation history, and sends to InsightActor for quality review. Returns PASS/NEEDS_IMPROVEMENT verdict with suggestions. Feedback is injected into the tool result so the orchestrator LLM can act on it (fix/re-render). Uses `build_insight_feedback_prompt()` (review-focused, not scientific analysis). Emits `INSIGHT_FEEDBACK` events (display + memory + console). Adds ~4-9s per render. Default off.

## Supported Spacecraft

### Primary Missions (54 CDAWeb + 17 PPI)

Mission JSON files are auto-generated: 54 CDAWeb missions from CDAS REST API + Master CDF metadata via `scripts/generate_mission_data.py`, plus 17 PPI missions from Metadex Solr via `scripts/generate_ppi_missions.py`. CDAWeb and PPI missions with matching stems are deep-merged at load time. Key missions include PSP, Solar Orbiter, ACE, OMNI, Wind, DSCOVR, MMS, STEREO-A, Cluster, THEMIS, Van Allen Probes, GOES, Voyager 1/2, Ulysses, Cassini, Juno, and more.

### Full CDAWeb Catalog Access (2000+ datasets)

All CDAWeb datasets are searchable via the `search_full_catalog` tool. New missions can be added by creating a JSON file in `knowledge/missions/cdaweb/` via `scripts/generate_mission_data.py --create-new`. The shared prefix map in `knowledge/mission_prefixes.py` maps dataset ID prefixes to mission identifiers.

## Time Range Parsing

Handled by `agent/time_utils.py`. Accepts:
- Relative: `"last week"`, `"last 3 days"`, `"last month"`, `"last year"`
- Month+year: `"January 2024"`
- Single date: `"2024-01-15"` (full day)
- Date range: `"2024-01-15 to 2024-01-20"`
- Datetime range: `"2024-01-15T06:00 to 2024-01-15T18:00"`
- Space-separated datetime: `"2024-01-15 12:00:00 to 2024-01-16"`
- Single datetime: `"2024-01-15T06:00"` (1-hour window)

All times are UTC. Outputs `TimeRange` objects with `start`/`end` datetimes.

## Key Implementation Details

### Tool Registry (`rendering/registry.py`)
- Describes 2 visualization tools: `render_plotly_json`, `manage_plot`
- Each tool has: name, description, typed parameters (with enums for constrained values)
- `_TOOL_MAP` dict for fast tool lookup by name

### Data Pipeline (`data_ops/`)
- `DataEntry` wraps a `pd.DataFrame` (DatetimeIndex + float64 columns) or `xr.DataArray`.
- `DataStore` is a singleton dict keyed by label. The LLM chains tools automatically: fetch -> custom_operation -> plot.
- `custom_ops.py`: AST-validated, sandboxed executor for LLM-generated pandas/numpy/scipy/pywt code. Replaces all hardcoded compute functions — the LLM writes the code directly. Sandbox includes `pd`, `np`, `xr`, `scipy` (full scipy), and `pywt` (PyWavelets).
- Data fetching uses the CDF backend exclusively — downloads CDF files from CDAWeb REST API, caches locally, reads with cdflib. Errors propagate directly for the agent to learn from.

### Saved Pipelines & PipelineStore
- **SavedPipeline** (`data_ops/pipeline.py`): Extracted replayable workflows from session operations logs. Each is a clean DAG with appropriation (fetch + transform) and presentation (render) phases. Saved as `~/.xhelio/pipelines/{pl_id}.json`, indexed in `_index.json`.
- **PipelineStore** (`agent/pipeline_store.py`): Searchable metadata index (`~/.xhelio/pipeline_store.json`). Built on `VersionedStore` for versioning, archival, and embedding-based search.
- **Extraction flow**: Session ops log → `SavedPipeline.from_session()` → `pipeline.save()` (disk) → `pipeline_store.register()` (search index, with vanilla filter and family dedup).
- **Vanilla detection** (`is_vanilla()`): Pipelines with <3 fetches and no transforms are trivial — saved to disk but not registered in store.
- **Family fingerprinting** (`appropriation_fingerprint()`): SHA-256 hash of the appropriation phase with canonical step IDs. Pipelines with identical data logic but different visualizations share one store entry, with all variant `pl_` IDs tracked in `variant_ids`.
- **Replay**: `SavedPipeline.execute(time_start, time_end)` replays with a new time range — no LLM needed. Loads by `pl_` ID from disk (unaffected by store registration).

### Timeseries vs General Data Mode (`DataEntry.is_timeseries`)
Each `DataEntry` has an `is_timeseries` boolean (default `True`) that controls how data is described, computed on, and rendered.

**How mode is set** — inferred from index type, not explicitly specified:
| Storage path | Mode | How determined |
|---|---|---|
| `fetch_data` (CDF/PDS) | Always `True` | Fetched data always has DatetimeIndex |
| `custom_operation` | Inferred | `isinstance(result.index, pd.DatetimeIndex)` or `"time" in da.dims` for xarray |
| `store_dataframe` | Inferred | `isinstance(result_df.index, pd.DatetimeIndex)` |

**Where mode matters:**
- **Data descriptions** (`store.py:describe_sources`): Timeseries entries report `cadence` and `time_range`; non-timeseries entries report `index_range` (no dates).
- **Computation** (`custom_ops.py`): The `source_timeseries` map controls whether source DataFrames get coerced to DatetimeIndex before sandbox execution. Result DatetimeIndex is only enforced when ALL sources are timeseries.
- **Rendering** (`plotly_renderer.py:_extract_index_data`): DatetimeIndex → ISO 8601 strings with `type: "date"` axis; other indices → raw values. The renderer infers axis type from the data structure rather than reading `is_timeseries` directly.
- **Visualization prompt** (`prompt_builder.py`): Tells the LLM to check `is_timeseries` from `list_fetched_data` and use appropriate axis formatting (no date formatting for non-timeseries data).
- **Persistence** (`store.py`): The flag is saved in `_index.json` and survives session round-trips.

### Custom Operations Library (`data_ops/ops_library.py`)
- Persists successful `custom_operation` code (5+ lines) to `~/.xhelio/custom_ops_library.json`.
- Simple operations (magnitude, smoothing, arithmetic) are cheap to regenerate — only complex multi-step pipelines are saved.
- Deduplicates by normalized description; bumps `use_count` on match.
- Evicts least-used entries (tiebreak: oldest `last_used_at`) when the library hits a cap (default 50, configurable via `ops_library_max_entries`).
- During the DataOps think phase, the top 20 entries are injected into the prompt so the LLM can reuse proven code.
- Reuse tracking: LLM includes `[from <id>]` in the description when adapting library code; the system bumps that entry's count.

### LLM Abstraction Layer (`agent/llm/`)
- **Phases 1-3 complete (February 2026)**: All LLM SDK calls go through `agent/llm/` adapter layer. Three adapters implemented.
- `agent/llm/base.py` — Abstract types: `ToolCall`, `UsageMetadata`, `LLMResponse`, `FunctionSchema`, `ChatSession` ABC, `LLMAdapter` ABC
- `agent/llm/gemini_adapter.py` — `GeminiAdapter` + `GeminiChatSession` wrapping `google-genai` SDK
- `agent/llm/openai_adapter.py` — `OpenAIAdapter` for OpenAI-compatible providers (OpenAI, DeepSeek, Qwen, Ollama, etc.)
- `agent/llm/anthropic_adapter.py` — `AnthropicAdapter` for Anthropic Claude models
- **Multimodal support**: `make_multimodal_message(text, image_bytes, mime_type)` factory method on all adapters builds provider-specific messages combining text + image for `ChatSession.send()`. Used by InsightActor for plot analysis via LLM vision.
- Escape hatches for provider-specific features via `LLMResponse.raw` field and adapter-specific methods
- Phase 4 pending: session persistence normalization, CLI `--provider` flag, UI provider selector

### Agent Loop (`agent/core.py`)
- LLM decides which tools to call via function calling (through `LLMAdapter` interface).
- Tool results are fed back via `adapter.make_tool_result_message()`.
- Orchestrator loop continues until LLM produces a text response (or 10 iterations), with consecutive delegation error tracking (breaks after 2 failures).
- Sub-agent loops limited to 5 iterations with duplicate call detection and consecutive error tracking.
- Token usage accumulated from `LLMResponse.usage` (input_tokens, output_tokens, thinking_tokens).

### Actor Model (`agent/actor.py`)

Each sub-agent is a persistent **Actor** with an inbox (`queue.Queue`), a dedicated thread, and a persistent LLM session.

- **`Actor` base class** (`agent/actor.py`): `Message` dataclass + `Actor` class with inbox, main loop thread, active tool tracking, and event bus integration. Tools can run synchronously (blocking, via `batch_sync` meta-tool) or asynchronously (background thread, result delivered via event bus → inbox).
- **Sub-agent actors**: `MissionActor`, `VizPlotlyActor`, `DataOpsActor`, `DataExtractionActor`, `InsightActor` — each extends `Actor` with specialized prompt builders and tool schemas.
- **Delegation**: Delegation tools (`delegate_to_mission`, `delegate_to_viz_plotly`, etc.) use `_get_or_create_*_actor()` + `_delegate_to_actor()`. Actors persist across delegations, preserving LLM context.
- **Serialization**: Each actor has one thread reading from its inbox — multiple requests to the same mission actor queue naturally without locks.

### LLM-Driven Routing (`agent/core.py`, `agent/mission_agent.py`, `agent/data_ops_agent.py`, `agent/viz_plotly_agent.py`)
- **Routing**: The OrchestratorAgent (LLM) decides whether to handle a request directly or delegate via `delegate_to_mission` (fetching), `delegate_to_data_ops` (computation), `delegate_to_data_extraction` (text-to-DataFrame), `delegate_to_viz_plotly` (visualization), or `delegate_to_insight` (multimodal plot analysis) tools. No regex-based routing — the LLM uses conversation context and the routing table to decide.
- **Mission sub-agents**: Each spacecraft has a data fetching specialist with rich system prompt (recommended datasets, analysis patterns). Agents are cached per session. Sub-agents have **fetch-only tools** (discovery, data_ops_fetch, conversation) — no compute, plot, or routing tools.
- **DataOps sub-agent**: Data transformation specialist with `custom_operation`, `describe_data`, `save_data` + `list_fetched_data`. System prompt includes computation patterns and code guidelines. Singleton, cached per session.
- **DataExtraction sub-agent**: Text-to-DataFrame specialist with `store_dataframe`, `read_document`, `ask_clarification` + `list_fetched_data`. System prompt includes extraction patterns and DataFrame creation guidelines. Singleton, cached per session.
- **Visualization sub-agent**: Visualization specialist with `render_plotly_json` + `manage_plot` + `list_fetched_data` tools. Owns all visualization operations (plotting, export, reset, zoom, traces). The LLM provides Plotly figure JSON with `data_label` placeholders, and the system fills in actual data arrays.
- **Tool separation**: Tools have a `category` field (`discovery`, `visualization`, `data_ops`, `data_ops_fetch`, `data_ops_compute`, `data_extraction`, `spice`, `function_docs`, `conversation`, `routing`, `document`, `data_export`, `memory`, `web_search`, `pipeline`, `pipeline_ops`). `get_tool_schemas(categories=..., extra_names=...)` filters tools by category. Orchestrator sees `["discovery", "web_search", "conversation", "routing", "document", "memory", "data_export", "spice", "pipeline", "pipeline_ops"]` + `list_fetched_data` extra. MissionActor sees `["discovery", "data_ops_fetch", "conversation"]` + `list_fetched_data` extra. DataOpsActor sees `["data_ops_compute", "conversation"]` + `list_fetched_data`, `search_function_docs`, `get_function_docs` extras. DataExtractionActor sees `["data_extraction", "document", "conversation"]` + `list_fetched_data` extra. VizPlotlyActor sees `["visualization"]` + `list_fetched_data`, `manage_plot` extras → `render_plotly_json` + `manage_plot` + `list_fetched_data`.
- **Post-delegation flow**: After `delegate_to_mission` returns data labels, the orchestrator uses `delegate_to_data_ops` for computation, `delegate_to_data_extraction` for text-to-DataFrame conversion, `delegate_to_viz_plotly` to visualize results, and optionally `delegate_to_insight` for scientific interpretation of the rendered plot.
- **Slim orchestrator**: System prompt contains a routing table (mission names + capabilities), orchestrator rules, error recovery patterns, and delegation instructions. Dataset IDs and analysis tips live in mission sub-agents.
- **Gemini context caching**: When using Gemini, the orchestrator creates an explicit context cache containing the full system prompt (with mission catalog) + tool schemas (~38K tokens). This exceeds the 32K threshold for Gemini's cached content API, giving a 75% discount on cached input tokens. The cache is created once per session (24h TTL). Non-Gemini providers use the slim prompt without caching.
- **Per-agent session history** (`ctx:` tags): Sub-agents get fresh blank chats per delegation and have no awareness of prior session activity. To fix this, the EventBus `DEFAULT_TAGS` registry includes `ctx:mission`, `ctx:viz`, `ctx:dataops`, `ctx:planner`, and `ctx:orchestrator` tags on relevant event types (data fetches, computes, renders, errors, sub-agent tool calls). At delegation time, `_build_agent_history(agent_type)` queries `get_events(tags={"ctx:{type}"})`, formats events into concise one-line summaries, and injects the result as "Session history (what happened earlier)" before the memory context. This gives agents awareness of prior fetches, failed operations, and rendered plots without replaying the full conversation. The `ctx:mission` tag covers all mission agents (one shared history for cross-mission comparison scenarios). The `ctx:viz` and `ctx:dataops` tags filter `SUB_AGENT_TOOL`/`SUB_AGENT_ERROR` events by agent name to show only that agent's prior tool calls. The `ctx:orchestrator` tag (on 9 event types: `SUB_AGENT_TOOL`, `SUB_AGENT_ERROR`, `DATA_FETCHED`, `DATA_COMPUTED`, `RENDER_EXECUTED`, `CUSTOM_OP_FAILURE`, `FETCH_ERROR`, `RENDER_ERROR`, `PLOT_ACTION`) gives the orchestrator and planner a terse status-only view of sub-agent activity (e.g. `[PSP_Agent] fetch_data: ok`, `Fetched: ACE.Bmag`). `DELEGATION`/`DELEGATION_DONE` are excluded from `ctx:orchestrator` because the orchestrator sees these as its own tool calls in chat history.
- **Token-budgeted history with LLM compaction**: History budgets are token-based (`history_budget_sub_agent`: 10k tokens default, `history_budget_orchestrator`: 20k tokens default, configurable via `config.json`). When formatted history exceeds the budget, a single LLM call (`adapter.generate()` with `SMART_MODEL`, temperature 0.1) compacts the log — preserving all errors/failures verbatim, keeping the most recent 5-10 events in full, and summarizing routine successes into groups. Compacted results are cached by `(agent_type, event_count)` to avoid redundant LLM calls. On LLM failure, falls back to simple truncation (most recent lines kept, oldest dropped). The orchestrator/planner use `_format_orchestrator_history_event()` for terse output; sub-agents use the existing `_format_history_event()` for detailed output.
- **Injection points**: Sub-agents (mission, viz, dataops) are injected at delegation time (6 existing injection points). The orchestrator is injected in `process_message()` after followup context. The planner is injected in `_handle_planning_request()` after time range injection.

### Multi-Step Requests (Hybrid Planning)
- Simple requests are handled by the orchestrator's conversation loop (up to 10 iterations, with consecutive delegation error guard)
- "Compare PSP and ACE" -> `delegate_to_mission("PSP", ...)` -> `delegate_to_mission("ACE", ...)` -> `delegate_to_viz_plotly(plot both)` — all in one `process_message` call
- Complex requests use **hybrid routing** to the **PlannerAgent** for plan-execute-replan:
  1. **Regex pre-filter**: `is_complex_request()` regex heuristics catch obvious complex cases (free, no API cost) and route directly to planner
  2. **Orchestrator override**: The orchestrator (with HIGH thinking) can also call `request_planning` tool for complex cases the regex missed
  3. PlannerAgent runs **discovery phase** (tool-calling) then **planning phase** (JSON-schema-enforced)
  4. Fetch tasks use **physics-intent instructions** + `candidate_datasets` list — planner does NOT specify parameter names
  5. Mission agents inspect candidates, select best dataset/parameters, handle all-NaN fallback
  6. Results (with actual stored labels) are fed back to the PlannerAgent, which decides to continue or finish
  7. Maximum 5 rounds of replanning (configurable via `MAX_ROUNDS`)
  8. If the planner fails, falls back to direct orchestrator execution
  9. See `docs/planning-workflow.md` for the full detailed flow
- Tasks are tagged with `mission="__visualization__"` for visualization dispatch, `mission="__data_ops__"` for compute dispatch, `mission="__data_extraction__"` for text-to-DataFrame dispatch

### Thinking Levels
- Controlled via `create_chat(thinking="high"|"low"|"default")` in the adapter layer
- **HIGH**: Orchestrator (`agent/core.py`) and PlannerAgent (`agent/planner.py`) — deep reasoning for routing decisions and plan decomposition
- **LOW**: MissionActor, VizPlotlyActor, DataOpsActor, DataExtractionActor, InsightActor — fast execution with minimal thinking overhead
- **OFF**: Inline tier (follow-up suggestions, ghost text autocomplete) — cheapest/fastest model, no thinking
- Thinking tokens tracked separately in `get_token_usage()` across all agents
- Verbose mode logs full thoughts to terminal/file, plus 500-char tagged previews for web UI via `agent/thinking.py` utilities
- Task plans persist to `~/.xhelio/tasks/` with round tracking for multi-round plans

### Async Delegation (`agent/async_delegation.py`)
- **Config**: `reasoning.async_delegation` (default: `false`)
- When enabled, eligible `delegate_to_*` calls launch sub-agents on daemon threads and return immediately with `{"status": "pending_async"}`
- **Eligible tools**: `delegate_to_mission`, `delegate_to_data_ops`, `delegate_to_data_extraction`
- **Not eligible** (shared state): `delegate_to_viz_plotly` (PlotlyRenderer), `delegate_to_insight` (image export)
- **Freeze/wake pattern**: If the LLM produces no tool calls but async delegations are pending, the orchestrator freezes (zero LLM cost) until at least one completes, then wakes with results
- `DelegationManager` coordinates via `threading.Condition` for efficient blocking
- Each completed delegation includes a **structured operation log** built from EventBus events (tool calls, fetch results, errors) — the orchestrator sees step-by-step what the sub-agent did
- Thread-local `_active_agent_name` and `_current_agent_type` via `threading.local()` ensure concurrent sub-agents don't interfere with each other's identity tracking

### Per-Mission JSON Knowledge (`knowledge/missions/cdaweb/*.json`, `knowledge/missions/ppi/*.json`)
- **54 CDAWeb + 17 PPI mission JSON files**, auto-generated from CDAS REST API + Master CDF metadata (CDAWeb) and Metadex Solr (PPI). Deep-merged at load time for overlapping missions. Profiles include instrument groupings, dataset parameters, and time ranges.
- **Shared prefix map**: `knowledge/mission_prefixes.py` maps CDAWeb dataset ID prefixes to mission identifiers (40+ mission groups).
- **CDAWeb InstrumentType grouping**: `knowledge/cdaweb_metadata.py` fetches the CDAWeb REST API to get authoritative InstrumentType per dataset (18+ categories like "Magnetic Fields (space)", "Plasma and Solar Wind"). Bootstrap uses this to group datasets into meaningful instrument categories with keywords, instead of dumping everything into "General".
- **Full catalog search**: `knowledge/catalog_search.py` provides `search_full_catalog` tool — searches all CDAWeb + PPI datasets by keyword, with 24-hour local cache for CDAWeb data.
- **Master CDF metadata**: `knowledge/master_cdf.py` downloads CDF skeleton files from CDAWeb and extracts parameter metadata (names, types, units, fill values, sizes). Cached to `~/.xhelio/master_cdfs/`. Used as the network source for parameter metadata.
- **3-layer metadata resolution**: `knowledge/metadata_client.py` resolves dataset metadata through: in-memory cache → local file cache → Master CDF download. Master CDF results are persisted to the local file cache for subsequent use.
- **Recommended datasets**: All datasets in the instrument section are shown as recommended. Additional datasets are discoverable via `browse_datasets`.
- **Calibration exclusion lists**: Per-mission `_calibration_exclude.json` files filter out calibration, housekeeping, and ephemeris datasets from browse results. Uses glob patterns and exact IDs.
- **Auto-generation**: `scripts/generate_mission_data.py` queries CDAS REST API for catalog + Master CDF for parameters. Use `--create-new` to create skeleton JSON files for new missions.
- **Loader**: `knowledge/mission_loader.py` provides in-memory cache, routing table, and dataset access. Routing table derives capabilities from instrument keywords (magnetic field, plasma, energetic particles, electric field, radio/plasma waves, geomagnetic indices, ephemeris, composition, coronagraph, imaging).

### Long-term Memory (`agent/memory.py`)
- Cross-session memory that persists user preferences, session summaries, operational pitfalls, and reflections
- Storage: `~/.xhelio/memory.json` (schema v7) — global, not per-session
- Four memory types:
  - `"preference"` — plot styles, spacecraft of interest, workflow habits
  - `"summary"` — what was analyzed in each session
  - `"pitfall"` — operational lessons learned (e.g., data gaps, fill values)
  - `"reflection"` — procedural knowledge learned from errors (Reflexion pattern)
- Memory fields: `tags` (keyword list for search/dedup), `source` (extracted/reflected/user_explicit/consolidated), `access_count`, `last_accessed`, `supersedes`, `version`, `archived`, `review` (single review dict or None)
- **Review system** (v6): Sub-agents (MissionActor, VizPlotlyActor, DataOpsActor, DataExtractionActor) review injected memories directly via the `review_memory` tool after completing their main task. Each memory has at most one review (`stars`, `comment`, `agent`, `model`, `session_id`, `created_at`). Star meanings: 5=prevented mistake, 4=useful context, 3=relevant but no impact, 2=irrelevant, 1=misleading. Reviewer identity (consuming agent name from `_last_injected_ids`) is stamped. When a memory is edited (supersede), old version keeps its review, new version starts fresh.
- **Multi-scope support**: Each memory has `scopes` (list of strings). A memory can belong to multiple scopes simultaneously (e.g., `["data_ops", "visualization"]`)
  - Scope values: `"generic"` (default), `"mission:<ID>"` (e.g., `"mission:PSP"`), `"visualization"`, `"data_ops"`
  - All agents (orchestrator and sub-agents) receive the same structured format via `MemoryStore.format_for_injection(scope, include_summaries, include_review_instruction)`:
    ```
    [CONTEXT FROM LONG-TERM MEMORY]
    ## Operational Knowledge
    ### Preferences
    - [id] content
    ### Past Sessions          (orchestrator only, include_summaries=True)
    - [id] (date) content
    ### Lessons Learned
    - [id] content
    ### Operational Reflections
    - [id] content
    After completing your main task, review each memory above by calling review_memory(...)
    [END MEMORY CONTEXT]
    ```
  - Orchestrator: `build_prompt_section()` (thin wrapper → `format_for_injection(scope="generic", include_summaries=True)`)
  - Sub-agents: `_inject_memory(request, scope)` → `format_for_injection(scope=scope)` — called fresh at delegation time, so mid-session memory extractions are immediately visible
  - Inline completion: `format_for_injection(scope="generic", include_summaries=True, include_review_instruction=False)` — no review instruction
  - Query methods use `scope in m.scopes` — a memory with `scopes=["data_ops", "visualization"]` appears in both sub-agents
- **Embedding-based search**: `MemoryStore.search()` uses fastembed (BAAI/bge-small-en-v1.5) with cosine similarity threshold 0.55; falls back to tag-based scoring (tag intersection ×2 + substring ×1) when fastembed is unavailable
- **Tag generation**: `generate_tags()` tokenizes content, removes stop words, adds scope-specific tags for all scopes in the list
- **Tag-based dedup**: New memories are checked against existing tag sets (Jaccard similarity) to avoid duplicates
- **Access tracking**: Each memory tracks how many times it was retrieved for injection and when
- **Version tracking**: Edits archive old entries (`archived=True`) and create new ones with `supersedes` chain and `version++`
- **In-file archival** (v4): Archived memories stored in-place with `archived=True` flag instead of separate cold storage file
- Token budget: 100000 tokens global cap (orchestrator and each sub-agent get independent budgets via `format_for_injection()`; sub-agents get 1/4 each)
- Global enable/disable toggle + per-memory enable/disable
- **Injection tracking**: `MemoryStore._last_injected_ids` tracks which memories were actually injected into agent prompts each turn. Cleared at start of `process_message()`, populated by `format_for_injection()`. Used by MemoryAgent to annotate memories with `[INJECTED]` for consolidation decisions, and by `review_memory` tool handler to stamp the consuming agent name on reviews.
- Schema migration: v1 → v2 → v3 → v4 → v5 → v6 → v7 automatically on first load
- Web UI: Memory page with dashboard stats, type/scope filters, search, timeline, archive browser, and version history
- CLI: memories extracted periodically during the session (no shutdown pass needed)

### MemoryAgent (`agent/memory_agent.py`)
- **Periodic extraction**: Triggered every N user turns (default 2) by `_maybe_extract_memories()` in `core.py`. Runs async on a daemon thread using INLINE_MODEL. Queries EventBus for memory-tagged events since last extraction (incremental slicing). Lock prevents concurrent extractions. Flushes to disk after each extraction.
- **Single LLM call per extraction**: Curates EventBus events via `build_curated_events()` into a chronological timeline (conversation turns, data ops, routing, errors interleaved), then analyzes via single LLM call. Extracts preferences, session summaries, and pitfalls (with multi-scope support), uses tag-based dedup against existing memories. Actions: add/edit/drop only — reviews are handled by sub-agents directly via `review_memory` tool. The agent's own review is injected into the memory prompt for consolidation decisions. DataOps-only missions are filtered from extraction to avoid noise.
- **Pipeline registration**: MemoryAgent also curates pipeline candidates via LLM-judged registration. Pipeline candidates from session scans are routed through the MemoryAgent for quality assessment before being registered in PipelineStore.
- **Two-phase consolidation** (conservative policy — when memory count exceeds budget):
  - Phase A — Rule-based pre-filter (no LLM): archives excess summaries, low-confidence (<0.3), old unaccessed (>30 days, access_count=0), tag-overlap dedup (≥80% Jaccard)
  - Phase B — Per-group LLM merge: groups remaining memories by (type, frozenset(scopes)), sends over-budget groups to LLM for merging
  - Conservative: never merges memories that represent distinct knowledge; 10x memory token budget (100k tokens) to reduce pressure
  - Per-type budgets: preferences 15, summaries 10, pitfalls 15, reflections 10
- All exceptions caught — never breaks the main agent flow


- **VersionedStore** (`agent/versioned_store.py`): Base class providing versioned JSON persistence with schema migration, used by MemoryStore and PipelineStore.

- `SessionManager` saves and restores chat history + DataStore across process restarts
- Storage layout: `~/.xhelio/sessions/{session_id}/` with `metadata.json`, `history.json`, and `data/*.pkl`
- Auto-save after every turn in `process_message()` — survives crashes
- DataStore persistence uses disk-backed `put()` calls with pickle + `_labels.json`
- Chat history round-trips via `Content.model_dump(exclude_none=True)` → JSON → `chats.create(history=...)`
- CLI flags: `--continue` / `-c` (resume latest), `--session` / `-s ID` (resume specific)
- CLI command: `sessions` — list saved sessions
- Web UI: Sessions sidebar with Load / New / Delete buttons
- Sub-agent state not persisted (fresh chats per request); PlotlyRenderer resets on load (user can re-plot)

### Auto-Clamping Time Ranges
- `_validate_time_range()` in `agent/core.py` auto-adjusts requested time ranges to fit dataset availability windows
- Handles partial overlaps (clamps to available range) and full mismatches (informs user of available range)
- Fail-open: if metadata call fails (Master CDF), proceeds without validation

### Default Plot Styling
- `_DEFAULT_LAYOUT` in `rendering/plotly_renderer.py` sets explicit white backgrounds (`paper_bgcolor`, `plot_bgcolor`) and dark font color
- Prevents dark theme CSS from making plots appear black
- Applied in `_ensure_figure()` and `_grow_panels()`

### Figure Sizing
- Renderer sets explicit defaults: `autosize=False`, 300px per panel height, 1100px width
- Prevents Plotly.js from recalculating dimensions on toolbar interactions (zoom, pan, reset)
- LLM can set explicit width/height in the layout JSON to override defaults

### Web UI Streaming
- FastAPI backend streams live progress via SSE (server-sent events)
- Agent output unified through Python logging
- Tag-based filtering via `WEBUI_VISIBLE_TAGS` — only agent lifecycle, plan events, data fetched, thinking previews, and errors are shown. See the Logging section for the full tag table and how to add new categories.

### Mission Data Startup (`knowledge/startup.py`)
- Shared startup logic used by `main.py` and the API server
- `get_mission_status()` scans mission JSONs and reports count, datasets, last refresh date
- `show_mission_menu()` presents interactive refresh options on startup
- `resolve_refresh_flags()` maps CLI flags (`--refresh`, `--refresh-full`, `--refresh-all`) to actions
- `run_mission_refresh()` invokes bootstrap to refresh time ranges or rebuild all missions
- After refresh, clears mission_loader and metadata_client caches

### Automatic Model Fallback (`agent/model_fallback.py`)
- When any Gemini API call hits a 429 RESOURCE_EXHAUSTED (quota/rate limit), all agents automatically switch to `FALLBACK_MODEL` for the remainder of the session
- Session-level global flag — once activated, every subsequent `client.chats.create()` and `models.generate_content()` call uses the fallback model
- The OrchestratorAgent's persistent chat is recreated with the fallback model on first 429 error
- Sub-agents (Actor subclasses, PlannerAgent, MemoryAgent) use `get_active_model()` at chat/call creation time, so they pick up the fallback automatically
- Configurable via `fallback_model` in `~/.xhelio/config.json` (default: `gemini-3-flash`)
- If the fallback model also fails, the error propagates normally (no retry chain)

### Empty Session Auto-Cleanup
- On startup, `SessionManager` auto-removes sessions with no chat history and no stored data
- Prevents clutter from abandoned or crashed sessions
- Session save is skipped when there's nothing to persist

### First-Run Full Download
- On first run (no mission JSONs exist), `ensure_missions_populated()` calls `populate_missions()` which downloads the full CDAWeb catalog + Master CDF parameter metadata (~5-10 minutes, one-time)
- Subsequent startups are instant (JSON files already exist)
- Two refresh paths: `--refresh` (lightweight time-range update) and `--refresh-full` (destructive rebuild)
- Shows progress via tqdm in terminal, logger-based progress in web UI live log

### Web Search (`google_search` tool)
- **Gemini provider**: Uses built-in Google Search grounding API via an isolated Gemini API call (Gemini API does not support google_search + function_declarations in the same call)
- **Non-Gemini providers** (OpenAI, Anthropic, OpenRouter, etc.): Falls back to Tavily web search (`TAVILY_API_KEY` env var required; `tavily-python` package)
- **No search backend available**: Warning logged, error returned to LLM — agent continues without search
- Returns grounded text with source URLs
- Search results can be turned into plottable datasets via the `store_dataframe` tool (google_search → delegate_to_data_extraction → store_dataframe → plot)

## Configuration

**`.env`** at project root (secret only):
```
GOOGLE_API_KEY=<gemini-api-key>
TAVILY_API_KEY=<tavily-api-key>  # Optional — enables web search for non-Gemini providers
```

**`~/.xhelio/config.json`** (user-editable, all optional — defaults shown):
```json
{
  "llm_provider": "gemini",
  "providers": {
    "gemini": {
      "model": "gemini-3-flash",
      "sub_agent_model": "gemini-3-flash",
      "insight_model": "gemini-3-flash",
      "inline_model": "gemini-2.5-flash-lite",
      "planner_model": "gemini-3-flash",
      "fallback_model": "gemini-3-flash"
    }
  },
  "data_backend": "cdf",
  "catalog_search_method": "semantic",
  "memory_extraction_interval": 2,
  "memory_token_budget": 100000
}
```

See `config.template.json` for a copyable template. If the file doesn't exist, built-in defaults are used.

## Running

```bash
python main.py               # Normal mode (auto-saves session)
python main.py --verbose     # Show tool calls, timing, errors
python main.py --continue    # Resume most recent session
python main.py --session ID  # Resume specific session by ID
python main.py -m MODEL      # Specify Gemini model (overrides .env)
python main.py "request"     # Single-command mode (non-interactive, exits after response)
python main.py --refresh     # Refresh dataset time ranges (fast — start/stop dates only)
python main.py --refresh-full  # Full rebuild of all mission data
python main.py --refresh-all   # Full rebuild of all missions (same as --refresh-full)
```

### React Frontend + FastAPI Backend

```bash
python api_server.py                # Start FastAPI backend on localhost:8000
cd frontend && npm run dev          # Start Vite dev server on localhost:5173 (proxies /api)
cd frontend && npm run build        # Production build → frontend/dist/ (served by FastAPI)
```

Five-page SPA with client-side routing (`react-router-dom`):

**Chat Page** (`/`):
- SSE-streamed agent responses with live tool call display
- Interactive Plotly figures above the chat
- Session sidebar with Load / New / Delete
- Follow-up suggestions after each response
- Input history with arrow key navigation
- LLM ghost-text autocomplete (debounced, Tab to accept, Esc to dismiss)
- Plan status display with retry/cancel controls
- Color-coded tool events by category (data, compute, viz, spice, catalog)
- Expandable token usage with per-agent breakdown and data store stats

**Data Tools Page** (`/data`):
- Catalog Browser: mission → dataset → parameter cascade selectors with time range
- Direct data fetch into session DataStore
- Data table with auto-refresh (4s interval)
- Data preview (head+tail rows with numeric rounding)
- Memory manager: global toggle, type-tagged list, delete selected, clear all

**Pipeline Page** (`/pipeline`):
- Session selector (sessions with operations.json)
- Interactive DAG visualization via Plotly (from `scripts/plot_pipeline.py`)
- Step table with tool badge colors and click-to-select
- Code viewer with Python syntax highlighting (`prism-react-renderer`)
- Prev/Next navigation for compute steps
- Replay controls: cached or fresh re-fetch

**Gallery Page** (`/gallery`):
- Saved data products from pipeline sessions

**Settings Page** (`/settings`):
- LLM provider selector (Gemini / OpenAI-compatible / Anthropic)
- Six model tier inputs (main, sub-agent, insight, inline, planner, fallback)
- Gemini thinking level controls (main + sub-agent)
- Data & search settings (catalog search method, parallel fetch, max workers)
- Memory limits (max preferences, summaries, pitfalls)
- Save to `~/.xhelio/config.json`

**Tech stack:** React 19, TypeScript, Zustand (6 stores), Plotly.js, Tailwind 4, Vite 7

### MCP Server

```bash
python mcp_server.py              # Start MCP server (stdio transport)
python mcp_server.py -v           # With verbose logging
python mcp_server.py -m MODEL     # Override LLM model
```

Exposes the agent as three MCP tools (`chat`, `reset_session`, `get_status`) over stdio transport. Any MCP-compatible client can connect — Claude Desktop, Claude Code, Cursor, etc. The `chat` tool returns text plus a PNG image when a plot is produced.

### SPICE MCP Server (heliospice)

```bash
heliospice-mcp                      # Start SPICE MCP server (stdio)
heliospice-mcp -v                   # With verbose logging
python -m heliospice.server         # Alternative invocation
```

Standalone SPICE ephemeris server — no LLM needed. Eight tools for spacecraft positions, trajectories, velocities, distances, coordinate transforms, and kernel management. Kernels auto-downloaded from NAIF. Installed via `pip install heliospice[mcp]`. Configure via `.mcp.json`.

### CLI Commands

| Command | Description |
|---------|-------------|
| `quit` / `exit` | Exit the program |
| `reset` | Clear conversation history (starts new session) |
| `status` | Show current multi-step plan progress |
| `retry` | Retry the first failed task in current plan |
| `cancel` | Cancel current plan, skip remaining tasks |
| `errors` | Show recent error from log files |
| `sessions` | List saved sessions (most recent 10) |
| `capabilities` / `caps` | Show detailed capability summary |
| `help` | Show welcome message and help |

### Logging (`agent/logging.py`)
- Log files stored in `~/.xhelio/logs/agent_YYYYMMDD_HHMMSS.log` (one per session)
- Detailed error logging with stack traces
- `log_error()`: Captures context and full stack traces for debugging
- `log_tool_call()` / `log_tool_result()`: Tracks all tool invocations
- `log_plan_event()`: Records plan lifecycle events (tagged for web UI)
- `print_recent_errors()`: CLI command to review recent errors

#### Tag-Based Log Filtering (Web UI vs Terminal)
All log calls can be tagged with `extra=tagged("category")`. The web UI live log handler only shows records whose `log_tag` is in `WEBUI_VISIBLE_TAGS` (plus all WARNING/ERROR). Terminal and file handlers show everything.

**`WEBUI_VISIBLE_TAGS`** (defined in `agent/logging.py`):
| Tag | Example message | Source |
|-----|-----------------|--------|
| `"delegation"` | `[Router] Delegating to PSP specialist` | `agent/core.py` |
| `"delegation_done"` | `[Router] PSP specialist finished` | `agent/core.py` |
| `"plan_event"` | `Plan created: a1b2c3d4...` | `agent/logging.py:log_plan_event()` |
| `"plan_task"` | `[Plan] [PSP]: Fetch magnetic field data` | `agent/core.py` |
| `"data_fetched"` | `[DataOps] Stored 'AC_H2_MFI.BGSEc' (10080 points)` | `agent/core.py` |
| `"thinking"` | `[Thinking] The user wants...` (first 500 chars) | `core.py`, `actor.py`, `planner.py` |
| `"error"` | Real errors with context/stack traces | `agent/logging.py:log_error()` |

**What is NOT shown in web UI** (terminal/file only): `[CDF]`, `[Gemini]`, `[Tool:]` calls, full thinking text, internal tool-result warnings/errors, DataOps plumbing. Only `log_error()` errors appear in web UI (tagged `"error"`); per-tool `logger.warning("Tool error: ...")` lines are untagged and filtered out.

**To add a new category to web UI:**
1. Tag the log call: `logger.debug("...", extra=tagged("my_tag"))`
2. Add `"my_tag"` to `WEBUI_VISIBLE_TAGS` in `agent/logging.py`

No filter logic changes needed. The `tagged()` helper returns `{"log_tag": tag}` for use as the `extra` kwarg.

**Thinking log records**: Each thought emits two records — the full untagged text (goes to terminal/file only) and a 500-char truncated preview tagged `"thinking"` (shown in web UI). This happens in `_track_usage()` in `core.py`, `actor.py`, and `planner.py`.

## Tests

```bash
python -m pytest tests/test_store.py tests/test_custom_ops.py   # Data ops tests
python -m pytest tests/test_session.py                           # Session persistence tests
python -m pytest tests/test_memory.py tests/test_memory_agent.py # Memory + MemoryAgent tests
# SPICE tests are now in the heliospice repo
python -m pytest tests/                                          # All tests
```

## Dependencies

```
google-genai>=1.60.0    # Gemini API (via agent/llm/gemini_adapter.py)
python-dotenv>=1.0.0    # .env loading
requests>=2.28.0        # HTTP calls (CDAS REST, Master CDF, CDF downloads)
cdflib>=1.3.0           # CDF file reading (Master CDF metadata, data files)
numpy>=1.24.0           # Array operations
scipy>=1.10.0           # Signal processing, FFT, interpolation, statistics
PyWavelets>=1.8.0       # Wavelet transforms (CWT, DWT, packets)
pandas>=2.0.0           # DataFrame-based data pipeline
plotly>=5.18.0          # Interactive scientific data visualization
kaleido>=0.2.1          # Static image export for Plotly (PNG, PDF)
matplotlib>=3.7.0       # Legacy plotting (unused in main pipeline)
tqdm>=4.60.0            # Progress bars for bootstrap/data downloads
pytest>=7.0.0           # Test framework
tavily-python>=0.5.0    # Tavily web search (fallback for non-Gemini providers)
mcp>=1.26.0             # MCP server (stdio transport for Claude Desktop / Claude Code)
heliospice[mcp]>=0.1.0  # SPICE ephemeris + MCP server (auto-managed kernels, wraps SpiceyPy)
```
