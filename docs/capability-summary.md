# Capability Summary

Current state of the XHelio project as of March 2026.

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
  |  - Single-command mode: python main.py "request"
  |  - Auto-saves session every turn; checks for incomplete plans on startup
  |
  v
mcp_server.py  (MCP server over stdio, for Claude Desktop / Claude Code / Cursor)
  |  - Wraps same OrchestratorAgent as main.py / api_server.py
  |  - Tools: chat (text + PNG image), reset_session, get_status
  |  - Flags: --model, --verbose
  |  - Lazy agent init, web_mode=True (suppresses auto-open)
  |
  v
agent/core.py  OrchestratorAgent  (LLM-driven orchestrator)
  |  - Routes: fetch -> mission agents, compute -> DataOps agent, viz -> visualization agent, analysis -> insight agent
  |  - Complex multi-mission requests -> planner -> sub-agents
  |  - Token usage tracking (input/output/thinking/api_calls, includes all sub-agents)
  |  - Four model tiers: smart (orchestrator + planner), sub-agent (mission/viz/data), insight, inline (follow-ups, autocomplete)
  |  - Configurable via ~/.xhelio/config.json (model / sub_agent_model / inline_model keys)
  |  - Thinking levels: HIGH (orchestrator + planner), LOW (all sub-agents), OFF (inline)
  |
  +---> agent/viz_agent.py            Visualization sub-agent
  |       VizAgent                   Unified visualization specialist (Plotly/matplotlib/JSX)
  |       render_plotly_json()       Create/update plots via Plotly figure JSON with data_label placeholders
  |       list_fetched_data()        Discover available data in memory
  |       Inbox queue + main loop    Persistent actor with dedicated thread
  |                                  System prompt with visualization examples per backend
  |
  |  (InsightAgent deleted — vision is now an intrinsic tool on BaseAgent)
  |
  +---> agent/data_ops_agent.py   DataOps sub-agent (compute/describe/export tools)
  |       DataOpsActor            Focused Gemini session for data transformations
  |       Inbox queue + main loop Persistent actor with dedicated thread
  |                               System prompt with computation patterns + code guidelines
  |                               No fetch or plot tools — operates on in-memory data
  |
  +---> agent/data_io_agent.py       DataIO sub-agent (text-to-DataFrame + local file import)
  |       DataIOAgent              Focused Gemini session for data extraction and file loading
  |       Inbox queue + main loop Persistent actor with dedicated thread
  |                               Tools: store_dataframe, load_file, read_document, ask_clarification
  |                               Turns search results, documents, event lists into DataFrames;
  |                               loads local CSV/JSON/Excel/Parquet/CDF files
  |
  +---> agent/envoy_agent.py      Envoy sub-agents (dynamic, per-mission)
  |       EnvoyAgent              Focused LLM session per spacecraft mission
  |       Inbox queue + main loop Persistent actor with dedicated thread
  |                               No envoys currently registered — infrastructure intact
  |                               Envoys are added dynamically at runtime via manage_envoy
  |                               No compute or plot tools — reports fetched labels to orchestrator
  |
  +---> agent/prompts.py           System prompt wrapper (lazy-cached, thread-safe)
  |       get_system_prompt()      Cached orchestrator system prompt
  |
  +---> agent/planner.py          Task planning
  |       is_complex_request()    Regex heuristics for complexity detection
  |       PlannerAgent            Research-only planner — researches datasets and returns
  |                               a complete plan for the orchestrator to execute
  |                               Single-round, produces complete plan with tasks
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
  |       prompt_builder.py        Assembly manifests — lists which markdown sections to load per agent
  |       prompt_loader.py         Loads and LRU-caches prompt sections from knowledge/prompts/*.md
  |       prompts/                 53 markdown files organized by agent type (_shared/, orchestrator/, planner/, etc.)
  |       function_catalog.py      Auto-generated searchable catalog of scipy/pywt functions (search + docstring retrieval)
  |       envoys/                  Empty directory — scan target for runtime envoy kinds
  |       catalog.py               Thin routing layer (loads from JSON, backward-compat SPACECRAFT dict)
  |       startup.py               Mission data startup: status check, interactive refresh menu, CLI flag resolution
  |
  +---> data_ops/                 Python-side data pipeline (pandas-backed)
  |       store.py                  In-memory DataStore singleton (label -> DataEntry w/ DataFrame)
  |       custom_ops.py             AST-validated sandboxed executor for LLM-generated pandas/numpy code
  |       dag.py                    PipelineDAG — networkx-backed directed acyclic graph for pipeline tracking
  |
  +---> rendering/                Plotly-based visualization engine
  |       plotly_renderer.py        Fills data_label placeholders in LLM-generated Plotly JSON, multi-panel, PNG/PDF export via kaleido
  |       registry.py               Tool registry (2 declarative tools) — single source of truth for viz capabilities
  |
  +---> scripts/                  Tooling
          agent_server.py           TCP socket server for multi-turn agent testing
          run_agent_tests.py        Integration test suite (6 scenarios)
          regression_test_20260207.py  Regression tests from 2026-02-07 session
          stress_test.py            Stress testing
```

## Tools (33 tool schemas)

### Dataset Discovery
| Tool | Purpose |
|------|---------|
| `envoy_query` | Generic envoy capability discovery. Three modes: list all envoys (no args), navigate envoy JSON tree (envoy + path), regex search across envoy trees (search). Currently returns empty results — no envoys registered. |
| `google_search` | Web search — each provider uses its native search capability (Gemini: Google Search grounding, OpenAI: search model, Anthropic: web_search tool, MiniMax: MCP web_search) |

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
| `review_memory` | Rate how useful an injected operational memory was (1-5 stars + structured fields: rating, criticism, suggestion, comment). Called by sub-agents after completing their main task. |

### Conversation
| Tool | Purpose |
|------|---------|
| `ask_clarification` | Ask user when request is ambiguous |

### Routing
| Tool | Purpose |
|------|---------|
| `delegate_to_envoy` | LLM-driven delegation to a mission specialist sub-agent (envoy list dynamically injected into schema at runtime) |
| `delegate_to_data_ops` | LLM-driven delegation to the data ops specialist sub-agent |
| `delegate_to_data_io` | LLM-driven delegation to the data I/O specialist sub-agent |
| `delegate_to_viz` | LLM-driven delegation to the visualization sub-agent (backend param selects plotly/matplotlib/jsx) |
| ~~`delegate_to_insight`~~ | Deleted — replaced by intrinsic `vision` tool on BaseAgent |
| `delegate_to_planner` | Activate multi-step planning system for complex requests (orchestrator can trigger dynamically) |

### Pipeline (Live DAG)
| Tool | Purpose |
|------|---------|
| `pipeline(action="info")` | Inspect the current DAG: compact summary (default), full node detail with code (`node_id`), or browse ops library (`list_library`). Node detail includes ops library match detection. |
| `pipeline(action="modify")` | Mutate the DAG: `sub_action` selects `update_params`, `remove`, `insert_after`, `apply_library_op` (reuse saved code), `save_to_library` (curate code). Marks affected nodes stale. |
| `pipeline(action="execute")` | Re-run stale/pending nodes with backdating (skip descendants if output unchanged). Optional `use_cache` flag. |

The pipeline DAG (`data_ops/dag.py`) is a graph-native, networkx-backed directed acyclic graph that records every pipeline operation as a node with automatic edge creation from label flow (producer → consumer). It tracks operations across all agent boundaries — both orchestrator and sub-agent tool executions emit pipeline events via the EventBus, which the `PipelineDAGListener` routes to `PipelineDAG.add_node()`. Persists as `pipeline.json` in the session directory.

**`run_code` strict contract:**
- **Input isolation**: Only data listed in `inputs` is staged as files in a fresh temporary directory per execution. Undeclared data is not accessible.
- **Multi-output**: The `outputs` parameter maps store labels to variable names (e.g., `outputs={"Bmag": "result", "Bangle": "angle"}`). Each output is written independently.
- **No implicit state**: Each execution is fully isolated — no shared state between calls.

### Event Feed (Pull-Based Session Context)
| Tool | Purpose |
|------|---------|
| `events(action="check")` | Check for session events since last check. Returns summaries of data fetches, computations, plots, errors, delegations. Cursor-based — first call returns all relevant events, subsequent calls return only new ones (no duplicates). Available to all agents. |
| `events(action="details")` | Get full details for specific events by ID. Use after events(action="check") for exact tool arguments, result data, or error details. |

## Sub-Agent Architecture (9 agents)

### OrchestratorAgent (agent/core.py)
- Sees tools: discovery, conversation, routing, document, pipeline + `list_fetched_data` extra
- Routes: data fetching -> EnvoyAgent, computation -> DataOpsActor, text-to-data/file-import -> DataIOAgent, visualization -> VizAgent, plot analysis -> InsightActor
- Handles multi-step plans with mission-tagged task dispatch (`__data_ops__`, `__data_io__`, `__visualization__`)

### EnvoyAgent (agent/envoy_agent.py)
- No envoys currently registered. The envoy infrastructure (EnvoyAgent, kind registry, delegation) is intact but empty.
- Envoys are added dynamically at runtime via `manage_envoy` (temporarily removed from orchestrator tool set).
- Each envoy kind has its own tool set, prompt templates, and permission rules defined in `knowledge/envoys/{kind}/`.
- `delegate_to_envoy` schema dynamically injects the current envoy list at runtime.
- One agent per spacecraft, cached per session.
- No compute tools — reports fetched data labels to orchestrator.

### DataOpsActor (agent/data_ops_agent.py)
- Sees tools: data_ops_compute (`custom_operation`, `describe_data`, `save_data`), function_docs (`search_function_docs`, `get_function_docs`), conversation + `list_fetched_data` extra
- **Two-phase compute**: Think phase (explore data + research function APIs) then Execute phase (write code with enriched context)
- Think phase uses ephemeral chat with function_docs + data inspection tools (same pattern as PlannerAgent discovery)
- Sandbox includes full `scipy` and `pywt` (PyWavelets) for signal processing, wavelets, filtering, interpolation, etc.
- Function documentation catalog auto-generated from scipy submodules and pywt docstrings
- Singleton, cached per session
- System prompt with computation patterns and code guidelines
- No fetch tools — operates on already-fetched data in memory

### DataIOAgent (agent/data_io_agent.py)
- Sees tools: data_io (`store_dataframe`, `load_file`), document (`read_document`), conversation (`ask_clarification`) + `list_fetched_data` extra
- Singleton, cached per session
- System prompt with extraction patterns, DataFrame creation guidelines, file loading workflow, and document reading workflow
- Turns unstructured text (search results, document tables, event catalogs) into structured DataFrames
- Loads local files (CSV, JSON, Excel, Parquet, CDF) into the DataStore
- No fetch, compute, or plot tools — creates/imports data only

### VizAgent (agent/viz_agent.py)
- Unified visualization specialist parameterized by `VIZ_BACKENDS` config dict
- Plotly backend: `render_plotly_json` + `manage_plot` + `list_fetched_data`
- Matplotlib backend: `generate_mpl_script` + `manage_mpl_output` + `list_fetched_data`
- JSX backend: `generate_jsx_component` + `manage_jsx_output` + `list_fetched_data`
- `render_plotly_json`: LLM provides Plotly figure JSON with `data_label` placeholders; system fills in actual data
- The viz agent owns all visualization operations for the active backend

### Intrinsic Tools (agent/base_agent.py)

All agents inherit two intrinsic tools from BaseAgent (opt-out available):

- **`vision`** — Analyze an image file using the model's vision capability. Takes `image_path` + `question`, reads the file, and calls `LLMService.generate_vision()`. MIME type auto-detected from extension. No `xhelio__` prefix.
- **`web_search`** — Search the web for real-world context. Takes `query`, calls `LLMService.web_search()`. No `xhelio__` prefix.

Three-tier tool model:
1. **Intrinsic tools** (BaseAgent, no prefix, only need LLMService) — `vision`, `web_search`
2. **Orchestrator-private tools** (OrchestratorAgent._local_tools, xhelio policy)
3. **Tool server tools** (MCP-exposable, domain state) — all `xhelio__` prefixed tools

MemoryAgent opts out of intrinsic tools.

- **Automatic Figure Feedback** (`sync_insight_review()` in `eureka_hooks.py`): opt-in via `reasoning.insight_feedback: true` in config. After every successful `render_plotly_json`, exports the figure to PNG and calls `LLMService.generate_vision()` directly for quality review. Returns PASS/NEEDS_IMPROVEMENT verdict. Emits `INSIGHT_FEEDBACK` events (display + memory + console). Default off.

## Supported Spacecraft

No mission data is currently bundled. CDAWeb and PPI mission data have been removed. New mission support will be added via standalone MCP packages (`cdawebmcp`, `ppimcp`) registered as envoy kinds through `manage_envoy`.

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
- Data fetching infrastructure has been removed — will be re-added via MCP-backed envoys.

### PipelineDAG (`data_ops/dag.py`)
- **Graph-native**: networkx `DiGraph` with thread-safe access (`threading.Lock`). Nodes are operations, edges represent data flow via label ownership.
- **Automatic edges**: When a node declares `inputs=["label"]`, an edge is created from the producer of that label to the consumer. Label ownership is updated on success.
- **Cross-agent tracking**: Tool handlers emit pipeline events (`DATA_COMPUTED`, `RENDER_EXECUTED`, etc.) directly. The `PipelineDAGListener` (subscribed in `session_lifecycle.py`) routes pipeline-tagged events to `dag.add_node_auto()`.
- **Query API**: `predecessors()`, `successors()`, `ancestors()`, `descendants()`, `roots()`, `leaves()`, `path()`, `producer_of()`, `consumers_of()`, `topological_order()`, `subgraph()`.
- **Persistence**: Atomic save to `pipeline.json` in session directory. `PipelineDAG.load()` restores from disk.

### Timeseries vs General Data Mode (`DataEntry.is_timeseries`)
Each `DataEntry` has an `is_timeseries` boolean (default `True`) that controls how data is described, computed on, and rendered.

**How mode is set** — inferred from index type, not explicitly specified:
| Storage path | Mode | How determined |
|---|---|---|
| `fetch_data` (CDF) | Always `True` | Fetched data always has DatetimeIndex |
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
- **Phases 1-3 complete (February 2026)**: All LLM SDK calls go through `agent/llm/` adapter layer. Four adapters implemented.
- `agent/llm/base.py` — Abstract types: `ToolCall`, `UsageMetadata`, `LLMResponse`, `FunctionSchema`, `ChatSession` ABC, `LLMAdapter` ABC
- `agent/llm/gemini_adapter.py` — `GeminiAdapter` + `GeminiChatSession` wrapping `google-genai` SDK
- `agent/llm/openai_adapter.py` — `OpenAIAdapter` for OpenAI and OpenAI-compatible providers
- `agent/llm/anthropic_adapter.py` — `AnthropicAdapter` for Anthropic Claude models
- `agent/llm/minimax_adapter.py` — `MiniMaxAdapter` (thin `AnthropicAdapter` subclass, Anthropic-compatible API at `api.minimaxi.com`)
- `agent/minimax_mcp_client.py` — MCP client for `minimax-coding-plan-mcp` server (web search + image understanding)
- **Multimodal support**: `make_multimodal_message(text, image_bytes, mime_type)` factory method on all adapters builds provider-specific messages combining text + image for `ChatSession.send()`. Used by InsightActor for plot analysis via LLM vision. MiniMax routes image analysis through its MCP `understand_image` tool since the Anthropic-compatible API does not support image input.
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

- **`SubAgent` base class** (`agent/sub_agent.py`): `Message` dataclass + `SubAgent` class with inbox, main loop thread, active tool tracking, and event bus integration. Tools run synchronously (blocking). When the LLM emits multiple tool calls in a single response and all are in `_PARALLEL_SAFE_TOOLS`, they execute concurrently via ThreadPoolExecutor; otherwise sequentially.
- **Sub-agent actors**: `EnvoyAgent`, `VizAgent`, `DataOpsActor`, `DataIOAgent` — each extends `BaseAgent` with specialized prompt builders and tool schemas.
- **Delegation**: Delegation tools (`delegate_to_envoy`, `delegate_to_viz`, etc.) use `_get_or_create_*_agent()` + `_delegate_to_sub_agent()`. Agents persist across delegations, preserving LLM context.
- **Serialization**: Each actor has one thread reading from its inbox — multiple requests to the same mission actor queue naturally without locks.

### LLM-Driven Routing (`agent/core.py`, `agent/envoy_agent.py`, `agent/data_ops_agent.py`, `agent/viz_agent.py`)
- **Routing**: The OrchestratorAgent (LLM) decides whether to handle a request directly or delegate via `delegate_to_envoy` (fetching), `delegate_to_data_ops` (computation), `delegate_to_data_io` (text-to-DataFrame, file import), `delegate_to_viz` (visualization), tools. The orchestrator uses the intrinsic `vision` tool directly for plot analysis. The orchestrator can discover envoy capabilities via `envoy_query` (list, navigate, search) before delegating. No regex-based routing — the LLM uses conversation context to decide.
- **Mission sub-agents**: No envoys currently registered. When envoys are added via `manage_envoy`, each spacecraft gets a data fetching specialist. Agents are cached per session. Sub-agents have **fetch-only tools** (discovery, data_ops_fetch, conversation) — no compute, plot, or routing tools.
- **DataOps sub-agent**: Data transformation specialist with `custom_operation`, `describe_data`, `save_data` + `list_fetched_data`. System prompt includes computation patterns and code guidelines. Singleton, cached per session.
- **DataIO sub-agent**: Text-to-DataFrame and file import specialist with `store_dataframe`, `load_file`, `read_document`, `ask_clarification` + `list_fetched_data`. System prompt includes extraction patterns, DataFrame creation guidelines, and file loading workflow. Singleton, cached per session.
- **Visualization sub-agent**: Visualization specialist with `render_plotly_json` + `manage_plot` + `list_fetched_data` tools. Owns all visualization operations (plotting, export, reset, zoom, traces). The LLM provides Plotly figure JSON with `data_label` placeholders, and the system fills in actual data arrays.
- **Tool separation**: Tools have a `category` field (`discovery`, `visualization`, `data_ops`, `data_ops_fetch`, `data_ops_compute`, `data_extraction`, `function_docs`, `conversation`, `routing`, `document`, `data_export`, `memory`, `web_search`, `pipeline`, `pipeline_ops`). `get_tool_schemas(categories=..., extra_names=...)` filters tools by category. Orchestrator sees `["web_search", "conversation", "routing", "document", "memory", "data_export", "pipeline", "pipeline_ops"]` + `list_fetched_data`, `envoy_query` extras (discovery tools are envoy-internal only). EnvoyAgent sees `["discovery", "data_ops_fetch", "conversation"]` + `list_fetched_data` extra. DataOpsActor sees `["data_ops_compute", "conversation"]` + `list_fetched_data`, `search_function_docs`, `get_function_docs` extras. DataIOAgent sees `["data_extraction", "document", "conversation"]` + `list_fetched_data` extra. VizAgent sees tools per active backend: Plotly → `render_plotly_json` + `manage_plot` + `list_fetched_data`; matplotlib → `generate_mpl_script` + `manage_mpl_output` + `list_fetched_data`; JSX → `generate_jsx_component` + `manage_jsx_output` + `list_fetched_data`.
- **Post-delegation flow**: After `delegate_to_envoy` returns data labels, the orchestrator uses `delegate_to_data_ops` for computation, `delegate_to_data_io` for text-to-DataFrame conversion or file import, `delegate_to_viz` to visualize results, and optionally `vision` for scientific interpretation of the rendered plot.
- **Slim orchestrator**: System prompt contains orchestrator rules, error recovery patterns, and delegation instructions.
- **Gemini context caching**: When using Gemini, the orchestrator creates an explicit context cache containing the full system prompt (with mission catalog) + tool schemas (~38K tokens). This exceeds the 32K threshold for Gemini's cached content API, giving a 75% discount on cached input tokens. The cache is created once per session (24h TTL). Non-Gemini providers use the slim prompt without caching.
- **Per-agent session history** (`ctx:` tags): Sub-agents get fresh blank chats per delegation and have no awareness of prior session activity. To fix this, the EventBus `DEFAULT_TAGS` registry includes `ctx:mission`, `ctx:viz`, `ctx:dataops`, `ctx:planner`, and `ctx:orchestrator` tags on relevant event types (data fetches, computes, renders, errors, sub-agent tool calls). At delegation time, `_build_agent_history(agent_type)` queries `get_events(tags={"ctx:{type}"})`, formats events into concise one-line summaries, and injects the result as "Session history (what happened earlier)" before the memory context. This gives agents awareness of prior fetches, failed operations, and rendered plots without replaying the full conversation. The `ctx:mission` tag covers all mission agents (one shared history for cross-mission comparison scenarios). The `ctx:viz` and `ctx:dataops` tags filter `SUB_AGENT_TOOL`/`SUB_AGENT_ERROR` events by agent name to show only that agent's prior tool calls. The `ctx:orchestrator` tag (on 9 event types: `SUB_AGENT_TOOL`, `SUB_AGENT_ERROR`, `DATA_FETCHED`, `DATA_COMPUTED`, `RENDER_EXECUTED`, `CUSTOM_OP_FAILURE`, `FETCH_ERROR`, `RENDER_ERROR`, `PLOT_ACTION`) gives the orchestrator and planner a terse status-only view of sub-agent activity (e.g. `[PSP_Agent] fetch_data: ok`, `Fetched: ACE.Bmag`). `DELEGATION`/`DELEGATION_DONE` are excluded from `ctx:orchestrator` because the orchestrator sees these as its own tool calls in chat history.
- **Message-level context compaction**: For client-side history adapters (Anthropic, OpenAI Chat Completions), the full message list is resent every call. When estimated context tokens reach 80% of the model's context window, the `_check_and_compact()` method triggers compaction:
  1. Context windows are discovered via the [litellm community registry](https://github.com/BerriAI/litellm) (`model_prices_and_context_window.json`), cached locally in `~/.xhelio/model_context_windows.json` (24h TTL). Falls back to hardcoded `MODEL_CONTEXT_LIMITS` in `agent/llm_utils.py`.
  2. Token estimation uses the Gemini sentencepiece tokenizer (`agent/token_counter.py`) for all providers.
  3. Compaction summarizes older messages via a one-shot `adapter.generate()` call (temperature 0.1), keeping the last 3 complete turns intact. Thinking blocks are dropped from older turns. Tool-use/tool-result pairs are never split.
  4. Server-side adapters (Gemini Interactions API, OpenAI Responses API) skip compaction — they have 1M+ windows or built-in compaction.
  5. Emits `CONTEXT_COMPACTION` event with before/after token counts and context window size.
- **Injection points**: Sub-agents (mission, viz, dataops) are injected at delegation time (6 existing injection points). The orchestrator is injected in `process_message()` after followup context. The planner is injected in `_handle_planning_request()` after time range injection.

### Multi-Step Requests (Planning)
- Simple requests are handled by the orchestrator's conversation loop (up to 10 iterations, with consecutive delegation error guard)
- Multi-mission comparisons use sequential `delegate_to_envoy` calls for each mission, then `delegate_to_viz` — all in one `process_message` call
- Complex requests use **planning** via the **PlannerAgent**:
  1. **Regex pre-filter**: `is_complex_request()` regex heuristics catch obvious complex cases (free, no API cost) and route directly to planner
  2. **Orchestrator override**: The orchestrator (with HIGH thinking) can also call `delegate_to_planner` tool for complex cases the regex missed
  3. PlannerAgent runs **research phase** (tool-calling to search/browse datasets) then **planning phase** (JSON-schema-enforced `produce_plan` call)
  4. Planner returns a **complete plan** with all tasks (fetch, compute, visualization)
  5. Orchestrator **executes** the plan by calling delegation tools: `delegate_to_envoy` for fetch tasks, `delegate_to_data_ops` for compute, `delegate_to_viz` for visualization
  6. Fetch tasks execute in **parallel** (multiple `delegate_to_envoy` calls in one response)
  7. Compute and visualization tasks execute **after** fetches complete (dependency order)
  8. If the planner fails, falls back to direct orchestrator execution
  9. See planning flow details in `knowledge/prompts/planner/` (planner prompt sections)
- Tasks are tagged with `mission="__visualization__"` for visualization dispatch, `mission="__data_ops__"` for compute dispatch, `mission="__data_io__"` for text-to-DataFrame/file-import dispatch

### Thinking Levels
- Controlled via `create_chat(thinking="high"|"low"|"default")` in the adapter layer
- **HIGH**: Orchestrator (`agent/core.py`) and PlannerAgent (`agent/planner.py`) — deep reasoning for routing decisions and plan decomposition
- **LOW**: EnvoyAgent, VizAgent, DataOpsActor, DataIOAgent, InsightActor — fast execution with minimal thinking overhead
- **OFF**: Inline tier (follow-up suggestions, ghost text autocomplete) — cheapest/fastest model, no thinking
- Thinking tokens tracked separately in `get_token_usage()` across all agents
- Verbose mode logs full thoughts to terminal/file, plus 500-char tagged previews for web UI via `agent/thinking.py` utilities
- Task plans persist to `~/.xhelio/tasks/` with round tracking for multi-round plans

### Async Delegation (`agent/async_delegation.py`)
- **Config**: `reasoning.async_delegation` (default: `false`)
- When enabled, eligible `delegate_to_*` calls launch sub-agents on daemon threads and return immediately with `{"status": "pending_async"}`
- **Eligible tools**: `delegate_to_envoy`, `delegate_to_data_ops`, `delegate_to_data_io`
- **Not eligible** (shared state): `delegate_to_viz` (PlotlyRenderer)
- **Freeze/wake pattern**: If the LLM produces no tool calls but async delegations are pending, the orchestrator freezes (zero LLM cost) until at least one completes, then wakes with results
- `DelegationManager` coordinates via `threading.Condition` for efficient blocking
- Each completed delegation includes a **structured operation log** built from EventBus events (tool calls, fetch results, errors) — the orchestrator sees step-by-step what the sub-agent did
- Thread-local `_active_agent_name` and `_current_agent_type` via `threading.local()` ensure concurrent sub-agents don't interfere with each other's identity tracking

### Long-term Memory (`agent/memory.py`)
- Cross-session memory that persists user preferences, session summaries, operational pitfalls, and reflections
- Storage: `~/.xhelio/memory.json` (schema v7) — global, not per-session
- Four memory types:
  - `"preference"` — plot styles, spacecraft of interest, workflow habits
  - `"summary"` — what was analyzed in each session
  - `"pitfall"` — operational lessons learned (e.g., data gaps, fill values)
  - `"reflection"` — procedural knowledge learned from errors (Reflexion pattern)
- Memory fields: `tags` (keyword list for search/dedup), `source` (extracted/reflected/user_explicit/consolidated), `access_count`, `last_accessed`, `supersedes`, `version`, `archived`, `review` (single review dict or None)
- **Review system** (v7): Sub-agents review injected memories via the `review_memory` tool after completing their main task. The tool has structured parameters (`stars`, `rating`, `criticism`, `suggestion`, `comment`) — each is a required field, enforced server-side. Reviews are stored as separate Memory entries with `type="review"` and `review_of` linking to the target. Star meanings: 5=prevented mistake, 4=useful context, 3=relevant but no impact, 2=irrelevant, 1=misleading. When a memory is edited (supersede), old version keeps its reviews, new version starts fresh.
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
- **Pipeline registration**: Stubbed pending DAG-native reimplementation.
- **Two-phase consolidation** (conservative policy — when memory count exceeds budget):
  - Phase A — Rule-based pre-filter (no LLM): archives excess summaries, low-confidence (<0.3), old unaccessed (>30 days, access_count=0), tag-overlap dedup (≥80% Jaccard)
  - Phase B — Per-group LLM merge: groups remaining memories by (type, frozenset(scopes)), sends over-budget groups to LLM for merging
  - Conservative: never merges memories that represent distinct knowledge; 10x memory token budget (100k tokens) to reduce pressure
  - Per-type budgets: preferences 15, summaries 10, pitfalls 15, reflections 10
- All exceptions caught — never breaks the main agent flow


- **VersionedStore** (`agent/versioned_store.py`): Base class providing versioned JSON persistence with schema migration, used by MemoryStore.

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
- Fail-open: if metadata call fails, proceeds without validation

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
- Currently minimal — no mission data is bundled
- Will be re-enabled when MCP-backed envoys are registered

### Empty Session Auto-Cleanup
- On startup, `SessionManager` auto-removes sessions with no chat history and no stored data
- Prevents clutter from abandoned or crashed sessions
- Session save is skipped when there's nothing to persist

### First-Run Setup
- No mission data download required — envoys are registered dynamically at runtime via `manage_envoy`

### Web Search (`google_search` tool)
Each provider implements `adapter.web_search()` using its native search capability:
- **Gemini**: Google Search grounding API via an isolated Gemini API call (enabled by default)
- **OpenAI**: `gpt-4o-search-preview` model (native OpenAI only, disabled by default)
- **Anthropic**: `web_search_20250305` tool (native, disabled by default)
- **MiniMax**: MCP `web_search` via `minimax-coding-plan-mcp` subprocess (enabled by default)
- **No search backend available**: Warning logged, error returned to LLM — agent continues without search
- Returns grounded text with source URLs
- Search results can be turned into plottable datasets via the `store_dataframe` tool (google_search → delegate_to_data_io → store_dataframe → plot)

## Configuration

**`.env`** at project root (secret only):
```
GOOGLE_API_KEY=<gemini-api-key>
OPENAI_API_KEY=<openai-api-key>        # Optional — for OpenAI provider
ANTHROPIC_API_KEY=<anthropic-api-key>  # Optional — for Anthropic provider
MINIMAX_API_KEY=<minimax-api-key>      # Optional — for MiniMax provider
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
      "inline_model": "gemini-2.5-flash-lite"
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
- Color-coded tool events by category (data, compute, viz, catalog)
- Expandable token usage with per-agent breakdown and data store stats
- Export session to Markdown with figure thumbnails (dropdown: embed as base64 or local path)

**Data Tools Page** (`/data`):
- Catalog Browser: mission → dataset → parameter cascade selectors with time range
- Direct data fetch into session DataStore
- Data table with auto-refresh (4s interval)
- Data preview (head+tail rows with numeric rounding)
- Memory manager: global toggle, type-tagged list, delete selected, clear all

**Pipeline Page** (`/pipeline`):
- Session selector (sessions with pipeline.json)
- Interactive DAG visualization via Plotly
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
| `/fork` | Fork session with independent history and fresh agents |
| `/branch` | Fork into a new session branch (shares server-side history) |

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
python -m pytest tests/                                          # All tests
```

## Dependencies

```
google-genai>=1.60.0    # Gemini API (via agent/llm/gemini_adapter.py)
python-dotenv>=1.0.0    # .env loading
requests>=2.28.0        # HTTP calls
cdflib>=1.3.0           # CDF file reading
numpy>=1.24.0           # Array operations
scipy>=1.10.0           # Signal processing, FFT, interpolation, statistics
PyWavelets>=1.8.0       # Wavelet transforms (CWT, DWT, packets)
pandas>=2.0.0           # DataFrame-based data pipeline
plotly>=5.18.0          # Interactive scientific data visualization
kaleido>=0.2.1          # Static image export for Plotly (PNG, PDF)
matplotlib>=3.7.0       # Legacy plotting (unused in main pipeline)
tqdm>=4.60.0            # Progress bars for bootstrap/data downloads
pytest>=7.0.0           # Test framework
minimax-coding-plan-mcp # MiniMax MCP server for web search + image understanding (via uvx)
mcp>=1.26.0             # MCP server (stdio transport for Claude Desktop / Claude Code)
```
