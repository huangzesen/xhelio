# XHelio — Explore Heliosphere!

> Talk to NASA's spacecraft data. Powered by Google Gemini.

An autonomous AI agent that replaces complex analysis scripts with a single conversation. Ask for spacecraft data in plain English — the agent navigates NASA's heliophysics archive (70+ missions, 3,000+ datasets, decades of observations), handles the entire data pipeline, and produces interactive visualizations on demand. It eliminates the tooling barrier that normally takes months to overcome: opaque dataset IDs, mission-specific naming conventions, data access protocols, coordinate systems, unit conversions, and multi-panel plot boilerplate.

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for the SolidJS frontend)
- A Google Gemini API key

### Setup

```bash
git clone https://github.com/huangzesen/xhelio.git
cd xhelio
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

On first run, you'll be prompted for an API key. Mission data (70+ spacecraft profiles with dataset IDs, parameter metadata, and instrument groupings) is included in the repository — no download step required.

### Run (Web UI — recommended)

```bash
xhelio
```

This launches both the FastAPI backend (`:8000`) and the SolidJS frontend (`:5173`), installing frontend npm dependencies automatically on first run. Opens **http://localhost:5173** in your browser. Press `Ctrl+C` to stop both.

The web UI provides five pages:
- **Chat** — Conversational interface with streaming responses, interactive Plotly plots, follow-up suggestions, and example prompts
- **Data Tools** — Dataset catalog browser, data table preview, and memory manager
- **Pipeline** — Pipeline explorer for replaying saved sessions
- **Gallery** — Saved data products from pipeline sessions
- **Settings** — Model configuration, search settings, and memory limits

### Other modes

```bash
xhelio serve                    # Backend only (uvicorn on :8000)
xhelio cli                      # Text CLI (interactive agent)
xhelio cli --verbose            # Show tool calls and timing
xhelio mcp                      # MCP server over stdio
```

## Features

- **70+ spacecraft, 3,000+ datasets** — Parker Solar Probe, Solar Orbiter, ACE, MMS, Wind, DSCOVR, STEREO, Cluster, THEMIS, Van Allen Probes, GOES, Voyager, Cassini, Juno, MAVEN, Ulysses, and more (54 CDAWeb + 17 PDS PPI missions)
- **Full CDAWeb catalog search** — access any of NASA's 2,000+ heliophysics datasets, not just pre-configured ones
- **Google Gemini** — powered by Gemini models with function calling, extended thinking, and multimodal vision
- **Autonomous multi-step planning** — complex requests decomposed into task batches with dynamic replanning
- **Physics-aware computation** — magnitude, Alfven speed, plasma beta, spectral analysis, wavelets, smoothing, resampling, derivatives — the LLM writes the code in an AST-validated sandbox
- **SPICE ephemeris** — spacecraft positions, trajectories, coordinate transforms via the `heliospice` package
- **Cross-mission comparison** — automatically handles different dataset naming conventions, coordinate systems, and cadences
- **Web search** — find real-time space weather events, ICME catalogs, solar flare lists and feed them into the analysis pipeline (Google Search grounding)
- **Document ingestion** — upload PDFs or images of data tables, extract structured data via LLM vision
- **Multimodal plot analysis** — InsightAgent analyzes rendered plots via LLM vision to provide scientific interpretation (feature identification, data quality assessment, coordinate system awareness)
- **Long-term memory with consumer-side review** — learns preferences, pitfalls, and scientific discoveries across sessions; consuming agents rate each injected memory after task completion (see [Memory System](#memory-system) below)
- **Interactive Plotly plots** — zoom, pan, hover tooltips, multi-panel subplots, WebGL for large datasets
- **Pipeline DAG** — inspect, modify, and re-execute data pipelines with staleness tracking
- **Saved pipelines** — extract replayable workflows from sessions, parameterized by time range, with family-based dedup and searchable index
- **Session persistence** — auto-saves every turn, resume with `--continue`
- **PNG/PDF export** — publication-ready static images via kaleido

## Example Workflows

### Basic: Fetch and visualize

> "Show me Parker Solar Probe magnetic field data for its closest perihelion in 2024"

The agent knows PSP's dataset naming convention, finds the right time window, fetches RTN magnetic field components, and plots an interactive 3-component time series.

### Intermediate: Compute derived quantities

> "Fetch ACE magnetic field and solar wind plasma data for last month. Compute the magnetic field magnitude, the Alfven speed, and the plasma beta. Plot everything on separate panels."

Six autonomous steps: two data fetches (different instruments), three physics computations (each requiring the correct formula and unit handling), and a multi-panel Plotly figure.

### Advanced: Cross-mission event analysis

> "What were the major geomagnetic storms in 2024? For the strongest one, compare solar wind conditions at ACE and Wind, showing magnetic field magnitude and proton density on aligned time axes."

The agent searches the web for storm catalogs, identifies the May 2024 event, fetches data from two spacecraft (different dataset IDs, different parameter names), computes magnitudes, aligns time axes, and produces a publication-ready comparison plot — all autonomously through the PlannerAgent's replan loop.

### Research: Document-driven analysis

> [Upload a PDF table of ICME events from Richardson & Cane catalog]
> "Extract the events from 2023-2024 and plot their transit speeds as a time series"

LLM vision reads the PDF, the DataExtractionActor converts the table to a structured DataFrame, and the VizPlotlyActor renders the result.

## Architecture

Nine specialized agents, each with domain-specific tools and system prompts:

```
User Request
    |
    v
OrchestratorAgent (Gemini, HIGH thinking)
    |--- Routing table: 70+ missions × instrument types → specialist selection
    |--- Activates PlannerAgent for complex multi-step requests
    |
    +---> MissionActor (per-spacecraft)      Knows dataset IDs, parameter names,
    |                                        coordinate systems, time ranges
    +---> DataOpsActor                      Writes pandas/numpy/scipy/pywt code
    |                                        in an AST-validated sandbox
    +---> DataExtractionActor               Converts search results, PDFs,
    |                                        event catalogs into DataFrames
    +---> VizPlotlyActor                          Interactive Plotly figures with
    |                                        domain-appropriate defaults
    +---> InsightAgent                      Analyzes rendered plots via
    |                                        LLM vision (multimodal)
    +---> PlannerAgent                      Decomposes, executes, observes,
    |                                        replans (up to 5 rounds)
    +---> MemoryAgent                       Extracts preferences, pitfalls,
                                             and summaries across sessions
```

**Data pipeline:** Natural language &rarr; dataset discovery &rarr; data fetch (CDF/PDS) &rarr; pandas DataFrame &rarr; LLM-generated computation &rarr; interactive Plotly plot

**Key design decisions:**
- **LLM-driven routing** — the orchestrator uses conversation context and a routing table to decide which specialist handles each request. No regex dispatching.
- **Code generation sandbox** — the LLM writes pandas/numpy/scipy/pywt code for data transformations. All generated code is AST-validated before execution (blocks imports, exec/eval, os/sys access). Visualization uses 2 declarative tools (`render_plotly_json`, `manage_plot`) — no free-form code generation for plots.
- **Per-mission knowledge base** — 70+ auto-generated mission JSON files (54 CDAWeb + 17 PPI) with instrument groupings, dataset IDs, parameter metadata, and time ranges. CDAWeb and PPI missions with matching stems are deep-merged at load time. The orchestrator sees only a routing table; sub-agents receive rich domain-specific prompts with recommended datasets and analysis patterns.
- **Three-tier caching** — metadata: memory → local JSON file → Master CDF download. Mission data ships pre-built in the repository; new datasets are auto-cached on first access.

## Project Structure

```
agent/                  Core agent layer (9 agents + 46 tools)
  core.py                 OrchestratorAgent — routes, dispatches, plans
  mission_agent.py        MissionActor — per-spacecraft data fetching
  data_ops_agent.py       DataOpsActor — pandas/numpy/scipy/pywt computation
  data_extraction_agent.py DataExtractionActor — text/PDF to DataFrames
  viz_plotly_agent.py  VizPlotlyActor — Plotly rendering
  insight_agent.py        InsightAgent — multimodal plot analysis via LLM vision
  planner.py              PlannerAgent — plan-execute-replan loop
  memory_agent.py         MemoryAgent — cross-session memory extraction
  tools.py                46 tool schemas (categories defined in tool_catalog.py)
  session.py              Session persistence (auto-save every turn)
  memory.py               Long-term memory storage and management
  llm/                    LLM abstraction layer (Gemini)

knowledge/              70+ mission knowledge base + prompt generation
  missions/cdaweb/*.json  54 CDAWeb mission profiles (auto-generated)
  missions/ppi/*.json     17 PDS PPI mission profiles (auto-generated)
  catalog.py              Mission catalog with keyword search
  catalog_search.py       Full CDAWeb + PPI catalog search
  mission_loader.py       Lazy-loading cache and routing table
  prompt_builder.py       Dynamic system prompts per agent type
  metadata_client.py      3-layer metadata cache (memory → file → Master CDF)
  bootstrap.py            Mission data auto-download and refresh

data_ops/               pandas-backed data pipeline
  fetch.py                Data fetching (CDF/PDS) → DataFrames
  store.py                In-memory DataStore singleton
  custom_ops.py           AST-validated sandbox for LLM-generated code
  pipeline.py             Pipeline DAG + SavedPipeline (replayable workflows)

rendering/              Plotly visualization engine
  plotly_renderer.py      Multi-panel figures, WebGL, PNG/PDF export
  registry.py             2 declarative visualization tools (render_plotly_json, manage_plot)

api/                    FastAPI backend (routes, session manager, SSE streaming)
frontend/               SolidJS frontend (Vite + TypeScript + Tailwind)

xhelio_cli.py           CLI entry point (xhelio command)
api_server.py           FastAPI server entry point
mcp_server.py           MCP server over stdio
```

## Mission Data

The repository ships with pre-built mission data for 70+ spacecraft (3,456 JSON files covering 54 CDAWeb and 17 PDS PPI missions). This includes:

- **Mission profiles** — instrument groupings, dataset IDs, time ranges, and recommended analysis patterns for each spacecraft
- **Parameter metadata** — variable names, descriptions, units, and dimensions for every dataset, cached from CDAWeb Master CDF files

Users get a working system immediately after cloning — no first-run download step. When the agent encounters a dataset not yet in the cache, `metadata_client.py` fetches its metadata from CDAWeb on demand and saves it locally for future use (lazy auto-cache).

### Updating the mission cache

After regenerating mission data (via `scripts/generate_mission_data.py`), update the cache archive that the public mirror sync uses:

```bash
cd knowledge/missions
COPYFILE_DISABLE=1 tar czf /tmp/missions-cache.tar.gz .
gh release upload missions-cache /tmp/missions-cache.tar.gz --clobber --repo huangzesen/xhelio-dev
```

The CI workflow (`.github/workflows/sync-public.yml`) downloads this archive and includes it in the public mirror. The `COPYFILE_DISABLE=1` flag prevents macOS resource fork files from being included in the tarball.

## Configuration

API keys are stored in `.env` at the project root:

| Environment Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | (required) | Gemini API key |

All other settings are in `~/.xhelio/config.json` (see `config.template.json` for all options):

| Setting | Default | Description |
|---|---|---|
| `providers.gemini.model` | `"gemini-3-flash"` | Model for orchestrator + planner |
| `providers.gemini.sub_agent_model` | `"gemini-3-flash"` | Model for sub-agents |
| `providers.gemini.insight_model` | `"gemini-2.5-flash"` | Model for InsightAgent (multimodal plot analysis) |
| `providers.gemini.inline_model` | `"gemini-2.5-flash-lite"` | Model for follow-ups, autocomplete |
| `providers.gemini.planner_model` | `"gemini-3-flash-preview"` | Model for PlannerAgent |
| `providers.gemini.fallback_model` | `"gemini-2.5-flash"` | Fallback when primary model fails |

## Memory System

XHelio features a long-term memory system that persists operational knowledge across sessions. The system has three phases — **extraction**, **injection**, and **review** — forming a closed feedback loop where consuming agents evaluate the memories they receive.

### Memory Types

Four operational memory types, each with a structured content format:

| Type | Purpose | Format |
|------|---------|--------|
| **Preference** | User habits and style choices | 1-2 sentences (freeform) |
| **Pitfall** | Lessons from past mistakes | `Trigger:` / `Problem:` / `Fix:` |
| **Reflection** | Procedural insights from errors | `Trigger:` / `Problem:` / `Fix:` |
| **Summary** | Session analysis records | `Data:` / `Analysis:` / `Finding:` |

A fifth type, **review**, stores consumer feedback on other memories (see [Consumer-Side Memory Review](#consumer-side-memory-review) below).

### Multi-Scope Injection

Each memory belongs to one or more scopes: `generic`, `mission:<ID>` (e.g., `mission:PSP`), `visualization`, or `data_ops`. When a sub-agent is activated, only memories matching its scope are injected:

```
MissionActor[PSP]      ← receives memories scoped to "mission:PSP" + "generic"
VizPlotlyActor               ← receives memories scoped to "visualization" + "generic"
DataOpsActor           ← receives memories scoped to "data_ops" + "generic"
OrchestratorAgent      ← receives all "generic" memories + session summaries
```

Injection is controlled by a global token budget (default 100,000 tokens) to prevent prompt explosion. Memories are added in priority order (preferences, summaries, pitfalls, reflections) until the budget is exhausted.

### Extraction

The **MemoryAgent** runs periodically (configurable interval, default every 2 user turns). It receives curated events from the session's EventBus — user messages, agent responses, data fetches, computation results, errors — and outputs structured JSON actions:

- **add** — create a new memory with type, scopes, and structured content
- **edit** — archive the old version and create a new one (version chain via `supersedes` field)
- **drop** — archive a memory that is no longer relevant

### Consumer-Side Memory Review

After completing a task, each consuming agent is required to review the memories it was given by calling `review_memory(memory_id, stars, comment)`. This is a closed feedback loop where the **consumer** — not the creator — evaluates each memory's usefulness in context.

**How it differs from existing approaches:**

Most AI memory systems use **write-time importance scoring** (the creator rates a memory when it's created, e.g., [Generative Agents, Park et al. 2023](https://arxiv.org/abs/2304.03442)) or **passive decay signals** (access frequency and recency, e.g., [FadeMem](https://arxiv.org/abs/2601.18642), [Mem0](https://arxiv.org/html/2504.19413v1)). XHelio's approach is different:

1. **Read-time evaluation** — the agent that *uses* a memory rates it after the task, not the agent that *created* it. A pitfall about NaN handling might be rated 5 stars by DataOpsActor but 1 star by VizPlotlyActor.

2. **Per-agent reviews** — each agent type gets its own review for the same memory. A MissionActor and a DataOpsActor can independently rate the same pitfall, and both reviews coexist.

3. **Structured feedback** — reviews include a star rating (1-5) plus a structured comment with labeled sections:
   - *Rating*: why this star count
   - *Criticism*: what's wrong or could be better
   - *Suggestion*: how to improve the memory's content or scope
   - *Comment*: any extra observation

4. **Version-controlled ratings** — reviews are stored as Memory entries (type `review`, linked via `review_of` field). When an agent re-reviews a memory, the old review is archived and a new one supersedes it, forming a version chain with timestamps — enabling time-series analysis of how a memory's perceived usefulness evolves.

**Star scale:**

| Stars | Meaning |
|-------|---------|
| 5 | Directly prevented a mistake |
| 4 | Useful context for the task |
| 3 | Relevant but no measurable impact |
| 2 | Irrelevant to this task |
| 1 | Actively misleading |

### Persistence

- All memories are stored in `~/.xhelio/memory.json` (single file, schema version 7)
- Every save is auto-committed to a local git repository (`~/.xhelio/.git`), providing full change history
- Archived memories (old versions, dropped entries, superseded reviews) remain in the file with `archived: true` for audit trails
- Schema migrations (v1 through v7) run automatically on load

## Tech Stack

- **Google Gemini** — function calling, extended thinking, structured output, and multimodal vision
- **FastAPI + SSE** — backend API with server-sent events for streaming
- **SolidJS + TypeScript + Vite** — frontend with Tailwind CSS
- **Plotly** — interactive scientific data visualization with WebGL
- **pandas / numpy / scipy / PyWavelets** — data pipeline, computation, spectral analysis, wavelets
- **heliospice** — SPICE ephemeris for spacecraft positions and coordinate transforms
- **CDAWeb + PDS PPI** — NASA data archives (CDF file downloads, PDS file archive)
- **kaleido** — static image export (PNG, PDF)

## License

MIT
