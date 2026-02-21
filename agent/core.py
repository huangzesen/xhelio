"""
Core agent logic - orchestrates Gemini calls and tool execution.

The OrchestratorAgent routes requests to:
- MissionAgent sub-agents for data operations (per spacecraft)
- VisualizationAgent sub-agent for all visualization
"""

import contextvars
import math
import time
import threading
from datetime import datetime
import pandas as pd
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Optional

import config
from config import get_data_dir, get_api_key
from .llm import LLMAdapter, GeminiAdapter, OpenAIAdapter, AnthropicAdapter, LLMResponse, FunctionSchema
from .tools import get_tool_schemas
from .prompts import get_system_prompt
from .time_utils import parse_time_range, TimeRangeError
from .tasks import (
    Task, TaskPlan, TaskStatus, PlanStatus,
    get_task_store, create_task, create_plan,
)
from .planner import PlannerAgent, format_plan_for_display, MAX_ROUNDS
from .session import SessionManager
from .memory import MemoryStore, estimate_tokens
from .mission_agent import MissionAgent
from .visualization_agent import VisualizationAgent
from .data_ops_agent import DataOpsAgent
from .data_extraction_agent import DataExtractionAgent
from .logging import (
    setup_logging, attach_log_file, get_logger, log_error, log_tool_call,
    log_tool_result, log_plan_event, log_session_end,
    set_session_id, tagged, LOG_DIR,
)
from .event_bus import (
    EventBus, get_event_bus, set_event_bus,
    DebugLogListener, SSEEventListener, OperationsLogListener, DisplayLogBuilder,
    EventLogWriter, TokenLogListener, load_event_log,
    # Event types
    USER_MESSAGE, AGENT_RESPONSE, TOOL_CALL, TOOL_RESULT,
    DATA_FETCHED, DATA_COMPUTED, DATA_CREATED, RENDER_EXECUTED, PLOT_ACTION,
    DELEGATION, DELEGATION_DONE, SUB_AGENT_TOOL, SUB_AGENT_ERROR,
    PLAN_CREATED, PLAN_TASK, PLAN_COMPLETED, PROGRESS,
    THINKING, LLM_CALL, LLM_RESPONSE,
    FETCH_ERROR, HIGH_NAN, CUSTOM_OP_FAILURE, RECOVERY, RENDER_ERROR, TOOL_ERROR,
    SESSION_START, SESSION_END, DEBUG,
    MEMORY_EXTRACTION_START, MEMORY_EXTRACTION_DONE, MEMORY_EXTRACTION_ERROR,
    STM_COMPACTION,
)
from .memory_agent import MemoryAgent, MemoryContext
from .discovery_store import DiscoveryStore
from .discovery_agent import DiscoveryAgent

from .loop_guard import LoopGuard, make_call_key
from .model_fallback import activate_fallback, get_active_model, is_quota_error
from .base_agent import (
    _LLM_WARN_INTERVAL, _LLM_RETRY_TIMEOUT, _LLM_MAX_RETRIES,
    send_with_timeout, _CancelledDuringLLM,
    execute_tools_batch, track_llm_usage,
)
from rendering.registry import get_method
from rendering.plotly_renderer import PlotlyRenderer
from knowledge.catalog import search_by_keywords
from knowledge.catalog_search import search_catalog as search_full_catalog
from knowledge.metadata_client import (
    list_parameters,
    get_dataset_time_range,
    list_missions,
    validate_dataset_id,
    validate_parameter_id,
)
from data_ops.store import get_store, DataEntry, build_source_map, describe_sources
from data_ops.fetch import fetch_data
from data_ops.custom_ops import run_custom_operation, run_multi_source_operation, run_dataframe_creation
from data_ops.operations_log import get_operations_log

# Orchestrator sees discovery, web search, conversation, and routing tools
# (NOT data fetching or data_ops — handled by sub-agents)
ORCHESTRATOR_CATEGORIES = ["discovery", "web_search", "conversation", "routing", "document", "memory", "data_export", "spice", "pipeline"]
ORCHESTRATOR_EXTRA_TOOLS = ["list_fetched_data", "preview_data"]


# Roles that map to "user" or "assistant/model" across all LLM adapters
_USER_ROLES = {"user"}
_AGENT_ROLES = {"model", "assistant"}


def _extract_turns(history_entries: list, *, max_text: int = 300) -> list[str]:
    """Extract user/agent text turns from adapter-specific history formats.

    Works with Gemini (role="model", parts=[{text}]),
    OpenAI (role="assistant", content=str), and
    Anthropic (role="assistant", content=str|list).
    """
    turns = []
    for content in history_entries:
        if isinstance(content, dict):
            role = content.get("role", "")
        else:
            role = getattr(content, "role", "")

        if role in _USER_ROLES:
            label = "User"
        elif role in _AGENT_ROLES:
            label = "Agent"
        else:
            continue

        # Extract text: try Gemini-style parts first, then OpenAI/Anthropic content
        text = None
        if isinstance(content, dict):
            parts = content.get("parts")
            if parts:
                for part in parts:
                    t = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                    if t:
                        text = t
                        break
            if not text:
                c = content.get("content", "")
                if isinstance(c, str):
                    text = c
                elif isinstance(c, list):
                    for block in c:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text")
                            break
        else:
            parts = getattr(content, "parts", None) or []
            for part in parts:
                t = getattr(part, "text", None)
                if t:
                    text = t
                    break
            if not text:
                text = getattr(content, "content", None)
                if isinstance(text, list):
                    text = None

        if text:
            turns.append(f"{label}: {text[:max_text]}")
    return turns


def _create_adapter() -> LLMAdapter:
    """Create the LLM adapter based on config (llm_provider, llm_base_url, etc.)."""
    provider = config.LLM_PROVIDER.lower()
    api_key = get_api_key(provider)
    if provider == "openai":
        return OpenAIAdapter(api_key=api_key, base_url=config.LLM_BASE_URL)
    elif provider == "anthropic":
        return AnthropicAdapter(api_key=api_key)
    else:
        return GeminiAdapter(api_key=api_key)


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None for JSON safety.

    Gemini's API rejects function_response containing NaN or Inf values
    (400 INVALID_ARGUMENT). This ensures all tool results are safe.
    """
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


class OrchestratorAgent:
    """Main orchestrator agent that routes to mission and visualization sub-agents."""

    # Class-level fallbacks so tests using __new__ don't crash
    logger = get_logger()
    _event_bus = get_event_bus()

    def __init__(self, verbose: bool = False, gui_mode: bool = False, model: str | None = None, defer_chat: bool = False):
        """Initialize the orchestrator agent.

        Args:
            verbose: If True, print debug info about tool calls.
            gui_mode: If True, launch with visible GUI window.
            model: LLM model name (default: config.SMART_MODEL).
            defer_chat: If True, skip creating the initial chat session.
                Used for session resume where load_session() creates its own.
        """
        self.verbose = verbose
        self.gui_mode = gui_mode
        self.web_mode = False  # Set True by MCP server to suppress auto-open
        self._cancel_event = threading.Event()

        # Initialize logging (console-only until a session is started)
        self.logger = setup_logging(verbose=verbose)
        self._token_log_listener = None  # set by start_session() / load_session()
        # EventBus — single source of truth for all session activity
        self._event_bus = EventBus()
        set_event_bus(self._event_bus)
        self._debug_log_listener = DebugLogListener(self.logger)
        self._event_bus.subscribe(self._debug_log_listener)
        self._display_log_builder = DisplayLogBuilder()
        self._event_bus.subscribe(self._display_log_builder)
        self._event_log_writer: Optional[EventLogWriter] = None
        self._ops_log_listener = OperationsLogListener(get_operations_log)
        self._event_bus.subscribe(self._ops_log_listener)

        self._event_bus.emit(SESSION_START, level="info", msg="Initializing OrchestratorAgent")

        # Initialize LLM adapter (wraps all provider SDK calls)
        self.adapter: LLMAdapter = _create_adapter()

        # Build tool schemas for the orchestrator
        self._tool_schemas = [
            FunctionSchema(
                name=t["name"],
                description=t["description"],
                parameters=t["parameters"],
            )
            for t in get_tool_schemas(categories=ORCHESTRATOR_CATEGORIES, extra_names=ORCHESTRATOR_EXTRA_TOOLS)
        ]

        # Store model name and system prompt for chat creation
        self.model_name = model or config.SMART_MODEL
        self._system_prompt = get_system_prompt(gui_mode=gui_mode)

        # Explicit context caching — caches system prompt + tool schemas so they
        # are charged at 75% discount on every subsequent API call in this session.
        # Deferred until first API call (see _ensure_cache).
        self._cache_name = None
        self._cache_attempted = False

        if not defer_chat:
            self._ensure_cache()
            # Create chat session (use get_active_model in case fallback was already activated)
            self.chat = self.adapter.create_chat(
                model=get_active_model(self.model_name),
                system_prompt=self._system_prompt,
                tools=self._tool_schemas,
                thinking="high",
                cached_content=self._cache_name,
            )
        else:
            # Chat will be created by load_session() or _send_message()
            self.chat = None

        # Plotly renderer for visualization
        self._renderer = PlotlyRenderer(verbose=self.verbose, gui_mode=self.gui_mode)
        self._deferred_figure_state: Optional[dict] = None  # set by load_session() for lazy restore

        # Token usage tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_thinking_tokens = 0
        self._total_cached_tokens = 0
        self._api_calls = 0
        self._last_tool_context = "send_message"

        # Current plan being executed (if any)
        self._current_plan: Optional[TaskPlan] = None

        # SSE event listener (subscribed by api/routes.py when streaming)
        self._sse_listener: SSEEventListener | None = None

        # Cache of mission sub-agents, reused across requests in the session
        self._mission_agents: dict[str, MissionAgent] = {}
        self._mission_agents_lock = threading.Lock()

        # Cached visualization sub-agent
        self._viz_agent: Optional[VisualizationAgent] = None

        # Cached data ops sub-agent
        self._dataops_agent: Optional[DataOpsAgent] = None

        # Cached data extraction sub-agent
        self._data_extraction_agent: Optional[DataExtractionAgent] = None

        # Cached planner agent
        self._planner_agent: Optional[PlannerAgent] = None

        # Canonical time range for the current plan (reset after plan completes)
        self._plan_time_range: Optional['TimeRange'] = None

        # Session persistence
        self._session_id: Optional[str] = None
        self._session_manager = SessionManager()
        self._auto_save: bool = False

        # Long-term memory
        self._memory_store = MemoryStore()
        self._memory_agent: Optional[MemoryAgent] = None

        # Discovery memory (scientific knowledge from data exploration)
        self._discovery_store = DiscoveryStore()
        self._discovery_turn_counter = 0
        self._last_discovery_op_count = 0
        self._discovery_lock = threading.Lock()

        # Periodic memory extraction (mirrors discovery pattern)
        self._memory_turn_counter = 0
        self._last_memory_op_index = 0
        self._memory_lock = threading.Lock()

        # Track recent custom_operation failures for recovery detection
        self._recent_custom_op_failures: dict = {}

        # Cached pipeline DAG (invalidated when new ops are recorded)
        self._pipeline = None

        # LLM-compacted session history cache: {agent_type: (event_count, text)}
        self._compacted_history_cache: dict[str, tuple[int, str]] = {}

        # Thread pool for timeout-wrapped Gemini calls
        self._timeout_pool = ThreadPoolExecutor(max_workers=1)

    # ---- SSE event listener management ----

    def subscribe_sse(self, callback) -> SSEEventListener:
        """Subscribe an SSE callback to receive display events.

        Returns the SSEEventListener so it can be unsubscribed later.
        """
        listener = SSEEventListener(callback)
        self._sse_listener = listener
        self._event_bus.subscribe(listener)
        return listener

    def unsubscribe_sse(self):
        """Unsubscribe the current SSE listener, if any."""
        if self._sse_listener:
            self._event_bus.unsubscribe(self._sse_listener)
            self._sse_listener = None

    # ---- Cancellation API ----

    def request_cancel(self):
        """Signal the agent to stop after the current atomic operation."""
        self._cancel_event.set()
        self._event_bus.emit(DEBUG, level="info", msg="[Cancel] Cancellation requested")

    def clear_cancel(self):
        """Clear the cancellation flag (called at start of process_message)."""
        self._cancel_event.clear()

    def is_cancelled(self) -> bool:
        """Check whether cancellation has been requested."""
        return self._cancel_event.is_set()

    # ---- Pipeline DAG ----

    def _get_or_build_pipeline(self):
        """Return the cached Pipeline, building from OperationsLog if needed."""
        if self._pipeline is None:
            from data_ops.pipeline import Pipeline
            store = get_store()
            final_labels = set(store.list_entries())
            log = get_operations_log()
            self._pipeline = Pipeline.from_operations_log(log, final_labels)
        return self._pipeline

    def _invalidate_pipeline(self):
        """Invalidate the cached Pipeline (called after new ops are recorded)."""
        self._pipeline = None

    # ---- Parallel tool execution ----

    # Tools safe to run concurrently (I/O-bound, no shared mutable state conflicts)
    _PARALLEL_SAFE_TOOLS = {"fetch_data", "delegate_to_mission"}

    def _execute_tools_parallel(
        self, function_calls: list
    ) -> list[tuple[str | None, str, dict, dict]]:
        """Execute a batch of tool calls, parallelizing when safe."""
        from config import PARALLEL_FETCH, PARALLEL_MAX_WORKERS
        return execute_tools_batch(
            function_calls=function_calls,
            tool_executor=self._execute_tool_safe,
            parallel_safe_tools=self._PARALLEL_SAFE_TOOLS,
            parallel_enabled=PARALLEL_FETCH,
            max_workers=PARALLEL_MAX_WORKERS,
            agent_name="Orchestrator",
            logger=self.logger,
        )

    @property
    def memory_store(self) -> MemoryStore:
        """Return the long-term memory store (for web UI access)."""
        return self._memory_store

    def get_plotly_figure(self):
        """Return the current Plotly figure (or None)."""
        return self._renderer.get_figure()

    def _track_usage(self, response: LLMResponse):
        """Accumulate token usage from an LLMResponse."""
        token_state = {
            "input": self._total_input_tokens,
            "output": self._total_output_tokens,
            "thinking": self._total_thinking_tokens,
            "cached": self._total_cached_tokens,
            "api_calls": self._api_calls,
        }
        track_llm_usage(
            response=response,
            token_state=token_state,
            agent_name="OrchestratorAgent",
            last_tool_context=self._last_tool_context,
        )
        self._total_input_tokens = token_state["input"]
        self._total_output_tokens = token_state["output"]
        self._total_thinking_tokens = token_state["thinking"]
        self._total_cached_tokens = token_state["cached"]
        self._api_calls = token_state["api_calls"]

    def _ensure_cache(self):
        """Create Gemini context cache on first use (lazy).

        Called once — either in ``__init__`` (normal path) or on the first
        ``_send_message()`` call (resume path where ``defer_chat=True``).
        """
        if self._cache_attempted:
            return
        self._cache_attempted = True
        if not isinstance(self.adapter, GeminiAdapter):
            return
        try:
            self._cache_name = self.adapter.create_cache(
                model=self.model_name,
                system_prompt=self._system_prompt,
                tools=self._tool_schemas,
            )
            self._event_bus.emit(DEBUG, level="info", msg=f"Context cache created: {self._cache_name}")
        except Exception as exc:
            self._event_bus.emit(DEBUG, level="warning", msg=f"Cache creation failed: {exc}")
            self._cache_name = None

    def _send_message(self, message) -> LLMResponse:
        """Send a message on self.chat with timeout/retry and model fallback on 429."""
        # Lazily create context cache + chat on first API call (resume path)
        if not self._cache_attempted:
            self._ensure_cache()
            if self._cache_name and self.chat is not None:
                # Recreate chat with cache for cheaper subsequent calls
                try:
                    history = self.chat.get_history()
                except Exception:
                    history = None
                self.chat = self.adapter.create_chat(
                    model=self.model_name,
                    system_prompt=self._system_prompt,
                    tools=self._tool_schemas,
                    thinking="high",
                    cached_content=self._cache_name,
                    history=history,
                )

        try:
            return self._send_with_timeout(self.chat, message)
        except Exception as exc:
            if is_quota_error(exc, adapter=self.adapter) and config.FALLBACK_MODEL:
                activate_fallback(config.FALLBACK_MODEL)
                self._cache_name = None  # cache is model-specific, invalidate
                self.chat = self.adapter.create_chat(
                    model=config.FALLBACK_MODEL,
                    system_prompt=self._system_prompt,
                    tools=self._tool_schemas,
                    thinking="high",
                )
                self.model_name = config.FALLBACK_MODEL
                return self._send_with_timeout(self.chat, message)
            raise

    def _send_with_timeout(self, chat, message) -> LLMResponse:
        """Send a message to the LLM with periodic warnings and retry on timeout."""
        return send_with_timeout(
            chat=chat,
            message=message,
            timeout_pool=self._timeout_pool,
            cancel_event=self._cancel_event,
            retry_timeout=_LLM_RETRY_TIMEOUT,
            agent_name="Orchestrator",
            logger=self.logger,
        )

    def _extract_grounding_sources(self, response: LLMResponse) -> str:
        """Extract source citations from Google Search grounding metadata."""
        raw = response.raw
        if not raw or not getattr(raw, "candidates", None):
            return ""
        candidate = raw.candidates[0]
        meta = getattr(candidate, "grounding_metadata", None)
        if not meta:
            return ""
        chunks = getattr(meta, "grounding_chunks", None) or []
        sources = []
        seen = set()
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if web:
                uri = getattr(web, "uri", None)
                title = getattr(web, "title", None)
                if uri and uri not in seen:
                    seen.add(uri)
                    sources.append(f"- [{title or uri}]({uri})")
        if not sources:
            return ""
        return "\n\nSources:\n" + "\n".join(sources)

    def _log_grounding_queries(self, response: LLMResponse) -> None:
        """Log Google Search grounding queries from the raw response (if any)."""
        raw = response.raw
        if raw and getattr(raw, "candidates", None):
            meta = getattr(raw.candidates[0], "grounding_metadata", None)
            if meta and getattr(meta, "web_search_queries", None):
                self._event_bus.emit(DEBUG, level="debug", msg=f"[Search] Queries: {meta.web_search_queries}")

    def _google_search(self, query: str) -> dict:
        """Execute a web search query.

        Routes to Gemini grounded search (built-in Google Search) when using a
        Gemini adapter, otherwise falls back to Tavily web search.

        Args:
            query: The search query string.

        Returns:
            Dict with status, answer text, and source URLs.
        """
        if hasattr(self.adapter, "google_search"):
            return self._gemini_grounded_search(query)
        return self._tavily_search(query)

    def _gemini_grounded_search(self, query: str) -> dict:
        """Execute a Google Search query via Gemini's grounding API.

        The generateContent API does not support combining google_search with
        function_declarations in the same request.  This method makes an
        isolated call with only the GoogleSearch tool so Gemini can ground its
        response in real web results.
        """
        try:
            response = self.adapter.google_search(
                query=query,
                model=get_active_model(self.model_name),
            )
            self._last_tool_context = "google_search"
            self._track_usage(response)

            # Extract text
            text = response.text or ""

            # Extract sources from grounding metadata
            sources_text = self._extract_grounding_sources(response)

            raw = response.raw
            if raw and getattr(raw, "candidates", None):
                meta = getattr(raw.candidates[0], "grounding_metadata", None)
                if meta and getattr(meta, "web_search_queries", None):
                    self._event_bus.emit(DEBUG, level="debug", msg=f"[Search] Queries: {meta.web_search_queries}")

            return {
                "status": "success",
                "answer": text + sources_text,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Google Search failed: {e}",
            }

    def _tavily_search(self, query: str) -> dict:
        """Execute a web search query via Tavily API.

        Used as a fallback when the LLM provider does not have built-in web
        search (i.e. any non-Gemini provider).
        """
        try:
            from tavily import TavilyClient  # lazy import
        except ImportError:
            self._event_bus.emit(DEBUG, level="warning", msg="[Search] Web search unavailable — tavily-python not installed")
            return {
                "status": "error",
                "message": "Web search unavailable: tavily-python is not installed. "
                           "Install it with: pip install tavily-python",
            }

        import os
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            self._event_bus.emit(DEBUG, level="warning", msg="[Search] Web search unavailable — TAVILY_API_KEY not set")
            return {
                "status": "error",
                "message": "Web search unavailable: TAVILY_API_KEY environment variable not set.",
            }

        try:
            client = TavilyClient(api_key=api_key)
            result = client.search(query, include_answer="basic")

            answer = result.get("answer", "")
            sources = result.get("results", [])

            # Format sources
            sources_text = ""
            if sources:
                source_lines = [f"- [{s.get('title', 'Source')}]({s['url']})" for s in sources if s.get("url")]
                if source_lines:
                    sources_text = "\n\nSources:\n" + "\n".join(source_lines)

            return {
                "status": "success",
                "answer": (answer or "") + sources_text,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Tavily search failed: {e}",
            }

    def get_token_usage(self) -> dict:
        """Return cumulative token usage for this session (including sub-agents)."""
        input_tokens = self._total_input_tokens
        output_tokens = self._total_output_tokens
        thinking_tokens = self._total_thinking_tokens
        cached_tokens = self._total_cached_tokens
        api_calls = self._api_calls

        # Include usage from cached mission agents
        for agent in self._mission_agents.values():
            usage = agent.get_token_usage()
            input_tokens += usage["input_tokens"]
            output_tokens += usage["output_tokens"]
            thinking_tokens += usage.get("thinking_tokens", 0)
            cached_tokens += usage.get("cached_tokens", 0)
            api_calls += usage["api_calls"]

        # Include usage from visualization agent
        if self._viz_agent:
            usage = self._viz_agent.get_token_usage()
            input_tokens += usage["input_tokens"]
            output_tokens += usage["output_tokens"]
            thinking_tokens += usage.get("thinking_tokens", 0)
            cached_tokens += usage.get("cached_tokens", 0)
            api_calls += usage["api_calls"]

        # Include usage from data ops agent
        if self._dataops_agent:
            usage = self._dataops_agent.get_token_usage()
            input_tokens += usage["input_tokens"]
            output_tokens += usage["output_tokens"]
            thinking_tokens += usage.get("thinking_tokens", 0)
            cached_tokens += usage.get("cached_tokens", 0)
            api_calls += usage["api_calls"]

        # Include usage from data extraction agent
        if self._data_extraction_agent:
            usage = self._data_extraction_agent.get_token_usage()
            input_tokens += usage["input_tokens"]
            output_tokens += usage["output_tokens"]
            thinking_tokens += usage.get("thinking_tokens", 0)
            cached_tokens += usage.get("cached_tokens", 0)
            api_calls += usage["api_calls"]

        # Include usage from planner agent
        if self._planner_agent:
            usage = self._planner_agent.get_token_usage()
            input_tokens += usage["input_tokens"]
            output_tokens += usage["output_tokens"]
            thinking_tokens += usage.get("thinking_tokens", 0)
            cached_tokens += usage.get("cached_tokens", 0)

        # Include usage from memory agent
        if self._memory_agent:
            usage = self._memory_agent.get_token_usage()
            input_tokens += usage["input_tokens"]
            output_tokens += usage["output_tokens"]
            thinking_tokens += usage.get("thinking_tokens", 0)
            cached_tokens += usage.get("cached_tokens", 0)
            api_calls += usage["api_calls"]

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "cached_tokens": cached_tokens,
            "total_tokens": input_tokens + output_tokens + thinking_tokens,
            "api_calls": api_calls,
        }

    def get_token_usage_breakdown(self) -> list[dict]:
        """Return per-agent token usage breakdown.

        Returns a list of dicts with keys: agent, input, output, thinking, cached, calls.
        Only includes agents that have made at least one API call.
        """
        rows = []

        def _add(name, usage):
            if usage["api_calls"] > 0:
                rows.append({
                    "agent": name,
                    "input": usage["input_tokens"],
                    "output": usage["output_tokens"],
                    "thinking": usage.get("thinking_tokens", 0),
                    "cached": usage.get("cached_tokens", 0),
                    "calls": usage["api_calls"],
                })

        # Orchestrator's own usage
        _add("Orchestrator", {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "thinking_tokens": self._total_thinking_tokens,
            "cached_tokens": self._total_cached_tokens,
            "api_calls": self._api_calls,
        })

        # Mission agents
        for mission_id, agent in self._mission_agents.items():
            _add(f"Mission/{mission_id}", agent.get_token_usage())

        # Visualization agent
        if self._viz_agent:
            _add("Visualization", self._viz_agent.get_token_usage())

        # Data ops agent
        if self._dataops_agent:
            _add("DataOps", self._dataops_agent.get_token_usage())

        # Data extraction agent
        if self._data_extraction_agent:
            _add("DataExtraction", self._data_extraction_agent.get_token_usage())

        # Planner agent
        if self._planner_agent:
            usage = self._planner_agent.get_token_usage()
            if usage["api_calls"] > 0 or usage["input_tokens"] > 0:
                rows.append({
                    "agent": "Planner",
                    "input": usage["input_tokens"],
                    "output": usage["output_tokens"],
                    "thinking": usage.get("thinking_tokens", 0),
                    "cached": usage.get("cached_tokens", 0),
                    "calls": usage.get("api_calls", 0),
                })

        # Memory agent
        if self._memory_agent:
            _add("Memory", self._memory_agent.get_token_usage())

        return rows

    def _validate_time_range(self, dataset_id: str, start, end) -> dict | None:
        """Check a requested time range against a dataset's availability.

        Validates and adjusts the time range:
        - Fully within range → returns None (no adjustment needed)
        - Partial overlap → clamps to available window
        - No overlap → returns error dict (does NOT silently shift)

        Note: This validates dataset-level availability only. Individual
        parameters may still return all-NaN data within a valid range.

        Args:
            dataset_id: CDAWeb dataset ID
            start: Requested start datetime (naive, implicitly UTC)
            end: Requested end datetime (naive, implicitly UTC)

        Returns:
            None if fully valid, or a dict with:
                - "start": clamped start datetime
                - "end": clamped end datetime
                - "note": human-readable note about the adjustment
            Returns error dict with "status"="error" if no overlap.
            Returns None if the metadata lookup fails (fail-open).
        """
        time_range = get_dataset_time_range(dataset_id)
        if time_range is None:
            return None  # fail-open

        try:
            avail_start_str = time_range.get("start")
            avail_stop_str = time_range.get("stop")
            if not avail_start_str or not avail_stop_str:
                return None

            # Parse date strings, strip any tz info (all implicitly UTC)
            avail_start = datetime.fromisoformat(avail_start_str).replace(tzinfo=None)
            avail_stop = datetime.fromisoformat(avail_stop_str).replace(tzinfo=None)

            req_start = start.replace(tzinfo=None) if start.tzinfo else start
            req_end = end.replace(tzinfo=None) if end.tzinfo else end

            avail_range_str = (
                f"{avail_start.strftime('%Y-%m-%d')} to "
                f"{avail_stop.strftime('%Y-%m-%d')}"
            )
            duration = req_end - req_start

            # No overlap — request is entirely after available data
            if req_start >= avail_stop:
                return {
                    "error": True,
                    "note": (
                        f"No data available for '{dataset_id}' in the requested period. "
                        f"Dataset covers {avail_range_str}. "
                        f"Try a different dataset or adjust your time range."
                    ),
                }

            # No overlap — request is entirely before available data
            if req_end <= avail_start:
                return {
                    "error": True,
                    "note": (
                        f"No data available for '{dataset_id}' in the requested period. "
                        f"Dataset covers {avail_range_str}. "
                        f"Try a different dataset or adjust your time range."
                    ),
                }

            # Partial overlap — clamp to available window
            if req_start < avail_start or req_end > avail_stop:
                new_start = max(req_start, avail_start)
                new_end = min(req_end, avail_stop)
                return {
                    "start": new_start,
                    "end": new_end,
                    "note": (
                        f"Requested range partially outside available data for "
                        f"'{dataset_id}' (available: {avail_range_str}). "
                        f"Clamped to {new_start.strftime('%Y-%m-%d')} to "
                        f"{new_end.strftime('%Y-%m-%d')}."
                    ),
                }

        except (ValueError, TypeError):
            return None  # fail-open on parse errors

        return None  # fully valid

    @staticmethod
    def _resolve_entry(store, label: str):
        """Resolve a label to a DataEntry, supporting column sub-selection.

        Handles labels like 'PSP_B_DERIVATIVE_FINAL.B_mag' where
        'PSP_B_DERIVATIVE_FINAL' is the store key and 'B_mag' is a column.

        Returns (DataEntry, resolved_label) or (None, None) if not found.
        """
        from data_ops.store import DataEntry

        # Exact match first
        entry = store.get(label)
        if entry is not None:
            return entry, label

        # Try column sub-selection: split from the right and check
        # progressively longer prefixes as parent labels.
        # E.g. "A.B.C" tries "A.B" with col "C", then "A" with col "B.C"
        parts = label.split(".")
        for i in range(len(parts) - 1, 0, -1):
            parent_label = ".".join(parts[:i])
            col_name = ".".join(parts[i:])
            parent = store.get(parent_label)
            if parent is not None:
                # Try string match first, then int (CDF vectors have integer column names)
                if col_name not in parent.columns:
                    try:
                        col_name = int(col_name)
                    except (ValueError, TypeError):
                        continue
                    if col_name not in parent.columns:
                        continue
                sub_entry = DataEntry(
                    label=label,
                    data=parent.data[[col_name]],
                    units=parent.units,
                    description=f"{parent.description} [{col_name}]" if parent.description else col_name,
                    source=parent.source,
                    metadata=parent.metadata,
                )
                return sub_entry, label
        return None, None

    def _handle_render_plotly_json(self, tool_args: dict) -> dict:
        """Handle render_plotly_json: fill data_label placeholders and render."""
        fig_json = tool_args.get("figure_json", {})
        data_traces = fig_json.get("data", [])
        if not data_traces:
            return {
                "status": "error",
                "message": (
                    "figure_json.data is required and must be a non-empty array of traces. "
                    "Each trace needs at least a 'data_label' key. Example: "
                    'render_plotly_json(figure_json={"data": [{"type": "scatter", '
                    '"data_label": "DATASET.Parameter"}], "layout": {}}). '
                    "Call list_fetched_data first to discover available labels."
                ),
            }

        # Guard: reject figures with too many layout objects (shapes + annotations).
        # LLMs struggle to generate large arrays of complex objects — the JSON
        # often arrives garbled (dicts collapsed to floats, arrays to integers).
        layout = fig_json.get("layout", {})
        n_shapes = len(layout.get("shapes", []))
        n_annotations = len(layout.get("annotations", []))
        _MAX_LAYOUT_OBJECTS = 30
        if n_shapes + n_annotations > _MAX_LAYOUT_OBJECTS:
            return {
                "status": "error",
                "message": (
                    f"Too many layout objects: {n_shapes} shapes + {n_annotations} annotations "
                    f"= {n_shapes + n_annotations} (limit: {_MAX_LAYOUT_OBJECTS}). "
                    f"Reduce the number of shapes/annotations. For many similar markers, "
                    f"consider: (1) showing only the most significant events, "
                    f"(2) using a single legend entry instead of per-event labels, or "
                    f"(3) omitting annotations and keeping only the shapes."
                ),
            }

        # Collect all data_label values and resolve entries
        store = get_store()
        entry_map: dict = {}
        for trace in data_traces:
            label = trace.get("data_label")
            if label and label not in entry_map:
                entry, _ = self._resolve_entry(store, label)
                if entry is None:
                    return {"status": "error", "message": f"data_label '{label}' not found in memory"}
                entry_map[label] = entry

        # Validate non-empty data
        for label, entry in entry_map.items():
            if len(entry.data) == 0:
                return {"status": "error",
                        "message": f"Entry '{label}' has no data points"}

        try:
            result = self._renderer.render_plotly_json(fig_json, entry_map)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        if result.get("status") == "success":
            self._event_bus.emit(RENDER_EXECUTED, agent="orchestrator",
                msg="[Render] render_plotly_json executed",
                data={"args": {"figure_json": fig_json}, "inputs": list(entry_map.keys()), "outputs": []})

        return result

    def _handle_manage_plot(self, tool_args: dict) -> dict:
        """Handle the manage_plot tool call."""
        action = tool_args.get("action")
        if not action:
            return {"status": "error", "message": "action is required"}

        if action == "reset":
            self._event_bus.emit(PLOT_ACTION, agent="orchestrator",
                msg="[Plot] reset",
                data={"args": {"action": "reset"}, "outputs": []})
            return self._renderer.reset()

        elif action == "get_state":
            return self._renderer.get_current_state()

        elif action == "export":
            filename = tool_args.get("filename", "output.png")
            fmt = tool_args.get("format", "png")
            result = self._renderer.export(filename, format=fmt)

            if result.get("status") == "success":
                self._event_bus.emit(PLOT_ACTION, agent="orchestrator",
                    msg=f"[Plot] export {filename}",
                    data={"args": {"action": "export", "filename": filename, "format": fmt}, "outputs": []})

            # Auto-open the exported file in default viewer (skip in GUI mode)
            if result.get("status") == "success" and not self.gui_mode and not self.web_mode:
                try:
                    import os
                    import platform
                    filepath = result["filepath"]
                    if platform.system() == "Windows":
                        os.startfile(filepath)
                    elif platform.system() == "Darwin":
                        import subprocess
                        subprocess.Popen(["open", filepath])
                    else:
                        import subprocess
                        subprocess.Popen(["xdg-open", filepath])
                    result["auto_opened"] = True
                except Exception as e:
                    self._event_bus.emit(DEBUG, level="debug", msg=f"[Export] Could not auto-open: {e}")
                    result["auto_opened"] = False

            return result

        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    # --- SPICE MCP dispatch ---

    def _handle_spice_tool(self, tool_name: str, tool_args: dict) -> dict:
        """Route a SPICE tool call through the MCP client.

        For trajectory/velocity calls, extracts full data from the response
        and stores it in DataStore for plotting.
        """
        from .mcp_client import get_spice_client

        # Agent always needs full data for the DataStore — bypass response
        # size limit enforced by the MCP server for external callers.
        if tool_name in ("get_spacecraft_trajectory", "get_spacecraft_velocity"):
            tool_args = {**tool_args, "allow_large_response": True}

        try:
            client = get_spice_client()
        except Exception as e:
            log_error(f"SPICE MCP connection failed: {e}", e)
            return {"status": "error", "message": f"SPICE MCP server unavailable: {e}"}

        try:
            result = client.call_tool(tool_name, tool_args)
        except Exception as e:
            log_error(f"SPICE MCP call failed for {tool_name}: {e}", e)
            # Try one reconnect
            try:
                client = get_spice_client()
                result = client.call_tool(tool_name, tool_args)
            except Exception as e2:
                log_error(f"SPICE MCP retry failed for {tool_name}: {e2}", e2)
                return {"status": "error", "message": f"SPICE MCP call failed: {e2}"}

        if result.get("status") == "error":
            return result

        # For trajectory/velocity calls, extract data and store in DataStore
        if tool_name in ("get_spacecraft_trajectory", "get_spacecraft_velocity"):
            try:
                records = result.get("data")
                if records:
                    import pandas as pd

                    df = pd.DataFrame(records)
                    df["time"] = pd.to_datetime(df["time"], format="ISO8601")
                    df = df.set_index("time")

                    sc_name = tool_args["spacecraft"].upper().replace(" ", "_")
                    suffix = "trajectory" if tool_name == "get_spacecraft_trajectory" else "velocity"
                    label = f"SPICE.{sc_name}_{suffix}"
                    units = "km" if suffix == "trajectory" else "km/s"
                    store = get_store()
                    entry = DataEntry(
                        label=label,
                        data=df,
                        units=units,
                        description=f"SPICE {suffix} of {sc_name} rel. {tool_args.get('observer', 'SUN')}",
                        source="spice",
                    )
                    store.put(entry)
                    result["label"] = label
                    result["note"] = f"Stored as '{label}' — use render_plotly_json to plot."

                    # Remove full data from result returned to LLM (it only needs
                    # the summary + label; full data is in the DataStore)
                    del result["data"]
            except Exception as e:
                self._event_bus.emit(DEBUG, level="warning", msg=f"[SPICE] Failed to store {tool_name} data: {e}")
                # Non-fatal: the summary result is still returned

        return result

    def _execute_tool(self, tool_name: str, tool_args: dict) -> dict:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool

        Returns:
            Dict with result data (varies by tool)
        """
        # Log the tool call
        log_tool_call(tool_name, tool_args)

        self._event_bus.emit(DEBUG, level="debug", msg=f"[Tool: {tool_name}({tool_args})]")

        if tool_name == "search_datasets":
            query = tool_args.get("query")
            if not query:
                return {"status": "error", "message": "Missing required parameter: query"}
            self._event_bus.emit(DEBUG, level="debug", msg=f"[Catalog] Searching for: {query}")
            result = search_by_keywords(query)
            if result:
                self._event_bus.emit(DEBUG, level="debug", msg="[Catalog] Found matches.")
                return {"status": "success", **result}
            else:
                self._event_bus.emit(DEBUG, level="debug", msg="[Catalog] No matches found.")
                return {"status": "success", "message": "No matching datasets found."}

        elif tool_name == "list_parameters":
            dataset_id = tool_args.get("dataset_id")
            if not dataset_id:
                return {"status": "error", "message": "Missing required parameter: dataset_id"}
            # Return CDF variable names from Master CDF skeleton
            try:
                from data_ops.fetch_cdf import list_cdf_variables
                cdf_vars = list_cdf_variables(dataset_id)
                self._event_bus.emit(DEBUG, level="debug", msg=f"[CDF] Listed {len(cdf_vars)} data variables for {dataset_id}")
                return {"status": "success", "parameters": cdf_vars}
            except Exception as e:
                self._event_bus.emit(DEBUG, level="debug", msg=f"[CDF] Could not list variables for {dataset_id}: {e}, using metadata cache")
                params = list_parameters(dataset_id)
                return {"status": "success", "parameters": params}

        elif tool_name == "inspect_dataset":
            dataset_id = tool_args.get("dataset_id")
            time_start = tool_args.get("time_start")
            time_end = tool_args.get("time_end")
            missing = [k for k in ("dataset_id", "time_start", "time_end") if not tool_args.get(k)]
            if missing:
                return {"status": "error", "message": f"Missing required parameter(s): {', '.join(missing)}"}
            try:
                from data_ops.fetch_cdf import inspect_dataset
                result = inspect_dataset(dataset_id, time_start, time_end)
                return {"status": "success", **result}
            except Exception as e:
                return {"status": "error", "message": f"Failed to inspect dataset '{dataset_id}': {e}"}

        elif tool_name == "get_data_availability":
            dataset_id = tool_args.get("dataset_id")
            if not dataset_id:
                return {"status": "error", "message": "Missing required parameter: dataset_id"}
            time_range = get_dataset_time_range(dataset_id)
            if time_range is None:
                return {"status": "error", "message": f"Could not fetch availability for '{dataset_id}'."}
            return {
                "status": "success",
                "dataset_id": dataset_id,
                "start": time_range.get("start"),
                "stop": time_range.get("stop"),
            }

        elif tool_name == "browse_datasets":
            from knowledge.metadata_client import browse_datasets
            from knowledge.mission_loader import load_mission as _load_mission
            from knowledge.catalog import SPACECRAFT, classify_instrument_type
            mission_id = tool_args.get("mission_id")
            if not mission_id:
                return {"status": "error", "message": "Missing required parameter: mission_id"}
            # Ensure metadata cache exists (triggers download if needed)
            try:
                _load_mission(mission_id)
            except FileNotFoundError:
                pass
            datasets = browse_datasets(mission_id)
            if datasets is None:
                return {"status": "error", "message": f"No dataset index for '{mission_id}'."}

            # Enrich with instrument/type from mission JSON
            sc = SPACECRAFT.get(mission_id, {})
            ds_to_instrument = {}
            for inst_id, inst in sc.get("instruments", {}).items():
                kws = inst.get("keywords", [])
                for ds_id in inst.get("datasets", []):
                    ds_to_instrument[ds_id] = {"instrument": inst_id, "keywords": kws}

            for ds in datasets:
                info = ds_to_instrument.get(ds["id"], {})
                ds["instrument"] = info.get("instrument", "")
                ds["type"] = classify_instrument_type(info.get("keywords", []))

            return {"status": "success", "mission_id": mission_id,
                    "dataset_count": len(datasets), "datasets": datasets}

        elif tool_name == "list_missions":
            missions = list_missions()
            return {"status": "success", "missions": missions, "count": len(missions)}

        elif tool_name == "get_dataset_docs":
            from knowledge.metadata_client import get_dataset_docs
            dataset_id = tool_args.get("dataset_id")
            if not dataset_id:
                return {"status": "error", "message": "Missing required parameter: dataset_id"}
            docs = get_dataset_docs(dataset_id)
            if docs.get("documentation"):
                return {"status": "success", **docs}
            else:
                result = {"status": "partial" if docs.get("contact") else "error",
                          "dataset_id": docs.get("dataset_id", dataset_id),
                          "message": "Could not fetch documentation."}
                if docs.get("contact"):
                    result["contact"] = docs["contact"]
                if docs.get("resource_url"):
                    result["resource_url"] = docs["resource_url"]
                return result

        elif tool_name == "search_full_catalog":
            query = tool_args.get("query")
            if not query:
                return {"status": "error", "message": "Missing required parameter: query"}
            max_results = int(tool_args.get("max_results", 20))
            self._event_bus.emit(DEBUG, level="debug", msg=f"[Catalog] Full catalog search: {query}")
            results = search_full_catalog(query, max_results=max_results)
            if results:
                return {
                    "status": "success",
                    "query": query,
                    "count": len(results),
                    "datasets": results,
                    "note": "Use fetch_data with any dataset ID above. Use list_parameters to see available parameters.",
                }
            else:
                return {
                    "status": "success",
                    "query": query,
                    "count": 0,
                    "datasets": [],
                    "message": f"No datasets found matching '{query}'. Try broader search terms.",
                }

        elif tool_name == "google_search":
            query = tool_args.get("query")
            if not query:
                return {"status": "error", "message": "Missing required parameter: query"}
            self._event_bus.emit(DEBUG, level="debug", msg=f"[Search] Query: {query}")
            return self._google_search(query)

        elif tool_name == "ask_clarification":
            # Return the question to show to user
            return {
                "status": "clarification_needed",
                "question": tool_args.get("question", ""),
                "options": tool_args.get("options", []),
                "context": tool_args.get("context", ""),
            }

        # --- Visualization (declarative tools) ---

        elif tool_name == "render_plotly_json":
            return self._handle_render_plotly_json(tool_args)

        elif tool_name == "manage_plot":
            return self._handle_manage_plot(tool_args)

        # --- Data Operations Tools ---

        elif tool_name == "fetch_data":
            dataset_id = tool_args.get("dataset_id")
            parameter_id = tool_args.get("parameter_id")
            if not dataset_id or not parameter_id:
                missing = [k for k in ("dataset_id", "parameter_id") if not tool_args.get(k)]
                return {"status": "error", "message": f"Missing required parameter(s): {', '.join(missing)}"}

            # Pre-fetch validation: reject dataset IDs not in local cache
            ds_validation = validate_dataset_id(dataset_id)
            if not ds_validation["valid"]:
                return {"status": "error", "message": ds_validation["message"]}

            # Pre-fetch validation: reject parameter IDs not in cached metadata
            param_validation = validate_parameter_id(dataset_id, parameter_id)
            if not param_validation["valid"]:
                return {"status": "error", "message": param_validation["message"]}

            try:
                fetch_start = datetime.fromisoformat(tool_args.get("time_start", ""))
                fetch_end = datetime.fromisoformat(tool_args.get("time_end", ""))
            except (ValueError, TypeError) as e:
                return {"status": "error", "message": f"Invalid time_start/time_end: {e}"}
            if fetch_start >= fetch_end:
                return {
                    "status": "error",
                    "message": (
                        f"time_start ({tool_args['time_start']}) must be before "
                        f"time_end ({tool_args['time_end']})."
                    ),
                }

            # Auto-clamp to available data window
            adjustment_note = None

            validation = self._validate_time_range(
                dataset_id, fetch_start, fetch_end
            )
            if validation is not None:
                if validation.get("error"):
                    return {"status": "error", "message": validation["note"]}
                fetch_start = validation["start"]
                fetch_end = validation["end"]
                adjustment_note = validation["note"]
                self._event_bus.emit(DEBUG, level="debug", msg=f"[DataOps] Time range adjusted for {tool_args['dataset_id']}: "
                    f"{adjustment_note}")

            # Dedup: skip fetch if identical data already exists in store
            label = f"{dataset_id}.{parameter_id}"
            store = get_store()
            existing = store.get(label)
            if existing is not None and len(existing.time) > 0:
                if existing.is_xarray:
                    existing_start = pd.Timestamp(existing.time[0]).to_pydatetime().replace(tzinfo=None)
                    existing_end = pd.Timestamp(existing.time[-1]).to_pydatetime().replace(tzinfo=None)
                else:
                    existing_start = existing.data.index[0].to_pydatetime().replace(tzinfo=None)
                    existing_end = existing.data.index[-1].to_pydatetime().replace(tzinfo=None)
                if existing_start <= fetch_start and existing_end >= fetch_end:
                    self._event_bus.emit(DEBUG, level="debug", msg=f"[DataOps] Dedup: '{label}' already in memory "
                        f"({existing_start} to {existing_end}), skipping fetch")
                    response = {
                        "status": "success",
                        "already_loaded": True,
                        **existing.summary(),
                    }
                    if adjustment_note:
                        response["time_range_note"] = adjustment_note
                    self._event_bus.emit(DATA_FETCHED, agent="orchestrator",
                        msg=f"[Fetch] {label} (already loaded)",
                        data={"args": {
                            "dataset_id": tool_args["dataset_id"],
                            "parameter_id": tool_args["parameter_id"],
                            "time_start": tool_args.get("time_start", ""),
                            "time_end": tool_args.get("time_end", ""),
                            "time_range_resolved": [fetch_start.isoformat(), fetch_end.isoformat()],
                            "already_loaded": True,
                        }, "outputs": [label]})
                    return response

            try:
                result = fetch_data(
                    dataset_id=tool_args["dataset_id"],
                    parameter_id=tool_args["parameter_id"],
                    # ISO 8601 with trailing Z
                    time_min=fetch_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    time_max=fetch_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    force=tool_args.get("force_large_download", False),
                )
            except Exception as e:
                return {"status": "error", "message": str(e)}

            # Large download needs user permission
            if isinstance(result, dict) and result.get("status") == "confirmation_required":
                return {
                    "status": "clarification_needed",
                    "question": result["message"],
                    "options": [
                        "Yes, proceed with the download",
                        "No, try a shorter time range",
                        "Other (please specify)",
                    ],
                    "context": (
                        f"Dataset {result['dataset_id']}: {result['download_mb']} MB "
                        f"across {result['n_files']} files to download "
                        f"({result['n_cached']} already cached)."
                    ),
                }

            import xarray as xr
            fetched_data = result["data"]
            is_xarray = isinstance(fetched_data, xr.DataArray)

            if is_xarray:
                # xarray DataArray (3D+ variable)
                import numpy as np
                n_time = fetched_data.sizes["time"]

                # Detect all-NaN fetches
                if n_time > 0 and np.all(np.isnan(fetched_data.values)):
                    return {
                        "status": "error",
                        "message": (
                            f"Parameter '{tool_args['parameter_id']}' in dataset "
                            f"'{tool_args['dataset_id']}' returned {n_time} time steps "
                            f"but ALL values are fill/NaN — no real data available "
                            f"for this parameter in the requested time range. "
                            f"Try a different parameter or dataset."
                        ),
                    }

                # NaN percentage
                total_cells = fetched_data.size
                nan_total = int(np.isnan(fetched_data.values).sum())
                nan_pct = round(100 * nan_total / total_cells, 1) if total_cells > 0 else 0.0

                from config import DATA_BACKEND
                entry = DataEntry(
                    label=label,
                    data=fetched_data,
                    units=result["units"],
                    description=result["description"],
                    source=DATA_BACKEND,
                )
                store.put(entry)
                self._event_bus.emit(DATA_FETCHED, agent="orchestrator",
                    msg=f"[Fetch] {label} (xarray {dict(fetched_data.sizes)})",
                    data={"args": {
                        "dataset_id": tool_args["dataset_id"],
                        "parameter_id": tool_args["parameter_id"],
                        "time_range": tool_args.get("time_range", ""),
                        "time_range_resolved": [fetch_start.isoformat(), fetch_end.isoformat()],
                        "already_loaded": False,
                    }, "outputs": [label], "status": "success", "nan_percentage": nan_pct})
                response = {"status": "success", **entry.summary()}
                response["note"] = (
                    f"This is a {fetched_data.ndim}D variable with dims {dict(fetched_data.sizes)}. "
                    f"Use custom_operation with xarray syntax (da_{tool_args['parameter_id']}) "
                    f"to slice/reduce it to a 2D DataFrame before plotting."
                )

                n_points = n_time
            else:
                # pandas DataFrame (1D/2D variable)
                df = fetched_data
                # Detect all-NaN fetches (parameter has no real data in range)
                numeric_cols = df.select_dtypes(include="number")
                if len(df) > 0 and len(numeric_cols.columns) > 0 and numeric_cols.isna().all(axis=None):
                    return {
                        "status": "error",
                        "message": (
                            f"Parameter '{tool_args['parameter_id']}' in dataset "
                            f"'{tool_args['dataset_id']}' returned {len(df)} rows "
                            f"but ALL values are fill/NaN — no real data available "
                            f"for this parameter in the requested time range. "
                            f"Try a different parameter or dataset."
                        ),
                    }

                # Check NaN percentage before storing
                nan_total = numeric_cols.isna().sum().sum()
                nan_pct = round(100 * nan_total / numeric_cols.size, 1) if numeric_cols.size > 0 else 0.0

                from config import DATA_BACKEND
                entry = DataEntry(
                    label=label,
                    data=df,
                    units=result["units"],
                    description=result["description"],
                    source=DATA_BACKEND,
                )
                store.put(entry)
                self._event_bus.emit(DATA_FETCHED, agent="orchestrator",
                    msg=f"[Fetch] {label} ({len(entry.time)} points)",
                    data={"args": {
                        "dataset_id": tool_args["dataset_id"],
                        "parameter_id": tool_args["parameter_id"],
                        "time_range": tool_args.get("time_range", ""),
                        "time_range_resolved": [fetch_start.isoformat(), fetch_end.isoformat()],
                        "already_loaded": False,
                    }, "outputs": [label], "status": "success", "nan_percentage": nan_pct})
                response = {"status": "success", **entry.summary()}

                n_points = len(df)

            # Warn about very large datasets that may cause slow operations
            if n_points > 500_000:
                response["size_warning"] = (
                    f"Very large dataset ({n_points:,} points). "
                    f"Consider using a shorter time range or a lower-cadence dataset "
                    f"to avoid slow downstream operations."
                )

            if adjustment_note:
                response["time_range_note"] = adjustment_note

            # Report NaN percentage for transparency
            if nan_pct > 0:
                response["nan_percentage"] = nan_pct
                if nan_pct >= 25:
                    response["quality_warning"] = (
                        f"High NaN/fill ratio ({nan_pct}%). Data was stored but "
                        f"quality is degraded. Consider trying a different "
                        f"parameter or dataset if one with better coverage exists."
                    )

            # Surface metadata discrepancies from override system
            from knowledge.metadata_client import get_dataset_quality_report
            quality = get_dataset_quality_report(tool_args["dataset_id"])
            if quality and (quality["metadata_only"] or quality["data_only"]):
                response["metadata_discrepancies"] = quality

            return response

        elif tool_name == "list_fetched_data":
            store = get_store()
            entries = store.list_entries()
            return {"status": "success", "entries": entries, "count": len(entries)}

        elif tool_name == "custom_operation":
            import numpy as np
            import xarray as xr

            store = get_store()
            labels = tool_args.get("source_labels", [])
            if not labels:
                return {"status": "error", "message": "source_labels is required"}

            sources, err = build_source_map(store, labels)
            if err:
                return {"status": "error", "message": err}

            # Determine if "df" alias exists (only when at least one source is a DataFrame)
            has_df_alias = any(not isinstance(v, xr.DataArray) for v in sources.values())
            df_extra = ["df"] if has_df_alias else []

            # Build source_timeseries map: variable name → is_timeseries flag
            source_ts = {}
            for label in labels:
                entry = store.get(label)
                if entry is not None:
                    suffix = label.rsplit(".", 1)[-1]
                    prefix = "da" if entry.is_xarray else "df"
                    var_name = f"{prefix}_{suffix}"
                    source_ts[var_name] = entry.is_timeseries

            try:
                op_result, warnings = run_multi_source_operation(
                    sources, tool_args["code"], source_timeseries=source_ts,
                )
            except (ValueError, RuntimeError) as e:
                prefix = "Validation" if isinstance(e, ValueError) else "Execution"
                err_msg = f"{prefix} error: {e}"
                self._event_bus.emit(CUSTOM_OP_FAILURE, agent="orchestrator", level="warning",
                    msg=f"[DataOps] custom_operation error: {err_msg}",
                    data={"args": {
                        "source_labels": labels,
                        "code": tool_args["code"],
                        "output_label": tool_args.get("output_label", ""),
                    }, "inputs": labels, "outputs": [], "status": "error", "error": err_msg})
                return {
                    "status": "error",
                    "message": err_msg,
                    "available_variables": list(sources.keys()) + df_extra,
                    "source_info": describe_sources(store, labels),
                }

            first_entry = store.get(labels[0])
            units = tool_args.get("units", first_entry.units if first_entry else "")
            desc = tool_args.get("description", f"Custom operation on {', '.join(labels)}")
            # Result is timeseries if it has a DatetimeIndex (or time dim for xarray)
            if isinstance(op_result, xr.DataArray):
                result_is_ts = "time" in op_result.dims
            else:
                result_is_ts = isinstance(op_result.index, pd.DatetimeIndex)
            entry = DataEntry(
                label=tool_args["output_label"],
                data=op_result,
                units=units,
                description=desc,
                source="computed",
                is_timeseries=result_is_ts,
            )
            store.put(entry)
            self._event_bus.emit(DATA_COMPUTED, agent="orchestrator",
                msg=f"[DataOps] custom_operation -> '{tool_args['output_label']}'",
                data={"args": {
                    "source_labels": labels,
                    "code": tool_args["code"],
                    "output_label": tool_args["output_label"],
                    "description": desc,
                    "units": units,
                }, "inputs": labels, "outputs": [tool_args["output_label"]]})

            # Save to persistent ops library (only complex code: 5+ lines)
            try:
                import re as _re_lib
                from data_ops.ops_library import get_ops_library
                lib = get_ops_library()
                code = tool_args["code"]
                # Track reuse when description contains [from <id>]
                ref_match = _re_lib.search(r'\[from ([a-f0-9]{8})\]', desc)
                if ref_match:
                    lib.record_reuse(ref_match.group(1))
                # Count logical lines (split on newlines and semicolons)
                line_count = sum(1 for line in code.replace(";", "\n").split("\n") if line.strip())
                if line_count >= 5:
                    lib.add_or_update(
                        description=desc,
                        code=code,
                        source_labels=labels,
                        units=units,
                        session_id=self._session_id or "",
                    )
            except Exception:
                self._event_bus.emit(DEBUG, level="debug", msg="[OpsLibrary] Failed to save operation")

            for w in warnings:
                self._event_bus.emit(DEBUG, level="debug", msg=f"[DataOpsValidation] {w}")

            # Warn on empty or all-NaN results (P0-1 symptom)
            is_xr_result = isinstance(op_result, xr.DataArray)
            n_points = op_result.sizes["time"] if is_xr_result else len(op_result)
            if n_points == 0:
                warnings.append("Result has 0 data points — possible time range mismatch or all-NaN input")
                self._event_bus.emit(DEBUG, level="warning", msg=f"[DataOps] custom_operation produced 0 points for '{tool_args['output_label']}'")
            elif is_xr_result:
                if np.all(np.isnan(op_result.values)):
                    warnings.append("Result is entirely NaN — check source data overlap and computation logic")
                    self._event_bus.emit(DEBUG, level="warning", msg=f"[DataOps] custom_operation produced all-NaN for '{tool_args['output_label']}'")
            elif op_result.isna().all(axis=None):
                warnings.append("Result is entirely NaN — check source data overlap and computation logic")
                self._event_bus.emit(DEBUG, level="warning", msg=f"[DataOps] custom_operation produced all-NaN for '{tool_args['output_label']}'")

            self._event_bus.emit(DEBUG, level="debug", msg=f"[DataOps] Custom operation -> '{tool_args['output_label']}' ({n_points} points)")
            result = {
                "status": "success",
                **entry.summary(),
                "source_info": describe_sources(store, labels),
                "available_variables": list(sources.keys()) + df_extra,
            }
            if warnings:
                result["warnings"] = warnings
            return result

        elif tool_name == "store_dataframe":
            try:
                result_df = run_dataframe_creation(tool_args["code"])
            except (ValueError, RuntimeError) as e:
                prefix = "Validation" if isinstance(e, ValueError) else "Execution"
                err_msg = f"{prefix} error: {e}"
                self._event_bus.emit(DATA_CREATED, agent="orchestrator", level="warning",
                    msg=f"[DataOps] store_dataframe error: {err_msg}",
                    data={"args": {
                        "code": tool_args["code"],
                        "output_label": tool_args.get("output_label", ""),
                    }, "outputs": [], "status": "error", "error": err_msg})
                return {"status": "error", "message": err_msg}
            entry = DataEntry(
                label=tool_args["output_label"],
                data=result_df,
                units=tool_args.get("units", ""),
                description=tool_args.get("description", "Created from code"),
                source="created",
                is_timeseries=isinstance(result_df.index, pd.DatetimeIndex),
            )
            store = get_store()
            store.put(entry)
            self._event_bus.emit(DATA_CREATED, agent="orchestrator",
                msg=f"[DataOps] Created DataFrame -> '{tool_args['output_label']}' ({len(result_df)} points)",
                data={"args": {
                    "code": tool_args["code"],
                    "output_label": tool_args["output_label"],
                    "description": tool_args.get("description", "Created from code"),
                    "units": tool_args.get("units", ""),
                }, "outputs": [tool_args["output_label"]]})
            return {"status": "success", **entry.summary()}

        # --- Function Documentation Tools ---

        elif tool_name == "search_function_docs":
            from knowledge.function_catalog import search_functions
            query = tool_args["query"]
            package = tool_args.get("package")
            results = search_functions(query, package=package)
            return {
                "status": "success",
                "query": query,
                "count": len(results),
                "functions": results,
            }

        elif tool_name == "get_function_docs":
            from knowledge.function_catalog import get_function_docstring
            package = tool_args.get("package")
            function_name = tool_args.get("function_name")
            if not package or not function_name:
                return {"status": "error", "message": "Both 'package' (e.g. 'scipy.signal') and 'function_name' are required"}
            result = get_function_docstring(package, function_name)
            if "error" in result:
                return {"status": "error", "message": result["error"]}
            return {"status": "success", **result}

        # --- Describe & Export Tools ---

        elif tool_name == "describe_data":
            store = get_store()
            entry = store.get(tool_args["label"])
            if entry is None:
                return {"status": "error", "message": f"Label '{tool_args['label']}' not found in memory"}

            # Optional time-range filter
            time_start = tool_args.get("time_start")
            time_end = tool_args.get("time_end")

            # --- xarray DataArray path ---
            if entry.is_xarray:
                import numpy as np
                da = entry.data
                if (time_start or time_end) and "time" in da.dims:
                    sel_kw = {}
                    if time_start and time_end:
                        sel_kw["time"] = slice(time_start, time_end)
                    elif time_start:
                        sel_kw["time"] = slice(time_start, None)
                    else:
                        sel_kw["time"] = slice(None, time_end)
                    da = da.sel(**sel_kw)
                dims = dict(da.sizes)
                n_time = dims.get("time", 0)

                # Coordinate info
                coords_info = {}
                for cname, coord in da.coords.items():
                    cvals = coord.values
                    info = {"size": len(cvals), "dtype": str(cvals.dtype)}
                    if np.issubdtype(cvals.dtype, np.number):
                        info["min"] = float(np.nanmin(cvals))
                        info["max"] = float(np.nanmax(cvals))
                    elif np.issubdtype(cvals.dtype, np.datetime64):
                        info["min"] = str(cvals[0])
                        info["max"] = str(cvals[-1])
                    coords_info[cname] = info

                # Global statistics on finite values
                flat = da.values.flatten()
                finite = flat[np.isfinite(flat)]
                nan_count = int(flat.size - finite.size)
                if finite.size > 0:
                    pcts = np.percentile(finite, [25, 50, 75])
                    statistics = {
                        "min": float(np.min(finite)),
                        "max": float(np.max(finite)),
                        "mean": float(np.mean(finite)),
                        "std": float(np.std(finite)),
                        "25%": float(pcts[0]),
                        "50%": float(pcts[1]),
                        "75%": float(pcts[2]),
                    }
                else:
                    statistics = {"min": None, "max": None, "mean": None, "std": None}

                # Time info
                time_start = time_end = time_span = median_cadence = None
                if n_time > 0:
                    times = da.coords["time"].values
                    time_start = str(times[0])
                    time_end = str(times[-1])
                    time_span = str(pd.Timestamp(times[-1]) - pd.Timestamp(times[0]))
                    if n_time > 1:
                        dt = pd.Series(times).diff().dropna()
                        median_cadence = str(dt.median())

                return {
                    "status": "success",
                    "label": entry.label,
                    "units": entry.units,
                    "storage_type": "xarray",
                    "dims": dims,
                    "coordinates": coords_info,
                    "num_points": n_time,
                    "time_start": time_start,
                    "time_end": time_end,
                    "time_span": time_span,
                    "median_cadence": median_cadence,
                    "nan_count": nan_count,
                    "nan_percentage": round(nan_count / flat.size * 100, 1) if flat.size > 0 else 0,
                    "statistics": statistics,
                }

            df = entry.data
            if (time_start or time_end) and entry.is_timeseries:
                try:
                    if time_start and time_end:
                        df = df.loc[pd.Timestamp(time_start):pd.Timestamp(time_end)]
                    elif time_start:
                        df = df.loc[pd.Timestamp(time_start):]
                    else:
                        df = df.loc[:pd.Timestamp(time_end)]
                except (ValueError, TypeError) as e:
                    return {"status": "error", "message": f"Invalid time range: {e}"}
            stats = {}

            # Per-column statistics (numeric columns get full stats, others get count/unique)
            desc = df.describe(percentiles=[0.25, 0.5, 0.75], include="all")
            for col in df.columns:
                if df[col].dtype.kind in ("f", "i", "u"):  # numeric
                    col_stats = {
                        "min": float(desc.loc["min", col]),
                        "max": float(desc.loc["max", col]),
                        "mean": float(desc.loc["mean", col]),
                        "std": float(desc.loc["std", col]),
                        "25%": float(desc.loc["25%", col]),
                        "50%": float(desc.loc["50%", col]),
                        "75%": float(desc.loc["75%", col]),
                    }
                else:  # string/object/categorical columns
                    col_stats = {
                        "type": str(df[col].dtype),
                        "count": int(desc.loc["count", col]),
                        "unique": int(desc.loc["unique", col]) if "unique" in desc.index else None,
                        "top": str(desc.loc["top", col]) if "top" in desc.index else None,
                    }
                stats[col] = col_stats

            # Global metadata
            nan_count = int(df.isna().sum().sum())
            total_points = len(df)
            if total_points == 0:
                return {
                    "status": "success",
                    "label": entry.label,
                    "units": entry.units,
                    "num_points": 0,
                    "message": "No data points in the requested range.",
                }
            time_span = str(df.index[-1] - df.index[0]) if total_points > 1 else "single point"

            # Cadence estimate (median time step)
            if total_points > 1:
                dt = df.index.to_series().diff().dropna()
                median_cadence = str(dt.median())
            else:
                median_cadence = "N/A"

            return {
                "status": "success",
                "label": entry.label,
                "units": entry.units,
                "num_points": total_points,
                "num_columns": len(df.columns),
                "columns": list(df.columns),
                "time_start": str(df.index[0]),
                "time_end": str(df.index[-1]),
                "time_span": time_span,
                "median_cadence": median_cadence,
                "nan_count": nan_count,
                "nan_percentage": round(nan_count / (total_points * len(df.columns)) * 100, 1) if total_points > 0 else 0,
                "statistics": stats,
            }

        elif tool_name == "preview_data":
            store = get_store()
            entry = store.get(tool_args["label"])
            if entry is None:
                return {"status": "error", "message": f"Label '{tool_args['label']}' not found in memory"}

            # Optional time-range filter
            time_start = tool_args.get("time_start")
            time_end = tool_args.get("time_end")

            # --- xarray DataArray path ---
            if entry.is_xarray:
                import numpy as np
                da = entry.data
                if (time_start or time_end) and "time" in da.dims:
                    sel_kw = {}
                    if time_start and time_end:
                        sel_kw["time"] = slice(time_start, time_end)
                    elif time_start:
                        sel_kw["time"] = slice(time_start, None)
                    else:
                        sel_kw["time"] = slice(None, time_end)
                    da = da.sel(**sel_kw)
                n_time = da.sizes.get("time", 0)
                n_rows = min(tool_args.get("n_rows", 3), 10)
                position = tool_args.get("position", "both")

                def _xr_time_slice(indices):
                    rows = []
                    for i in indices:
                        sl = da.isel(time=i)
                        vals = sl.values.flatten()
                        finite = vals[np.isfinite(vals)]
                        rows.append({
                            "timestamp": str(da.coords["time"].values[i]),
                            "shape": list(sl.shape),
                            "min": float(np.min(finite)) if finite.size > 0 else None,
                            "max": float(np.max(finite)) if finite.size > 0 else None,
                            "mean": float(np.mean(finite)) if finite.size > 0 else None,
                            "nan_count": int(vals.size - finite.size),
                        })
                    return rows

                result = {
                    "status": "success",
                    "label": entry.label,
                    "units": entry.units,
                    "storage_type": "xarray",
                    "dims": dict(da.sizes),
                    "total_time_steps": n_time,
                }
                if n_time > 0:
                    head_idx = list(range(min(n_rows, n_time)))
                    tail_idx = list(range(max(n_time - n_rows, 0), n_time))
                    if position in ("head", "both"):
                        result["head"] = _xr_time_slice(head_idx)
                    if position in ("tail", "both"):
                        result["tail"] = _xr_time_slice(tail_idx)

                return result

            df = entry.data
            if (time_start or time_end) and entry.is_timeseries:
                try:
                    if time_start and time_end:
                        df = df.loc[pd.Timestamp(time_start):pd.Timestamp(time_end)]
                    elif time_start:
                        df = df.loc[pd.Timestamp(time_start):]
                    else:
                        df = df.loc[:pd.Timestamp(time_end)]
                except (ValueError, TypeError) as e:
                    return {"status": "error", "message": f"Invalid time range: {e}"}
            n_rows = min(tool_args.get("n_rows", 5), 50)
            position = tool_args.get("position", "both")

            def _df_to_rows(sub_df):
                rows = []
                for ts, row in sub_df.iterrows():
                    d = {"timestamp": str(ts)}
                    for col in sub_df.columns:
                        v = row[col]
                        d[col] = float(v) if isinstance(v, (int, float)) else str(v)
                    rows.append(d)
                return rows

            result = {
                "status": "success",
                "label": entry.label,
                "units": entry.units,
                "total_rows": len(df),
                "columns": list(df.columns),
            }

            if position in ("head", "both"):
                result["head"] = _df_to_rows(df.head(n_rows))
            if position in ("tail", "both"):
                result["tail"] = _df_to_rows(df.tail(n_rows))

            return result

        elif tool_name == "save_data":
            store = get_store()
            entry = store.get(tool_args["label"])
            if entry is None:
                return {"status": "error", "message": f"Label '{tool_args['label']}' not found in memory"}

            from pathlib import Path

            # --- xarray DataArray path: export as NetCDF ---
            if entry.is_xarray:
                da = entry.data
                filename = tool_args.get("filename", "")
                if not filename:
                    safe_label = entry.label.replace(".", "_").replace("/", "_")
                    filename = f"{safe_label}.nc"
                if not filename.endswith(".nc"):
                    filename += ".nc"

                parent = Path(filename).parent
                if parent and str(parent) != "." and not parent.exists():
                    parent.mkdir(parents=True, exist_ok=True)

                da.to_netcdf(filename, encoding={"time": {"units": "nanoseconds since 1970-01-01"}})
                filepath = str(Path(filename).resolve())
                file_size = Path(filename).stat().st_size

                self._event_bus.emit(DEBUG, level="debug", msg=f"[DataOps] Exported xarray '{entry.label}' to {filepath} ({file_size:,} bytes)")

                return {
                    "status": "success",
                    "label": entry.label,
                    "filepath": filepath,
                    "format": "netcdf",
                    "dims": dict(da.sizes),
                    "file_size_bytes": file_size,
                }

            # Generate filename if not provided
            filename = tool_args.get("filename", "")
            if not filename:
                safe_label = entry.label.replace(".", "_").replace("/", "_")
                filename = f"{safe_label}.csv"
            if not filename.endswith(".csv"):
                filename += ".csv"

            # Ensure parent directory exists
            parent = Path(filename).parent
            if parent and str(parent) != "." and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)

            # Export with ISO 8601 timestamps
            df = entry.data.copy()
            df.index.name = "timestamp"
            df.to_csv(filename, date_format="%Y-%m-%dT%H:%M:%S.%fZ")

            filepath = str(Path(filename).resolve())
            file_size = Path(filename).stat().st_size

            self._event_bus.emit(DEBUG, level="debug", msg=f"[DataOps] Exported '{entry.label}' to {filepath} ({file_size:,} bytes)")

            return {
                "status": "success",
                "label": entry.label,
                "filepath": filepath,
                "num_points": len(df),
                "num_columns": len(df.columns),
                "file_size_bytes": file_size,
            }

        # --- Document Reading (Gemini multimodal) ---

        elif tool_name == "read_document":
            from pathlib import Path

            file_path = tool_args["file_path"]
            if not Path(file_path).is_file():
                return {"status": "error", "message": f"File not found: {file_path}"}

            # MIME type map for supported formats
            mime_map = {
                ".pdf": "application/pdf",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
                ".tiff": "image/tiff",
            }
            ext = Path(file_path).suffix.lower()
            mime_type = mime_map.get(ext)
            if not mime_type:
                supported = ", ".join(sorted(mime_map.keys()))
                return {
                    "status": "error",
                    "message": f"Unsupported file format '{ext}'. Supported: {supported}",
                }

            try:
                import shutil

                # Read file bytes
                file_bytes = Path(file_path).read_bytes()

                # Build extraction prompt
                custom_prompt = tool_args.get("prompt", "")
                if custom_prompt:
                    extraction_prompt = custom_prompt
                elif ext == ".pdf":
                    extraction_prompt = (
                        "Extract all text content from this document. "
                        "Preserve the document structure (headings, paragraphs, lists). "
                        "Render tables as markdown tables. "
                        "Describe any figures or charts briefly."
                    )
                else:
                    extraction_prompt = (
                        "Extract all text and data from this image. "
                        "If it contains a table or chart, transcribe the data. "
                        "If it contains text, transcribe it faithfully. "
                        "Describe any visual elements briefly."
                    )

                # Send to LLM as multimodal content
                if hasattr(self.adapter, "make_bytes_part") and hasattr(self.adapter, "generate_multimodal"):
                    doc_part = self.adapter.make_bytes_part(data=file_bytes, mime_type=mime_type)
                    response = self.adapter.generate_multimodal(
                        model=get_active_model(self.model_name),
                        contents=[doc_part, extraction_prompt],
                    )
                else:
                    response = self.adapter.generate(
                        model=get_active_model(self.model_name),
                        contents=extraction_prompt,
                    )
                self._last_tool_context = "extract_document"
                self._track_usage(response)

                full_text = response.text or ""

                # Save original + extracted text to data_dir/documents/{stem}/
                docs_dir = get_data_dir() / "documents"
                src = Path(file_path)
                stem = src.stem

                # Find a unique subfolder name
                folder = docs_dir / stem
                counter = 1
                while folder.exists():
                    folder = docs_dir / f"{stem}_{counter}"
                    counter += 1
                folder.mkdir(parents=True, exist_ok=True)

                # Copy original file
                original_copy = folder / src.name
                shutil.copy2(str(src), str(original_copy))

                # Save extracted text
                out_path = folder / f"{stem}.md"
                out_path.write_text(full_text, encoding="utf-8")
                self._event_bus.emit(DEBUG, level="debug", msg=f"[Document] Saved to {folder} ({len(full_text)} chars)")

                # Truncate for LLM context
                max_chars = 50_000
                text = full_text
                truncated = len(full_text) > max_chars
                if truncated:
                    text = full_text[:max_chars]

                return {
                    "status": "success",
                    "file": Path(file_path).name,
                    "original_saved_to": str(original_copy),
                    "text_saved_to": str(out_path),
                    "char_count": len(full_text),
                    "truncated": truncated,
                    "content": text,
                }
            except Exception as e:
                return {"status": "error", "message": f"Document reading failed: {e}"}

        # --- Routing ---

        elif tool_name == "delegate_to_mission":
            mission_id = tool_args["mission_id"]
            request = tool_args["request"]
            self._event_bus.emit(DELEGATION, level="debug", msg=f"[Router] Delegating to {mission_id} specialist")
            try:
                agent = self._get_or_create_mission_agent(mission_id)
                # Inject current data store contents so mission agent knows what's loaded
                store = get_store()
                entries = store.list_entries()
                if entries:
                    labels = [
                        f"  - {e['label']} ({e['num_points']} pts, {e['time_min']} to {e['time_max']})"
                        for e in entries
                    ]
                    request += (
                        "\n\nData currently in memory:\n"
                        + "\n".join(labels)
                        + "\nDo NOT re-fetch data that is already in memory with a matching label and time range."
                    )
                request += self._build_agent_history("mission")
                request = self._inject_memory(request, f"mission:{mission_id}")
                sub_result = agent.process_request(request)
                snapshot = get_store().list_entries()
                result = self._wrap_delegation_result(sub_result, store_snapshot=snapshot)
                result["mission"] = mission_id
                self._event_bus.emit(DELEGATION_DONE, level="debug", msg=f"[Router] {mission_id} specialist finished")
                return result
            except (KeyError, FileNotFoundError):
                return {
                    "status": "error",
                    "message": f"Unknown mission '{mission_id}'. Check the supported missions table.",
                }

        elif tool_name == "delegate_to_visualization":
            request = tool_args["request"]
            context = tool_args.get("context", "")
            self._event_bus.emit(DELEGATION, level="debug", msg="[Router] Delegating to Visualization specialist")

            # Intercept export requests — handle directly, no LLM needed
            req_lower = request.lower()
            if "export" in req_lower or ".png" in req_lower or ".pdf" in req_lower:
                import re as _re
                fn_match = _re.search(r'[\w.-]+\.(?:png|pdf|svg)', request, _re.IGNORECASE)
                filename = fn_match.group(0) if fn_match else "output.png"
                fmt = "pdf" if filename.endswith(".pdf") else "png"
                result = self._renderer.export(filename, format=fmt)
                if result.get("status") == "success" and not self.gui_mode and not self.web_mode:
                    try:
                        import os, platform, subprocess
                        fp = result["filepath"]
                        if platform.system() == "Darwin":
                            subprocess.Popen(["open", fp])
                        elif platform.system() == "Windows":
                            os.startfile(fp)
                        else:
                            subprocess.Popen(["xdg-open", fp])
                    except Exception:
                        pass
                return {"status": "success", "result": f"Exported plot to {result.get('filepath', filename)}"}

            agent = self._get_or_create_viz_agent()
            full_request = f"{request}\n\nContext: {context}" if context else request

            # Inject current plot state so the viz agent knows what's displayed
            state = self._renderer.get_current_state()
            if state["has_plot"]:
                if state.get("figure_json"):
                    import json
                    fig_json_str = json.dumps(state["figure_json"], indent=2)
                    full_request += (
                        f"\n\nCurrently displayed: {state['traces']}"
                        f"\n\nCurrent figure_json (modify this, don't rebuild from scratch):\n{fig_json_str}"
                    )
                else:
                    full_request += f"\n\nCurrently displayed: {state['traces']}"
            else:
                full_request += "\n\nNo plot currently displayed."

            full_request += self._build_agent_history("viz", agent_name="Visualization Agent")
            full_request = self._inject_memory(full_request, "visualization")
            sub_result = agent.process_request(full_request)
            self._event_bus.emit(DELEGATION_DONE, level="debug", msg="[Router] Visualization specialist finished")
            return self._wrap_delegation_result(sub_result, store_snapshot=get_store().list_entries())

        elif tool_name == "delegate_to_data_ops":
            request = tool_args["request"]
            context = tool_args.get("context", "")
            self._event_bus.emit(DELEGATION, level="debug", msg="[Router] Delegating to DataOps specialist")
            agent = self._get_or_create_dataops_agent()
            full_request = f"{request}\n\nContext: {context}" if context else request
            full_request += self._build_agent_history("dataops", agent_name="DataOps Agent")
            full_request = self._inject_memory(full_request, "data_ops")
            sub_result = agent.process_request(full_request)
            self._event_bus.emit(DELEGATION_DONE, level="debug", msg="[Router] DataOps specialist finished")
            return self._wrap_delegation_result(sub_result, store_snapshot=get_store().list_entries())

        elif tool_name == "delegate_to_data_extraction":
            request = tool_args["request"]
            context = tool_args.get("context", "")
            self._event_bus.emit(DELEGATION, level="debug", msg="[Router] Delegating to DataExtraction specialist")
            agent = self._get_or_create_data_extraction_agent()
            full_request = f"{request}\n\nContext: {context}" if context else request
            sub_result = agent.process_request(full_request)
            self._event_bus.emit(DELEGATION_DONE, level="debug", msg="[Router] DataExtraction specialist finished")
            return self._wrap_delegation_result(sub_result, store_snapshot=get_store().list_entries())

        # --- SPICE Ephemeris Tools (via MCP) ---

        elif tool_name in (
            "get_spacecraft_position", "get_spacecraft_trajectory",
            "get_spacecraft_velocity", "transform_coordinates",
            "compute_distance", "list_spice_missions",
            "list_coordinate_frames",
        ):
            return self._handle_spice_tool(tool_name, tool_args)

        elif tool_name == "search_discoveries":
            from dataclasses import asdict as _disc_asdict
            query = tool_args.get("query", "")
            mission = tool_args.get("mission")
            limit = tool_args.get("limit", 5)
            results = self._discovery_store.search(query, limit=limit, mission=mission)
            return {
                "status": "success",
                "count": len(results),
                "discoveries": [
                    {
                        "id": d.id,
                        "summary": d.summary,
                        "content": d.content,
                        "missions": d.missions,
                        "datasets": d.datasets,
                        "reasoning": d.reasoning,
                        "created_at": d.created_at,
                        "pipeline": d.pipeline,
                    }
                    for d in results
                ],
            }

        elif tool_name == "recall_memories":
            from dataclasses import asdict as _asdict
            query = tool_args.get("query", "")
            mem_type = tool_args.get("type")
            scope = tool_args.get("scope")
            limit = tool_args.get("limit", 20)
            if query:
                # Tag-based search across active + cold memories
                results = self._memory_store.search(
                    query, mem_type=mem_type, scope=scope, limit=limit,
                )
                results = [_asdict(m) for m in results]
            else:
                # No query: list recent memories
                all_memories = [
                    _asdict(m) for m in self._memory_store.get_enabled()
                ]
                if mem_type:
                    all_memories = [m for m in all_memories if m.get("type") == mem_type]
                if scope:
                    all_memories = [m for m in all_memories if scope in m.get("scopes", [])]
                results = all_memories[-limit:]
            return {
                "status": "success",
                "count": len(results),
                "memories": results,
            }

        elif tool_name == "review_memory":
            from datetime import datetime as _dt
            from agent.memory import Memory, generate_tags
            memory_id = tool_args.get("memory_id", "")
            stars_raw = tool_args.get("stars")
            try:
                stars = int(stars_raw)
            except (TypeError, ValueError):
                stars = 0
            comment = tool_args.get("comment", "")
            if not memory_id or stars < 1 or stars > 5:
                return {"status": "error", "message": "Invalid memory_id or stars (must be 1-5)"}
            if not isinstance(comment, str) or not comment.strip():
                return {"status": "error", "message": "Comment required"}
            entry = self._memory_store.get_by_id(memory_id)
            if entry is None or entry.archived:
                return {"status": "error", "message": f"Memory {memory_id} not found"}
            agent_name = self._memory_store._last_injected_ids.get(memory_id, "unknown")
            model_name = get_active_model(self.model_name)
            content = f"{stars}★ {comment.strip()}"
            tags = [f"review:{memory_id}", agent_name, f"stars:{stars}"]
            # If the same agent already reviewed this memory, supersede that review
            existing_review = self._memory_store.get_review_for(memory_id, agent=agent_name)
            supersedes = ""
            version = 1
            if existing_review:
                supersedes = existing_review.id
                version = existing_review.version + 1
                existing_review.archived = True
            review_memory = Memory(
                type="review",
                scopes=list(entry.scopes),
                content=content,
                source="extracted",
                source_session=self._session_id or "",
                tags=tags,
                review_of=memory_id,
                supersedes=supersedes,
                version=version,
            )
            self._memory_store.add(review_memory)
            return {"status": "success", "memory_id": memory_id, "stars": stars}

        elif tool_name == "request_planning":
            request = tool_args["request"]
            reasoning = tool_args.get("reasoning", "")
            time_start = tool_args.get("time_start", "")
            time_end = tool_args.get("time_end", "")
            structured_time_range = f"{time_start} to {time_end}" if time_start and time_end else ""
            self._event_bus.emit(DEBUG, level="debug", msg=f"[Planner] Planning requested: {reasoning}")
            summary = self._handle_planning_request(
                request, structured_time_range=structured_time_range
            )
            return {"status": "success", "result": summary, "planning_used": True}

        # --- Pipeline DAG Tools ---

        elif tool_name == "get_pipeline_info":
            # Mode: list ops library
            if tool_args.get("list_library"):
                from data_ops.ops_library import get_ops_library
                entries = get_ops_library().get_top_entries()
                return {"status": "success", "library_entries": entries}

            pipeline = self._get_or_build_pipeline()
            if len(pipeline) == 0:
                return {"status": "success", "message": "No pipeline operations recorded yet."}

            # Mode: single node detail
            node_id = tool_args.get("node_id")
            if node_id:
                detail = pipeline.node_detail(node_id)
                if detail is None:
                    return {"status": "error", "message": f"Node '{node_id}' not found"}
                # Check ops library for matching code
                if detail.get("code"):
                    from data_ops.ops_library import get_ops_library
                    match = get_ops_library().find_matching_code(detail["code"])
                    if match:
                        detail["library_match"] = {
                            "id": match["id"],
                            "description": match["description"],
                            "use_count": match.get("use_count", 1),
                        }
                return {"status": "success", **detail}

            # Default: compact summary
            return {"status": "success", **pipeline.to_summary()}

        elif tool_name == "modify_pipeline_node":
            pipeline = self._get_or_build_pipeline()
            action = tool_args.get("action", "")

            if action == "update_params":
                node_id = tool_args.get("node_id", "")
                params = tool_args.get("params", {})
                if not node_id:
                    return {"status": "error", "message": "node_id is required for update_params"}
                if not params:
                    return {"status": "error", "message": "params is required for update_params"}
                try:
                    affected = pipeline.update_node_params(node_id, params)
                    return {
                        "status": "success",
                        "action": "update_params",
                        "node_id": node_id,
                        "affected_nodes": sorted(affected),
                        "stale_count": len(pipeline.get_stale_nodes()),
                    }
                except KeyError as e:
                    return {"status": "error", "message": str(e)}

            elif action == "remove":
                node_id = tool_args.get("node_id", "")
                if not node_id:
                    return {"status": "error", "message": "node_id is required for remove"}
                try:
                    result = pipeline.remove_node(node_id)
                    return {"status": "success", "action": "remove", **result}
                except KeyError as e:
                    return {"status": "error", "message": str(e)}

            elif action == "insert_after":
                after_id = tool_args.get("after_id", "")
                tool_type = tool_args.get("tool", "custom_operation")
                params = tool_args.get("params", {})
                output_label = tool_args.get("output_label", "")
                if not after_id:
                    return {"status": "error", "message": "after_id is required for insert_after"}
                if not output_label:
                    return {"status": "error", "message": "output_label is required for insert_after"}
                try:
                    new_id = pipeline.insert_node(after_id, tool_type, params, output_label)
                    return {
                        "status": "success",
                        "action": "insert_after",
                        "new_node_id": new_id,
                        "after_id": after_id,
                        "stale_count": len(pipeline.get_stale_nodes()),
                    }
                except KeyError as e:
                    return {"status": "error", "message": str(e)}

            elif action == "apply_library_op":
                node_id = tool_args.get("node_id", "")
                library_entry_id = tool_args.get("library_entry_id", "")
                if not node_id:
                    return {"status": "error", "message": "node_id is required for apply_library_op"}
                if not library_entry_id:
                    return {"status": "error", "message": "library_entry_id is required for apply_library_op"}
                node = pipeline.get_node(node_id)
                if node is None:
                    return {"status": "error", "message": f"Node '{node_id}' not found"}
                if node.tool != "custom_operation":
                    return {"status": "error", "message": f"Node '{node_id}' is not a compute node (tool={node.tool})"}
                from data_ops.ops_library import get_ops_library
                lib = get_ops_library()
                entry = lib.get_entry_by_id(library_entry_id)
                if entry is None:
                    return {"status": "error", "message": f"Library entry '{library_entry_id}' not found"}
                new_params = {"code": entry["code"], "description": entry["description"]}
                affected = pipeline.update_node_params(node_id, new_params)
                lib.record_reuse(library_entry_id)
                return {
                    "status": "success",
                    "action": "apply_library_op",
                    "node_id": node_id,
                    "library_entry_id": library_entry_id,
                    "affected_nodes": sorted(affected),
                    "stale_count": len(pipeline.get_stale_nodes()),
                }

            elif action == "save_to_library":
                node_id = tool_args.get("node_id", "")
                if not node_id:
                    return {"status": "error", "message": "node_id is required for save_to_library"}
                node = pipeline.get_node(node_id)
                if node is None:
                    return {"status": "error", "message": f"Node '{node_id}' not found"}
                if node.tool != "custom_operation":
                    return {"status": "error", "message": f"Node '{node_id}' is not a compute node (tool={node.tool})"}
                code = node.params.get("code", "")
                if not code.strip():
                    return {"status": "error", "message": f"Node '{node_id}' has no code to save"}
                description = node.params.get("description", "")
                source_labels = list(node.inputs)
                units = node.params.get("units", "")
                from data_ops.ops_library import get_ops_library
                entry = get_ops_library().add_or_update(
                    description=description,
                    code=code,
                    source_labels=source_labels,
                    units=units,
                )
                return {
                    "status": "success",
                    "action": "save_to_library",
                    "node_id": node_id,
                    "library_entry": entry,
                }

            else:
                return {"status": "error", "message": f"Unknown action: {action}. Use 'update_params', 'remove', 'insert_after', 'apply_library_op', or 'save_to_library'."}

        elif tool_name == "execute_pipeline":
            pipeline = self._get_or_build_pipeline()
            stale = pipeline.get_stale_nodes()
            if not stale:
                return {"status": "success", "message": "No stale nodes to execute.", "executed": 0}

            use_cache = tool_args.get("use_cache", True)
            cache_store = get_store() if use_cache else None
            store = get_store()

            result = pipeline.execute_stale(
                store, cache_store=cache_store, renderer=self._renderer,
            )
            return {"status": "success", **result}

        else:
            result = {"status": "error", "message": f"Unknown tool: {tool_name}"}
            log_error(
                f"Unknown tool called: {tool_name}",
                context={"tool_name": tool_name, "tool_args": tool_args}
            )
            return result

    def _execute_tool_safe(self, tool_name: str, tool_args: dict) -> dict:
        """Execute a tool with error handling and logging.

        Wraps _execute_tool to catch unexpected exceptions and log them.
        After execution, checks for hot-path memory extraction opportunities.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool

        Returns:
            Dict with result data (varies by tool)
        """
        self._event_bus.emit(TOOL_CALL, agent="orchestrator", msg=f"[Tool] {tool_name}({tool_args})",
            data={"tool_name": tool_name, "tool_args": tool_args})

        try:
            result = self._execute_tool(tool_name, tool_args)
            result = _sanitize_for_json(result)

            # Log the result
            is_success = result.get("status") != "error"
            log_tool_result(tool_name, result, is_success)

            # If error, log with more detail
            if not is_success:
                log_error(
                    f"Tool {tool_name} returned error: {result.get('message', 'Unknown')}",
                    context={"tool_name": tool_name, "tool_args": tool_args, "result": result}
                )

            # Invalidate cached pipeline when data-producing tools run
            _PIPELINE_INVALIDATING_TOOLS = {
                "fetch_data", "custom_operation", "store_dataframe",
                "render_plotly_json", "manage_plot",
            }
            if is_success and tool_name in _PIPELINE_INVALIDATING_TOOLS:
                self._invalidate_pipeline()

            _tr_event = TOOL_RESULT
            self._event_bus.emit(_tr_event, agent="orchestrator",
                msg=f"[Tool Result] {tool_name}: {'success' if is_success else 'error'}",
                data={"tool_name": tool_name, "status": "success" if is_success else "error"})

            return result

        except Exception as e:
            # Unexpected exception - log with full stack trace
            log_error(
                f"Unexpected exception in tool {tool_name}",
                exc=e,
                context={"tool_name": tool_name, "tool_args": tool_args}
            )
            self._event_bus.emit(TOOL_ERROR, agent="orchestrator", level="error",
                msg=f"[Tool] {tool_name} internal error: {e}",
                data={"tool_name": tool_name, "error": str(e)})
            return {"status": "error", "message": f"Internal error: {e}"}

    # ---- Prefix → mission ID mapping for dataset-based scope detection ----
    _DATASET_PREFIX_MAP = {
        "PSP": "PSP", "AC": "ACE", "SOLO": "SolO", "SO": "SolO",
        "OMNI": "OMNI", "WI": "WIND", "DSCOVR": "DSCOVR",
        "MMS1": "MMS", "MMS2": "MMS", "MMS3": "MMS", "MMS4": "MMS",
        "MMS": "MMS", "STA": "STEREO_A",
    }

    def _dataset_scopes(self, dataset_id: str) -> list[str]:
        """Determine mission scopes from a dataset ID prefix."""
        if not dataset_id:
            return ["generic"]
        prefix = dataset_id.split("_")[0].upper()
        mission = self._DATASET_PREFIX_MAP.get(prefix, "")
        return [f"mission:{mission}"] if mission else ["generic"]

    def _execute_task(self, task: Task) -> str:
        """Execute a single task and return the result.

        Sends the task instruction to Gemini and handles tool calls.
        Updates the task status and records tool calls made.

        Args:
            task: The task to execute

        Returns:
            The text response from Gemini after completing the task
        """
        task.status = TaskStatus.IN_PROGRESS
        task.tool_calls = []

        self._event_bus.emit(DEBUG, level="debug", msg=f"[Task] Executing: {task.description}")
        self._event_bus.emit(DEBUG, level="debug", msg="[Gemini] Sending task instruction...")

        try:
            # Create a fresh chat session for task execution with forced function calling
            task_chat = self.adapter.create_chat(
                model=get_active_model(self.model_name),
                system_prompt=self._system_prompt,
                tools=self._tool_schemas,
                force_tool_call=True,
                thinking="low",
            )
            task_prompt = (
                f"Execute this task: {task.instruction}\n\n"
                "CRITICAL: Do ONLY what the instruction says. Do NOT add extra steps.\n"
                "- If the task says 'search', just search and report the results as text.\n"
                "- If the task says 'fetch', just fetch the data.\n"
                "- Do NOT create DataFrames, plots, or visualizations unless the instruction explicitly asks for it.\n"
                "- Do NOT delegate to other agents unless the instruction explicitly asks for it.\n"
                "- Return results as concise text, not as tool calls."
            )
            self._last_tool_context = "task:" + task.description[:50]
            response = self._send_with_timeout(task_chat, task_prompt)
            self._track_usage(response)

            # Process tool calls with loop guard
            guard = LoopGuard(max_total_calls=20, max_iterations=10)
            last_stop_reason = None
            had_successful_tool = False

            while True:
                stop_reason = guard.check_iteration()
                if stop_reason:
                    self._event_bus.emit(DEBUG, level="debug", msg=f"[Task] Stopping: {stop_reason}")
                    last_stop_reason = stop_reason
                    break

                if self._cancel_event.is_set():
                    self._event_bus.emit(DEBUG, level="info", msg="[Cancel] Stopping task execution loop")
                    last_stop_reason = "cancelled by user"
                    break

                if not response.tool_calls:
                    break

                function_calls = response.tool_calls

                # Break if LLM is trying to ask for clarification (not supported in task execution)
                if any(fc.name == "ask_clarification" for fc in function_calls):
                    self._event_bus.emit(DEBUG, level="debug", msg="[Task] Skipping clarification request")
                    break

                # Check for loops/duplicates/cycling
                call_keys = set()
                for fc in function_calls:
                    call_keys.add(make_call_key(fc.name, fc.args))
                stop_reason = guard.check_calls(call_keys)
                if stop_reason:
                    self._event_bus.emit(DEBUG, level="debug", msg=f"[Task] Stopping: {stop_reason}")
                    last_stop_reason = stop_reason
                    break

                function_responses = []
                for fc in function_calls:
                    tool_name = fc.name
                    tool_args = fc.args

                    task.tool_calls.append(tool_name)
                    result = self._execute_tool_safe(tool_name, tool_args)

                    if result.get("status") == "success":
                        had_successful_tool = True
                    elif result.get("status") == "error":
                        self._event_bus.emit(DEBUG, level="warning", msg=f"[Tool Result: ERROR] {result.get('message', '')}")

                    function_responses.append(
                        self.adapter.make_tool_result_message(
                            tool_name, result, tool_call_id=fc.id
                        )
                    )

                guard.record_calls(call_keys)

                self._event_bus.emit(DEBUG, level="debug", msg=f"[LLM] Sending {len(function_responses)} tool result(s) back...")
                tool_names = [fc.name for fc in function_calls]
                self._last_tool_context = "+".join(tool_names)
                response = self._send_with_timeout(task_chat, function_responses)
                self._track_usage(response)

            # Warn if no tools were called (LLM just responded with text)
            if not task.tool_calls:
                log_error(
                    f"Task completed without any tool calls: {task.description}",
                    context={"task_instruction": task.instruction}
                )
                self._event_bus.emit(DEBUG, level="warning", msg="[WARNING] No tools were called for this task")

            # Extract text response
            result_text = response.text or "Done."

            if last_stop_reason:
                if last_stop_reason == "cancelled by user":
                    task.status = TaskStatus.FAILED
                    task.error = "Task cancelled by user"
                    result_text += " [CANCELLED]"
                elif had_successful_tool:
                    task.status = TaskStatus.COMPLETED
                    result_text += " [loop guard stopped extra calls]"
                else:
                    task.status = TaskStatus.FAILED
                    task.error = f"Task stopped by loop guard: {last_stop_reason}"
                    result_text += f" [STOPPED: {last_stop_reason}]"
            else:
                task.status = TaskStatus.COMPLETED

            task.result = result_text

            self._event_bus.emit(DEBUG, level="debug", msg=f"[Task] {'Failed' if last_stop_reason else 'Completed'}: {task.description}")

            return result_text

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self._event_bus.emit(DEBUG, level="warning", msg=f"[Task] Failed: {task.description} - {e}")
            return f"Error: {e}"

    def _summarize_plan_execution(self, plan: TaskPlan) -> str:
        """Generate a summary of the completed plan execution."""
        # Build context from completed tasks
        summary_parts = [f"I just executed a multi-step plan for: \"{plan.user_request}\""]
        summary_parts.append("")

        completed = plan.get_completed_tasks()
        failed = plan.get_failed_tasks()

        if completed:
            summary_parts.append("Completed tasks:")
            for task in completed:
                summary_parts.append(f"  - {task.description}")
                if task.result:
                    result_preview = task.result[:100] + "..." if len(task.result) > 100 else task.result
                    summary_parts.append(f"    Result: {result_preview}")

        if failed:
            summary_parts.append("")
            summary_parts.append("Failed tasks:")
            for task in failed:
                summary_parts.append(f"  - {task.description}")
                if task.error:
                    summary_parts.append(f"    Error: {task.error}")

        summary_parts.append("")
        summary_parts.append("Please provide a brief summary of what was accomplished for the user.")

        prompt = "\n".join(summary_parts)

        self._event_bus.emit(DEBUG, level="debug", msg="[LLM] Generating execution summary...")

        try:
            self._last_tool_context = "plan_summary"
            response = self._send_message(prompt)
            self._track_usage(response)

            text = response.text or plan.progress_summary()
            text += self._extract_grounding_sources(response)
            return text

        except Exception as e:
            log_error("Error generating plan summary", exc=e, context={"plan_id": plan.id})
            self._event_bus.emit(DEBUG, level="warning", msg=f"[Summary] Error generating summary: {e}")
            return plan.progress_summary()

    def _get_or_create_planner_agent(self) -> PlannerAgent:
        """Get the cached planner agent or create a new one."""
        if self._planner_agent is None:
            self._planner_agent = PlannerAgent(
                adapter=self.adapter,
                model_name=config.PLANNER_MODEL,
                tool_executor=self._execute_tool_safe,
                verbose=self.verbose,
                cancel_event=self._cancel_event,
                event_bus=self._event_bus,
            )
            self._event_bus.emit(DEBUG, level="debug", msg=f"[Router] Created PlannerAgent ({config.PLANNER_MODEL})")
        return self._planner_agent

    def _build_task_result_summary(
        self, task: Task, labels_before: set[str], labels_after: set[str]
    ) -> dict:
        """Build an informative result summary dict for a completed plan task."""
        result_text = (task.result or "")[:500]
        if result_text in ("", "Done.") and task.tool_calls:
            result_text = f"Tools called: {', '.join(task.tool_calls)}"

        new_labels = labels_after - labels_before
        if new_labels:
            new_label_parts = []
            for lbl in sorted(new_labels):
                entry_info = next(
                    (e for e in get_store().list_entries() if e["label"] == lbl),
                    None,
                )
                if entry_info:
                    new_label_parts.append(
                        f"{lbl} ({entry_info.get('shape', '?')}, "
                        f"{entry_info.get('num_points', '?')} pts, "
                        f"units={entry_info.get('units', '?')}, "
                        f"columns={entry_info.get('columns', [])})"
                    )
                else:
                    new_label_parts.append(lbl)
            result_text += f" | New data: {'; '.join(new_label_parts)}"
        elif task.status == TaskStatus.FAILED:
            result_text += " | No new data added."

        warnings = []
        for tr in getattr(task, "tool_results", []):
            if tr.get("quality_warning"):
                warnings.append(f"{tr.get('label', '?')}: {tr['quality_warning']}")
            if tr.get("time_range_note"):
                warnings.append(tr["time_range_note"])
        if warnings:
            result_text += f" | Warnings: {'; '.join(warnings)}"

        return {
            "description": task.description,
            "status": task.status.value,
            "result_summary": result_text,
            "error": task.error,
        }

    def _execute_plan_task(self, task: Task, plan: TaskPlan) -> None:
        """Execute a single plan task, routing to the appropriate agent.

        Updates the task status in place.

        Args:
            task: The task to execute
            plan: The parent plan (for logging context)
        """
        mission_tag = f" [{task.mission}]" if task.mission else ""
        self._event_bus.emit(PROGRESS, level="debug", msg=f"[Plan]{mission_tag}: {task.description}")

        # Inject canonical time range so all tasks use the same dates
        if self._plan_time_range:
            tr_str = self._plan_time_range.to_time_range_string()
            task.instruction += f"\n\nCanonical time range for this plan: {tr_str}"

        # Inject current data-store contents so sub-agents know what's available
        store = get_store()
        entries = store.list_entries()
        if entries:
            labels = [
                f"  - {e['label']} (columns: {e.get('columns', [])}, {e['num_points']} pts)"
                for e in entries
            ]
            task.instruction += "\n\nData currently in memory:\n" + "\n".join(labels)

        special_missions = {"__visualization__", "__data_ops__", "__data_extraction__"}

        if task.mission == "__visualization__":
            instr_lower = task.instruction.lower()
            is_export = "export" in instr_lower or ".png" in instr_lower or ".pdf" in instr_lower

            if is_export:
                # Export is a simple dispatch — handle directly, no need for LLM
                self._handle_export_task(task)
            else:
                # Plot tasks: ensure instruction includes actual labels
                has_tool_ref = "render_plotly_json" in instr_lower
                if not has_tool_ref and entries:
                    all_labels = ",".join(e["label"] for e in entries)
                    task.instruction = (
                        f"Use render_plotly_json to plot {all_labels}. "
                        f"Original request: {task.instruction}"
                    )

                # Inject current plot state for the viz agent
                state = self._renderer.get_current_state()
                if state["has_plot"]:
                    if state.get("figure_json"):
                        import json
                        fig_json_str = json.dumps(state["figure_json"], indent=2)
                        task.instruction += (
                            f"\n\nCurrently displayed: {state['traces']}"
                            f"\n\nCurrent figure_json (modify this, don't rebuild from scratch):\n{fig_json_str}"
                        )
                    else:
                        task.instruction += f"\n\nCurrently displayed: {state['traces']}"
                else:
                    task.instruction += "\n\nNo plot currently displayed."

                task.instruction += self._build_agent_history("viz", agent_name="Visualization Agent")
                task.instruction = self._inject_memory(task.instruction, "visualization")
                self._get_or_create_viz_agent().execute_task(task)
        elif task.mission == "__data_ops__":
            task.instruction += self._build_agent_history("dataops", agent_name="DataOps Agent")
            task.instruction = self._inject_memory(task.instruction, "data_ops")
            self._get_or_create_dataops_agent().execute_task(task)
        elif task.mission == "__data_extraction__":
            self._get_or_create_data_extraction_agent().execute_task(task)
        elif task.mission and task.mission not in special_missions:
            # Inject candidate datasets into instruction for mission agent
            if task.candidate_datasets:
                ds_list = ", ".join(task.candidate_datasets)
                task.instruction += f"\n\nCandidate datasets to inspect: {ds_list}"
            task.instruction += self._build_agent_history("mission")
            task.instruction = self._inject_memory(task.instruction, f"mission:{task.mission}")
            try:
                agent = self._get_or_create_mission_agent(task.mission)
                agent.execute_task(task)
            except (KeyError, FileNotFoundError):
                self._event_bus.emit(DEBUG, level="debug", msg=f"[Plan] Unknown mission '{task.mission}', using main agent")
                self._execute_task(task)
        else:
            self._execute_task(task)

    def _handle_export_task(self, task: Task) -> None:
        """Handle an export task directly without the VisualizationAgent.

        Export is a simple dispatch call — no LLM reasoning needed.
        Extracts the filename from the task instruction and calls the
        renderer's export method directly.

        Args:
            task: The export task to execute
        """
        import re

        task.status = TaskStatus.IN_PROGRESS
        task.tool_calls = []

        # Extract filename from instruction
        fn_match = re.search(r'[\w.-]+\.(?:png|pdf|svg)', task.instruction, re.IGNORECASE)
        filename = fn_match.group(0) if fn_match else "output.png"

        self._event_bus.emit(DEBUG, level="debug", msg=f"[Plan] Direct export: {filename}")
        task.tool_calls.append("export")

        result = self._renderer.export(filename)
        # Auto-open in non-GUI/non-web mode
        if result.get("status") == "success" and not self.gui_mode and not self.web_mode:
            try:
                import os
                import platform
                filepath = result["filepath"]
                if platform.system() == "Windows":
                    os.startfile(filepath)
                elif platform.system() == "Darwin":
                    import subprocess
                    subprocess.Popen(["open", filepath])
                else:
                    import subprocess
                    subprocess.Popen(["xdg-open", filepath])
                result["auto_opened"] = True
            except Exception as e:
                self._event_bus.emit(DEBUG, level="debug", msg=f"[Export] Could not auto-open: {e}")
                result["auto_opened"] = False
        if result.get("status") == "success":
            task.status = TaskStatus.COMPLETED
            task.result = f"Exported plot to {result.get('filepath', filename)}"
        else:
            task.status = TaskStatus.FAILED
            task.error = result.get("message", "Export failed")
            task.result = f"Export failed: {task.error}"

    def _extract_time_range(self, text: str):
        """Try to extract a resolved TimeRange from a user message.

        Uses parse_time_range() on the full text and common sub-patterns.
        Returns a TimeRange on success, or None if parsing fails.
        """
        import re as _re

        # Try the whole text first (works for "ACE mag for 2024-01-01 to 2024-01-15")
        try:
            return parse_time_range(text)
        except (TimeRangeError, ValueError):
            pass

        # Try to extract a "for <time_expr>" or "from <time_expr>" clause
        for pattern in [
            r'\bfor\s+(.+?)(?:\s*$)',
            r'\bfrom\s+(\d{4}.+?)(?:\s*$)',
            r'\bduring\s+(.+?)(?:\s*$)',
            r'(\d{4}-\d{2}-\d{2}\s+to\s+\d{4}-\d{2}-\d{2})',
            r'(\d{4}-\d{2}-\d{2}T[\d:]+\s+to\s+\d{4}-\d{2}-\d{2}T[\d:]+)',
            r'((?:last\s+(?:\d+\s+)?(?:week|day|month|year))s?)',
            r'((?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+\d{4})',
        ]:
            match = _re.search(pattern, text, _re.IGNORECASE)
            if match:
                try:
                    return parse_time_range(match.group(1).strip())
                except (TimeRangeError, ValueError):
                    continue

        return None

    def _handle_planning_request(
        self, user_message: str, *, structured_time_range: str = ""
    ) -> str:
        """Process a complex multi-step request using the plan-execute-replan loop."""
        self._event_bus.emit(DEBUG, level="debug", msg="[PlannerAgent] Starting planner for complex request...")

        # Prefer the structured time_range from the tool call (Gemini-resolved).
        # Fall back to regex extraction only when the structured param is empty.
        self._plan_time_range = None
        if structured_time_range:
            try:
                self._plan_time_range = parse_time_range(structured_time_range)
                self._event_bus.emit(DEBUG, level="debug", msg=f"[PlannerAgent] Resolved time range (structured): "
                    f"{self._plan_time_range.to_time_range_string()}")
            except (TimeRangeError, ValueError) as e:
                self._event_bus.emit(DEBUG, level="debug", msg=f"[PlannerAgent] Structured time_range parse failed: {e}")

        if not self._plan_time_range:
            # Strip memory context before extracting time range — memory contains
            # date references from past sessions that confuse the regex.
            import re as _re
            clean_msg = _re.sub(
                r'\[CONTEXT FROM LONG-TERM MEMORY\].*?\[END MEMORY CONTEXT\]\s*',
                '', user_message, flags=_re.DOTALL
            )
            self._plan_time_range = self._extract_time_range(clean_msg)
            if self._plan_time_range:
                self._event_bus.emit(DEBUG, level="debug", msg=f"[PlannerAgent] Resolved time range (regex fallback): "
                    f"{self._plan_time_range.to_time_range_string()}")

        planner = self._get_or_create_planner_agent()

        # Build planning message, injecting resolved time range if available
        planning_msg = user_message
        if self._plan_time_range:
            tr_str = self._plan_time_range.to_time_range_string()
            planning_msg = f"{user_message}\n\nResolved time range: {tr_str}. Use this exact range for ALL fetch tasks."

        # Inject planner-level session history
        planner_history = self._build_agent_history("planner")
        if planner_history:
            planning_msg += planner_history

        # Round 1: initial planning
        response = planner.start_planning(planning_msg)
        if response is None:
            self._event_bus.emit(DEBUG, level="debug", msg="[PlannerAgent] Planner failed, falling back to direct execution")
            return self._process_single_message(user_message)

        plan = create_plan(user_message, [])
        self._current_plan = plan
        plan.status = PlanStatus.EXECUTING
        store = get_task_store()
        store.save(plan)

        log_plan_event("created", plan.id, f"Dynamic plan for: {user_message[:50]}...")

        # Surface plan summary in web UI live log
        plan_summary = response.get("summary") or response.get("reasoning", "")
        if plan_summary:
            if len(plan_summary) > 300:
                plan_summary = plan_summary[:300] + "..."
            self._event_bus.emit(PROGRESS, level="debug", msg=f"[Planning] {plan_summary}")

        tasks_preview = response.get("tasks", [])
        if tasks_preview:
            task_lines = []
            for i, t in enumerate(tasks_preview, 1):
                desc = t.get("description", "?")
                ds = t.get("candidate_datasets")
                ds_str = f" ({', '.join(ds)})" if ds else ""
                task_lines.append(f"  {i}. {desc}{ds_str}")
            self._event_bus.emit(DEBUG, level="debug", msg="[Planning] Tasks:\n" + "\n".join(task_lines))

        round_num = 0
        while round_num < MAX_ROUNDS:
            round_num += 1

            if self._cancel_event.is_set():
                self._event_bus.emit(DEBUG, level="info", msg="[Cancel] Stopping plan loop between rounds")
                break

            tasks_data = response.get("tasks", [])

            if not tasks_data and response.get("status") == "done":
                break

            if not tasks_data:
                # Empty tasks with "continue" — treat as done
                break

            # Create Task objects for this batch
            new_tasks = []
            all_candidates_invalid = False
            for td in tasks_data:
                mission = td.get("mission")
                if isinstance(mission, str) and mission.lower() in ("null", "none", ""):
                    mission = None
                task = create_task(
                    description=td["description"],
                    instruction=td["instruction"],
                    mission=mission,
                )
                task.round = round_num
                task.candidate_datasets = td.get("candidate_datasets")

                # Validate candidate_datasets against local metadata cache
                if task.candidate_datasets:
                    valid = []
                    invalid = []
                    for ds_id in task.candidate_datasets:
                        v = validate_dataset_id(ds_id)
                        if v["valid"]:
                            valid.append(ds_id)
                        else:
                            invalid.append(ds_id)
                    if invalid:
                        self._event_bus.emit(DEBUG, level="debug", msg=f"[Plan] Stripped invalid candidate_datasets "
                            f"from '{task.description}': {invalid}")
                    if valid:
                        task.candidate_datasets = valid
                    else:
                        # ALL candidates invalid — flag for re-prompt
                        self._event_bus.emit(DEBUG, level="warning", msg=f"[Plan] ALL candidate_datasets invalid for "
                            f"'{task.description}': {invalid}")
                        all_candidates_invalid = True
                new_tasks.append(task)

            # Log validation summary (round 1 is the long one)
            if round_num == 1:
                valid_count = sum(1 for t in new_tasks if t.candidate_datasets)
                total = len(new_tasks)
                self._event_bus.emit(DEBUG, level="debug", msg=f"[Planning] Validated datasets for {valid_count}/{total} tasks")

            # If any task has all-invalid candidates, re-prompt the planner
            if all_candidates_invalid:
                invalid_ids = []
                for t in new_tasks:
                    if t.candidate_datasets:
                        for ds_id in t.candidate_datasets:
                            v = validate_dataset_id(ds_id)
                            if not v["valid"]:
                                invalid_ids.append(ds_id)
                correction_msg = (
                    "VALIDATION ERROR: The following dataset IDs do not exist "
                    f"in the local metadata cache: {invalid_ids}. "
                    "Re-emit the same tasks using ONLY dataset IDs from the "
                    "Discovery Results. Do NOT invent dataset IDs."
                )
                self._event_bus.emit(DEBUG, level="debug", msg=f"[Plan] Sending correction to planner: {correction_msg}")
                response = planner.continue_planning(
                    [{"description": "Dataset ID validation",
                      "status": "failed",
                      "result_summary": correction_msg,
                      "error": correction_msg}],
                    round_num=round_num,
                    max_rounds=MAX_ROUNDS,
                )
                if response is None:
                    self._event_bus.emit(DEBUG, level="debug", msg="[PlannerAgent] Planner error after correction, finalizing")
                    break
                # Re-process the corrected response in the next loop iteration
                continue

            plan.add_tasks(new_tasks)
            store.save(plan)

            self._event_bus.emit(DEBUG, level="debug", msg=f"[PlannerAgent] Round {round_num}: {len(new_tasks)} tasks "
                f"(status={response['status']})")
            self._event_bus.emit(DEBUG, level="debug", msg=format_plan_for_display(plan))
            self._event_bus.emit(DEBUG, level="debug", msg=f"[Plan] Executing {len(new_tasks)} task(s) (round {round_num})...")

            # Execute batch — partition into parallelizable fetch tasks and serial tasks
            special_missions = {"__visualization__", "__data_ops__", "__data_extraction__"}
            fetch_tasks = [t for t in new_tasks if t.mission and t.mission not in special_missions]
            other_tasks = [t for t in new_tasks if t not in fetch_tasks]

            round_results = []
            cancelled = False

            # Run fetch tasks in parallel if multiple independent missions
            from config import PARALLEL_FETCH
            if PARALLEL_FETCH and len(fetch_tasks) > 1 and not self._cancel_event.is_set():
                self._event_bus.emit(DEBUG, level="debug", msg=f"[Parallel] Executing {len(fetch_tasks)} fetch tasks concurrently: "
                    f"{[t.mission for t in fetch_tasks]}")
                labels_before = set(e['label'] for e in get_store().list_entries())
                max_workers = min(len(fetch_tasks), 3)

                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {
                        pool.submit(contextvars.copy_context().run, self._execute_plan_task, t, plan): t
                        for t in fetch_tasks
                    }
                    for future in as_completed(futures):
                        t = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            t.status = TaskStatus.FAILED
                            t.error = str(e)

                labels_after = set(e['label'] for e in get_store().list_entries())
                # Build result summaries for all parallel tasks
                for task in fetch_tasks:
                    round_results.append(
                        self._build_task_result_summary(task, labels_before, labels_after)
                    )
                store.save(plan)
            else:
                # Run fetch tasks serially (0 or 1 task)
                other_tasks = list(new_tasks)  # all tasks go serial
                fetch_tasks = []

            # Run remaining tasks serially (viz, data_ops, data_extraction, or single fetches)
            for i, task in enumerate(other_tasks):
                if self._cancel_event.is_set():
                    self._event_bus.emit(DEBUG, level="info", msg="[Cancel] Stopping plan mid-batch")
                    for remaining in other_tasks[i:]:
                        remaining.status = TaskStatus.SKIPPED
                        remaining.error = "Cancelled by user"
                    cancelled = True
                    break
                labels_before = set(e['label'] for e in get_store().list_entries())
                self._execute_plan_task(task, plan)
                labels_after = set(e['label'] for e in get_store().list_entries())
                round_results.append(
                    self._build_task_result_summary(task, labels_before, labels_after)
                )
                store.save(plan)

            if cancelled:
                store.save(plan)
                break

            # Append current store state so planner knows what data exists
            store_entries = get_store().list_entries()
            if store_entries:
                store_details = [
                    {"label": e["label"], "columns": e.get("columns", []),
                     "shape": e.get("shape", ""), "units": e.get("units", ""),
                     "num_points": e.get("num_points", 0)}
                    for e in store_entries
                ]
                for r in round_results:
                    r["data_in_memory"] = store_details

            if response.get("status") == "done":
                break

            # Replan: send results back to planner with round budget
            response = planner.continue_planning(
                round_results,
                round_num=round_num,
                max_rounds=MAX_ROUNDS,
            )
            if response is None:
                self._event_bus.emit(DEBUG, level="debug", msg="[PlannerAgent] Planner error mid-plan, finalizing")
                break

        # Finalize
        if self._cancel_event.is_set():
            plan.status = PlanStatus.CANCELLED
            log_plan_event("cancelled", plan.id, plan.progress_summary())
        elif plan.get_failed_tasks():
            plan.status = PlanStatus.FAILED
            log_plan_event("failed", plan.id, plan.progress_summary())
        else:
            plan.status = PlanStatus.COMPLETED
            log_plan_event("completed", plan.id, plan.progress_summary())
        store.save(plan)
        planner.reset()

        summary = self._summarize_plan_execution(plan)
        self._current_plan = None
        self._plan_time_range = None

        return summary

    def _process_single_message(self, user_message: str) -> str:
        """Process a single (non-complex) user message."""
        self._event_bus.emit(DEBUG, level="debug", msg="[LLM] Sending message to model...")
        self._last_tool_context = "initial_message"
        response = self._send_message(user_message)
        self._track_usage(response)

        self._event_bus.emit(DEBUG, level="debug", msg="[LLM] Response received.")
        self._log_grounding_queries(response)

        max_iterations = 10
        iteration = 0
        consecutive_delegation_errors = 0

        while iteration < max_iterations:
            iteration += 1

            if self._cancel_event.is_set():
                self._event_bus.emit(DEBUG, level="info", msg="[Cancel] Stopping orchestrator loop")
                break

            if not response.tool_calls:
                break

            function_calls = response.tool_calls

            # Execute tools — parallel when safe, serial otherwise
            tool_results = self._execute_tools_parallel(function_calls)

            function_responses = []
            has_delegation_error = False

            # First pass: logging, delegation tracking, observations
            for tc_id, tool_name, tool_args, result in tool_results:
                if result.get("status") == "error":
                    self._event_bus.emit(DEBUG, level="warning", msg=f"[Tool Result: ERROR] {result.get('message', '')}")

                # Track delegation failures (sub-agent stopped due to errors)
                if tool_name.startswith("delegate_to_") and result.get("status") == "error":
                    has_delegation_error = True
                    sub_text = result.get("result", "")
                    if sub_text:
                        self._event_bus.emit(DEBUG, level="debug", msg=f"[Delegation Failed] {tool_name} sub-agent response: {sub_text}")

                # Inject structured observation summary
                if config.OBSERVATION_SUMMARIES:
                    from .observations import generate_observation
                    result["observation"] = generate_observation(tool_name, tool_args, result)

            # Append reflection hint when ALL tools in a round failed
            if config.SELF_REFLECTION and len(tool_results) > 0:
                all_errors = all(r.get("status") == "error" for _, _, _, r in tool_results)
                if all_errors:
                    last_result = tool_results[-1][3]
                    reflection = (
                        " ALL tool calls in this round failed. "
                        "Before retrying, analyze what went wrong "
                        "and try a different approach — different parameters, "
                        "datasets, or strategy."
                    )
                    last_result["observation"] = last_result.get("observation", "") + reflection

            # Second pass: build function responses (after observations are finalized)
            for tc_id, tool_name, tool_args, result in tool_results:
                # Handle clarification specially - return immediately
                if result.get("status") == "clarification_needed":
                    question = result["question"]
                    if result.get("context"):
                        question = f"{result['context']}\n\n{question}"
                    if result.get("options"):
                        question += "\n\nOptions:\n" + "\n".join(
                            f"  {i+1}. {opt}" for i, opt in enumerate(result["options"])
                        )
                    return question

                function_responses.append(
                    self.adapter.make_tool_result_message(
                        tool_name, result, tool_call_id=tc_id
                    )
                )

            # Track consecutive delegation failures
            if has_delegation_error:
                consecutive_delegation_errors += 1
            else:
                consecutive_delegation_errors = 0

            tool_names = [fc.name for fc in function_calls]
            self._last_tool_context = "+".join(tool_names)

            if consecutive_delegation_errors >= 2:
                self._event_bus.emit(DEBUG, level="debug", msg=f"[Orchestrator] {consecutive_delegation_errors} consecutive delegation failures, stopping retries")
                # Send results back one more time so LLM can produce a final text answer
                response = self._send_message(function_responses)
                self._track_usage(response)
        
                break

            self._event_bus.emit(DEBUG, level="debug", msg=f"[LLM] Sending {len(function_responses)} tool result(s) back to model...")
            response = self._send_message(function_responses)
            self._track_usage(response)
    
            self._event_bus.emit(DEBUG, level="debug", msg="[LLM] Response received.")
            self._log_grounding_queries(response)

        # Extract text response
        text = response.text or "Done."
        text += self._extract_grounding_sources(response)
        return text

    def _inject_memory(self, request: str, scope: str) -> str:
        """Append scoped memory section to a delegation request string."""
        section = self._memory_store.format_for_injection(scope=scope)
        return f"{request}\n\n{section}" if section else request

    def _build_agent_history(self, agent_type: str, agent_name: str | None = None) -> str:
        """Build session history context for an agent from EventBus events.

        Queries events tagged with ``ctx:{agent_type}`` and formats them into
        a concise summary injected before delegation so the agent knows
        what happened earlier in the session.

        For orchestrator/planner: uses terse status-only formatting.
        For sub-agents (mission/viz/dataops): uses detailed formatting.

        When the formatted history exceeds the token budget, an LLM call
        compacts it into a shorter summary. Results are cached by
        (agent_type, event_count) to avoid redundant LLM calls.

        Args:
            agent_type: One of "mission", "viz", "dataops", "planner",
                "orchestrator".
            agent_name: Optional agent name filter for SUB_AGENT_TOOL/ERROR
                events. For mission agents this is None (all mission activity
                is relevant). For viz/dataops, pass the agent name to filter
                to that agent's events.

        Returns:
            Formatted history string, or "" if no relevant events.
        """
        events = self._event_bus.get_events(tags={f"ctx:{agent_type}"})
        if not events:
            return ""

        # Check cache — return cached if (agent_type, event_count) matches
        cache_key = agent_type
        cached = self._compacted_history_cache.get(cache_key)
        if cached and cached[0] == len(events):
            return cached[1]

        # Format events — orchestrator/planner use terse formatting
        use_terse = agent_type in ("orchestrator", "planner")
        lines = []
        for ev in events:
            if use_terse:
                line = self._format_orchestrator_history_event(ev)
            else:
                line = self._format_history_event(ev, agent_type, agent_name)
            if line is not None:
                lines.append(line)

        if not lines:
            return ""

        raw_text = "\n".join(lines)

        # Select budget based on agent type
        if agent_type in ("orchestrator", "planner"):
            budget = config.get("history_budget_orchestrator", 20000)
        else:
            budget = config.get("history_budget_sub_agent", 10000)

        tokens = estimate_tokens(raw_text)
        if tokens <= budget:
            result = "\n\nSession history (what happened earlier):\n" + raw_text
        else:
            # Over budget — try LLM compaction, fall back to truncation
            compacted = self._compact_history(raw_text, agent_type, budget)
            result = "\n\nSession history (what happened earlier):\n" + compacted

        # Cache result
        self._compacted_history_cache[cache_key] = (len(events), result)
        return result

    def _compact_history(self, raw_text: str, agent_type: str, budget: int) -> str:
        """Compact session history via a single LLM call.

        Uses SMART_MODEL with temperature 0.1 to summarize the history
        while preserving all errors/failures and recent events.
        LLM errors propagate naturally — if the LLM is unreachable,
        the entire agent can't function anyway.
        """
        role_descriptions = {
            "mission": "a spacecraft data fetching specialist",
            "viz": "a data visualization specialist",
            "dataops": "a data transformation/computation specialist",
            "planner": "a multi-step planning orchestrator",
            "orchestrator": "the top-level orchestrator that routes user requests to sub-agents",
        }
        role = role_descriptions.get(agent_type, "an AI agent")
        target_tokens = budget // 2
        before_tokens = estimate_tokens(raw_text)

        prompt = (
            f"You are compacting a session activity log for {role}. "
            f"A fresh agent instance needs this context to continue the session.\n\n"
            f"Rules:\n"
            f"- Preserve ALL errors, failures, and their details verbatim\n"
            f"- Preserve the most recent 5-10 events in full detail\n"
            f"- Summarize routine successes into groups (e.g. 'Fetched 5 ACE datasets')\n"
            f"- Keep chronological order\n"
            f"- Output ONLY the compacted log lines, no commentary\n"
            f"- Target length: ~{target_tokens} tokens\n\n"
            f"Session log to compact:\n{raw_text}"
        )

        response = self.adapter.generate(
            model=config.SMART_MODEL,
            contents=prompt,
            temperature=0.1,
        )

        # Track token cost so it appears in the WebUI token panel
        token_state = {
            "input": self._total_input_tokens,
            "output": self._total_output_tokens,
            "thinking": self._total_thinking_tokens,
            "cached": self._total_cached_tokens,
            "api_calls": self._api_calls,
        }
        track_llm_usage(
            response=response,
            token_state=token_state,
            agent_name="OrchestratorAgent",
            last_tool_context="stm_compaction",
        )
        self._total_input_tokens = token_state["input"]
        self._total_output_tokens = token_state["output"]
        self._total_thinking_tokens = token_state["thinking"]
        self._total_cached_tokens = token_state["cached"]
        self._api_calls = token_state["api_calls"]

        compacted = response.text.strip() if response and response.text else ""
        if not compacted:
            return raw_text

        after_tokens = estimate_tokens(compacted)
        self._event_bus.emit(
            STM_COMPACTION,
            agent="ShortTermMemory",
            level="info",
            msg=f"Short-term memory compacted for {agent_type}: {before_tokens}\u2192{after_tokens} tokens",
            data={
                "agent_type": agent_type,
                "before_tokens": before_tokens,
                "after_tokens": after_tokens,
                "compacted_text": compacted,
            },
        )
        return compacted

    def _format_orchestrator_history_event(self, ev) -> str | None:
        """Format a single event into a terse status-only line for orchestrator/planner.

        The orchestrator doesn't need detailed args — just outcomes and errors.
        Returns None to skip the event.
        """
        d = ev.data
        args = d.get("args", {})
        status = d.get("status", "")
        error = d.get("error")

        if ev.type == SUB_AGENT_TOOL:
            tool = d.get("tool_name", "?")
            if error or status == "error":
                return f"[{ev.agent}] {tool}: FAILED — {(error or ev.msg)[:100]}"
            return f"[{ev.agent}] {tool}: ok"

        elif ev.type == SUB_AGENT_ERROR:
            return f"[{ev.agent}] ERROR: {ev.msg[:100]}"

        elif ev.type == DATA_FETCHED:
            if args.get("already_loaded"):
                return None
            label = d.get("outputs", ["?"])[0] if d.get("outputs") else "?"
            return f"Fetched: {label}"

        elif ev.type == DATA_COMPUTED:
            label = d.get("outputs", ["?"])[0] if d.get("outputs") else "?"
            return f"Computed: {label}"

        elif ev.type == CUSTOM_OP_FAILURE:
            return f"Compute FAILED: {(error or ev.msg)[:100]}"

        elif ev.type == RENDER_EXECUTED:
            if error or status == "error":
                return "Viz: FAILED"
            return f"Viz: rendered ({status or 'ok'})"

        elif ev.type == RENDER_ERROR:
            return f"Viz: FAILED — {(error or ev.msg)[:80]}"

        elif ev.type == PLOT_ACTION:
            action = args.get("action", d.get("action", "?"))
            return f"Plot: {action}"

        elif ev.type == FETCH_ERROR:
            return f"Fetch FAILED: {ev.msg[:80]}"

        return None

    def _format_history_event(self, ev, agent_type: str, agent_name: str | None) -> str | None:
        """Format a single EventBus event into a concise history line.

        Returns None to skip the event.

        Follows the formatting conventions from ``MemoryAgent.build_curated_events()``.
        """
        d = ev.data
        args = d.get("args", {})
        status = d.get("status", "")
        error = d.get("error")

        if ev.type == SUB_AGENT_TOOL:
            # For mission context: include all agents' tool calls
            # For viz/dataops: only include events from that specific agent
            if agent_name and ev.agent != agent_name:
                return None
            tool = d.get("tool_name", "?")
            tool_args = str(d.get("tool_args", ""))[:100]
            return f"[{ev.agent}] called {tool}({tool_args})"

        elif ev.type == SUB_AGENT_ERROR:
            if agent_name and ev.agent != agent_name:
                return None
            return f"[{ev.agent}] ERROR: {ev.msg[:150]}"

        elif ev.type == DATA_FETCHED:
            # Skip already-loaded entries (cache hits)
            if args.get("already_loaded"):
                return None
            label = d.get("outputs", ["?"])[0] if d.get("outputs") else "?"
            ds = args.get("dataset_id", "?")
            param = args.get("parameter_id", "?")
            return f"Fetched: {label} ({ds}/{param})"

        elif ev.type == DATA_COMPUTED:
            label = d.get("outputs", ["?"])[0] if d.get("outputs") else "?"
            code = args.get("code", "")[:120]
            return f"Computed: {label} (code: {code})"

        elif ev.type == CUSTOM_OP_FAILURE:
            err = (error or ev.msg)[:150]
            code = args.get("code", "")[:120]
            return f"Compute FAILED: {err} (code: {code})"

        elif ev.type == RENDER_EXECUTED:
            inputs = d.get("inputs", [])
            label_str = ", ".join(str(i) for i in inputs) if inputs else "?"
            return f"Rendered: {label_str}"

        elif ev.type == RENDER_ERROR:
            err = (error or ev.msg)[:150]
            return f"Render FAILED: {err}"

        elif ev.type == PLOT_ACTION:
            action = args.get("action", d.get("action", "?"))
            return f"Plot action: {action}"

        elif ev.type == FETCH_ERROR:
            return f"Fetch FAILED: {ev.msg[:150]}"

        elif ev.type in (DELEGATION, DELEGATION_DONE):
            return f"Delegated: {ev.msg[:100]}"

        return None

    @staticmethod
    def _wrap_delegation_result(sub_result, store_snapshot=None) -> dict:
        """Convert a sub-agent process_request result into a tool result dict.

        If the sub-agent reported failure (stopped due to errors/loops),
        return status='error' so the orchestrator knows not to retry.

        Args:
            sub_result: Dict from sub-agent's process_request ({text, failed, errors}).
            store_snapshot: Optional list of store entry summaries to include,
                so the orchestrator LLM sees concrete data state after delegation.
        """
        if isinstance(sub_result, dict):
            text = sub_result.get("text", "")
            failed = sub_result.get("failed", False)
            errors = sub_result.get("errors", [])
        else:
            # Legacy: plain string (shouldn't happen, but be safe)
            text = str(sub_result)
            failed = False
            errors = []

        if failed and errors:
            error_summary = "; ".join(errors[-3:])  # last 3 errors
            result = {
                "status": "error",
                "message": f"Sub-agent failed. Errors: {error_summary}",
                "result": text,
            }
        else:
            result = {"status": "success", "result": text}

        if store_snapshot is not None:
            result["data_in_memory"] = [
                {"label": e["label"], "columns": e.get("columns", []),
                 "shape": e.get("shape", ""), "units": e.get("units", ""),
                 "num_points": e.get("num_points", 0)}
                for e in store_snapshot
            ]
        return result

    def _try_create_cache(self, agent, label: str) -> None:
        """Attempt to create an explicit context cache for a sub-agent."""
        if not isinstance(self.adapter, GeminiAdapter):
            return
        try:
            agent._cache_name = self.adapter.create_cache(
                model=agent.model_name,
                system_prompt=agent.system_prompt,
                tools=agent._tool_schemas,
            )
            self._event_bus.emit(DEBUG, level="debug", msg=f"Cache created for {label}: {agent._cache_name}")
        except Exception:
            pass  # fall back to no cache

    def _get_or_create_mission_agent(self, mission_id: str) -> MissionAgent:
        """Get a cached mission agent or create a new one. Thread-safe."""
        with self._mission_agents_lock:
            if mission_id not in self._mission_agents:
                agent = MissionAgent(
                    mission_id=mission_id,
                    adapter=self.adapter,
                    model_name=config.SUB_AGENT_MODEL,
                    tool_executor=self._execute_tool_safe,
                    verbose=self.verbose,
                    cancel_event=self._cancel_event,
                    event_bus=self._event_bus,
                )
                self._try_create_cache(agent, f"{mission_id} mission")
                self._mission_agents[mission_id] = agent
                self._event_bus.emit(DEBUG, level="debug", msg=f"[Router] Created {mission_id} mission agent ({config.SUB_AGENT_MODEL})")
            return self._mission_agents[mission_id]

    def _get_or_create_viz_agent(self) -> VisualizationAgent:
        """Get the cached visualization agent or create a new one."""
        if self._viz_agent is None:
            self._viz_agent = VisualizationAgent(
                adapter=self.adapter,
                model_name=config.SUB_AGENT_MODEL,
                tool_executor=self._execute_tool_safe,
                verbose=self.verbose,
                gui_mode=self.gui_mode,
                cancel_event=self._cancel_event,
                event_bus=self._event_bus,
            )
            self._try_create_cache(self._viz_agent, "visualization")
            self._event_bus.emit(DEBUG, level="debug", msg=f"[Router] Created Visualization agent ({config.SUB_AGENT_MODEL})")
        return self._viz_agent

    def _get_or_create_dataops_agent(self) -> DataOpsAgent:
        """Get the cached data ops agent or create a new one."""
        if self._dataops_agent is None:
            self._dataops_agent = DataOpsAgent(
                adapter=self.adapter,
                model_name=config.SUB_AGENT_MODEL,
                tool_executor=self._execute_tool_safe,
                verbose=self.verbose,
                cancel_event=self._cancel_event,
                event_bus=self._event_bus,
            )
            self._try_create_cache(self._dataops_agent, "dataops")
            self._event_bus.emit(DEBUG, level="debug", msg="[Router] Created DataOps agent")
        return self._dataops_agent

    def _get_or_create_data_extraction_agent(self) -> DataExtractionAgent:
        """Get the cached data extraction agent or create a new one."""
        if self._data_extraction_agent is None:
            self._data_extraction_agent = DataExtractionAgent(
                adapter=self.adapter,
                model_name=config.SUB_AGENT_MODEL,
                tool_executor=self._execute_tool_safe,
                verbose=self.verbose,
                cancel_event=self._cancel_event,
                event_bus=self._event_bus,
            )
            self._event_bus.emit(DEBUG, level="debug", msg="[Router] Created DataExtraction agent")
        return self._data_extraction_agent

    # ---- Long-term memory (end-of-session) ----

    def _maybe_extract_discoveries(self) -> None:
        """Trigger async discovery extraction if enough new ops have occurred.

        Runs on a daemon thread using INLINE_MODEL (cheapest).
        Lock prevents concurrent extractions.
        """
        ops_log = get_operations_log()
        current_count = len(ops_log)
        if current_count <= self._last_discovery_op_count:
            return  # No new operations since last extraction

        if not self._discovery_lock.acquire(blocking=False):
            return  # Another extraction already running

        try:
            # Snapshot data on main thread
            ops_records = ops_log.get_records()
            try:
                history = self.chat.get_history()
                turns = _extract_turns(history, max_text=500)
            except Exception:
                turns = []

            session_id = self._session_id or "unknown"
            self._last_discovery_op_count = current_count

            def _run():
                try:
                    agent = DiscoveryAgent(
                        adapter=self.adapter,
                        model_name=config.INLINE_MODEL,
                        discovery_store=self._discovery_store,
                        session_id=session_id,
                    )
                    agent.extract_discoveries(ops_records, turns)
                    # Flush discoveries to disk
                    self._discovery_store.save()
                except Exception as e:
                    self._event_bus.emit(DEBUG, level="debug", msg=f"[Discovery] Async extraction failed: {e}")
                finally:
                    self._discovery_lock.release()

            t = threading.Thread(target=_run, daemon=True)
            t.start()
        except Exception:
            self._discovery_lock.release()

    def _maybe_extract_memories(self) -> None:
        """Trigger async memory extraction with full session context.

        Runs on a daemon thread using INLINE_MODEL (cheapest).
        Lock prevents concurrent extractions. The MemoryAgent sees
        memory-tagged events from the EventBus, curated into concise summaries.
        """
        # Check if there are new memory-tagged events since last extraction
        memory_events = self._event_bus.get_events(tags={"memory"}, since_index=self._last_memory_op_index)
        if not memory_events:
            return  # No new events since last extraction

        if not self._memory_lock.acquire(blocking=False):
            return  # Another extraction already running

        try:
            # Detect active scopes from active sub-agents
            active_scopes = ["generic"]
            if self._viz_agent is not None:
                active_scopes.append("visualization")
            if self._dataops_agent is not None:
                active_scopes.append("data_ops")
            for mission_id in self._mission_agents:
                active_scopes.append(f"mission:{mission_id}")

            # Convert EventBus events to dicts and curate
            raw_events = [
                {"event": ev.type, "agent": ev.agent, "msg": ev.msg, **(ev.data or {})}
                for ev in memory_events
            ]
            curated = MemoryAgent.build_curated_events(raw_events)

            # Load active memories for active scopes only.
            # Skip review-type entries but attach their feedback to the target.
            from .memory import MEMORY_TOKEN_BUDGET
            injected_ids = self._memory_store._last_injected_ids
            active_memories = []
            for m in self._memory_store.get_enabled():
                if m.type == "review":
                    continue
                if not any(s in m.scopes for s in active_scopes):
                    continue
                entry = {
                    "id": m.id, "type": m.type,
                    "scopes": m.scopes, "content": m.content,
                    "injected": m.id in injected_ids,
                    "version": m.version,
                    "access_count": m.access_count,
                    "created_at": m.created_at,
                }
                # Attach review feedback
                reviews = self._memory_store.get_reviews_for(m.id)
                if reviews:
                    entry["reviews"] = []
                    for r in reviews:
                        agent_tag = next(
                            (t for t in r.tags if t and not t.startswith("review:") and not t.startswith("stars:")),
                            "",
                        )
                        entry["reviews"].append({"agent": agent_tag, "feedback": r.content, "date": r.created_at})
                # Attach version history (previous versions)
                if m.supersedes:
                    history = []
                    prev_id = m.supersedes
                    seen = set()
                    while prev_id and prev_id not in seen:
                        seen.add(prev_id)
                        prev = self._memory_store.get_by_id(prev_id)
                        if prev is None:
                            break
                        history.append({"version": prev.version, "content": prev.content, "date": prev.created_at})
                        prev_id = prev.supersedes
                    if history:
                        entry["previous_versions"] = history
                active_memories.append(entry)

            self._last_memory_op_index = len(self._event_bus._events)

            context = MemoryContext(
                events=curated,
                active_memories=active_memories,
                active_scopes=active_scopes,
                token_budget=MEMORY_TOKEN_BUDGET,
                total_memory_tokens=self._memory_store.total_tokens(),
            )

            session_id = self._session_id or ""
            bus = self._event_bus  # capture before thread (ContextVar won't propagate)

            def _run():
                set_event_bus(bus)  # propagate session bus to daemon thread
                try:
                    bus.emit(MEMORY_EXTRACTION_START,
                             agent="Memory", level="info",
                             msg="[Memory] Extraction started",
                             data={"curated_events": len(curated), "active_scopes": active_scopes})

                    if self._memory_agent is None:
                        self._memory_agent = MemoryAgent(
                            adapter=self.adapter,
                            model_name=config.SMART_MODEL,
                            memory_store=self._memory_store,
                            verbose=self.verbose,
                            session_id=session_id,
                            event_bus=bus,
                        )
                    executed = self._memory_agent.run(context)

                    # Tally actions by type
                    counts = {}
                    for action in (executed or []):
                        atype = action.get("action", "unknown")
                        counts[atype] = counts.get(atype, 0) + 1

                    bus.emit(MEMORY_EXTRACTION_DONE,
                             agent="Memory", level="info",
                             msg=f"[Memory] Extraction complete: {counts}" if counts else "[Memory] Extraction complete: no changes",
                             data={"actions": counts})
                except Exception as e:
                    bus.emit(MEMORY_EXTRACTION_ERROR,
                             agent="Memory", level="warning",
                             msg=f"[Memory] Extraction failed: {e}")
                finally:
                    self._memory_lock.release()

            t = threading.Thread(target=_run, daemon=True)
            t.start()
        except Exception:
            self._memory_lock.release()

    def generate_follow_ups(self, max_suggestions: int = 3) -> list[str]:
        """Generate contextual follow-up suggestions based on the conversation.

        Uses a lightweight single-shot Gemini call (Flash model) to produce
        2-3 short, actionable follow-up questions the user might ask next.

        Returns:
            List of suggestion strings, or [] on any failure.
        """
        try:
            history = self.chat.get_history()
        except Exception:
            return []

        # Build context from last 6 turns
        turns = _extract_turns(history[-6:], max_text=300)

        if not turns:
            return []

        conversation_text = "\n".join(turns)

        # DataStore context
        store = get_store()
        labels = [e["label"] for e in store.list_entries()]
        data_context = f"Data in memory: {', '.join(labels)}" if labels else "No data in memory yet."

        has_plot = self._renderer.get_figure() is not None
        plot_context = "A plot is currently displayed." if has_plot else "No plot is displayed."

        prompt = f"""Based on this conversation, suggest {max_suggestions} short follow-up questions the user might ask next.

{conversation_text}

{data_context}
{plot_context}

Respond with a JSON array of strings only (no markdown fencing). Each suggestion should be:
- A natural, conversational question (max 12 words)
- Actionable — something the agent can actually do
- Different from what was already asked
- Related to the current context (data, plots, spacecraft)

Example: ["Compare this with solar wind speed", "Zoom in to January 10-15", "Export the plot as PDF"]"""

        try:
            response = self.adapter.generate(
                model=get_active_model(config.INLINE_MODEL),
                contents=prompt,
                temperature=0.7,
            )
            self._last_tool_context = "follow_up_suggestions"
            self._track_usage(response)

            text = (response.text or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text[:-3].strip()

            import json
            suggestions = json.loads(text)
            if isinstance(suggestions, list):
                return [s for s in suggestions if isinstance(s, str)][:max_suggestions]
        except Exception as e:
            self._event_bus.emit(DEBUG, level="debug", msg=f"[FollowUp] Generation failed: {e}")

        return []

    def generate_session_title(self) -> Optional[str]:
        """Generate a short title from the first exchange via INLINE_MODEL."""
        try:
            history = self.chat.get_history()
        except Exception:
            return None
        turns = _extract_turns(history[:4], max_text=500)
        if not turns:
            return None
        conversation_text = "\n".join(turns)
        prompt = (
            "Generate a short title (5-8 words) for this conversation. "
            "Summarize the user's main intent.\n\n"
            f"{conversation_text}\n\n"
            "Respond with ONLY the title text, no quotes, no punctuation at the end."
        )
        try:
            response = self.adapter.generate(
                model=get_active_model(config.INLINE_MODEL),
                contents=prompt,
                temperature=0.3,
            )
            self._track_usage(response)
            text = (response.text or "").strip().strip("\"'")
            if text and len(text) <= 100:
                return text
        except Exception as e:
            self._event_bus.emit(DEBUG, level="debug", msg=f"[SessionTitle] Generation failed: {e}")
        return None

    def generate_inline_completions(
        self, partial: str, max_completions: int = 3
    ) -> list[str]:
        """Complete the user's partial input using the LLM.

        Returns full sentences that start with or continue from *partial*.
        Uses the cheapest model for low latency.
        """
        from knowledge.prompt_builder import build_inline_completion_prompt

        try:
            history = self.chat.get_history()
        except Exception:
            history = []

        # Last 4 turns for context
        turns = _extract_turns((history or [])[-4:], max_text=200)

        store = get_store()
        labels = [e["label"] for e in store.list_entries()]

        prompt = build_inline_completion_prompt(
            partial,
            conversation_context="\n".join(turns),
            memory_section=self._memory_store.format_for_injection(
                scope="generic", include_summaries=True, include_review_instruction=False,
            ),
            data_labels=labels,
            max_completions=max_completions,
        )

        try:
            response = self.adapter.generate(
                model=get_active_model(config.INLINE_MODEL),
                contents=prompt,
                temperature=0.5,
                max_output_tokens=100,
            )
            self._last_tool_context = "inline_completion"
            self._track_usage(response)

            text = (response.text or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text[:-3].strip()

            import json
            completions = json.loads(text)
            if isinstance(completions, list):
                # Return full completions — the frontend caches them
                # and matches as the user types, stripping the prefix
                # itself for ghost text display
                valid = []
                for c in completions:
                    if isinstance(c, str) and c.startswith(partial) and len(c) > len(partial):
                        valid.append(c)
                return valid[:max_completions]
        except Exception as e:
            self._event_bus.emit(DEBUG, level="debug", msg=f"[InlineComplete] Generation failed: {e}")

        return []

    # ---- Session persistence ----

    def _start_event_log_writer(self) -> None:
        """Create and subscribe an EventLogWriter for the current session directory."""
        # Close any existing writer first
        if self._event_log_writer is not None:
            self._event_bus.unsubscribe(self._event_log_writer)
            self._event_log_writer.close()
        session_dir = self._session_manager.base_dir / self._session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        self._event_log_writer = EventLogWriter(session_dir / "events.jsonl")
        self._event_bus.subscribe(self._event_log_writer)

    def _start_token_log_listener(self) -> None:
        """Create and subscribe a TokenLogListener for the current session."""
        # Close any existing listener first
        if self._token_log_listener is not None:
            self._event_bus.unsubscribe(self._token_log_listener)
            self._token_log_listener.close()
        token_log_path = LOG_DIR / f"token_{self._session_id}.jsonl"
        self._token_log_listener = TokenLogListener(token_log_path)
        self._event_bus.subscribe(self._token_log_listener)

    def start_session(self) -> str:
        """Create a new session on disk, attach the log file, and enable auto-save.

        The session directory and metadata are created immediately so the
        session appears in the sidebar right away (with a default name).
        Empty sessions from previous runs are cleaned up first.

        Returns:
            The new session_id.
        """
        self._session_manager.cleanup_empty_sessions()
        self._session_id = self._session_manager.create_session(self.model_name)
        self._auto_save = True
        self._session_title_generated = False
        set_session_id(self._session_id)
        attach_log_file(self._session_id)
        self._start_token_log_listener()
        self._event_bus.session_id = self._session_id

        # Start writing structured event log to disk
        self._start_event_log_writer()

        self._event_bus.emit(DEBUG, level="debug", msg=f"[Session] Started: {self._session_id}")
        return self._session_id

    def save_session(self) -> None:
        """Persist the current chat history and DataStore to disk."""
        if not self._session_id:
            return
        try:
            history_dicts = self.chat.get_history()
        except Exception:
            history_dicts = []

        store = get_store()
        usage = self.get_token_usage()

        # Use EventBus as source of truth for turn count and preview —
        # chat history contains augmented messages (with injected context
        # headers like [ACTIVE SESSION CONTEXT]) which corrupt previews.
        bus_user_msgs = self._event_bus.get_events(types={USER_MESSAGE})
        turn_count = len(bus_user_msgs)

        # Fallback to history count if EventBus is empty (e.g. resumed session)
        if turn_count == 0:
            turn_count = sum(1 for h in history_dicts if h.get("role") == "user")

        # Don't persist empty sessions (no user messages, no data, no events)
        if turn_count == 0 and len(store) == 0:
            return

        # Preview from last user message (original, not augmented)
        last_preview = ""
        if bus_user_msgs:
            last_event = bus_user_msgs[-1]
            last_text = (last_event.data or {}).get("text", "") if hasattr(last_event, "data") else ""
            if last_text:
                last_preview = last_text[:80]
        # Fallback to history if EventBus had no text
        if not last_preview:
            for h in reversed(history_dicts):
                if h.get("role") == "user":
                    parts = h.get("parts", [])
                    for p in parts:
                        text = p.get("text", "") if isinstance(p, dict) else ""
                        if text:
                            last_preview = text[:80]
                            break
                    if last_preview:
                        break

        # Generate display_log from the EventBus (single source of truth)
        display_log = self._display_log_builder.entries

        metadata_updates = {
            "turn_count": turn_count,
            "last_message_preview": last_preview,
            "token_usage": usage,
            "model": self.model_name,
        }

        # Auto-generate a session title after the first turn
        if turn_count >= 1 and not getattr(self, '_session_title_generated', False):
            session_name = self.generate_session_title()
            if session_name:
                metadata_updates["name"] = session_name
                self._session_title_generated = True

        self._session_manager.save_session(
            session_id=self._session_id,
            chat_history=history_dicts,
            data_store=store,
            metadata_updates=metadata_updates,
            figure_state=self._renderer.save_state(),
            figure_obj=self._renderer.get_figure(),
            operations=get_operations_log().get_records(),
            display_log=display_log,
        )
        self._event_bus.emit(DEBUG, level="debug", msg=f"[Session] Saved ({turn_count} turns, {len(store)} data entries)")

    def load_session(self, session_id: str) -> tuple[dict, list[dict] | None, list[dict] | None]:
        """Restore chat history and DataStore from a saved session.

        Args:
            session_id: The session to load.

        Returns:
            Tuple of (metadata dict, display_log list or None, event_log list or None).
        """
        history_dicts, data_dir, metadata, figure_state, operations, display_log, event_log = self._session_manager.load_session(session_id)

        # Restore chat with saved history — fall back to fresh chat if
        # the adapter can't reconstruct function_call/function_response parts
        if history_dicts:
            try:
                self.chat = self.adapter.create_chat(
                    model=self.model_name,
                    system_prompt=self._system_prompt,
                    tools=self._tool_schemas,
                    history=history_dicts,
                    thinking="high",
                    cached_content=self._cache_name,
                )
            except Exception as e:
                self._event_bus.emit(DEBUG, level="warning", msg=f"[Session] Could not restore chat history: {e}. "
                    "Starting fresh chat (data still restored).")
                self.chat = self.adapter.create_chat(
                    model=self.model_name,
                    system_prompt=self._system_prompt,
                    tools=self._tool_schemas,
                    thinking="high",
                    cached_content=self._cache_name,
                )
        else:
            self.chat = self.adapter.create_chat(
                model=self.model_name,
                system_prompt=self._system_prompt,
                tools=self._tool_schemas,
                thinking="high",
                cached_content=self._cache_name,
            )

        # Restore DataStore (lazy — pickle/netcdf files are loaded on first access)
        store = get_store()
        store.clear()
        if data_dir.exists():
            count = store.load_from_directory_lazy(data_dir)
            self._event_bus.emit(DEBUG, level="debug", msg=f"[Session] Registered {count} data entries (lazy)")

        # Restore operations log
        ops_log = get_operations_log()
        ops_log.clear()
        if operations:
            ops_log.load_from_records(operations)
            self._event_bus.emit(DEBUG, level="debug", msg=f"[Session] Restored {len(operations)} operation records")

        # Clear sub-agent caches (they'll be recreated on next use)
        self._mission_agents.clear()
        self._viz_agent = None
        self._dataops_agent = None
        self._data_extraction_agent = None
        self._planner_agent = None
        self._renderer.reset()

        # Defer figure restore — the full Plotly figure is built lazily when
        # the frontend first requests GET /figure (via _restore_deferred_figure).
        # This avoids loading all data pickle files during session resume.
        if figure_state:
            self._deferred_figure_state = figure_state
            # Restore lightweight metadata on the renderer (no data loading)
            self._renderer._panel_count = figure_state.get("panel_count", 0)
            self._renderer._trace_labels = figure_state.get("trace_labels", [])
            self._renderer._last_fig_json = figure_state.get("last_fig_json")
            self._event_bus.emit(DEBUG, level="debug", msg="[Session] Figure restore deferred until first access")
        else:
            self._deferred_figure_state = None

        self._session_id = session_id
        self._auto_save = True
        self._session_title_generated = bool(metadata.get("name"))
        set_session_id(session_id)
        attach_log_file(session_id)
        self._start_token_log_listener()
        self._event_bus.session_id = session_id

        # Start writing structured event log (append mode — resumes keep adding)
        self._start_event_log_writer()

        self._event_bus.emit(DEBUG, level="debug", msg=f"[Session] Loaded: {session_id}")
        return metadata, display_log, event_log

    def get_session_id(self) -> Optional[str]:
        """Return the current session ID, or None."""
        return self._session_id

    def _restore_deferred_figure(self) -> None:
        """Restore the Plotly figure from deferred state (lazy).

        Called when the frontend first requests the figure via
        GET /figure, or when the user sends their first message.
        """
        figure_state = self._deferred_figure_state
        if figure_state is None:
            return
        self._deferred_figure_state = None  # clear so we only do this once

        try:
            entries = None
            last_fig_json = figure_state.get("last_fig_json")
            if last_fig_json:
                store = get_store()
                entries = {}
                for trace in last_fig_json.get("data", []):
                    label = trace.get("data_label")
                    if label and label not in entries:
                        entry, _ = self._resolve_entry(store, label)
                        if entry is not None:
                            entries[label] = entry
                # Only pass entries if we resolved all of them
                all_labels = {t.get("data_label") for t in last_fig_json.get("data", []) if t.get("data_label")}
                if not all_labels.issubset(entries.keys()):
                    entries = None  # fall back to legacy path

            self._renderer.restore_state(figure_state, entries=entries)
            self._event_bus.emit(DEBUG, level="debug", msg="[Session] Deferred figure restore complete")
        except Exception as e:
            self._event_bus.emit(DEBUG, level="warning", msg=f"[Session] Could not restore deferred figure: {e}")

    def _build_followup_context(self) -> str:
        """Build a compact context string from current data store state.

        Returns empty string on first turn or if store is empty.
        """
        store = get_store()
        entries = store.list_entries()
        if not entries:
            return ""

        from knowledge.mission_prefixes import match_dataset_to_mission, get_canonical_id

        missions: dict[str, set[str]] = {}  # stem -> set of dataset_ids
        for e in entries:
            label = e["label"]
            dataset_id = label.split(".")[0]
            stem, _ = match_dataset_to_mission(dataset_id)
            if stem:
                missions.setdefault(stem, set()).add(dataset_id)

        if not missions:
            return ""

        # Build compact context
        lines = ["[ACTIVE SESSION CONTEXT]"]
        mission_ids = [get_canonical_id(s) for s in sorted(missions)]
        lines.append(f"Active mission(s): {', '.join(mission_ids)}")
        lines.append(f"Data in memory: {len(entries)} entries")

        for stem in sorted(missions):
            mid = get_canonical_id(stem)
            labels = [e["label"] for e in entries
                      if e["label"].split(".")[0] in missions[stem]]
            if labels:
                lines.append(f"  {mid}: {', '.join(labels[:10])}")
                if len(labels) > 10:
                    lines.append(f"    ... and {len(labels) - 10} more")

        lines.append("")
        lines.append(
            "For follow-up requests on this data, delegate to the mission "
            "agent directly — do NOT re-search datasets or list parameters yourself."
        )
        lines.append("[END SESSION CONTEXT]")
        return "\n".join(lines)

    def process_message(self, user_message: str) -> str:
        """Process a user message and return the agent's response.

        All messages go through the orchestrator LLM, which decides whether to
        invoke the planner via the ``request_planning`` tool when multi-step
        coordination is needed.
        """
        self.clear_cancel()
        self._memory_store._last_injected_ids.clear()
        self._event_bus.emit(USER_MESSAGE, level="info", msg=f"[User] {user_message}",
            data={"text": user_message})

        # Inject long-term memory context
        memory_section = self._memory_store.build_prompt_section()
        if memory_section:
            augmented = f"{memory_section}\n\n{user_message}"
        else:
            augmented = user_message

        # Inject active session context for faster follow-up routing
        followup_ctx = self._build_followup_context()
        if followup_ctx:
            augmented = f"{followup_ctx}\n\n{augmented}"

        # Inject orchestrator-level session history (terse sub-agent activity summary)
        orchestrator_history = self._build_agent_history("orchestrator")
        if orchestrator_history:
            augmented = f"{augmented}{orchestrator_history}"

        from .base_agent import _CancelledDuringLLM
        try:
            result = self._process_single_message(augmented)
        except _CancelledDuringLLM:
            result = "Interrupted by user."
        except Exception:
            # Auto-save before re-raising so the session is not lost
            if self._auto_save and self._session_id:
                try:
                    self.save_session()
                except Exception:
                    pass
            raise

        self._event_bus.emit(AGENT_RESPONSE, level="info", msg=f"[Agent] {result}",
            data={"text": result})

        # Auto-save after each turn
        if self._auto_save and self._session_id:
            try:
                self.save_session()
            except Exception as e:
                self._event_bus.emit(DEBUG, level="warning", msg=f"Auto-save failed: {e}")

        # Discovery extraction disabled — code is 70% complete (consolidation
        # paths are dormant).  Re-enable by setting discovery_extraction_interval > 0.
        # self._discovery_turn_counter += 1
        # interval = config.get("discovery_extraction_interval", 2)
        # if interval > 0 and self._discovery_turn_counter % interval == 0:
        #     self._maybe_extract_discoveries()

        # Maybe trigger async memory extraction (first round, then every N more)
        self._memory_turn_counter += 1
        mem_interval = config.get("memory_extraction_interval", 2)
        if mem_interval > 0 and (self._memory_turn_counter - 1) % mem_interval == 0:
            self._maybe_extract_memories()

        return result

    def _cleanup_caches(self):
        """Delete all explicit caches (orchestrator + sub-agents) to stop storage charges."""
        if not isinstance(self.adapter, GeminiAdapter):
            return
        # Collect all cache names
        cache_names = []
        if self._cache_name:
            cache_names.append(("orchestrator", self._cache_name))
        for mid, agent in self._mission_agents.items():
            if getattr(agent, "_cache_name", None):
                cache_names.append((f"mission:{mid}", agent._cache_name))
        for label, agent in [("viz", self._viz_agent), ("dataops", self._dataops_agent)]:
            if agent and getattr(agent, "_cache_name", None):
                cache_names.append((label, agent._cache_name))
        # Delete them
        for label, name in cache_names:
            try:
                self.adapter.delete_cache(name)
                self._event_bus.emit(DEBUG, level="debug", msg=f"Deleted {label} cache: {name}")
            except Exception as exc:
                self._event_bus.emit(DEBUG, level="debug", msg=f"Cache cleanup failed for {label}: {exc}")

    def close(self):
        """Clean up resources before shutdown."""
        if self._event_log_writer is not None:
            self._event_bus.unsubscribe(self._event_log_writer)
            self._event_log_writer.close()
            self._event_log_writer = None
        if self._token_log_listener is not None:
            self._event_bus.unsubscribe(self._token_log_listener)
            self._token_log_listener.close()
            self._token_log_listener = None
        self._cleanup_caches()

    def reset(self):
        """Reset conversation history, mission agent cache, and sub-agents."""
        self._cancel_event.clear()
        # Delete sub-agent caches before clearing references
        self._cleanup_caches()
        # Keep the orchestrator cache (still valid for the same session)
        self.chat = self.adapter.create_chat(
            model=get_active_model(self.model_name),
            system_prompt=self._system_prompt,
            tools=self._tool_schemas,
            thinking="high",
            cached_content=self._cache_name,
        )
        self._current_plan = None
        self._plan_time_range = None
        self._mission_agents.clear()
        self._viz_agent = None
        self._dataops_agent = None
        self._data_extraction_agent = None
        self._planner_agent = None
        self._renderer.reset()
        get_operations_log().clear()

        # Reset discovery turn counter (do NOT clear discovery store)
        self._discovery_turn_counter = 0
        self._last_discovery_op_count = 0

        # Reset memory turn counter and agent (do NOT clear memory store)
        self._memory_agent = None
        self._memory_turn_counter = 0
        self._last_memory_op_index = 0
        self._recent_custom_op_failures = {}

        # Close the event log writer and token log listener before clearing events
        if self._event_log_writer is not None:
            self._event_bus.unsubscribe(self._event_log_writer)
            self._event_log_writer.close()
            self._event_log_writer = None
        if self._token_log_listener is not None:
            self._event_bus.unsubscribe(self._token_log_listener)
            self._token_log_listener.close()
            self._token_log_listener = None

        self._event_bus.clear()

        # Start a fresh session if auto-save was active
        if self._auto_save:
            self._session_id = self._session_manager.generate_session_id()
            set_session_id(self._session_id)
            attach_log_file(self._session_id)
            self._start_token_log_listener()
            self._event_bus.session_id = self._session_id
            self._start_event_log_writer()
            self._event_bus.emit(DEBUG, level="debug", msg=f"[Session] New session after reset: {self._session_id}")

    def get_current_plan(self) -> Optional[TaskPlan]:
        """Get the currently executing plan, if any."""
        return self._current_plan

    def get_plan_status(self) -> Optional[str]:
        """Get a formatted status of the current plan."""
        if self._current_plan is None:
            store = get_task_store()
            incomplete = store.get_incomplete_plans()
            if incomplete:
                plan = sorted(incomplete, key=lambda p: p.created_at, reverse=True)[0]
                return format_plan_for_display(plan)
            return None
        return format_plan_for_display(self._current_plan)

    def cancel_plan(self) -> str:
        """Cancel the current plan and mark remaining tasks as skipped."""
        if self._current_plan is None:
            return "No active plan to cancel."

        plan = self._current_plan
        skipped_count = 0

        for task in plan.tasks:
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.SKIPPED
                skipped_count += 1

        plan.status = PlanStatus.CANCELLED
        store = get_task_store()
        store.save(plan)

        completed = len(plan.get_completed_tasks())
        self._current_plan = None
        self._plan_time_range = None

        return f"Plan cancelled. {completed} task(s) completed, {skipped_count} skipped."

    def retry_failed_task(self) -> str:
        """Retry the first failed task in the current plan."""
        if self._current_plan is None:
            store = get_task_store()
            incomplete = store.get_incomplete_plans()
            failed_plans = [p for p in incomplete if p.get_failed_tasks()]
            if not failed_plans:
                return "No failed tasks to retry."
            self._current_plan = sorted(failed_plans, key=lambda p: p.created_at, reverse=True)[0]

        plan = self._current_plan
        failed = plan.get_failed_tasks()
        if not failed:
            return "No failed tasks to retry."

        task = failed[0]
        task.status = TaskStatus.PENDING
        task.error = None
        task.result = None
        task.tool_calls = []

        plan.status = PlanStatus.EXECUTING
        store = get_task_store()
        store.save(plan)

        self._event_bus.emit(DEBUG, level="debug", msg=f"[Retry] Retrying task: {task.description}")

        result = self._execute_task(task)
        store.save(plan)

        if plan.is_complete():
            if plan.get_failed_tasks():
                plan.status = PlanStatus.FAILED
            else:
                plan.status = PlanStatus.COMPLETED
            store.save(plan)

        return f"Retried: {task.description}\nResult: {result}"

    def resume_plan(self, plan: TaskPlan) -> str:
        """Resume an incomplete plan from storage."""
        self._current_plan = plan
        plan.status = PlanStatus.EXECUTING
        store = get_task_store()

        self._event_bus.emit(DEBUG, level="debug", msg=f"[Resume] Resuming plan: {plan.user_request[:50]}...")
        self._event_bus.emit(DEBUG, level="debug", msg=format_plan_for_display(plan))

        pending = plan.get_pending_tasks()
        if not pending:
            plan.status = PlanStatus.COMPLETED if not plan.get_failed_tasks() else PlanStatus.FAILED
            store.save(plan)
            return self._summarize_plan_execution(plan)

        for i, task in enumerate(plan.tasks):
            if task.status != TaskStatus.PENDING:
                continue

            plan.current_task_index = i
            store.save(plan)

            self._event_bus.emit(PLAN_TASK, level="debug", msg=f"[Plan] Resuming step {i+1}/{len(plan.tasks)}: {task.description}")

            self._execute_task(task)
            store.save(plan)

        if plan.get_failed_tasks():
            plan.status = PlanStatus.FAILED
        else:
            plan.status = PlanStatus.COMPLETED
        store.save(plan)

        summary = self._summarize_plan_execution(plan)
        self._current_plan = None
        self._plan_time_range = None

        return summary

    def discard_plan(self, plan: TaskPlan) -> str:
        """Discard an incomplete plan."""
        store = get_task_store()
        store.delete(plan.id)
        return f"Discarded plan: {plan.user_request[:50]}..."


def create_agent(verbose: bool = False, gui_mode: bool = False, model: str | None = None, defer_chat: bool = False) -> OrchestratorAgent:
    """Factory function to create a new agent instance.

    Args:
        verbose: If True, print debug info about tool calls.
        gui_mode: If True, launch with visible GUI window.
        model: Gemini model name (default: gemini-2.5-flash).
        defer_chat: If True, skip creating the initial chat session.

    Returns:
        Configured OrchestratorAgent instance.
    """
    return OrchestratorAgent(verbose=verbose, gui_mode=gui_mode, model=model, defer_chat=defer_chat)
