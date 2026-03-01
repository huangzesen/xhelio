"""
Core agent logic - orchestrates Gemini calls and tool execution.

The OrchestratorAgent routes requests to:
- MissionAgent sub-agents for data operations (per mission)
- VizAgent[Plotly] / VizAgent[Mpl] sub-agents for visualization
"""

import contextvars
import json
import math
import queue
import time
import threading
from collections import defaultdict
from datetime import datetime
import pandas as pd
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import config
from config import get_data_dir, get_api_key
from .llm import (
    LLMAdapter,
    GeminiAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    LLMResponse,
    FunctionSchema,
)
from .tools import (
    get_tool_schemas,
    get_function_schemas,
    get_spice_tool_names,
    TOOLS as _ALL_TOOLS,
)
from .tool_catalog import get_browse_result, resolve_tools, META_TOOL_NAMES
from .truncation import trunc, trunc_items, join_labels, get_limit, get_item_limit
from .prompts import get_system_prompt
from .time_utils import parse_time_range, TimeRangeError
from .tool_timing import ToolTimer, stamp_tool_result
from .tasks import (
    Task,
    TaskPlan,
    TaskStatus,
    PlanStatus,
    get_task_store,
    create_task,
    create_plan,
)
from .planner import PlannerAgent, format_plan_for_display
from .turn_limits import get_limit as get_turn_limit
from .session import SessionManager
from .memory import MemoryStore
from .token_counter import count_tokens as estimate_tokens, count_tool_tokens
from .sub_agent import SubAgent, AgentState, Message, _make_message
from .mission_agent import MissionAgent
from .viz_plotly_agent import VizPlotlyAgent
from .viz_mpl_agent import VizMplAgent
from .viz_jsx_agent import VizJsxAgent
from .data_ops_agent import DataOpsAgent
from .data_extraction_agent import DataExtractionAgent
from .insight_agent import InsightAgent
from .logging import (
    setup_logging,
    attach_log_file,
    get_logger,
    log_error,
    log_tool_call,
    log_tool_result,
    log_plan_event,
    log_session_end,
    set_session_id,
    tagged,
    LOG_DIR,
)
from .event_bus import (
    EventBus,
    get_event_bus,
    set_event_bus,
    DebugLogListener,
    SSEEventListener,
    OperationsLogListener,
    DisplayLogBuilder,
    EventLogWriter,
    TokenLogListener,
    load_event_log,
    # Event types
    USER_MESSAGE,
    AGENT_RESPONSE,
    TOOL_CALL,
    TOOL_RESULT,
    DATA_FETCHED,
    DATA_COMPUTED,
    DATA_CREATED,
    RENDER_EXECUTED,
    MPL_RENDER_EXECUTED,
    JSX_RENDER_EXECUTED,
    PLOT_ACTION,
    DELEGATION,
    DELEGATION_DONE,
    SUB_AGENT_TOOL,
    SUB_AGENT_ERROR,
    PLAN_CREATED,
    PLAN_TASK,
    PLAN_COMPLETED,
    PROGRESS,
    THINKING,
    LLM_CALL,
    LLM_RESPONSE,
    FETCH_ERROR,
    HIGH_NAN,
    CUSTOM_OP_FAILURE,
    RECOVERY,
    RENDER_ERROR,
    TOOL_ERROR,
    SESSION_START,
    SESSION_END,
    DEBUG,
    MEMORY_EXTRACTION_START,
    MEMORY_EXTRACTION_DONE,
    MEMORY_EXTRACTION_ERROR,
    STM_COMPACTION,
    INSIGHT_RESULT,
    WORK_CANCELLED,
    USER_AMENDMENT,
    TEXT_DELTA,
    ROUND_START,
    ROUND_END,
    CYCLE_START,
    CYCLE_END,
    SESSION_TITLE,
    AGENT_STATE_CHANGE,
    INSIGHT_FEEDBACK,
)
from .control_center import ControlCenter, WorkUnit, WorkStatus
from .memory_agent import MemoryAgent, MemoryContext
from .pipeline_store import PipelineStore
from .context_tracker import ContextTracker

from .loop_guard import LoopGuard, DupVerdict
from .model_fallback import (
    activate_fallback,
    reset_fallback,
    get_active_model,
    is_quota_error,
)
from .llm_utils import (
    _LLM_WARN_INTERVAL,
    _LLM_RETRY_TIMEOUT,
    _LLM_MAX_RETRIES,
    send_with_timeout,
    send_with_timeout_stream,
    _CancelledDuringLLM,
    execute_tools_batch,
    track_llm_usage,
    build_outcome_summary,
)
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
from data_ops.store import (
    get_store,
    set_store,
    DataStore,
    DataEntry,
    build_source_map,
    describe_sources,
)
from data_ops.fetch import fetch_data
from data_ops.custom_ops import (
    run_custom_operation,
    run_multi_source_operation,
    run_dataframe_creation,
)
from data_ops.operations_log import set_operations_log, OperationsLog

from .agent_registry import (
    ORCHESTRATOR_TOOLS,
    MISSION_TOOL_REGISTRY,
)

_VIZ_DELEGATION_TOOLS = {
    "plotly": "delegate_to_viz_plotly",
    "matplotlib": "delegate_to_viz_mpl",
    "jsx": "delegate_to_viz_jsx",
}


def _active_viz_tool() -> str:
    """Return the name of the active viz delegation tool."""
    return _VIZ_DELEGATION_TOOLS.get(
        config.PREFER_VIZ_BACKEND, "delegate_to_viz_mpl"
    )


def _inactive_viz_tools() -> set[str]:
    """Return the set of viz delegation tools that should be hidden."""
    active = _active_viz_tool()
    return {t for t in _VIZ_DELEGATION_TOOLS.values() if t != active}


# Roles that map to "user" or "assistant/model" across all LLM adapters
_USER_ROLES = {"user"}
_AGENT_ROLES = {"model", "assistant"}


def _extract_turns(history_entries: list, *, max_text: int | None = None) -> list[str]:
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
                    t = (
                        part.get("text")
                        if isinstance(part, dict)
                        else getattr(part, "text", None)
                    )
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
            limit = max_text if max_text is not None else get_limit("context.turn_text")
            turns.append(f"{label}: {text[:limit]}")
    return turns


def _create_adapter() -> LLMAdapter:
    """Create the LLM adapter based on config (llm_provider, llm_base_url, etc.)."""
    provider = config.LLM_PROVIDER.lower()
    api_key = get_api_key(provider)
    if provider == "openai":
        return OpenAIAdapter(api_key=api_key, base_url=config.LLM_BASE_URL)
    elif provider == "anthropic":
        return AnthropicAdapter(api_key=api_key, base_url=config.LLM_BASE_URL)
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

    def __init__(
        self,
        verbose: bool = False,
        gui_mode: bool = False,
        model: str | None = None,
        defer_chat: bool = False,
    ):
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
        self._shutdown_event = threading.Event()
        self._text_already_streamed = False  # set by _send_message_streaming
        self._intermediate_text_streamed = False  # set by _send_message_streaming

        # Sub-agent inbox — primary coordination mechanism for the orchestrator
        self._inbox: queue.Queue[Message] = queue.Queue()

        # Lifecycle state machine (mirrors SubAgent's AgentState)
        self._state = AgentState.SLEEPING
        self._cycle_number: int = 0
        self._turn_number: int = 0

        # Disk-backed data store (created at start_session / load_session)
        self._store: DataStore | None = None

        # Sub-agents: agent_id → SubAgent
        self._sub_agents: dict[str, SubAgent] = {}
        self._sub_agents_lock = threading.Lock()
        self._dataops_seq: int = 0  # Counter for ephemeral DataOps agents
        self._mission_seq: int = 0  # Counter for ephemeral Mission agents

        # Turnless orchestrator: shared Condition + Control Center
        # RLock so that has_pending()/drain() can re-acquire inside wait loops
        self._shared_cond = threading.Condition(threading.RLock())
        self._control_center = ControlCenter(self._shared_cond)

        # Incremental context injection tracker — avoids re-sending identical
        # context to actors whose persistent LLM sessions already contain it.
        self._ctx_tracker = ContextTracker()

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
        self._ops_log_listener = OperationsLogListener(lambda: self._ops_log)
        self._event_bus.subscribe(self._ops_log_listener)

        self._event_bus.emit(
            SESSION_START, level="info", msg="Initializing OrchestratorAgent"
        )

        # Initialize LLM adapter (wraps all provider SDK calls)
        self.adapter: LLMAdapter = _create_adapter()

        # Discover SPICE tools from MCP server (lazy, non-fatal).
        # Must happen before building tool schemas so the LLM sees them.
        self._ensure_spice_tools()

        # Build tool schemas for the orchestrator
        self._all_tool_schemas = get_function_schemas(names=ORCHESTRATOR_TOOLS)

        # Filter out inactive viz tools (settings-driven)
        inactive = _inactive_viz_tools()
        self._all_tool_schemas = [
            s for s in self._all_tool_schemas if s.name not in inactive
        ]
        self._viz_backend = config.PREFER_VIZ_BACKEND

        # Meta-tools (always active): browse_tools, load_tools, ask_clarification
        self._meta_tool_schemas = [
            s for s in self._all_tool_schemas if s.name in META_TOOL_NAMES
        ]

        # Tool store: model browses and loads tools on demand.
        # Only active when Interactions API is in use (tools are per-call).
        self._use_tool_store = config.USE_TOOL_STORE and config.USE_INTERACTIONS_API
        self._loaded_tool_names: set[str] = set()

        if self._use_tool_store:
            # Pre-load essential categories so the model can act immediately
            from .tool_catalog import DEFAULT_TOOL_CATEGORIES, DEFAULT_EXTRA_TOOLS

            default_names = set(
                resolve_tools(
                    DEFAULT_TOOL_CATEGORIES + DEFAULT_EXTRA_TOOLS,
                    agent_context="ctx:orchestrator",
                )
            )
            self._loaded_tool_names = default_names
            default_schemas = [
                s
                for s in self._all_tool_schemas
                if s.name in default_names and s.name not in META_TOOL_NAMES
            ]
            self._tool_schemas = list(self._meta_tool_schemas) + default_schemas
        else:
            # Legacy: all tools active from the start
            self._tool_schemas = list(self._all_tool_schemas)

        # Store model name and system prompt for chat creation
        self.model_name = model or config.SMART_MODEL
        self._system_prompt = get_system_prompt(gui_mode=gui_mode)

        if not defer_chat:
            # Create chat session (use get_active_model in case fallback was already activated)
            self.chat = self.adapter.create_chat(
                model=get_active_model(self.model_name),
                system_prompt=self._system_prompt,
                tools=self._tool_schemas,
                thinking="high",
            )
        else:
            # Chat will be created by load_session() or _send_message()
            self.chat = None

        # Plotly renderer for visualization
        self._renderer = PlotlyRenderer(verbose=self.verbose, gui_mode=self.gui_mode)
        self._deferred_figure_state: Optional[dict] = (
            None  # set by load_session() for lazy restore
        )

        # Token usage tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_thinking_tokens = 0
        self._total_cached_tokens = 0
        self._api_calls = 0
        self._last_tool_context = "send_message"
        self._round_start_tokens: dict | None = None  # snapshot at CYCLE_START

        # Token decomposition (system/tools/history breakdown)
        self._system_prompt_tokens = 0
        self._tools_tokens = 0
        self._token_decomp_dirty = True
        self._latest_input_tokens = 0

        # Thread-local storage for per-thread agent identity (async delegation)
        self._tls = threading.local()
        self._tls.active_agent_name = "OrchestratorAgent"
        self._tls.current_agent_type = "orchestrator"
        self._tls.current_tool_call_id = None

        # Inline model token usage (tracked separately for breakdown)
        self._inline_input_tokens = 0
        self._inline_output_tokens = 0
        self._inline_thinking_tokens = 0
        self._inline_cached_tokens = 0
        self._inline_api_calls = 0

        # Retired ephemeral agent token usage (preserved after cleanup)
        self._retired_agent_usage: list[dict] = []

        # Current plan being executed (if any)
        self._current_plan: Optional[TaskPlan] = None

        # SSE event listener (subscribed by api/routes.py when streaming)
        self._sse_listener: SSEEventListener | None = None

        # Cached planner agent
        self._planner_agent: Optional[PlannerAgent] = None

        # Canonical time range for the current plan (reset after plan completes)
        self._plan_time_range: Optional["TimeRange"] = None

        # Session persistence
        self._session_id: Optional[str] = None
        self._session_manager = SessionManager()
        self._auto_save: bool = False

        # Load persisted informed-tool overrides
        from .agent_registry import AGENT_INFORMED_REGISTRY as _informed_reg

        _informed_reg.load(get_data_dir() / "informed_tools.json")

        # Long-term memory
        self._memory_store = MemoryStore()
        self._memory_agent: Optional[MemoryAgent] = None

        # Eureka discovery (Step 2)
        self._eureka_lock = threading.Lock()
        self._eureka_agent: Optional["EurekaAgent"] = None
        self._eureka_turn_counter: int = 0

        # Pipeline template index (searchable metadata for saved templates)
        self._pipeline_store = PipelineStore()

        # Periodic memory extraction (mirrors discovery pattern)
        self._memory_turn_counter = 0
        self._last_memory_op_index = 0
        self._memory_lock = threading.Lock()

        # Track recent custom_operation failures for recovery detection
        self._recent_custom_op_failures: dict = {}

        # Inline completion circuit breaker: disable after repeated failures
        self._inline_fail_count: int = 0
        self._inline_disabled_until: float = 0.0

        # Insight review iteration counter (reset per user turn)
        self._insight_review_iter: int = 0

        # Cached pipeline DAG (invalidated when new ops are recorded)
        self._pipeline = None

        # Pull-based event feed for orchestrator (sub-agent feeds created on demand)
        from agent.event_feed import EventFeedBuffer

        self._event_feed = EventFeedBuffer(
            self._event_bus,
            "ctx:orchestrator",
            token_quota=config.get("history_budget_orchestrator", 40000),
            compact_fn=self._compact_history,
            agent_type="orchestrator",
        )

        # Thread pool for timeout-wrapped Gemini calls
        self._timeout_pool = ThreadPoolExecutor(max_workers=1)

    # ---- Thread-local agent identity (safe for async delegation) ----

    @property
    def _active_agent_name(self) -> str:
        return getattr(self._tls, "active_agent_name", "OrchestratorAgent")

    @_active_agent_name.setter
    def _active_agent_name(self, value: str) -> None:
        self._tls.active_agent_name = value

    @property
    def _current_agent_type(self) -> str:
        return getattr(self._tls, "current_agent_type", "orchestrator")

    @_current_agent_type.setter
    def _current_agent_type(self, value: str) -> None:
        self._tls.current_agent_type = value

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

            store = self._store
            final_labels = {e["label"] for e in store.list_entries()}
            self._pipeline = Pipeline.from_operations_log(self._ops_log, final_labels)
        return self._pipeline

    def _invalidate_pipeline(self):
        """Invalidate the cached Pipeline (called after new ops are recorded)."""
        self._pipeline = None

    # ---- Parallel tool execution ----

    # Tools safe to run concurrently. All delegate_to_* tools return
    # immediately (pending_async) and do their work on background threads.
    _PARALLEL_SAFE_TOOLS = {
        "fetch_data",
        "delegate_to_mission",
        "delegate_to_viz_plotly",
        "delegate_to_viz_mpl",
        "delegate_to_viz_jsx",
        "delegate_to_data_ops",
        "delegate_to_data_extraction",
        "delegate_to_insight",
    }

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

    # ---- Lifecycle state machine ----

    @property
    def state(self) -> AgentState:
        """Current lifecycle state of the orchestrator."""
        return self._state

    @property
    def cycle_number(self) -> int:
        """Current cycle number (sleep→sleep interval count)."""
        return self._cycle_number

    @property
    def turn_number(self) -> int:
        """Current turn number (agent response count)."""
        return self._turn_number

    def _set_state(self, new_state: AgentState, reason: str = "") -> None:
        """Transition orchestrator to a new lifecycle state."""
        old = self._state
        if old == new_state:
            return
        self._state = new_state
        suffix = f" ({reason})" if reason else ""
        self._event_bus.emit(
            AGENT_STATE_CHANGE,
            agent="orchestrator",
            level="debug",
            msg=f"[orchestrator] {old.value} → {new_state.value}{suffix}",
            data={
                "agent_id": "orchestrator",
                "old": old.value,
                "new": new_state.value,
                "reason": reason,
                "cycle": self._cycle_number,
                "turn": self._turn_number,
            },
        )

    def orchestrator_status(self) -> dict:
        """Return orchestrator status for monitoring and API exposure."""
        return {
            "agent_id": "orchestrator",
            "state": self._state.value,
            "cycle": self._cycle_number,
            "turn": self._turn_number,
            "active_work": self._control_center.list_active(),
            "tokens": self.get_token_usage(),
        }

    @property
    def memory_store(self) -> MemoryStore:
        """Return the long-term memory store (for web UI access)."""
        return self._memory_store

    def get_plotly_figure(self):
        """Return the current Plotly figure (or None)."""
        return self._renderer.get_figure()

    def _update_token_decomposition(self) -> None:
        """Recompute cached system prompt and tools token counts."""
        self._system_prompt_tokens = estimate_tokens(self._system_prompt)
        self._tools_tokens = count_tool_tokens(self._tool_schemas)
        self._token_decomp_dirty = False

    def _track_usage(self, response: LLMResponse):
        """Accumulate token usage from an LLMResponse."""
        if self._token_decomp_dirty:
            self._update_token_decomposition()
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
            system_tokens=self._system_prompt_tokens,
            tools_tokens=self._tools_tokens,
        )
        self._total_input_tokens = token_state["input"]
        self._total_output_tokens = token_state["output"]
        self._total_thinking_tokens = token_state["thinking"]
        self._total_cached_tokens = token_state["cached"]
        self._api_calls = token_state["api_calls"]
        self._latest_input_tokens = response.usage.input_tokens

    def _track_inline_usage(self, response: LLMResponse):
        """Track token usage from inline model calls (follow-ups, titles, completions).

        Tokens are accumulated in separate inline counters so they appear as
        their own row in the breakdown, while still being emitted via EventBus
        for the overall session totals.
        """
        token_state = {
            "input": self._inline_input_tokens,
            "output": self._inline_output_tokens,
            "thinking": self._inline_thinking_tokens,
            "cached": self._inline_cached_tokens,
            "api_calls": self._inline_api_calls,
        }
        track_llm_usage(
            response=response,
            token_state=token_state,
            agent_name="Inline",
            last_tool_context=self._last_tool_context,
        )
        self._inline_input_tokens = token_state["input"]
        self._inline_output_tokens = token_state["output"]
        self._inline_thinking_tokens = token_state["thinking"]
        self._inline_cached_tokens = token_state["cached"]
        self._inline_api_calls = token_state["api_calls"]

    def _send_message(self, message) -> LLMResponse:
        """Send a message on self.chat with timeout/retry and model fallback on 429."""
        try:
            return self._send_with_timeout(self.chat, message)
        except Exception as exc:
            if is_quota_error(exc, adapter=self.adapter) and config.FALLBACK_MODEL:
                activate_fallback(config.FALLBACK_MODEL)
                self.chat = self.adapter.create_chat(
                    model=config.FALLBACK_MODEL,
                    system_prompt=self._system_prompt,
                    tools=self._tool_schemas,
                    thinking="high",
                )
                self.model_name = config.FALLBACK_MODEL
                return self._send_with_timeout(self.chat, message)
            raise

    def _send_message_streaming(self, message) -> LLMResponse:
        """Send a message with streaming text deltas emitted via EventBus.

        Text tokens are emitted as ``TEXT_DELTA`` events as they arrive,
        giving the user near-instant first-token feedback.  The complete
        ``LLMResponse`` is returned at the end (same contract as
        ``_send_message``).

        Sets ``self._text_already_streamed = True`` so that ``run_loop()``
        knows to skip the final bulk ``TEXT_DELTA`` emission.
        """

        def _on_chunk(text_delta: str) -> None:
            self._event_bus.emit(
                TEXT_DELTA,
                agent="orchestrator",
                level="info",
                msg=text_delta,
                data={"text": text_delta, "streaming": True},
            )

        try:
            response = send_with_timeout_stream(
                chat=self.chat,
                message=message,
                timeout_pool=self._timeout_pool,
                cancel_event=self._cancel_event,
                retry_timeout=_LLM_RETRY_TIMEOUT,
                agent_name="Orchestrator",
                logger=self.logger,
                on_chunk=_on_chunk,
            )
        except Exception as exc:
            if is_quota_error(exc, adapter=self.adapter) and config.FALLBACK_MODEL:
                activate_fallback(config.FALLBACK_MODEL)
                self.chat = self.adapter.create_chat(
                    model=config.FALLBACK_MODEL,
                    system_prompt=self._system_prompt,
                    tools=self._tool_schemas,
                    thinking="high",
                )
                self.model_name = config.FALLBACK_MODEL
                # Fallback without streaming — non-critical
                return self._send_with_timeout(self.chat, message)
            raise

        # Mark that text was already streamed so downstream doesn't re-emit
        if response.text:
            if response.tool_calls:
                # Intermediate text alongside tool calls — already streamed
                self._intermediate_text_streamed = True
            else:
                # Final text — already streamed, run_loop() should skip bulk emit
                self._text_already_streamed = True

        return response

    def _emit_intermediate_text(self, response: LLMResponse) -> None:
        """Emit intermediate LLM text (returned alongside tool calls) as commentary."""
        if response.text and response.tool_calls:
            if self._intermediate_text_streamed:
                # Text was already emitted incrementally via streaming
                self._intermediate_text_streamed = False
                return
            self._event_bus.emit(
                TEXT_DELTA,
                agent="orchestrator",
                level="info",
                msg=f"[Orchestrator] {response.text}",
                data={"text": response.text + "\n\n", "commentary": True},
            )

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
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Search] Queries: {meta.web_search_queries}",
                )

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
                    self._event_bus.emit(
                        DEBUG,
                        level="debug",
                        msg=f"[Search] Queries: {meta.web_search_queries}",
                    )

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
            self._event_bus.emit(
                DEBUG,
                level="warning",
                msg="[Search] Web search unavailable — tavily-python not installed",
            )
            return {
                "status": "error",
                "message": "Web search unavailable: tavily-python is not installed. "
                "Install it with: pip install tavily-python",
            }

        import os

        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            self._event_bus.emit(
                DEBUG,
                level="warning",
                msg="[Search] Web search unavailable — TAVILY_API_KEY not set",
            )
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
                source_lines = [
                    f"- [{s.get('title', 'Source')}]({s['url']})"
                    for s in sources
                    if s.get("url")
                ]
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
        input_tokens = self._total_input_tokens + self._inline_input_tokens
        output_tokens = self._total_output_tokens + self._inline_output_tokens
        thinking_tokens = self._total_thinking_tokens + self._inline_thinking_tokens
        cached_tokens = self._total_cached_tokens + self._inline_cached_tokens
        api_calls = self._api_calls + self._inline_api_calls

        # Include usage from all sub-agent-based sub-agents
        with self._sub_agents_lock:
            for sub_agent in self._sub_agents.values():
                usage = sub_agent.get_token_usage()
                input_tokens += usage["input_tokens"]
                output_tokens += usage["output_tokens"]
                thinking_tokens += usage.get("thinking_tokens", 0)
                cached_tokens += usage.get("cached_tokens", 0)
                api_calls += usage["api_calls"]

        # Include usage from retired ephemeral agents
        for retired in self._retired_agent_usage:
            input_tokens += retired["input_tokens"]
            output_tokens += retired["output_tokens"]
            thinking_tokens += retired.get("thinking_tokens", 0)
            cached_tokens += retired.get("cached_tokens", 0)
            api_calls += retired["api_calls"]

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
                rows.append(
                    {
                        "agent": name,
                        "input": usage["input_tokens"],
                        "output": usage["output_tokens"],
                        "thinking": usage.get("thinking_tokens", 0),
                        "cached": usage.get("cached_tokens", 0),
                        "calls": usage["api_calls"],
                        "ctx_system": usage.get("ctx_system_tokens", 0),
                        "ctx_tools": usage.get("ctx_tools_tokens", 0),
                        "ctx_history": usage.get("ctx_history_tokens", 0),
                        "ctx_total": usage.get("ctx_total_tokens", 0),
                    }
                )

        # Orchestrator's own usage
        _add(
            "Orchestrator",
            {
                "input_tokens": self._total_input_tokens,
                "output_tokens": self._total_output_tokens,
                "thinking_tokens": self._total_thinking_tokens,
                "cached_tokens": self._total_cached_tokens,
                "api_calls": self._api_calls,
                "ctx_system_tokens": self._system_prompt_tokens,
                "ctx_tools_tokens": self._tools_tokens,
                "ctx_history_tokens": max(0, self._latest_input_tokens - self._system_prompt_tokens - self._tools_tokens),
                "ctx_total_tokens": self._latest_input_tokens,
            },
        )

        # Inline model usage (follow-ups, session titles, completions)
        _add(
            "Inline",
            {
                "input_tokens": self._inline_input_tokens,
                "output_tokens": self._inline_output_tokens,
                "thinking_tokens": self._inline_thinking_tokens,
                "cached_tokens": self._inline_cached_tokens,
                "api_calls": self._inline_api_calls,
            },
        )

        # Sub-agent-based sub-agents
        with self._sub_agents_lock:
            for agent_id, sub_agent in self._sub_agents.items():
                _add(agent_id, sub_agent.get_token_usage())

        # Retired ephemeral agents (cleaned up but token usage preserved)
        for retired in self._retired_agent_usage:
            rows.append(
                {
                    "agent": retired["agent"],
                    "input": retired["input_tokens"],
                    "output": retired["output_tokens"],
                    "thinking": retired.get("thinking_tokens", 0),
                    "cached": retired.get("cached_tokens", 0),
                    "calls": retired["api_calls"],
                    "ctx_system": retired.get("ctx_system_tokens", 0),
                    "ctx_tools": retired.get("ctx_tools_tokens", 0),
                    "ctx_history": retired.get("ctx_history_tokens", 0),
                    "ctx_total": retired.get("ctx_total_tokens", 0),
                }
            )

        # Planner agent
        if self._planner_agent:
            usage = self._planner_agent.get_token_usage()
            if usage["api_calls"] > 0 or usage["input_tokens"] > 0:
                rows.append(
                    {
                        "agent": "Planner",
                        "input": usage["input_tokens"],
                        "output": usage["output_tokens"],
                        "thinking": usage.get("thinking_tokens", 0),
                        "cached": usage.get("cached_tokens", 0),
                        "calls": usage.get("api_calls", 0),
                        "ctx_system": usage.get("ctx_system_tokens", 0),
                        "ctx_tools": usage.get("ctx_tools_tokens", 0),
                        "ctx_history": usage.get("ctx_history_tokens", 0),
                        "ctx_total": usage.get("ctx_total_tokens", 0),
                    }
                )

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

        Delegates to the shared ``data_ops.store.resolve_entry()`` function.
        """
        from data_ops.store import resolve_entry

        return resolve_entry(store, label)

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
        store = self._store
        entry_map: dict = {}
        for trace in data_traces:
            label = trace.get("data_label")
            if label and label not in entry_map:
                entry, _ = self._resolve_entry(store, label)
                if entry is None:
                    available = [e["label"] for e in store.list_entries()]
                    return {
                        "status": "error",
                        "message": (
                            f"data_label '{label}' not found in memory. "
                            f"Available labels: {available}"
                        ),
                    }
                entry_map[label] = entry

        # Validate non-empty data
        for label, entry in entry_map.items():
            if len(entry.data) == 0:
                return {
                    "status": "error",
                    "message": f"Entry '{label}' has no data points",
                }

        try:
            result = self._renderer.render_plotly_json(fig_json, entry_map)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        if result.get("status") == "success":
            # Record render op inline so we can include op_id in the event
            ops_log = self._ops_log
            op_record = ops_log.record(
                tool="render_plotly_json",
                args={"figure_json": fig_json},
                outputs=[],
                inputs=list(entry_map.keys()),
                status="success",
            )
            self._event_bus.emit(
                RENDER_EXECUTED,
                agent="orchestrator",
                msg="[Render] render_plotly_json executed",
                data={
                    "args": {"figure_json": fig_json},
                    "inputs": list(entry_map.keys()),
                    "outputs": [],
                    "op_id": op_record["id"],
                },
            )

            review = self._sync_insight_review()
            if review is not None and not review.get("failed", False):
                passed = review.get("passed", True)
                verdict = "PASS" if passed else "NEEDS_IMPROVEMENT"
                result["insight_review"] = {
                    "verdict": verdict,
                    "feedback": review.get("text", ""),
                    "suggestions": review.get("suggestions", []),
                }
                if not passed:
                    result["insight_review"]["action_needed"] = (
                        "The automatic figure review found issues. "
                        "Review the feedback and suggestions above, then "
                        "re-render with improvements."
                    )

        return result

    def _handle_manage_plot(self, tool_args: dict) -> dict:
        """Handle the manage_plot tool call."""
        action = tool_args.get("action")
        if not action:
            return {"status": "error", "message": "action is required"}

        if action == "reset":
            self._event_bus.emit(
                PLOT_ACTION,
                agent="orchestrator",
                msg="[Plot] reset",
                data={"args": {"action": "reset"}, "outputs": []},
            )
            return self._renderer.reset()

        elif action == "get_state":
            return self._renderer.get_current_state()

        elif action == "export":
            filename = tool_args.get("filename", "output.png")
            fmt = tool_args.get("format", "png")
            result = self._renderer.export(filename, format=fmt)

            if result.get("status") == "success":
                self._event_bus.emit(
                    PLOT_ACTION,
                    agent="orchestrator",
                    msg=f"[Plot] export {filename}",
                    data={
                        "args": {
                            "action": "export",
                            "filename": filename,
                            "format": fmt,
                        },
                        "outputs": [],
                    },
                )

            # Auto-open the exported file in default viewer (skip in GUI mode)
            if (
                result.get("status") == "success"
                and not self.gui_mode
                and not self.web_mode
            ):
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
                    self._event_bus.emit(
                        DEBUG, level="debug", msg=f"[Export] Could not auto-open: {e}"
                    )
                    result["auto_opened"] = False

            return result

        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    # --- Matplotlib Visualization Tool Handlers ---

    def _handle_generate_mpl_script(self, tool_args: dict) -> dict:
        """Handle the generate_mpl_script tool call."""
        code = tool_args.get("script")
        if not code:
            return {"status": "error", "message": "script parameter is required"}

        description = tool_args.get("description", "Matplotlib plot")

        # Resolve session data directory and labels index
        session_dir = self._session_manager.base_dir / self._session_id
        data_dir = session_dir / "data"
        labels_path = data_dir / "_labels.json"

        if not labels_path.exists():
            return {
                "status": "error",
                "message": "No data in session. Fetch data first before generating a plot.",
            }

        import json as _json

        labels_index = _json.loads(labels_path.read_text())

        # Generate script_id
        from datetime import datetime as _dt
        import secrets

        script_id = _dt.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)

        # Execute script in sandbox
        from rendering.mpl_sandbox import run_mpl_script

        output_dir = session_dir / "mpl_outputs"
        result = run_mpl_script(
            code=code,
            data_dir=data_dir,
            output_dir=output_dir,
            labels_index=labels_index,
            script_id=script_id,
            timeout=60.0,
        )

        if result.success:
            # Record in operations log
            self._event_bus.emit(
                MPL_RENDER_EXECUTED,
                agent="VizAgent[Mpl]",
                msg=f"[MplViz] Script executed: {description}",
                data={
                    "script_id": script_id,
                    "description": description,
                    "output_path": result.output_path,
                    "script_path": result.script_path,
                },
            )
            response = {
                "status": "success",
                "script_id": script_id,
                "output_path": result.output_path,
                "message": f"Matplotlib plot saved successfully. Script ID: {script_id}",
            }
            if result.stdout.strip():
                response["stdout"] = result.stdout
            return response
        else:
            response = {
                "status": "error",
                "script_id": script_id,
                "message": "Script execution failed. See stderr for details.",
                "stderr": result.stderr,
            }
            if result.stdout.strip():
                response["stdout"] = result.stdout
            if result.script_path:
                response["script_path"] = result.script_path
            return response

    def _handle_manage_mpl_output(self, tool_args: dict) -> dict:
        """Handle the manage_mpl_output tool call."""
        action = tool_args.get("action")
        if not action:
            return {"status": "error", "message": "action is required"}

        session_dir = self._session_manager.base_dir / self._session_id
        scripts_dir = session_dir / "mpl_scripts"
        outputs_dir = session_dir / "mpl_outputs"

        if action == "list":
            items = []
            if scripts_dir.exists():
                for script_file in sorted(scripts_dir.glob("*.py")):
                    script_id = script_file.stem
                    output_file = outputs_dir / f"{script_id}.png"
                    items.append(
                        {
                            "script_id": script_id,
                            "has_output": output_file.exists(),
                            "script_path": str(script_file),
                            "output_path": str(output_file)
                            if output_file.exists()
                            else None,
                        }
                    )
            return {"status": "success", "items": items, "count": len(items)}

        elif action == "get_script":
            script_id = tool_args.get("script_id")
            if not script_id:
                return {
                    "status": "error",
                    "message": "script_id is required for get_script",
                }
            script_file = scripts_dir / f"{script_id}.py"
            if not script_file.exists():
                return {"status": "error", "message": f"Script not found: {script_id}"}
            return {
                "status": "success",
                "script_id": script_id,
                "script": script_file.read_text(encoding="utf-8"),
            }

        elif action == "rerun":
            script_id = tool_args.get("script_id")
            if not script_id:
                return {"status": "error", "message": "script_id is required for rerun"}
            script_file = scripts_dir / f"{script_id}.py"
            if not script_file.exists():
                return {"status": "error", "message": f"Script not found: {script_id}"}

            # Read and re-execute the saved script directly (it's already wrapped)
            import subprocess as _sp
            from rendering.mpl_sandbox import _find_python, MplSandboxResult
            import tempfile

            python_exe = _find_python()
            output_path = outputs_dir / f"{script_id}.png"
            env = {
                "PATH": __import__("os").environ.get("PATH", "/usr/bin:/bin"),
                "HOME": __import__("os").environ.get("HOME", "/tmp"),
                "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
            }
            with tempfile.TemporaryDirectory(prefix="mpl_") as mpl_config:
                env["MPLCONFIGDIR"] = mpl_config
                try:
                    proc = _sp.run(
                        [python_exe, str(script_file)],
                        capture_output=True,
                        text=True,
                        timeout=60.0,
                        env=env,
                        cwd=str(Path(__file__).resolve().parent.parent),
                    )
                    if proc.returncode == 0 and output_path.exists():
                        self._event_bus.emit(
                            MPL_RENDER_EXECUTED,
                            agent="VizAgent[Mpl]",
                            msg=f"[MplViz] Script re-executed: {script_id}",
                            data={
                                "script_id": script_id,
                                "description": f"Rerun of {script_id}",
                            },
                        )
                        return {
                            "status": "success",
                            "script_id": script_id,
                            "message": "Script re-executed successfully",
                        }
                    return {
                        "status": "error",
                        "stderr": proc.stderr,
                        "stdout": proc.stdout,
                    }
                except _sp.TimeoutExpired:
                    return {
                        "status": "error",
                        "message": "Script timed out during rerun",
                    }

        elif action == "delete":
            script_id = tool_args.get("script_id")
            if not script_id:
                return {
                    "status": "error",
                    "message": "script_id is required for delete",
                }
            deleted = []
            script_file = scripts_dir / f"{script_id}.py"
            output_file = outputs_dir / f"{script_id}.png"
            if script_file.exists():
                script_file.unlink()
                deleted.append("script")
            if output_file.exists():
                output_file.unlink()
                deleted.append("output")
            if not deleted:
                return {
                    "status": "error",
                    "message": f"No files found for script_id: {script_id}",
                }
            return {"status": "success", "deleted": deleted, "script_id": script_id}

        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    # --- JSX/Recharts Visualization ---

    def _handle_generate_jsx_component(self, tool_args: dict) -> dict:
        """Handle the generate_jsx_component tool call."""
        code = tool_args.get("code")
        if not code:
            return {"status": "error", "message": "code parameter is required"}

        description = tool_args.get("description", "JSX component")

        # Generate script_id
        from datetime import datetime as _dt
        import secrets

        script_id = _dt.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)

        # Run the JSX pipeline
        from rendering.jsx_sandbox import run_jsx_pipeline
        import config as _config

        session_dir = self._session_manager.base_dir / self._session_id
        output_dir = session_dir / "jsx_outputs"

        result = run_jsx_pipeline(
            code=code,
            store=self._store,
            output_dir=output_dir,
            script_id=script_id,
            timeout=30.0,
            max_points=_config.MAX_PLOT_POINTS,
        )

        if result.success:
            # Record in operations log
            self._event_bus.emit(
                JSX_RENDER_EXECUTED,
                agent="VizAgent[JSX]",
                msg=f"[JsxViz] Component compiled: {description}",
                data={
                    "script_id": script_id,
                    "description": description,
                    "output_path": result.output_path,
                    "data_path": result.data_path,
                    "script_path": result.script_path,
                },
            )
            return {
                "status": "success",
                "script_id": script_id,
                "output_path": result.output_path,
                "data_path": result.data_path,
                "message": f"JSX component compiled successfully. Script ID: {script_id}",
            }
        else:
            response = {
                "status": "error",
                "script_id": script_id,
                "message": "JSX compilation failed. See stderr for details.",
                "stderr": result.stderr,
            }
            if result.script_path:
                response["script_path"] = result.script_path
            return response

    def _handle_manage_jsx_output(self, tool_args: dict) -> dict:
        """Handle the manage_jsx_output tool call."""
        action = tool_args.get("action")
        if not action:
            return {"status": "error", "message": "action is required"}

        session_dir = self._session_manager.base_dir / self._session_id
        scripts_dir = session_dir / "jsx_scripts"
        outputs_dir = session_dir / "jsx_outputs"

        if action == "list":
            items = []
            if scripts_dir.exists():
                for script_file in sorted(scripts_dir.glob("*.tsx")):
                    script_id = script_file.stem
                    bundle_file = outputs_dir / f"{script_id}.js"
                    data_file = outputs_dir / f"{script_id}.data.json"
                    items.append(
                        {
                            "script_id": script_id,
                            "has_bundle": bundle_file.exists(),
                            "has_data": data_file.exists(),
                            "script_path": str(script_file),
                        }
                    )
            return {"status": "success", "items": items, "count": len(items)}

        elif action == "get_source":
            script_id = tool_args.get("script_id")
            if not script_id:
                return {
                    "status": "error",
                    "message": "script_id is required for get_source",
                }
            script_file = scripts_dir / f"{script_id}.tsx"
            if not script_file.exists():
                return {"status": "error", "message": f"Script not found: {script_id}"}
            return {
                "status": "success",
                "script_id": script_id,
                "source": script_file.read_text(encoding="utf-8"),
            }

        elif action == "recompile":
            script_id = tool_args.get("script_id")
            if not script_id:
                return {
                    "status": "error",
                    "message": "script_id is required for recompile",
                }
            script_file = scripts_dir / f"{script_id}.tsx"
            if not script_file.exists():
                return {"status": "error", "message": f"Script not found: {script_id}"}

            # Re-run the pipeline with the saved source
            code = script_file.read_text(encoding="utf-8")
            from rendering.jsx_sandbox import run_jsx_pipeline
            import config as _config

            result = run_jsx_pipeline(
                code=code,
                store=self._store,
                output_dir=outputs_dir,
                script_id=script_id,
                timeout=30.0,
                max_points=_config.MAX_PLOT_POINTS,
            )
            if result.success:
                self._event_bus.emit(
                    JSX_RENDER_EXECUTED,
                    agent="VizAgent[JSX]",
                    msg=f"[JsxViz] Component recompiled: {script_id}",
                    data={
                        "script_id": script_id,
                        "description": f"Recompile of {script_id}",
                    },
                )
                return {
                    "status": "success",
                    "script_id": script_id,
                    "message": "Component recompiled successfully",
                }
            return {"status": "error", "stderr": result.stderr}

        elif action == "delete":
            script_id = tool_args.get("script_id")
            if not script_id:
                return {
                    "status": "error",
                    "message": "script_id is required for delete",
                }
            deleted = []
            script_file = scripts_dir / f"{script_id}.tsx"
            bundle_file = outputs_dir / f"{script_id}.js"
            data_file = outputs_dir / f"{script_id}.data.json"
            for f, label in [
                (script_file, "source"),
                (bundle_file, "bundle"),
                (data_file, "data"),
            ]:
                if f.exists():
                    f.unlink()
                    deleted.append(label)
            if not deleted:
                return {
                    "status": "error",
                    "message": f"No files found for script_id: {script_id}",
                }
            return {"status": "success", "deleted": deleted, "script_id": script_id}

        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    # --- SPICE MCP dispatch ---

    _spice_tools_ready = False

    def _ensure_spice_tools(self) -> None:
        """Lazily connect the SPICE MCP client and register discovered tools.

        Called once before the first SPICE tool dispatch. Discovers tool
        schemas from the MCP server, registers them into tools.py and
        agent_registry.py so the LLM sees them and dispatch works.
        """
        if OrchestratorAgent._spice_tools_ready:
            return

        from .mcp_client import get_spice_client
        from .tools import register_dynamic_tools, get_spice_tool_names
        from .agent_registry import register_spice_tools
        from .observations import register_spice_tool_names

        try:
            client = get_spice_client()
        except Exception as e:
            log_error(f"SPICE MCP connection failed during tool discovery: {e}", e)
            return

        # Get tool schemas from MCP server
        schemas = client.get_tool_schemas()
        names = [s["name"] for s in schemas]

        if not names:
            self._event_bus.emit(
                DEBUG, level="warning", msg="[SPICE] MCP server returned no tools"
            )
            return

        # Register into tools.py (TOOLS list + tool catalog)
        register_dynamic_tools(schemas)

        # Register into agent_registry.py (call lists + informed registry)
        register_spice_tools(names)

        # Register into observations.py (_SPICE_TOOLS set)
        register_spice_tool_names(names)

        OrchestratorAgent._spice_tools_ready = True
        self._event_bus.emit(
            DEBUG,
            level="info",
            msg=f"[SPICE] Discovered {len(names)} tools: {', '.join(names)}",
        )

        # Update SPICE catalog entry from live MCP data (non-fatal)
        try:
            from knowledge.catalog import update_spice_from_mcp

            missions_result = client.call_tool("list_spice_missions", {})
            frames_result = client.call_tool("list_coordinate_frames", {})
            missions_list = missions_result.get("missions", [])
            frames_list = frames_result.get("frames", [])
            if missions_list or frames_list:
                update_spice_from_mcp(missions_list, frames_list)
                self._event_bus.emit(
                    DEBUG,
                    level="info",
                    msg=f"[SPICE] Updated catalog: {len(missions_list)} missions, "
                    f"{len(frames_list)} frames",
                )
        except Exception as e:
            self._event_bus.emit(
                DEBUG,
                level="warning",
                msg=f"[SPICE] Failed to update catalog from MCP: {e}",
            )

    def _handle_spice_tool(self, tool_name: str, tool_args: dict) -> dict:
        """Route a SPICE tool call through the MCP client.

        For data-producing calls (responses containing a "data" key),
        extracts full data and stores it in DataStore for plotting.
        """
        from .mcp_client import get_spice_client

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

        # Detect data-producing responses: if the result has a "data" key
        # with a list of records, store it in the DataStore for plotting.
        records = result.get("data")
        if isinstance(records, list) and records:
            try:
                df = pd.DataFrame(records)
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"], format="ISO8601")
                    df = df.set_index("time")

                sc_name = (
                    tool_args.get("spacecraft") or tool_args.get("target1") or "unknown"
                )
                sc_name = sc_name.upper().replace(" ", "_")
                # Derive a descriptive suffix from the tool name
                suffix = (
                    tool_name.replace("get_spacecraft_", "")
                    .replace("get_", "")
                    .replace("compute_", "")
                )
                label = f"SPICE.{sc_name}_{suffix}"
                # Guess units from columns
                cols = list(df.columns)
                if any("km_s" in c for c in cols):
                    units = "km/s"
                elif any("au" in c.lower() for c in cols):
                    units = "AU"
                else:
                    units = "km"
                store = self._store
                entry = DataEntry(
                    label=label,
                    data=df,
                    units=units,
                    description=f"SPICE {suffix} of {sc_name} rel. {tool_args.get('observer', tool_args.get('target2', 'SUN'))}",
                    source="spice",
                )
                store.put(entry)
                result["label"] = label
                result["note"] = (
                    f"Stored as '{label}' — use render_plotly_json to plot."
                )

                # Remove full data from result returned to LLM (it only needs
                # the summary + label; full data is in the DataStore)
                del result["data"]
            except Exception as e:
                self._event_bus.emit(
                    DEBUG,
                    level="warning",
                    msg=f"[SPICE] Failed to store {tool_name} data: {e}",
                )
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

        self._event_bus.emit(
            DEBUG, level="debug", msg=f"[Tool: {tool_name}({tool_args})]"
        )

        # ── Permission check ──
        from agent.agent_registry import AGENT_CALL_REGISTRY
        from agent.tools import get_spice_tool_names

        agent_ctx = f"ctx:{self._current_agent_type}"
        allowed = AGENT_CALL_REGISTRY.get(agent_ctx, frozenset())
        spice_names = get_spice_tool_names()
        if tool_name not in allowed and tool_name not in spice_names:
            msg = f"Tool '{tool_name}' is not available for {self._current_agent_type}."
            log_error(msg, context={"tool_name": tool_name, "tool_args": tool_args})
            return {"status": "error", "message": msg}

        # ── Registry dispatch ──
        from agent.tool_handlers import TOOL_REGISTRY

        handler = TOOL_REGISTRY.get(tool_name)
        if handler:
            return handler(self, tool_args)


        # ── Remaining inline handlers (not yet in registry) ──
        if tool_name == "manage_tool_logs":
            return self._handle_manage_tool_logs(tool_args)

        # --- SPICE Ephemeris Tools (via MCP, dynamically discovered) ---
        elif tool_name in get_spice_tool_names():
            return self._handle_spice_tool(tool_name, tool_args)

        else:
            result = {"status": "error", "message": f"Unknown tool: {tool_name}"}
            log_error(
                f"Unknown tool called: {tool_name}",
                context={"tool_name": tool_name, "tool_args": tool_args},
            )
            return result

    # ---- Query event log handler ----

    def _handle_query_event_log(self, args: dict) -> dict:
        """Handle query_event_log tool: introspect session events."""
        event_types = args.get("event_types")
        tool_name_filter = args.get("tool_name")
        last_n = min(args.get("last_n", 20), get_item_limit("items.query_event_log"))
        since_turn = args.get("since_turn")

        # Find turn boundaries by scanning for USER_MESSAGE events
        all_events = self._event_bus.get_events()
        turn_starts: list[int] = []
        for i, ev in enumerate(all_events):
            if ev.type == USER_MESSAGE:
                turn_starts.append(i)

        # Resolve since_index from since_turn
        since_index = 0
        if since_turn is not None and turn_starts:
            # since_turn=0 → current turn, 1 → previous, etc.
            target_idx = len(turn_starts) - 1 - since_turn
            if 0 <= target_idx < len(turn_starts):
                since_index = turn_starts[target_idx]

        # Query with filters
        type_set = set(event_types) if event_types else None
        events = self._event_bus.get_events(
            types=type_set,
            since_index=since_index,
        )

        # Filter by tool_name if specified
        if tool_name_filter:
            events = [
                e
                for e in events
                if e.data.get("tool_name") == tool_name_filter
                or tool_name_filter in e.msg
            ]

        # Annotate with turn number and format
        formatted = []
        for ev in events[-last_n:]:
            # Find which turn this event belongs to
            turn_num = 0
            for ti, start_idx in enumerate(turn_starts):
                # Find this event's index
                try:
                    ev_idx = all_events.index(ev)
                except ValueError:
                    break
                if ev_idx >= start_idx:
                    turn_num = ti
            entry: dict = {
                "type": ev.type,
                "ts": ev.ts,
                "agent": ev.agent,
                "turn": turn_num,
                "status": ev.data.get("status", ""),
            }
            if ev.data.get("tool_name"):
                entry["tool_name"] = ev.data["tool_name"]
            # Truncate msg to keep response compact
            entry["msg"] = trunc(ev.msg, "detail.request")
            formatted.append(entry)

        return {
            "status": "success",
            "count": len(formatted),
            "total_events": len(all_events),
            "total_turns": len(turn_starts),
            "events": formatted,
        }

    # ---- Async delegation infrastructure ----

    def _build_mission_request(self, mission_id: str, request: str, agent=None) -> str:
        """Build a full request string for a mission delegation."""
        agent_id = agent.agent_id if agent else f"MissionAgent[{mission_id}]"
        store = self._store
        entries = store.list_entries()
        if entries:
            new_entries, removed_labels, store_hash = self._ctx_tracker.get_store_delta(
                agent_id, entries
            )
            if new_entries or removed_labels:
                labels = [
                    f"  - {e['label']} ({e['num_points']} pts, {e['time_min']} to {e['time_max']})"
                    for e in new_entries
                ]
                if new_entries and not removed_labels:
                    request += (
                        "\n\nNew data added to memory:\n"
                        + "\n".join(labels)
                        + "\nDo NOT re-fetch data that is already in memory with a matching label and time range."
                    )
                elif removed_labels and not new_entries:
                    request += (
                        "\n\nData removed from memory: "
                        + ", ".join(removed_labels)
                        + "\nDo NOT re-fetch data that is already in memory with a matching label and time range."
                    )
                else:
                    request += "\n\nData store updated:\n" + "\n".join(labels)
                    if removed_labels:
                        request += "\nRemoved: " + ", ".join(removed_labels)
                    request += "\nDo NOT re-fetch data that is already in memory with a matching label and time range."
                self._ctx_tracker.record(
                    agent_id, store_entries=entries, store_hash=store_hash
                )
            # else: store unchanged — skip injection (actor already has it)
        if not (agent and agent._interaction_id):
            request += "\n\n[Tip: Call check_events to see what happened earlier in this session.]"
        return request

    def _build_dataops_request(self, request: str, context: str, agent=None) -> str:
        """Build a full request string for a DataOps delegation."""
        import json

        _agent_id = agent.agent_id if agent else "DataOpsAgent"
        full_request = f"{request}\n\nContext: {context}" if context else request
        store = self._store
        entries = store.list_entries()
        if entries:
            new_entries, removed_labels, store_hash = self._ctx_tracker.get_store_delta(
                _agent_id, entries
            )
            if new_entries or removed_labels:
                store_text = json.dumps(new_entries, indent=2, default=str)
                if new_entries and not removed_labels:
                    full_request += (
                        "\n\nNew data added to memory:\n```json\n"
                        + store_text
                        + "\n```"
                    )
                elif removed_labels and not new_entries:
                    full_request += "\n\nData removed from memory: " + ", ".join(
                        removed_labels
                    )
                else:
                    full_request += (
                        "\n\nData store updated:\n```json\n" + store_text + "\n```"
                    )
                    if removed_labels:
                        full_request += "\nRemoved: " + ", ".join(removed_labels)
                self._ctx_tracker.record(
                    _agent_id, store_entries=entries, store_hash=store_hash
                )
            # else: store unchanged — skip injection
        if not (agent and agent._interaction_id):
            full_request += "\n\n[Tip: Call check_events to see what happened earlier in this session.]"
        return full_request

    def _build_extraction_request(self, request: str, context: str) -> str:
        """Build a full request string for a DataExtraction delegation."""
        full_request = f"{request}\n\nContext: {context}" if context else request
        return full_request

    def _build_operation_log(
        self,
        since_event_index: int,
        agent_name_filter: str | None = None,
    ) -> list[dict]:
        """Build a structured operation log from EventBus events since a given index.

        Returns a list of dicts summarizing each tool call/result pair,
        data fetch, computation, or error that occurred during the delegation.

        Args:
            since_event_index: Only include events at or after this index.
            agent_name_filter: If provided, only include events whose ``agent``
                field contains this string. Used by concurrent work units to
                isolate their own events from those of other units.
        """
        relevant_types = {
            SUB_AGENT_TOOL,
            DATA_FETCHED,
            DATA_COMPUTED,
            DATA_CREATED,
            RENDER_EXECUTED,
            FETCH_ERROR,
            CUSTOM_OP_FAILURE,
            SUB_AGENT_ERROR,
            TOOL_CALL,
            TOOL_RESULT,
            TOOL_ERROR,
        }
        events = self._event_bus.get_events(
            types=relevant_types,
            since_index=since_event_index,
        )
        if agent_name_filter:
            events = [e for e in events if agent_name_filter in e.agent]

        log_entries: list[dict] = []
        for ev in events:
            tool = ev.data.get("tool_name", "")
            if not tool and ev.type in (SUB_AGENT_TOOL, SUB_AGENT_ERROR):
                # Extract tool name from msg like "[MissionAgent] fetch_data(...)"
                msg = ev.msg
                if "]" in msg:
                    after_bracket = msg.split("]", 1)[1].strip()
                    tool = (
                        after_bracket.split("(")[0].strip()
                        if "(" in after_bracket
                        else after_bracket.split()[0]
                        if after_bracket
                        else ""
                    )

            status = "success"
            error = ""
            if ev.type in (FETCH_ERROR, CUSTOM_OP_FAILURE, SUB_AGENT_ERROR, TOOL_ERROR):
                status = "error"
                error = ev.data.get("error", trunc(ev.msg, "history.error.brief"))
            elif ev.data.get("status") == "error":
                status = "error"
                error = ev.data.get("message", trunc(ev.msg, "history.error.brief"))

            outputs: list[str] = []
            if ev.data.get("label"):
                outputs.append(ev.data["label"])
            if ev.data.get("labels"):
                outputs.extend(ev.data["labels"])

            # Summarize args
            args_summary = ""
            tool_args = ev.data.get("tool_args", {})
            if tool_args:
                parts = []
                _shown_items, _ = trunc_items(
                    list(tool_args.items()), "items.tool_args"
                )
                for k, v in _shown_items:
                    sv = trunc(str(v), "console.args.value")
                    parts.append(f"{k}={sv}")
                args_summary = ", ".join(parts)

            entry: dict = {
                "tool": tool or ev.type,
                "status": status,
            }
            if args_summary:
                entry["args_summary"] = args_summary
            if error:
                entry["error"] = error
            if outputs:
                entry["outputs"] = outputs
            log_entries.append(entry)

        return log_entries

    def _build_work_unit_result(self, unit: "WorkUnit") -> dict:
        """Combine a work unit's result with its operation log."""
        if unit.status == WorkStatus.CANCELLED:
            result = {
                "status": "cancelled",
                "agent_name": unit.agent_name,
                "message": f"Cancelled: {unit.task_summary}",
            }
        elif unit.result:
            result = dict(unit.result)
        else:
            result = {"status": "error", "message": "No result"}
        result["operation_log"] = unit.operation_log
        result["work_unit_id"] = unit.id
        if unit.completed_at is not None:
            result["duration_s"] = round(unit.completed_at - unit.started_at, 1)
        return result

    # ---- Turnless orchestrator: input queue and control center tools ----

    def push_input(self, message: str) -> None:
        """Push a user message into the input queue. Called by API layer.

        Non-blocking — always succeeds. The message will be processed as
        a separate turn after the current turn completes.

        Delivers the message to the actor inbox for processing.
        """
        self._inbox.put(_make_message("user_input", "user", message))
        self._event_bus.emit(
            USER_AMENDMENT,
            level="info",
            msg=f"[Turnless] User input queued ({len(message)} chars)",
            data={"message_preview": trunc(message, "history.error.short")},
        )

    # ---- Persistent event loop (turnless orchestrator) ----

    def run_loop(self) -> None:
        """Persistent event loop — runs for the session's lifetime in a daemon thread.

        Uses the actor inbox (queue.Queue) for message delivery.
        """
        self._run_loop_actor()

    def _run_loop_actor(self) -> None:
        """Agent-based event loop: reads from inbox queue."""
        self._event_bus.emit(DEBUG, level="info", msg="[RunLoop] Started (actor mode)")
        self._set_state(AgentState.SLEEPING, reason="run loop started")

        while not self._shutdown_event.is_set():
            try:
                msg = self._inbox.get(timeout=1.0)
            except queue.Empty:
                continue

            if self._shutdown_event.is_set():
                break

            if msg.type == "user_input":
                # Track cycle: SLEEPING → ACTIVE starts a new cycle
                if self._state == AgentState.SLEEPING:
                    self._cycle_number += 1
                self._set_state(AgentState.ACTIVE, reason="user input")

                # CYCLE START
                self._round_start_tokens = self.get_token_usage()
                self._event_bus.emit(
                    CYCLE_START,
                    level="info",
                    msg="Cycle started",
                    data={"cycle": self._cycle_number},
                )

                turns_in_cycle = 0
                user_message = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                try:
                    response_text = self.process_message(user_message)
                    self._turn_number += 1
                    turns_in_cycle += 1
                    if self._text_already_streamed:
                        # Text was already emitted incrementally via streaming
                        self._text_already_streamed = False
                    else:
                        self._event_bus.emit(
                            TEXT_DELTA,
                            level="info",
                            msg=response_text,
                            data={"text": response_text},
                        )
                except Exception as e:
                    self._text_already_streamed = False
                    self.logger.error(f"[RunLoop] Error: {e}", exc_info=True)
                    self._event_bus.emit(
                        DEBUG,
                        level="error",
                        msg=f"[RunLoop] Error processing message: {e}",
                    )

                # Drain any additional queued messages (new turns within the same cycle)
                while not self._shutdown_event.is_set():
                    try:
                        extra = self._inbox.get_nowait()
                    except queue.Empty:
                        break
                    if extra.type == "user_input":
                        # Emit round boundary so the frontend resets its timer
                        mid_tokens = self.get_token_usage()
                        mid_start = self._round_start_tokens or {}
                        mid_delta = {
                            k: mid_tokens.get(k, 0) - mid_start.get(k, 0)
                            for k in mid_tokens
                        }
                        self._event_bus.emit(
                            CYCLE_END,
                            level="info",
                            msg="Round complete (queued follow-up)",
                            data={
                                "cycle": self._cycle_number,
                                "turns_in_cycle": turns_in_cycle,
                                "token_usage": mid_tokens,
                                "round_token_usage": mid_delta,
                            },
                        )
                        self._round_start_tokens = mid_tokens
                        turns_in_cycle = 0
                        self._event_bus.emit(
                            CYCLE_START,
                            level="info",
                            msg="Round started (queued follow-up)",
                            data={"cycle": self._cycle_number},
                        )

                        extra_msg = (
                            extra.content
                            if isinstance(extra.content, str)
                            else str(extra.content)
                        )
                        try:
                            response_text = self.process_message(extra_msg)
                            self._turn_number += 1
                            turns_in_cycle += 1
                            if self._text_already_streamed:
                                self._text_already_streamed = False
                            else:
                                self._event_bus.emit(
                                    TEXT_DELTA,
                                    level="info",
                                    msg=response_text,
                                    data={"text": response_text},
                                )
                        except Exception as e:
                            self._text_already_streamed = False
                            self.logger.error(f"[RunLoop] Error: {e}", exc_info=True)

                # CYCLE END — cycle complete, back to SLEEPING
                current_tokens = self.get_token_usage()
                start = self._round_start_tokens or {}
                round_delta = {
                    k: current_tokens.get(k, 0) - start.get(k, 0)
                    for k in current_tokens
                }
                self._event_bus.emit(
                    CYCLE_END,
                    level="info",
                    msg="Cycle complete",
                    data={
                        "cycle": self._cycle_number,
                        "turns_in_cycle": turns_in_cycle,
                        "token_usage": current_tokens,
                        "round_token_usage": round_delta,
                    },
                )
                self._set_state(
                    AgentState.SLEEPING,
                    reason=f"cycle {self._cycle_number} complete, {turns_in_cycle} turn(s)",
                )

        self._event_bus.emit(DEBUG, level="info", msg="[RunLoop] Stopped (actor mode)")

    def shutdown_loop(self) -> None:
        """Signal the persistent event loop to stop."""
        self._shutdown_event.set()
        # Wake actor loop by putting a sentinel
        if hasattr(self, "_inbox"):
            self._inbox.put(_make_message("cancel", "system", "shutdown"))

    def _handle_list_active_work(self, tool_args: dict) -> dict:
        """Tool handler: list all running work units."""
        active = self._control_center.list_active()
        return {
            "status": "success",
            "count": len(active),
            "work_units": active,
        }

    def _handle_cancel_work(self, tool_args: dict) -> dict:
        """Tool handler: cancel work units."""
        unit_id = tool_args.get("unit_id")
        agent_type = tool_args.get("agent_type")
        do_all = tool_args.get("cancel_all", False)

        if do_all:
            count = self._control_center.cancel_all()
            self._event_bus.emit(
                WORK_CANCELLED,
                level="info",
                msg=f"[ControlCenter] Cancelled all work ({count} units)",
                data={"cancel_scope": "all", "count": count},
            )
            return {"status": "ok", "cancelled": count}
        elif agent_type:
            count = self._control_center.cancel_by_type(agent_type)
            self._event_bus.emit(
                WORK_CANCELLED,
                level="info",
                msg=f"[ControlCenter] Cancelled {count} {agent_type} units",
                data={"cancel_scope": "type", "agent_type": agent_type, "count": count},
            )
            return {"status": "ok", "cancelled": count}
        elif unit_id:
            unit = self._control_center.get(unit_id)
            ok = self._control_center.cancel(unit_id)
            if ok and unit:
                self._event_bus.emit(
                    WORK_CANCELLED,
                    level="info",
                    msg=f"[ControlCenter] Cancelled {unit.agent_name} (id={unit_id})",
                    data={"cancel_scope": "unit", "unit_id": unit_id},
                )
            return {"status": "ok" if ok else "not_found"}
        else:
            return {
                "status": "error",
                "message": "Provide unit_id, agent_type, or cancel_all",
            }

    def _execute_tool_safe(
        self, tool_name: str, tool_args: dict, tool_call_id: str | None = None
    ) -> dict:
        """Execute a tool with error handling and logging.

        Wraps _execute_tool to catch unexpected exceptions and log them.
        After execution, checks for hot-path memory extraction opportunities.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            tool_call_id: Optional LLM tool_call_id (used for async delegation mapping)

        Returns:
            Dict with result data (varies by tool)
        """
        # Pop and emit commentary before tool execution
        commentary = tool_args.pop("commentary", None)
        if commentary:
            self._event_bus.emit(
                TEXT_DELTA,
                agent="orchestrator",
                level="info",
                msg=f"[orchestrator] {commentary}",
                data={"text": commentary + "\n\n", "commentary": True},
            )

        self._event_bus.emit(
            TOOL_CALL,
            agent="orchestrator",
            msg=f"[Tool] {tool_name}({tool_args})",
            data={"tool_name": tool_name, "tool_args": tool_args},
        )

        # Store tool_call_id in thread-local so delegation handlers can read it
        self._tls.current_tool_call_id = tool_call_id

        try:
            timer = ToolTimer()
            with timer:
                result = self._execute_tool(tool_name, tool_args)
            result = _sanitize_for_json(result)
            stamp_tool_result(result, timer.elapsed_ms)

            # Log the result
            is_success = result.get("status") != "error"
            log_tool_result(tool_name, result, is_success)

            # If error, log with more detail
            if not is_success:
                log_error(
                    f"Tool {tool_name} returned error: {result.get('message', 'Unknown')}",
                    context={
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "result": result,
                    },
                )

            # Invalidate cached pipeline when data-producing tools run
            _PIPELINE_INVALIDATING_TOOLS = {
                "fetch_data",
                "custom_operation",
                "store_dataframe",
                "render_plotly_json",
                "manage_plot",
            }
            if is_success and tool_name in _PIPELINE_INVALIDATING_TOOLS:
                self._invalidate_pipeline()

            _tr_event = TOOL_RESULT
            self._event_bus.emit(
                _tr_event,
                agent="orchestrator",
                msg=f"[Tool Result] {tool_name}: {'success' if is_success else 'error'}",
                data={
                    "tool_name": tool_name,
                    "status": "success" if is_success else "error",
                    "elapsed_ms": timer.elapsed_ms,
                },
            )

            return result

        except Exception as e:
            # Unexpected exception - log with full stack trace
            log_error(
                f"Unexpected exception in tool {tool_name}",
                exc=e,
                context={"tool_name": tool_name, "tool_args": tool_args},
            )
            self._event_bus.emit(
                TOOL_ERROR,
                agent="orchestrator",
                level="error",
                msg=f"[Tool] {tool_name} internal error: {e}",
                data={"tool_name": tool_name, "error": str(e)},
            )
            error_result = {"status": "error", "message": f"Internal error: {e}"}
            stamp_tool_result(error_result, timer.elapsed_ms)
            return error_result

    def _execute_tool_for_actor(
        self, tool_name: str, tool_args: dict, tool_call_id: str | None = None
    ) -> dict:
        """Event-free tool executor for actor sub-agents.

        Identical to _execute_tool_safe but emits NO events. Agents own their
        event lifecycle and emit SUB_AGENT_TOOL / SUB_AGENT_ERROR themselves.
        """
        # Pop commentary — actors emit their own TEXT_DELTA before calling us
        tool_args.pop("commentary", None)
        try:
            timer = ToolTimer()
            with timer:
                result = self._execute_tool(tool_name, tool_args)
            result = _sanitize_for_json(result)
            stamp_tool_result(result, timer.elapsed_ms)

            is_success = result.get("status") != "error"
            if not is_success:
                log_error(
                    f"Tool {tool_name} returned error: {result.get('message', 'Unknown')}",
                    context={
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "result": result,
                    },
                )

            _PIPELINE_INVALIDATING_TOOLS = {
                "fetch_data",
                "custom_operation",
                "store_dataframe",
                "render_plotly_json",
                "manage_plot",
            }
            if is_success and tool_name in _PIPELINE_INVALIDATING_TOOLS:
                self._invalidate_pipeline()

            return result

        except Exception as e:
            log_error(
                f"Unexpected exception in tool {tool_name}",
                exc=e,
                context={"tool_name": tool_name, "tool_args": tool_args},
            )
            error_result = {"status": "error", "message": f"Internal error: {e}"}
            stamp_tool_result(error_result, timer.elapsed_ms)
            return error_result

    def _handle_manage_tool_logs(self, args: dict) -> dict:
        """Handle manage_tool_logs tool: view/add/drop tool log subscriptions."""
        from .agent_registry import AGENT_INFORMED_REGISTRY

        action = args.get("action")
        tool_names = args.get("tool_names", [])
        reasoning = args.get("reasoning", "")

        if not reasoning:
            return {"status": "error", "message": "reasoning is required"}
        if not tool_names:
            return {"status": "error", "message": "tool_names is required"}

        # Determine which agent context is making the request
        ctx = f"ctx:{self._current_agent_type}"

        if action == "view":
            # Bypass tag system — scan ALL events for matching tool names
            target = set(tool_names)
            max_events = args.get("max_events", 20)
            all_events = self._event_bus.get_events(
                types={
                    SUB_AGENT_TOOL,
                    TOOL_CALL,
                    TOOL_RESULT,
                    DATA_FETCHED,
                    DATA_COMPUTED,
                    DATA_CREATED,
                    RENDER_EXECUTED,
                    PLOT_ACTION,
                }
            )
            matched = []
            for ev in reversed(all_events):
                ev_tool = ev.data.get("tool_name", "")
                if ev_tool in target:
                    matched.append(
                        {
                            "type": ev.type,
                            "agent": ev.agent,
                            "tool_name": ev_tool,
                            "tool_args": ev.data.get("tool_args", {}),
                            "result_summary": trunc(
                                str(ev.data.get("tool_result", ev.msg)),
                                "history.task_result",
                            ),
                            "ts": ev.ts,
                        }
                    )
                    if len(matched) >= max_events:
                        break
            return {"status": "success", "events": matched, "count": len(matched)}

        elif action == "add":
            all_valid = {t["name"] for t in _ALL_TOOLS}
            results = []
            for tn in tool_names:
                if tn not in all_valid:
                    results.append(
                        {"tool": tn, "status": "error", "message": "Unknown tool"}
                    )
                    continue
                added = AGENT_INFORMED_REGISTRY.add(ctx, tn, reasoning)
                results.append(
                    {"tool": tn, "status": "added" if added else "already_present"}
                )

        elif action == "drop":
            results = []
            for tn in tool_names:
                ok, err = AGENT_INFORMED_REGISTRY.drop(ctx, tn, reasoning)
                results.append(
                    {"tool": tn, "status": "dropped" if ok else "error", "message": err}
                )

        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

        # For add/drop: persist
        persist_path = get_data_dir() / "informed_tools.json"
        AGENT_INFORMED_REGISTRY.save(persist_path)

        return {
            "status": "success",
            "action": action,
            "results": results,
            "reasoning": reasoning,
        }

    # ---- Prefix → mission ID mapping for dataset-based scope detection ----
    _DATASET_PREFIX_MAP = {
        "PSP": "PSP",
        "AC": "ACE",
        "SOLO": "SolO",
        "SO": "SolO",
        "OMNI": "OMNI",
        "WI": "WIND",
        "DSCOVR": "DSCOVR",
        "MMS1": "MMS",
        "MMS2": "MMS",
        "MMS3": "MMS",
        "MMS4": "MMS",
        "MMS": "MMS",
        "STA": "STEREO_A",
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

        self._event_bus.emit(
            DEBUG, level="debug", msg=f"[Task] Executing: {task.description}"
        )
        self._event_bus.emit(
            DEBUG, level="debug", msg="[Gemini] Sending task instruction..."
        )

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
            self._last_tool_context = "task:" + trunc(
                task.description, "api.session_preview"
            )
            response = self._send_with_timeout(task_chat, task_prompt)
            self._track_usage(response)

            # Process tool calls with loop guard
            guard = LoopGuard(
                max_total_calls=get_turn_limit("orchestrator.task.max_total_calls"),
                dup_free_passes=get_turn_limit("orchestrator.dup_free_passes"),
                dup_hard_block=get_turn_limit("orchestrator.dup_hard_block"),
            )
            last_stop_reason = None
            had_successful_tool = False

            while True:
                if self._cancel_event.is_set():
                    self._event_bus.emit(
                        DEBUG, level="info", msg="[Cancel] Stopping task execution loop"
                    )
                    last_stop_reason = "cancelled by user"
                    break

                if not response.tool_calls:
                    break

                function_calls = response.tool_calls

                # Break if LLM is trying to ask for clarification (not supported in task execution)
                if any(fc.name == "ask_clarification" for fc in function_calls):
                    self._event_bus.emit(
                        DEBUG,
                        level="debug",
                        msg="[Task] Skipping clarification request",
                    )
                    break

                # Total call limit check
                stop_reason = guard.check_limit(len(function_calls))
                if stop_reason:
                    self._event_bus.emit(
                        DEBUG, level="debug", msg=f"[Task] Stopping: {stop_reason}"
                    )
                    last_stop_reason = stop_reason
                    break

                function_responses = []
                for fc in function_calls:
                    tool_name = fc.name
                    tool_args = dict(fc.args) if fc.args else {}

                    # Duplicate call tracking
                    verdict = guard.record_tool_call(tool_name, tool_args)
                    if verdict.blocked:
                        result = {
                            "status": "blocked",
                            "_duplicate_warning": verdict.warning,
                            "message": f"Execution skipped — duplicate call #{verdict.count}",
                        }
                    else:
                        task.tool_calls.append(tool_name)
                        result = self._execute_tool_safe(tool_name, tool_args)
                        if verdict.warning and isinstance(result, dict):
                            result["_duplicate_warning"] = verdict.warning

                    if result.get("status") == "success":
                        had_successful_tool = True
                    elif result.get("status") == "error":
                        self._event_bus.emit(
                            DEBUG,
                            level="warning",
                            msg=f"[Tool Result: ERROR] {result.get('message', '')}",
                        )

                    function_responses.append(
                        self.adapter.make_tool_result_message(
                            tool_name, result, tool_call_id=fc.id
                        )
                    )

                guard.record_calls(len(function_calls))

                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[LLM] Sending {len(function_responses)} tool result(s) back...",
                )
                tool_names = [fc.name for fc in function_calls]
                self._last_tool_context = "+".join(tool_names)
                response = self._send_with_timeout(task_chat, function_responses)
                self._track_usage(response)

            # Warn if no tools were called (LLM just responded with text)
            if not task.tool_calls:
                log_error(
                    f"Task completed without any tool calls: {task.description}",
                    context={"task_instruction": task.instruction},
                )
                self._event_bus.emit(
                    DEBUG,
                    level="warning",
                    msg="[WARNING] No tools were called for this task",
                )

            # Extract text response
            result_text = response.text
            if not result_text and task.tool_results:
                result_text = build_outcome_summary(task.tool_results)
            if not result_text and task.tool_calls:
                result_text = f"Completed. Tools called: {', '.join(task.tool_calls)}"
            if not result_text:
                result_text = "Completed with no output."

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

            self._event_bus.emit(
                DEBUG,
                level="debug",
                msg=f"[Task] {'Failed' if last_stop_reason else 'Completed'}: {task.description}",
            )

            return result_text

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self._event_bus.emit(
                DEBUG, level="warning", msg=f"[Task] Failed: {task.description} - {e}"
            )
            return f"Error: {e}"

    def _summarize_plan_execution(self, plan: TaskPlan) -> str:
        """Generate a summary of the completed plan execution."""
        # Build context from completed tasks
        summary_parts = [
            f'I just executed a multi-step plan for: "{plan.user_request}"'
        ]
        summary_parts.append("")

        completed = plan.get_completed_tasks()
        failed = plan.get_failed_tasks()

        if completed:
            summary_parts.append("Completed tasks:")
            for task in completed:
                summary_parts.append(f"  - {task.description}")
                if task.result:
                    result_preview = trunc(task.result, "history.error.short")
                    summary_parts.append(f"    Result: {result_preview}")

        if failed:
            summary_parts.append("")
            summary_parts.append("Failed tasks:")
            for task in failed:
                summary_parts.append(f"  - {task.description}")
                if task.error:
                    summary_parts.append(f"    Error: {task.error}")

        # Check if viz tasks failed and no plot exists
        viz_failed = any(
            t.mission == "__visualization__" and t.status == TaskStatus.FAILED
            for t in (list(completed) + list(failed))
        )
        has_plot = self._renderer.get_current_state().get("has_plot", False)
        if viz_failed and not has_plot:
            summary_parts.append("")
            summary_parts.append(
                "IMPORTANT: The visualization task FAILED and NO plot is displayed. "
                "You MUST delegate to the visualization agent to create the plot "
                "using the data currently available in memory."
            )

        summary_parts.append("")
        summary_parts.append(
            "Please provide a brief summary of what was accomplished for the user."
        )

        prompt = "\n".join(summary_parts)

        self._event_bus.emit(
            DEBUG, level="debug", msg="[LLM] Generating execution summary..."
        )

        try:
            self._last_tool_context = "plan_summary"
            response = self._send_message(prompt)
            self._track_usage(response)

            text = response.text or plan.progress_summary()
            text += self._extract_grounding_sources(response)
            return text

        except Exception as e:
            log_error(
                "Error generating plan summary", exc=e, context={"plan_id": plan.id}
            )
            self._event_bus.emit(
                DEBUG, level="warning", msg=f"[Summary] Error generating summary: {e}"
            )
            return plan.progress_summary()

    def _get_or_create_planner_agent(self) -> PlannerAgent:
        """Get the cached planner agent or create a new one."""
        if self._planner_agent is None:
            self._planner_agent = PlannerAgent(
                adapter=self.adapter,
                model_name=config.PLANNER_MODEL,
                tool_executor=self._execute_tool_for_actor,
                verbose=self.verbose,
                cancel_event=self._cancel_event,
                event_bus=self._event_bus,
                ctx_tracker=self._ctx_tracker,
            )
            self._event_bus.emit(
                DEBUG,
                level="debug",
                msg=f"[Router] Created PlannerAgent ({config.PLANNER_MODEL})",
            )
        return self._planner_agent

    def _build_task_result_summary(
        self, task: Task, labels_before: set[str], labels_after: set[str]
    ) -> dict:
        """Build an informative result summary dict for a completed plan task."""
        result_text = trunc(task.result or "", "history.task_result")
        if (
            result_text in ("", "Done.", "Completed with no output.")
            and task.tool_calls
        ):
            result_text = f"Tools called: {', '.join(task.tool_calls)}"

        new_labels = labels_after - labels_before
        if new_labels:
            new_label_parts = []
            for lbl in sorted(new_labels):
                entry_info = next(
                    (e for e in self._store.list_entries() if e["label"] == lbl),
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
            if tr.get("time_coverage"):
                cov = tr["time_coverage"]
                if cov["coverage_pct"] < 50:
                    warnings.append(
                        f"{tr.get('label', '?')}: data covers only {cov['coverage_pct']}% "
                        f"of requested range (actual: {cov['actual_start']} to {cov['actual_end']})"
                    )
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
        self._event_bus.emit(
            PROGRESS, level="debug", msg=f"[Plan]{mission_tag}: {task.description}"
        )

        # Inject canonical time range so all tasks use the same dates
        if self._plan_time_range:
            tr_str = self._plan_time_range.to_time_range_string()
            task.instruction += f"\n\nCanonical time range for this plan: {tr_str}"

        # Inject current data-store contents so sub-agents know what's available
        store = self._store
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
            is_export = (
                "export" in instr_lower
                or ".png" in instr_lower
                or ".pdf" in instr_lower
            )

            if is_export:
                # Export is a simple dispatch — handle directly, no need for LLM
                self._handle_export_task(task)
            else:
                # Plot tasks: ensure instruction includes actual labels
                active_viz_render = (
                    "generate_mpl_script"
                    if config.PREFER_VIZ_BACKEND == "matplotlib"
                    else "render_plotly_json"
                )
                has_tool_ref = active_viz_render in instr_lower
                if not has_tool_ref and entries:
                    all_labels = ",".join(e["label"] for e in entries)
                    task.instruction = (
                        f"Use {active_viz_render} to plot {all_labels}. "
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
                        task.instruction += (
                            f"\n\nCurrently displayed: {state['traces']}"
                        )
                else:
                    task.instruction += "\n\nNo plot currently displayed."

                if config.PREFER_VIZ_BACKEND == "matplotlib":
                    actor = self._get_or_create_viz_mpl_actor()
                else:
                    actor = self._get_or_create_viz_plotly_actor()
                result = actor.send_and_wait(
                    task.instruction, sender="planner", timeout=300.0
                )
                task.result = result.get("text", "")
                task.status = (
                    TaskStatus.COMPLETED
                    if not result.get("failed")
                    else TaskStatus.FAILED
                )
                if result.get("errors"):
                    task.error = "; ".join(result["errors"])
        elif task.mission == "__data_ops__":
            actor = self._get_available_dataops_actor()
            _eph = actor.agent_id != "DataOpsAgent"
            result = actor.send_and_wait(
                task.instruction, sender="planner", timeout=300.0
            )
            if _eph:
                self._cleanup_ephemeral_actor(actor.agent_id)
            task.result = result.get("text", "")
            task.status = (
                TaskStatus.COMPLETED if not result.get("failed") else TaskStatus.FAILED
            )
            if result.get("errors"):
                task.error = "; ".join(result["errors"])
        elif task.mission == "__data_extraction__":
            actor = self._get_or_create_extraction_actor()
            result = actor.send_and_wait(
                task.instruction, sender="planner", timeout=300.0
            )
            task.result = result.get("text", "")
            task.status = (
                TaskStatus.COMPLETED if not result.get("failed") else TaskStatus.FAILED
            )
            if result.get("errors"):
                task.error = "; ".join(result["errors"])
        elif task.mission and task.mission not in special_missions:
            # Inject candidate datasets into instruction for mission agent
            if task.candidate_datasets:
                ds_list = ", ".join(task.candidate_datasets)
                task.instruction += f"\n\nCandidate datasets to inspect: {ds_list}"
            try:
                primary = self._get_or_create_mission_agent(task.mission)
            except (KeyError, FileNotFoundError):
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Plan] Unknown mission '{task.mission}', using main agent",
                )
                self._execute_task(task)
                return
            # Overflow to ephemeral if primary is busy
            if primary.state != AgentState.SLEEPING or primary.inbox.qsize() > 0:
                actor = self._create_ephemeral_mission_actor(task.mission)
                is_ephemeral = True
            else:
                actor = primary
                is_ephemeral = False
            result = actor.send_and_wait(
                task.instruction, sender="planner", timeout=300.0
            )
            if is_ephemeral:
                self._cleanup_ephemeral_actor(actor.agent_id)
            task.result = result.get("text", "")
            task.status = (
                TaskStatus.COMPLETED if not result.get("failed") else TaskStatus.FAILED
            )
            if result.get("errors"):
                task.error = "; ".join(result["errors"])
        else:
            self._execute_task(task)

    def _handle_export_task(self, task: Task) -> None:
        """Handle an export task directly without the VizAgent.

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
        fn_match = re.search(
            r"[\w.-]+\.(?:png|pdf|svg)", task.instruction, re.IGNORECASE
        )
        filename = fn_match.group(0) if fn_match else "output.png"

        self._event_bus.emit(
            DEBUG, level="debug", msg=f"[Plan] Direct export: {filename}"
        )
        task.tool_calls.append("export")

        result = self._renderer.export(filename)
        # Auto-open in non-GUI/non-web mode
        if (
            result.get("status") == "success"
            and not self.gui_mode
            and not self.web_mode
        ):
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
                self._event_bus.emit(
                    DEBUG, level="debug", msg=f"[Export] Could not auto-open: {e}"
                )
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
            r"\bfor\s+(.+?)(?:\s*$)",
            r"\bfrom\s+(\d{4}.+?)(?:\s*$)",
            r"\bduring\s+(.+?)(?:\s*$)",
            r"(\d{4}-\d{2}-\d{2}\s+to\s+\d{4}-\d{2}-\d{2})",
            r"(\d{4}-\d{2}-\d{2}T[\d:]+\s+to\s+\d{4}-\d{2}-\d{2}T[\d:]+)",
            r"((?:last\s+(?:\d+\s+)?(?:week|day|month|year))s?)",
            r"((?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+\d{4})",
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
        self._event_bus.emit(
            DEBUG,
            level="debug",
            msg="[PlannerAgent] Starting planner for complex request...",
        )

        # Prefer the structured time_range from the tool call (Gemini-resolved).
        # Fall back to regex extraction only when the structured param is empty.
        self._plan_time_range = None
        if structured_time_range:
            try:
                self._plan_time_range = parse_time_range(structured_time_range)
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[PlannerAgent] Resolved time range (structured): "
                    f"{self._plan_time_range.to_time_range_string()}",
                )
            except (TimeRangeError, ValueError) as e:
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[PlannerAgent] Structured time_range parse failed: {e}",
                )

        if not self._plan_time_range:
            # Strip memory context before extracting time range — memory contains
            # date references from past sessions that confuse the regex.
            import re as _re

            clean_msg = _re.sub(
                r"\[CONTEXT FROM LONG-TERM MEMORY\].*?\[END MEMORY CONTEXT\]\s*",
                "",
                user_message,
                flags=_re.DOTALL,
            )
            self._plan_time_range = self._extract_time_range(clean_msg)
            if self._plan_time_range:
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[PlannerAgent] Resolved time range (regex fallback): "
                    f"{self._plan_time_range.to_time_range_string()}",
                )

        planner = self._get_or_create_planner_agent()

        # Build planning message, injecting resolved time range if available
        planning_msg = user_message
        if self._plan_time_range:
            tr_str = self._plan_time_range.to_time_range_string()
            planning_msg = f"{user_message}\n\nResolved time range: {tr_str}. Use this exact range for ALL fetch tasks."

        planning_msg += (
            "\n\n[Tip: Call check_events to see what happened earlier in this session.]"
        )

        # Round 1: initial planning
        response = planner.start_planning(planning_msg)
        if response is None:
            self._event_bus.emit(
                DEBUG,
                level="debug",
                msg="[PlannerAgent] Planner failed, falling back to direct execution",
            )
            return self._process_single_message(user_message)

        plan = create_plan(user_message, [])
        self._current_plan = plan
        plan.status = PlanStatus.EXECUTING
        store = get_task_store()
        store.save(plan)

        log_plan_event(
            "created",
            plan.id,
            f"Dynamic plan for: {trunc(user_message, 'api.session_preview')}",
        )

        # Surface plan summary in web UI live log
        plan_summary = response.get("summary") or response.get("reasoning", "")
        if plan_summary:
            plan_summary = trunc(plan_summary, "detail.code")
            self._event_bus.emit(
                PROGRESS, level="debug", msg=f"[Planning] {plan_summary}"
            )

        tasks_preview = response.get("tasks", [])
        if tasks_preview:
            task_lines = []
            for i, t in enumerate(tasks_preview, 1):
                desc = t.get("description", "?")
                ds = t.get("candidate_datasets")
                ds_str = f" ({', '.join(ds)})" if ds else ""
                task_lines.append(f"  {i}. {desc}{ds_str}")
            self._event_bus.emit(
                DEBUG, level="debug", msg="[Planning] Tasks:\n" + "\n".join(task_lines)
            )

        round_num = 0
        while round_num < get_turn_limit("planner.max_rounds"):
            round_num += 1

            if self._cancel_event.is_set():
                self._event_bus.emit(
                    DEBUG,
                    level="info",
                    msg="[Cancel] Stopping plan loop between rounds",
                )
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
                        self._event_bus.emit(
                            DEBUG,
                            level="debug",
                            msg=f"[Plan] Stripped invalid candidate_datasets "
                            f"from '{task.description}': {invalid}",
                        )
                    if valid:
                        task.candidate_datasets = valid
                    else:
                        # ALL candidates invalid — flag for re-prompt
                        self._event_bus.emit(
                            DEBUG,
                            level="warning",
                            msg=f"[Plan] ALL candidate_datasets invalid for "
                            f"'{task.description}': {invalid}",
                        )
                        all_candidates_invalid = True
                new_tasks.append(task)

            # Log validation summary (round 1 is the long one)
            if round_num == 1:
                valid_count = sum(1 for t in new_tasks if t.candidate_datasets)
                total = len(new_tasks)
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Planning] Validated datasets for {valid_count}/{total} tasks",
                )

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
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Plan] Sending correction to planner: {correction_msg}",
                )
                response = planner.continue_planning(
                    [
                        {
                            "description": "Dataset ID validation",
                            "status": "failed",
                            "result_summary": correction_msg,
                            "error": correction_msg,
                        }
                    ],
                    round_num=round_num,
                    max_rounds=get_turn_limit("planner.max_rounds"),
                )
                if response is None:
                    self._event_bus.emit(
                        DEBUG,
                        level="debug",
                        msg="[PlannerAgent] Planner error after correction, finalizing",
                    )
                    break
                # Re-process the corrected response in the next loop iteration
                continue

            plan.add_tasks(new_tasks)
            store.save(plan)

            self._event_bus.emit(
                DEBUG,
                level="debug",
                msg=f"[PlannerAgent] Round {round_num}: {len(new_tasks)} tasks "
                f"(status={response['status']})",
            )
            self._event_bus.emit(
                DEBUG, level="debug", msg=format_plan_for_display(plan)
            )
            self._event_bus.emit(
                DEBUG,
                level="debug",
                msg=f"[Plan] Executing {len(new_tasks)} task(s) (round {round_num})...",
            )

            # Execute batch — partition into parallelizable fetch tasks and serial tasks
            special_missions = {
                "__visualization__",
                "__data_ops__",
                "__data_extraction__",
            }
            fetch_tasks = [
                t for t in new_tasks if t.mission and t.mission not in special_missions
            ]
            other_tasks = [t for t in new_tasks if t not in fetch_tasks]

            # Coalesce same-mission tasks so one agent handles all quantities
            mission_groups: dict[str, list[Task]] = defaultdict(list)
            for t in fetch_tasks:
                mission_groups[t.mission].append(t)

            coalesced_fetch_tasks: list[Task] = []
            absorbed_map: dict[int, Task] = {}  # id(absorbed_task) -> parent_task
            for mission_id, tasks in mission_groups.items():
                if len(tasks) == 1:
                    coalesced_fetch_tasks.append(tasks[0])
                else:
                    # Combine instructions into the first task
                    primary = tasks[0]
                    parts = [primary.instruction]
                    for t in tasks[1:]:
                        parts.append(t.instruction)
                        absorbed_map[id(t)] = primary
                    primary.instruction = "\n\n---\n\nAlso handle this request:\n".join(
                        parts
                    )
                    primary.description = " + ".join(t.description for t in tasks)
                    # Merge candidate_datasets from all tasks
                    if any(t.candidate_datasets for t in tasks):
                        all_candidates: list[str] = []
                        for t in tasks:
                            if t.candidate_datasets:
                                all_candidates.extend(t.candidate_datasets)
                        primary.candidate_datasets = list(dict.fromkeys(all_candidates))
                    coalesced_fetch_tasks.append(primary)
                    self._event_bus.emit(
                        DEBUG,
                        level="debug",
                        msg=f"[Plan] Coalesced {len(tasks)} tasks for {mission_id} into one delegation",
                    )
            fetch_tasks = coalesced_fetch_tasks

            round_results = []
            cancelled = False

            # Run fetch tasks in parallel if multiple independent missions
            from config import PARALLEL_FETCH

            if (
                PARALLEL_FETCH
                and len(fetch_tasks) > 1
                and not self._cancel_event.is_set()
            ):
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Parallel] Executing {len(fetch_tasks)} fetch tasks concurrently: "
                    f"{[t.mission for t in fetch_tasks]}",
                )
                labels_before = set(e["label"] for e in self._store.list_entries())
                max_workers = min(len(fetch_tasks), 3)

                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {
                        pool.submit(
                            contextvars.copy_context().run,
                            self._execute_plan_task,
                            t,
                            plan,
                        ): t
                        for t in fetch_tasks
                    }
                    for future in as_completed(futures):
                        t = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            t.status = TaskStatus.FAILED
                            t.error = str(e)

                labels_after = set(e["label"] for e in self._store.list_entries())
                # Build result summaries for all parallel tasks
                for task in fetch_tasks:
                    round_results.append(
                        self._build_task_result_summary(
                            task, labels_before, labels_after
                        )
                    )
                store.save(plan)
            else:
                # Run fetch tasks serially (0 or 1 task)
                other_tasks = list(new_tasks)  # all tasks go serial
                fetch_tasks = []

            # Run remaining tasks serially (viz, data_ops, data_extraction, or single fetches)
            for i, task in enumerate(other_tasks):
                if self._cancel_event.is_set():
                    self._event_bus.emit(
                        DEBUG, level="info", msg="[Cancel] Stopping plan mid-batch"
                    )
                    for remaining in other_tasks[i:]:
                        remaining.status = TaskStatus.SKIPPED
                        remaining.error = "Cancelled by user"
                    cancelled = True
                    break
                labels_before = set(e["label"] for e in self._store.list_entries())
                self._execute_plan_task(task, plan)
                labels_after = set(e["label"] for e in self._store.list_entries())
                round_results.append(
                    self._build_task_result_summary(task, labels_before, labels_after)
                )
                store.save(plan)

            # Propagate results to tasks that were coalesced into a parent
            for task in new_tasks:
                parent = absorbed_map.get(id(task))
                if parent is not None:
                    task.result = parent.result
                    task.status = parent.status
                    task.error = parent.error

            if cancelled:
                store.save(plan)
                break

            # Append current store state so planner knows what data exists
            store_entries = self._store.list_entries()
            if store_entries:
                store_details = [
                    {
                        "label": e["label"],
                        "columns": e.get("columns", []),
                        "shape": e.get("shape", ""),
                        "units": e.get("units", ""),
                        "num_points": e.get("num_points", 0),
                    }
                    for e in store_entries
                ]
                for r in round_results:
                    r["data_in_memory"] = store_details

            if response.get("status") == "done":
                # Override: reject "done" if viz tasks failed and no plot exists
                viz_failed = any(
                    t.mission == "__visualization__" and t.status == TaskStatus.FAILED
                    for t in new_tasks
                )
                has_plot = self._renderer.get_current_state().get("has_plot", False)
                if (
                    viz_failed
                    and not has_plot
                    and round_num < get_turn_limit("planner.max_rounds")
                ):
                    self._event_bus.emit(
                        DEBUG,
                        level="debug",
                        msg="[Plan] Orchestrator rejecting 'done' — viz failed, no plot exists",
                    )
                    response = planner.continue_planning(
                        round_results,
                        round_num=round_num,
                        max_rounds=get_turn_limit("planner.max_rounds"),
                    )
                    if response is not None:
                        continue
                break

            # Replan: send results back to planner with round budget
            response = planner.continue_planning(
                round_results,
                round_num=round_num,
                max_rounds=get_turn_limit("planner.max_rounds"),
            )
            if response is None:
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg="[PlannerAgent] Planner error mid-plan, finalizing",
                )
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
        """Process a single (non-complex) user message.

        Uses the Control Center for async delegation tracking.
        """
        self._insight_review_iter = 0
        self._event_bus.emit(
            DEBUG, level="debug", msg="[LLM] Sending message to model..."
        )
        self._last_tool_context = "initial_message"
        response = self._send_message_streaming(user_message)
        self._track_usage(response)
        self._emit_intermediate_text(response)

        self._event_bus.emit(DEBUG, level="debug", msg="[LLM] Response received.")
        self._log_grounding_queries(response)

        cc = self._control_center  # shorthand

        guard = LoopGuard(
            max_total_calls=get_turn_limit("orchestrator.max_total_calls"),
            dup_free_passes=get_turn_limit("orchestrator.dup_free_passes"),
            dup_hard_block=get_turn_limit("orchestrator.dup_hard_block"),
        )

        while True:
            if self._cancel_event.is_set():
                self._event_bus.emit(
                    DEBUG, level="info", msg="[Cancel] Stopping orchestrator loop"
                )
                break

            if not response.tool_calls:
                # FREEZE/WAKE: if async work is pending, wait for completion
                if cc.has_pending():
                    self._event_bus.emit(
                        DEBUG,
                        level="debug",
                        msg="[Turnless] No tool calls but work pending — waiting",
                    )

                    completed = cc.wait_for_any(
                        cancel_event=self._cancel_event,
                    )
                    completed.extend(cc.drain_completed())

                    if completed:
                        function_responses = []
                        has_async_failure = False

                        for unit in completed:
                            # Normal delegation → tool result
                            result = self._build_work_unit_result(unit)
                            if result.get("status") == "error":
                                has_async_failure = True
                            if config.OBSERVATION_SUMMARIES:
                                from .observations import generate_observation

                                tool_name = self._work_unit_tool_name(unit)
                                tool_args = {"request": unit.task_summary}
                                result["observation"] = generate_observation(
                                    tool_name, tool_args, result
                                )
                            function_responses.append(
                                self.adapter.make_tool_result_message(
                                    self._work_unit_tool_name(unit),
                                    result,
                                    tool_call_id=unit.tool_call_id,
                                )
                            )

                        if function_responses:
                            response = self._send_message_streaming(function_responses)
                            self._track_usage(response)
                            self._emit_intermediate_text(response)

                        # If delegation failed and LLM gave up, nudge it to continue
                        if has_async_failure and not response.tool_calls:
                            nudge = (
                                "The delegation above FAILED. The user's request is NOT fulfilled. "
                                "Do NOT say 'Done'. Analyze the error details and try a different "
                                "approach: different parameters, alternative dataset, or handle "
                                "the operation directly with your own tools."
                            )
                            self._event_bus.emit(
                                DEBUG,
                                level="debug",
                                msg="[Orchestrator] Nudging LLM to continue after async delegation failure",
                            )
                            response = self._send_message(nudge)
                            self._track_usage(response)
                            self._emit_intermediate_text(response)

                        # If delegation succeeded but no plot exists, nudge viz
                        elif not has_async_failure and not response.tool_calls:
                            has_plot = self._renderer.get_current_state().get(
                                "has_plot", False
                            )
                            if not has_plot:
                                if self._store and self._store.list_entries():
                                    nudge = (
                                        "Data delegation completed successfully, but NO visualization has been "
                                        "produced yet. If the user's request requires a plot, delegate to the "
                                        "visualization agent now."
                                    )
                                    self._event_bus.emit(
                                        DEBUG,
                                        level="debug",
                                        msg="[Orchestrator] Nudging LLM — data ready but no plot",
                                    )
                                    response = self._send_message(nudge)
                                    self._track_usage(response)
                                    self._emit_intermediate_text(response)

                        continue
                break

            function_calls = response.tool_calls

            # Total call limit check
            stop_reason = guard.check_limit(len(function_calls))
            if stop_reason:
                self._event_bus.emit(
                    DEBUG, level="debug", msg=f"[Orchestrator] Stopping: {stop_reason}"
                )
                break

            # Duplicate call tracking — get verdicts, partition blocked vs executable
            verdicts: list[DupVerdict] = []
            to_execute: list = []
            for fc in function_calls:
                args = dict(fc.args) if fc.args else {}
                verdict = guard.record_tool_call(fc.name, args)
                verdicts.append(verdict)
                if verdict.blocked:
                    self._event_bus.emit(
                        DEBUG,
                        level="debug",
                        msg=f"[Orchestrator] Blocked duplicate: {fc.name} (count={verdict.count})",
                    )
                else:
                    to_execute.append(fc)

            # Execute only non-blocked tools
            executed_results = (
                self._execute_tools_parallel(to_execute) if to_execute else []
            )

            # Reassemble tool_results in original order, injecting warnings
            exec_iter = iter(executed_results)
            tool_results: list[tuple[str | None, str, dict, dict]] = []
            for fc, verdict in zip(function_calls, verdicts):
                tc_id = getattr(fc, "id", None)
                args = dict(fc.args) if fc.args else {}
                if verdict.blocked:
                    result = {
                        "status": "blocked",
                        "_duplicate_warning": verdict.warning,
                        "message": f"Execution skipped — duplicate call #{verdict.count}",
                    }
                    tool_results.append((tc_id, fc.name, args, result))
                else:
                    ex_tc_id, ex_name, ex_args, result = next(exec_iter)
                    if verdict.warning and isinstance(result, dict):
                        result["_duplicate_warning"] = verdict.warning
                    tool_results.append((ex_tc_id, ex_name, ex_args, result))

            function_responses = []

            # First pass: logging, delegation tracking, observations
            for tc_id, tool_name, tool_args, result in tool_results:
                if result.get("status") == "error":
                    self._event_bus.emit(
                        DEBUG,
                        level="warning",
                        msg=f"[Tool Result: ERROR] {result.get('message', '')}",
                    )

                if config.OBSERVATION_SUMMARIES and result.get("status") not in (
                    "pending_async",
                    "blocked",
                ):
                    from .observations import generate_observation

                    result["observation"] = generate_observation(
                        tool_name, tool_args, result
                    )

            # Drain completed work units that finished during sync tool execution
            just_completed = cc.drain_completed()
            for unit in just_completed:
                result = self._build_work_unit_result(unit)
                if config.OBSERVATION_SUMMARIES:
                    from .observations import generate_observation

                    t_name = self._work_unit_tool_name(unit)
                    result["observation"] = generate_observation(
                        t_name, {"request": unit.task_summary}, result
                    )
                tool_results.append(
                    (unit.tool_call_id, t_name, {"request": unit.task_summary}, result)
                )

            # Append reflection hint when ALL tools in a round failed
            non_async_results = [
                r for _, _, _, r in tool_results if r.get("status") != "pending_async"
            ]
            if config.SELF_REFLECTION and len(non_async_results) > 0:
                all_errors = all(r.get("status") == "error" for r in non_async_results)
                if all_errors:
                    last_result = non_async_results[-1]
                    reflection = (
                        " ALL tool calls in this round failed. "
                        "Before retrying, analyze what went wrong "
                        "and try a different approach — different parameters, "
                        "datasets, or strategy."
                    )
                    last_result["observation"] = (
                        last_result.get("observation", "") + reflection
                    )

            # Second pass: build function responses
            for tc_id, tool_name, tool_args, result in tool_results:
                if result.get("status") == "clarification_needed":
                    question = result["question"]
                    if result.get("context"):
                        question = f"{result['context']}\n\n{question}"
                    if result.get("options"):
                        question += "\n\nOptions:\n" + "\n".join(
                            f"  {i + 1}. {opt}"
                            for i, opt in enumerate(result["options"])
                        )
                    return question

                function_responses.append(
                    self.adapter.make_tool_result_message(
                        tool_name, result, tool_call_id=tc_id
                    )
                )

            guard.record_calls(len(function_calls))

            tool_names = [fc.name for fc in function_calls]
            self._last_tool_context = "+".join(tool_names)

            self._event_bus.emit(
                DEBUG,
                level="debug",
                msg=f"[LLM] Sending {len(function_responses)} tool result(s) back to model...",
            )
            response = self._send_message(function_responses)
            self._track_usage(response)
            self._emit_intermediate_text(response)

            self._event_bus.emit(DEBUG, level="debug", msg="[LLM] Response received.")
            self._log_grounding_queries(response)

        # Post-loop continuation: drain remaining async work and let
        # the LLM react (e.g., delegate to viz after DataOps fails).
        post_budget = 8
        while post_budget > 0 and not self._cancel_event.is_set():
            # Wait for any remaining async work
            if not cc.has_pending():
                break
            self._event_bus.emit(
                DEBUG,
                level="debug",
                msg="[Turnless] Post-loop drain — waiting for remaining work",
            )
            remaining = cc.wait_for_any(cancel_event=self._cancel_event)
            remaining.extend(cc.drain_completed())
            if not remaining:
                break

            # Send results to LLM
            function_responses = []
            has_failure = False
            for unit in remaining:
                result = self._build_work_unit_result(unit)
                if result.get("status") == "error":
                    has_failure = True
                if config.OBSERVATION_SUMMARIES:
                    from .observations import generate_observation

                    t_name = self._work_unit_tool_name(unit)
                    result["observation"] = generate_observation(
                        t_name, {"request": unit.task_summary}, result
                    )
                function_responses.append(
                    self.adapter.make_tool_result_message(
                        self._work_unit_tool_name(unit),
                        result,
                        tool_call_id=unit.tool_call_id,
                    )
                )

            if function_responses:
                response = self._send_message_streaming(function_responses)
                self._track_usage(response)
                self._emit_intermediate_text(response)

            # Process any tool calls the LLM issues in response
            while response.tool_calls and post_budget > 0:
                post_budget -= 1
                tool_results = self._execute_tools_parallel(response.tool_calls)
                just_completed = cc.drain_completed()
                for unit in just_completed:
                    r = self._build_work_unit_result(unit)
                    if config.OBSERVATION_SUMMARIES:
                        from .observations import generate_observation

                        t_name = self._work_unit_tool_name(unit)
                        r["observation"] = generate_observation(
                            t_name, {"request": unit.task_summary}, r
                        )
                    tool_results.append(
                        (unit.tool_call_id, t_name, {"request": unit.task_summary}, r)
                    )
                fn_responses = []
                for tc_id, tool_name, tool_args, result in tool_results:
                    fn_responses.append(
                        self.adapter.make_tool_result_message(
                            tool_name, result, tool_call_id=tc_id
                        )
                    )
                response = self._send_message(fn_responses)
                self._track_usage(response)
                self._emit_intermediate_text(response)

            # Nudge if delegation failed and LLM gave up
            if has_failure and not response.tool_calls:
                nudge = (
                    "A delegation FAILED and the user's request is NOT fully "
                    "fulfilled. Do NOT say 'Done'. Continue: delegate to "
                    "another agent, retry, or handle remaining work directly."
                )
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg="[Orchestrator] Nudging after post-drain failure",
                )
                response = self._send_message(nudge)
                self._track_usage(response)
                self._emit_intermediate_text(response)
                if response.tool_calls:
                    post_budget -= 1
                    continue

            break

        # Extract text response
        text = response.text
        if not text:
            store_entries = self._store.list_entries() if self._store else []
            has_plot = self._renderer.get_current_state().get("has_plot", False)
            parts = []
            if store_entries:
                parts.append(f"Data loaded: {len(store_entries)} dataset(s) in memory")
            if has_plot:
                parts.append("Visualization: produced")
            else:
                parts.append("Visualization: not produced")
            text = (
                "Completed. " + ". ".join(parts) + "."
                if parts
                else "Completed with no output."
            )
        text += self._extract_grounding_sources(response)
        return text

    @staticmethod
    def _work_unit_tool_name(unit: "WorkUnit") -> str:
        """Derive the original delegation tool name from a WorkUnit's agent_type."""
        _type_to_tool = {
            "mission": "delegate_to_mission",
            "data_ops": "delegate_to_data_ops",
            "data_extraction": "delegate_to_data_extraction",
            "viz_plotly": "delegate_to_viz_plotly",
            "viz_mpl": "delegate_to_viz_mpl",
            "viz_jsx": "delegate_to_viz_jsx",
            "insight": "delegate_to_insight",
            "planner": "request_planning",
        }
        return _type_to_tool.get(unit.agent_type, "delegate_to_mission")

    def _get_active_mission_ids(self) -> set[str]:
        """Return set of active mission IDs from current actors. Thread-safe."""
        with self._sub_agents_lock:
            return {
                k.removeprefix("MissionAgent[").rstrip("]")
                for k in self._sub_agents
                if k.startswith("MissionAgent[")
            }

    def _inject_memory(self, request: str, scope: str) -> str:
        """Append scoped memory section to a delegation request string.

        Review instructions are suppressed because actors now handle memory
        reviews as deferred follow-ups after delivering the main result.
        """
        # For data_ops scope, only include memories for missions active this session
        active_missions: set[str] | None = None
        if scope == "data_ops":
            active_missions = self._get_active_mission_ids()
        section = self._memory_store.format_for_injection(
            scope=scope,
            active_missions=active_missions,
            include_review_instruction=False,
        )
        return f"{request}\n\n{section}" if section else request

    def _inject_memory_incremental(
        self, request: str, scope: str, agent_id: str
    ) -> str:
        """Like _inject_memory, but skips injection when memory is unchanged for agent_id."""
        active_missions: set[str] | None = None
        if scope == "data_ops":
            with self._sub_agents_lock:
                active_missions = {
                    k.removeprefix("MissionAgent[").rstrip("]")
                    for k in self._sub_agents
                    if k.startswith("MissionAgent[")
                }
        section = self._memory_store.format_for_injection(
            scope=scope,
            active_missions=active_missions,
            include_review_instruction=False,
        )
        if not section:
            return request
        if self._ctx_tracker.is_changed(agent_id, "memory", section):
            self._ctx_tracker.record(agent_id, memory=section)
            return f"{request}\n\n{section}"
        return request  # Memory unchanged — actor already has it

    def _compact_history(self, raw_text: str, agent_type: str, budget: int) -> str:
        """Compact session history via a single LLM call.

        Uses SMART_MODEL with temperature 0.1 to summarize the history
        while preserving all errors/failures and recent events.
        LLM errors propagate naturally — if the LLM is unreachable,
        the entire agent can't function anyway.
        """
        role_descriptions = {
            "mission": "a mission data specialist",
            "viz_plotly": "a data visualization specialist",
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

    @staticmethod
    def _wrap_delegation_result(sub_result, store_snapshot=None) -> dict:
        """Convert an actor send_and_wait result into a tool result dict.

        If the actor reported failure (stopped due to errors/loops),
        return status='error' so the orchestrator knows not to retry.

        Args:
            sub_result: Dict from actor's _handle_request ({text, failed, errors}).
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
            error_summary = "; ".join(errors[-get_item_limit("items.error_summary") :])
            result = {
                "status": "error",
                "message": f"Sub-agent failed. Errors: {error_summary}",
                "result": text,
            }
        else:
            result = {"status": "success", "result": text}

        if store_snapshot is not None:
            result["data_in_memory"] = [
                {
                    "label": e["label"],
                    "columns": e.get("columns", []),
                    "shape": e.get("shape", ""),
                    "units": e.get("units", ""),
                    "num_points": e.get("num_points", 0),
                }
                for e in store_snapshot
            ]
        return result

    def reset_sub_agents(self) -> None:
        """Invalidate all cached sub-agents so they are recreated with current config."""
        self._planner_agent = None
        # Stop and clear actors
        with self._sub_agents_lock:
            for actor in self._sub_agents.values():
                actor.stop(timeout=2.0)
            self._sub_agents.clear()
        MISSION_TOOL_REGISTRY.clear_active()
        # Reset context tracker so recreated actors get full context on first use
        self._ctx_tracker.reset_all()
        self._event_bus.emit(
            DEBUG,
            level="debug",
            msg="[Config] Sub-agents invalidated after config reload",
        )

    # ---- Agent-based sub-agent management ----

    def _get_or_create_mission_agent(self, mission_id: str) -> MissionAgent:
        """Get the persistent mission actor, creating it on first use."""
        actor_id = f"MissionAgent[{mission_id}]"
        with self._sub_agents_lock:
            if actor_id not in self._sub_agents:
                actor = MissionAgent(
                    mission_id=mission_id,
                    adapter=self.adapter,
                    model_name=config.SUB_AGENT_MODEL,
                    tool_executor=self._execute_tool_for_actor,
                    event_bus=self._event_bus,
                    memory_store=self._memory_store,
                    memory_scope=f"mission:{mission_id}",
                )
                actor.start()
                self._sub_agents[actor_id] = actor
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Router] Created {mission_id} mission actor ({config.SUB_AGENT_MODEL})",
                )
                # Register this mission's tools in the informed set so the
                # orchestrator sees logs from tools the mission actor uses.
                if MISSION_TOOL_REGISTRY.mark_active(mission_id):
                    from .agent_registry import AGENT_INFORMED_REGISTRY

                    for tool_name in MISSION_TOOL_REGISTRY.get_tools(mission_id):
                        AGENT_INFORMED_REGISTRY._registry.setdefault(
                            "ctx:orchestrator", set()
                        ).add(tool_name)
            return self._sub_agents[actor_id]

    def _create_ephemeral_mission_actor(self, mission_id: str) -> MissionAgent:
        """Create an ephemeral overflow mission actor for parallel delegation."""
        with self._sub_agents_lock:
            seq = self._mission_seq
            self._mission_seq = seq + 1
            ephemeral_id = f"MissionAgent[{mission_id}]#{seq}"
            actor = MissionAgent(
                mission_id=mission_id,
                adapter=self.adapter,
                model_name=config.SUB_AGENT_MODEL,
                tool_executor=self._execute_tool_for_actor,
                agent_id=ephemeral_id,
                event_bus=self._event_bus,
                memory_store=self._memory_store,
                memory_scope=f"mission:{mission_id}",
            )
            actor.start()
            self._sub_agents[ephemeral_id] = actor
        self._event_bus.emit(
            DEBUG,
            level="debug",
            msg=f"[Router] Created ephemeral mission actor {ephemeral_id}",
        )
        return actor

    def _get_or_create_viz_plotly_actor(self) -> VizPlotlyAgent:
        """Get the cached Plotly viz actor or create a new one. Thread-safe."""
        actor_id = "VizAgent[Plotly]"
        with self._sub_agents_lock:
            if actor_id not in self._sub_agents:
                actor = VizPlotlyAgent(
                    adapter=self.adapter,
                    model_name=config.SUB_AGENT_MODEL,
                    tool_executor=self._execute_tool_for_actor,
                    gui_mode=self.gui_mode,
                    event_bus=self._event_bus,
                    memory_store=self._memory_store,
                    memory_scope="visualization",
                )
                actor.start()
                self._sub_agents[actor_id] = actor
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Router] Created Plotly Visualization actor ({config.SUB_AGENT_MODEL})",
                )
            return self._sub_agents[actor_id]

    def _get_or_create_viz_mpl_actor(self) -> VizMplAgent:
        """Get the cached MPL viz actor or create a new one. Thread-safe."""
        actor_id = "VizAgent[Mpl]"
        with self._sub_agents_lock:
            if actor_id not in self._sub_agents:
                session_dir = self._session_manager.base_dir / self._session_id
                actor = VizMplAgent(
                    adapter=self.adapter,
                    model_name=config.SUB_AGENT_MODEL,
                    tool_executor=self._execute_tool_for_actor,
                    gui_mode=self.gui_mode,
                    event_bus=self._event_bus,
                    memory_store=self._memory_store,
                    memory_scope="visualization",
                    session_dir=session_dir,
                )
                actor.start()
                self._sub_agents[actor_id] = actor
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Router] Created MPL Visualization actor ({config.SUB_AGENT_MODEL})",
                )
            return self._sub_agents[actor_id]

    def _get_or_create_viz_jsx_actor(self) -> VizJsxAgent:
        """Get the cached JSX viz actor or create a new one. Thread-safe."""
        actor_id = "VizAgent[JSX]"
        with self._sub_agents_lock:
            if actor_id not in self._sub_agents:
                session_dir = self._session_manager.base_dir / self._session_id
                actor = VizJsxAgent(
                    adapter=self.adapter,
                    model_name=config.SUB_AGENT_MODEL,
                    tool_executor=self._execute_tool_for_actor,
                    gui_mode=self.gui_mode,
                    event_bus=self._event_bus,
                    memory_store=self._memory_store,
                    memory_scope="visualization",
                    session_dir=session_dir,
                )
                actor.start()
                self._sub_agents[actor_id] = actor
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Router] Created JSX Visualization actor ({config.SUB_AGENT_MODEL})",
                )
            return self._sub_agents[actor_id]

    def _get_available_dataops_actor(self) -> DataOpsAgent:
        """Get an idle DataOps actor or create a new ephemeral one.

        Priority: (1) idle primary actor, (2) create primary if it doesn't
        exist, (3) create an ephemeral overflow instance.  Ephemeral actors
        are cleaned up after their delegation completes.
        """
        primary_id = "DataOpsAgent"
        with self._sub_agents_lock:
            if primary_id in self._sub_agents:
                actor = self._sub_agents[primary_id]
                if actor.state == AgentState.SLEEPING and actor.inbox.qsize() == 0:
                    return actor
            else:
                # Create the primary (persistent) actor
                actor = DataOpsAgent(
                    adapter=self.adapter,
                    model_name=config.SUB_AGENT_MODEL,
                    tool_executor=self._execute_tool_for_actor,
                    event_bus=self._event_bus,
                    memory_store=self._memory_store,
                    memory_scope="data_ops",
                    active_missions_fn=self._get_active_mission_ids,
                )
                actor.start()
                self._sub_agents[primary_id] = actor
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg="[Router] Created DataOps actor",
                )
                return actor

            # Primary is busy — create an ephemeral overflow instance
            seq = self._dataops_seq
            self._dataops_seq = seq + 1
            ephemeral_id = f"DataOpsAgent#{seq}"
            actor = DataOpsAgent(
                adapter=self.adapter,
                model_name=config.SUB_AGENT_MODEL,
                tool_executor=self._execute_tool_for_actor,
                agent_id=ephemeral_id,
                event_bus=self._event_bus,
                memory_store=self._memory_store,
                memory_scope="data_ops",
                active_missions_fn=self._get_active_mission_ids,
            )
            actor.start()
            self._sub_agents[ephemeral_id] = actor
            self._event_bus.emit(
                DEBUG,
                level="debug",
                msg=f"[Router] Created ephemeral DataOps actor {ephemeral_id}",
            )
            return actor

    def _cleanup_ephemeral_actor(self, actor_id: str) -> None:
        """Shut down and remove an ephemeral actor from the registry.

        Preserves the actor's token usage in ``_retired_agent_usage`` so it
        is still included in ``get_token_usage()`` and
        ``get_token_usage_breakdown()`` after the actor is removed.
        """
        with self._sub_agents_lock:
            actor = self._sub_agents.pop(actor_id, None)
        if actor:
            # Preserve token usage before stopping the actor
            usage = actor.get_token_usage()
            if usage.get("api_calls", 0) > 0:
                self._retired_agent_usage.append(
                    {
                        "agent": actor_id,
                        "input_tokens": usage["input_tokens"],
                        "output_tokens": usage["output_tokens"],
                        "thinking_tokens": usage.get("thinking_tokens", 0),
                        "cached_tokens": usage.get("cached_tokens", 0),
                        "api_calls": usage["api_calls"],
                        "ctx_system_tokens": usage.get("ctx_system_tokens", 0),
                        "ctx_tools_tokens": usage.get("ctx_tools_tokens", 0),
                        "ctx_history_tokens": usage.get("ctx_history_tokens", 0),
                        "ctx_total_tokens": usage.get("ctx_total_tokens", 0),
                    }
                )
            actor.stop()
            self._ctx_tracker.reset(actor_id)
            self._event_bus.emit(
                DEBUG,
                level="debug",
                msg=f"[Router] Cleaned up ephemeral actor {actor_id}",
            )

    def _get_or_create_extraction_actor(self) -> DataExtractionAgent:
        """Get the cached extraction actor or create a new one. Thread-safe."""
        actor_id = "DataExtractionAgent"
        with self._sub_agents_lock:
            if actor_id not in self._sub_agents:
                actor = DataExtractionAgent(
                    adapter=self.adapter,
                    model_name=config.SUB_AGENT_MODEL,
                    tool_executor=self._execute_tool_for_actor,
                    event_bus=self._event_bus,
                )
                actor.start()
                self._sub_agents[actor_id] = actor
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg="[Router] Created DataExtraction actor",
                )
            return self._sub_agents[actor_id]

    def _get_or_create_insight_actor(self) -> InsightAgent:
        """Get the cached insight actor or create a new one. Thread-safe."""
        actor_id = "InsightAgent"
        with self._sub_agents_lock:
            if actor_id not in self._sub_agents:
                actor = InsightAgent(
                    adapter=self.adapter,
                    model_name=config.INSIGHT_MODEL,
                    tool_executor=self._execute_tool_for_actor,
                    event_bus=self._event_bus,
                )
                actor.start()
                self._sub_agents[actor_id] = actor
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Router] Created Insight actor ({config.INSIGHT_MODEL})",
                )
            return self._sub_agents[actor_id]

    def _delegate_to_sub_agent(
        self,
        actor: SubAgent,
        request,
        timeout: float = 300.0,
        store_snapshot=None,
        tool_call_id: str | None = None,
        agent_type: str = "",
        agent_name: str = "",
        task_summary: str = "",
        post_process: Callable | None = None,
        post_complete: Callable | None = None,
    ) -> dict:
        """Dispatch delegation as an async work unit via ControlCenter.

        The delegation runs on a background thread. The orchestrator's main
        loop picks up completed work units via ``wait_for_any()`` /
        ``drain_completed()``.

        Args:
            actor: The target Agent instance.
            request: String or dict payload for the actor.
            timeout: Max seconds to wait (inside background thread).
            store_snapshot: Optional list of store entries to include in result.
            tool_call_id: LLM tool_call_id for result mapping.
            agent_type: Agent type for ControlCenter (e.g. "mission", "viz_plotly").
            agent_name: Agent name for ControlCenter (e.g. "MissionAgent[ACE]").
            task_summary: Human-readable summary for ControlCenter.
            post_process: Optional callable(result) -> result to run after
                delegation completes but before marking work unit complete.
                Runs on the background thread.
            post_complete: Optional callable(result) to run AFTER
                ``mark_completed()`` fires. Use for non-critical work
                (e.g. PNG export) that should not block the orchestrator.
                Runs on the background thread (fire-and-forget).
        """
        cc = self._control_center
        summary = task_summary or (
            request[:200] if isinstance(request, str) else str(request)[:200]
        )
        unit = cc.register(
            kind="delegation",
            agent_type=agent_type,
            agent_name=agent_name or actor.agent_id,
            task_summary=summary,
            tool_call_id=tool_call_id,
        )

        # Capture operation log index before the delegation starts so we can
        # collect only the operations produced by this delegation thread.
        ops_log = self._ops_log
        ops_start_index = len(ops_log.get_records())

        def _run():
            try:
                # Check cancel before starting
                if unit.cancel_event.is_set():
                    return

                result = actor.send_and_wait(
                    request, sender="orchestrator", timeout=timeout
                )

                # Check cancel after completion — don't mark completed if cancelled
                if unit.cancel_event.is_set():
                    return

                wrapped = self._wrap_delegation_result(
                    result, store_snapshot=store_snapshot
                )
                if post_process is not None:
                    wrapped = post_process(wrapped)

                # Build operation log from records added during this delegation
                all_records = ops_log.get_records()
                operation_log = all_records[ops_start_index:]

                cc.mark_completed(unit.id, wrapped, operation_log=operation_log)

                # Fire-and-forget callback after marking complete — runs off
                # the critical path so it doesn't block the orchestrator.
                if post_complete is not None:
                    try:
                        post_complete(wrapped)
                    except Exception as pc_err:
                        get_logger().debug(f"post_complete callback failed: {pc_err}")
            except Exception as e:
                if not unit.cancel_event.is_set():
                    cc.mark_failed(unit.id, str(e))

        thread = threading.Thread(
            target=_run, daemon=True, name=f"delegation-{unit.id}"
        )
        unit.thread = thread
        thread.start()

        return {
            "status": "pending_async",
            "work_unit_id": unit.id,
            "agent_name": unit.agent_name,
        }

    def list_active_actors(self) -> list[dict]:
        """Return status of all actor-based sub-agents."""
        with self._sub_agents_lock:
            return [actor.status() for actor in self._sub_agents.values()]

    def hot_reload_config(self) -> dict:
        """Hot-reload agent to match current config module-level constants.

        Compares the agent's own cached state (adapter type, model name)
        against the current ``config.*`` values. If nothing diverges, this
        is a no-op. Otherwise it preserves chat history while rebuilding
        the adapter (if provider changed) and clearing all cached sub-agents
        so they pick up new model names on next use.

        Returns:
            dict with keys: status, provider_changed, old_model, new_model
        """
        # Determine what the agent currently has vs what config says
        _ADAPTER_PROVIDER = {
            GeminiAdapter: "gemini",
            OpenAIAdapter: "openai",
            AnthropicAdapter: "anthropic",
        }
        current_provider = _ADAPTER_PROVIDER.get(type(self.adapter), "unknown")
        current_base_url = getattr(self.adapter, "base_url", None)
        current_model = self.model_name

        target_provider = config.LLM_PROVIDER.lower()
        target_base_url = config.LLM_BASE_URL
        target_model = config.SMART_MODEL

        target_viz_backend = config.PREFER_VIZ_BACKEND

        provider_changed = current_provider != target_provider
        base_url_changed = current_base_url != target_base_url
        model_changed = current_model != target_model
        viz_backend_changed = self._viz_backend != target_viz_backend
        adapter_needs_rebuild = provider_changed or base_url_changed

        # Check if any sub-agent would use a stale model (compare what config
        # says now vs what the cached agents were built with)
        sub_agents_stale = self._planner_agent is not None
        if not sub_agents_stale:
            with self._sub_agents_lock:
                sub_agents_stale = len(self._sub_agents) > 0

        if not (
            adapter_needs_rebuild
            or model_changed
            or sub_agents_stale
            or viz_backend_changed
        ):
            self._event_bus.emit(
                DEBUG,
                level="debug",
                msg="[Config] Hot-reload: no changes detected, skipping",
            )
            return {
                "status": "unchanged",
                "provider_changed": False,
                "old_model": current_model,
                "new_model": target_model,
            }

        # 1. Viz Backend Hot-Reload (Filtering Tools + System Prompt Rebuild)
        if viz_backend_changed:
            self._event_bus.emit(
                DEBUG,
                level="info",
                msg=f"[Config] Hot-reload: Switching viz backend to {target_viz_backend}",
            )
            # Rebuild tool schemas from the full set, filtering out inactive viz tools
            self._all_tool_schemas = get_function_schemas(names=ORCHESTRATOR_TOOLS)
            inactive = _inactive_viz_tools()
            self._all_tool_schemas = [
                s for s in self._all_tool_schemas if s.name not in inactive
            ]

            # Rebuild _tool_schemas (respecting tool store mode)
            if self._use_tool_store:
                # Re-resolve and re-filter
                from .tool_catalog import DEFAULT_TOOL_CATEGORIES, DEFAULT_EXTRA_TOOLS

                default_names = set(
                    resolve_tools(
                        DEFAULT_TOOL_CATEGORIES + DEFAULT_EXTRA_TOOLS,
                        agent_context="ctx:orchestrator",
                    )
                )
                self._loaded_tool_names = default_names
                default_schemas = [
                    s
                    for s in self._all_tool_schemas
                    if s.name in default_names and s.name not in META_TOOL_NAMES
                ]
                self._tool_schemas = list(self._meta_tool_schemas) + default_schemas
            else:
                self._tool_schemas = list(self._all_tool_schemas)

            # Regenerate system prompt (it might have backend-specific instructions)
            self._system_prompt = get_system_prompt(gui_mode=self.gui_mode)

            # Update the current chat session's tools and prompt if it exists
            if self.chat is not None:
                self.chat.update_tools(self._tool_schemas)
                self.chat.update_system_prompt(self._system_prompt)

            self._viz_backend = target_viz_backend

        # 2. Extract chat history (only useful if adapter stays the same)
        history = None
        if not adapter_needs_rebuild and self.chat is not None:
            try:
                history = self.chat.get_history()
            except Exception:
                pass

        # 2. Rebuild adapter if provider/base_url changed
        if adapter_needs_rebuild:
            self.adapter = _create_adapter()
            self._event_bus.emit(
                DEBUG,
                level="info",
                msg=f"[Config] Adapter rebuilt: {current_provider} → {target_provider}",
            )

        # 3. Update model name
        self.model_name = target_model

        # 4. Recreate chat session with preserved history
        try:
            self.chat = self.adapter.create_chat(
                model=get_active_model(self.model_name),
                system_prompt=self._system_prompt,
                tools=self._tool_schemas,
                thinking="high",
                history=history,
            )
        except Exception as exc:
            # Fall back to fresh chat if history transfer fails
            self._event_bus.emit(
                DEBUG,
                level="warning",
                msg=f"[Config] Chat recreation with history failed ({exc}), starting fresh",
            )
            self.chat = self.adapter.create_chat(
                model=get_active_model(self.model_name),
                system_prompt=self._system_prompt,
                tools=self._tool_schemas,
                thinking="high",
            )

        # 5. Clear all cached sub-agents and memory agent
        self.reset_sub_agents()
        self._memory_agent = None

        # 8. Reset fallback mode (user explicitly chose new models)
        reset_fallback()

        status_msg = (
            f"[Config] Hot-reloaded: model {current_model} → {target_model}"
            + (
                f", provider {current_provider} → {target_provider}"
                if provider_changed
                else ""
            )
        )
        self._event_bus.emit(DEBUG, level="info", msg=status_msg)

        return {
            "status": "reloaded",
            "provider_changed": provider_changed,
            "old_model": current_model,
            "new_model": target_model,
        }

    def _build_insight_context(self) -> str:
        """Build data context string for the InsightAgent.

        Gathers renderer state (trace labels, panel count) and store entries
        (label, num_points, time range, units, columns) for the context.
        """
        lines = []

        # Renderer state
        state = self._renderer.get_current_state()
        if state.get("traces"):
            lines.append(f"Traces on plot: {state['traces']}")
        if state.get("num_panels"):
            lines.append(f"Number of panels: {state['num_panels']}")

        # Store entries
        entries = self._store.list_entries()
        if entries:
            lines.append("\nData in memory:")
            for e in entries:
                parts = [f"  - {e['label']}"]
                if e.get("num_points"):
                    parts.append(f"{e['num_points']} pts")
                if e.get("units"):
                    parts.append(f"units={e['units']}")
                if e.get("time_min") and e.get("time_max"):
                    parts.append(f"range={e['time_min']} to {e['time_max']}")
                if e.get("columns"):
                    cols = e["columns"]
                    if len(cols) <= 5:
                        parts.append(f"columns={cols}")
                    else:
                        parts.append(
                            f"columns=[{cols[0]}, ..., {cols[-1]}] ({len(cols)} cols)"
                        )
                lines.append(", ".join(parts))

        return "\n".join(lines) if lines else "No data context available."

    def _sync_insight_review(self) -> dict | None:
        """Synchronous InsightAgent figure review after a successful render.

        Exports the current figure to PNG, gathers context, and dispatches
        a synchronous review via actor.send_and_wait(). Blocks until the
        review completes, then returns the result.

        Returns:
            The review result dict, or None if the review was skipped
            (disabled, iteration cap, no figure, etc.).
        """
        if not config.INSIGHT_FEEDBACK:
            return None

        if self._insight_review_iter >= config.INSIGHT_FEEDBACK_MAX_ITERS:
            return None

        figure = self._renderer.get_figure()
        if figure is None:
            return None

        import io

        try:
            buf = io.BytesIO()
            figure.write_image(buf, format="png", width=1100, height=600, scale=2)
            image_bytes = buf.getvalue()
        except Exception as e:
            get_logger().warning(f"Insight feedback skipped: PNG export failed: {e}")
            return None

        actor = self._get_or_create_insight_actor()
        data_context = self._build_insight_context()

        user_msgs = self._event_bus.get_events(types={USER_MESSAGE})
        user_request = (
            user_msgs[-1].data.get("text", user_msgs[-1].msg) if user_msgs else ""
        )

        self._insight_review_iter += 1

        result = actor.send_and_wait(
            {
                "action": "review",
                "image_bytes": image_bytes,
                "data_context": data_context,
                "user_request": user_request,
            },
            sender="orchestrator",
            timeout=180,
        )

        passed = result.get("passed", True)
        review_text = result.get("text", "")
        verdict = "PASS" if passed else "NEEDS_IMPROVEMENT"
        self._event_bus.emit(
            INSIGHT_FEEDBACK,
            agent="InsightAgent",
            msg=f"[Figure Auto-Review — {verdict}]\n{review_text}",
            data={"verdict": verdict, "text": review_text},
        )
        return result

    # ---- Long-term memory (end-of-session) ----

    def _build_memory_context(self) -> MemoryContext:
        """Build a MemoryContext from the current session state.

        Shared by _maybe_extract_memories() (periodic) and
        _run_memory_agent_for_pipelines() (on-demand).
        """
        from .memory import MEMORY_TOKEN_BUDGET

        # Detect active scopes from actors
        active_scopes = ["generic"]
        with self._sub_agents_lock:
            if (
                "VizAgent[Plotly]" in self._sub_agents
                or "VizAgent[Mpl]" in self._sub_agents
            ):
                active_scopes.append("visualization")
            if "DataOpsAgent" in self._sub_agents:
                active_scopes.append("data_ops")
            for key in self._sub_agents:
                if key.startswith("MissionAgent["):
                    mission_id = key.removeprefix("MissionAgent[").rstrip("]")
                    active_scopes.append(f"mission:{mission_id}")

        # Convert EventBus events to dicts and curate.
        # All events are passed — build_curated_events handles prioritization
        # via registry + catch-all, and the token budget caps total size.
        all_events = self._event_bus.get_events()
        raw_events = [
            {"event": ev.type, "agent": ev.agent, "msg": ev.msg, **(ev.data or {})}
            for ev in all_events
        ]
        curated = MemoryAgent.build_curated_events(raw_events)

        # Load active memories for active scopes only.
        # Skip review-type entries but attach their feedback to the target.
        injected_ids = self._memory_store._last_injected_ids
        active_memories = []
        for m in self._memory_store.get_enabled():
            if m.type == "review":
                continue
            if not any(s in m.scopes for s in active_scopes):
                continue
            entry = {
                "id": m.id,
                "type": m.type,
                "scopes": m.scopes,
                "content": m.content,
                "injected": m.id in injected_ids,
                "version": m.version,
                "access_count": m.access_count,
                "created_at": m.created_at,
            }
            # Attach review feedback (recent 10 across version lineage to prevent bias from outdated reviews)
            reviews = self._memory_store.get_recent_reviews_for_lineage(m.id, n=10)
            if reviews:
                entry["reviews"] = []
                for r in reviews:
                    agent_tag = next(
                        (
                            t
                            for t in r.tags
                            if t
                            and not t.startswith("review:")
                            and not t.startswith("stars:")
                        ),
                        "",
                    )
                    entry["reviews"].append(
                        {
                            "agent": agent_tag,
                            "feedback": r.content,
                            "date": r.created_at,
                        }
                    )
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
                    history.append(
                        {
                            "version": prev.version,
                            "content": prev.content,
                            "date": prev.created_at,
                        }
                    )
                    prev_id = prev.supersedes
                if history:
                    entry["previous_versions"] = history
            active_memories.append(entry)

        return MemoryContext(
            events=curated,
            active_memories=active_memories,
            active_scopes=active_scopes,
            token_budget=MEMORY_TOKEN_BUDGET,
            total_memory_tokens=self._memory_store.total_tokens(),
        )

    def _enumerate_pipeline_candidates(self) -> list[dict]:
        """Identify ALL fresh (unprocessed) pipelines across all sessions.

        Scans the current in-memory log and all past sessions'
        ``operations.json`` files for render ops with ``pipeline_status``
        == ``"fresh"`` (or absent).  The MemoryAgent sees every fresh
        pipeline and decides each one — register or discard.
        """
        candidates = []

        # ── Current session ──
        ops_log = self._ops_log
        candidates.extend(self._candidates_from_log(ops_log))

        # ── Past sessions ──
        sessions_dir = config.get_data_dir() / "sessions"
        current_sid = self._session_id or ""
        if sessions_dir.exists():
            for sdir in sorted(
                sessions_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
            ):
                if not sdir.is_dir():
                    continue
                if sdir.name == current_sid:
                    continue  # already scanned above
                ops_file = sdir / "operations.json"
                if not ops_file.exists():
                    continue
                try:
                    past_log = OperationsLog(session_id=sdir.name)
                    past_log.load_from_file(ops_file)
                    past_candidates = self._candidates_from_log(past_log)
                    if past_candidates:
                        candidates.extend(past_candidates)
                except Exception:
                    continue  # skip corrupt files

        return candidates

    @staticmethod
    def _candidates_from_log(ops_log) -> list[dict]:
        """Extract fresh pipeline candidates from a single OperationsLog.

        Returns rich per-step detail for each candidate so the LLM can make
        informed registration decisions.  Only render ops with
        ``pipeline_status`` == ``"fresh"`` (or absent) are included.
        """
        from data_ops.pipeline import is_vanilla
        from knowledge.mission_prefixes import (
            match_dataset_to_mission,
            get_canonical_id,
        )

        records = ops_log.get_records()

        render_ops = [
            r
            for r in records
            if r["tool"] == "render_plotly_json"
            and r["status"] == "success"
            and r.get("pipeline_status", "fresh") == "fresh"
        ]

        if not render_ops:
            return []

        candidates = []
        all_labels = {
            l for r in records if r["status"] == "success" for l in r.get("outputs", [])
        }

        for render in render_ops:
            render_id = render["id"]
            sub_dag = ops_log.get_state_pipeline(render_id, all_labels)

            # Build step-like dicts for is_vanilla check
            step_dicts = [
                {"tool": op["tool"], "params": op.get("args", {})} for op in sub_dag
            ]

            # Auto-extract scopes from dataset IDs
            missions_set: set[str] = set()
            steps = []
            for op in sub_dag:
                tool = op["tool"]
                args = op.get("args", {})
                output_label = op.get("outputs", [""])[0] if op.get("outputs") else ""
                step: dict = {"tool": tool}

                if tool == "fetch_data":
                    ds = args.get("dataset_id", "")
                    param = args.get("parameter_id", "")
                    step["dataset_id"] = ds
                    step["parameter_id"] = param
                    if output_label:
                        step["output_label"] = output_label
                    if ds:
                        stem, _ = match_dataset_to_mission(ds)
                        if stem:
                            missions_set.add(get_canonical_id(stem))

                elif tool in ("custom_operation", "store_dataframe"):
                    if args.get("code"):
                        step["code"] = args["code"]
                    if args.get("description"):
                        step["description"] = args["description"]
                    if args.get("units"):
                        step["units"] = args["units"]
                    if output_label:
                        step["output_label"] = output_label

                elif tool == "render_plotly_json":
                    step["inputs"] = list(op.get("inputs", []))
                    # figure_json intentionally omitted — too large

                elif tool == "manage_plot":
                    step["action"] = args.get("action", "")
                    for k in ("plot_id", "title", "subplot"):
                        if args.get(k):
                            step[k] = args[k]

                else:
                    if output_label:
                        step["output_label"] = output_label

                steps.append(step)

            scopes = sorted(f"mission:{m}" for m in missions_set)

            candidates.append(
                {
                    "render_op_id": render_id,
                    "step_count": len(sub_dag),
                    "is_vanilla": is_vanilla(step_dicts),
                    "scopes": scopes,
                    "steps": steps,
                }
            )

        return candidates

    def _ensure_memory_agent(self, session_id: str = "", bus=None) -> MemoryAgent:
        """Lazily create or return the existing MemoryAgent."""
        if bus is None:
            bus = self._event_bus
        if session_id == "":
            session_id = self._session_id or ""
        if self._memory_agent is None:
            self._memory_agent = MemoryAgent(
                adapter=self.adapter,
                model_name=config.SMART_MODEL,
                memory_store=self._memory_store,
                pipeline_store=self._pipeline_store,
                verbose=self.verbose,
                session_id=session_id,
                event_bus=bus,
            )
        return self._memory_agent

    def _ensure_eureka_agent(self) -> "EurekaAgent":
        """Lazily create or return the existing EurekaAgent."""
        if self._eureka_agent is None:
            from .eureka_agent import EurekaAgent

            self._eureka_agent = EurekaAgent(
                adapter=self.adapter,
                model_name=config.INLINE_MODEL,
                event_bus=self._event_bus,
                orchestrator_ref=self,
            )
        return self._eureka_agent

    def _build_eureka_context(self) -> dict:
        """Build context dict for Eureka discovery."""
        user_msgs = self._event_bus.get_events(types={USER_MESSAGE})
        return {
            "session_id": self._session_id or "unknown",
            "data_store_keys": [e["label"] for e in self._store.list_entries()] if self._store else [],
            "has_figure": self._renderer.get_figure() is not None,
            "recent_messages": [m.msg for m in user_msgs[-5:]],
        }

    def _maybe_extract_eurekas(self) -> None:
        """Trigger async Eureka extraction on a daemon thread.

        Lock prevents concurrent extractions.
        """
        if not self._eureka_lock.acquire(blocking=False):
            return

        try:
            agent = self._ensure_eureka_agent()
            context = self._build_eureka_context()
        except Exception as e:
            self.logger.warning(f"Eureka setup failed: {e}")
            self._eureka_lock.release()
            return

        from .event_bus import (
            EUREKA_EXTRACTION_START,
            EUREKA_EXTRACTION_DONE,
            EUREKA_EXTRACTION_ERROR,
            set_event_bus,
        )

        bus = self._event_bus

        def _run():
            set_event_bus(bus)
            try:
                bus.emit(EUREKA_EXTRACTION_START, agent="Eureka", level="info")
                eurekas = agent.run(context)
                bus.emit(EUREKA_EXTRACTION_DONE, agent="Eureka", level="info", data={"count": len(eurekas)})
            except Exception as e:
                self.logger.warning(f"Eureka extraction failed: {e}")
                bus.emit(EUREKA_EXTRACTION_ERROR, agent="Eureka", level="warning", data={"error": str(e)})
            finally:
                self._eureka_lock.release()

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _run_memory_agent_for_pipelines(self) -> list[dict]:
        """Force a Memory Agent run focused on pipeline curation.

        Builds context with pipeline candidates from the current session,
        runs the Memory Agent synchronously, and returns pipeline actions.
        """
        context = self._build_memory_context()
        context.pipeline_candidates = self._enumerate_pipeline_candidates()

        if not context.pipeline_candidates:
            return []  # Nothing to curate

        agent = self._ensure_memory_agent()

        self._event_bus.emit(
            MEMORY_EXTRACTION_START,
            agent="Memory",
            level="info",
            msg="[Memory] Pipeline curation started",
            data={"pipeline_candidates": len(context.pipeline_candidates)},
        )

        try:
            executed = agent.run(context)
            # Persist ops log so pipeline_status changes are saved to disk
            self._persist_operations_log()
            pipeline_actions = [
                a
                for a in (executed or [])
                if a.get("action") in ("register_pipeline", "discard_pipeline")
            ]
            return pipeline_actions
        except Exception as e:
            self._event_bus.emit(
                MEMORY_EXTRACTION_ERROR,
                agent="Memory",
                level="warning",
                msg=f"[Memory] Pipeline curation failed: {e}",
            )
            return []

    def _persist_operations_log(self) -> None:
        """Save the current operations log to the session directory on disk."""
        if not self._session_id:
            return
        try:
            session_dir = config.get_data_dir() / "sessions" / self._session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            self._ops_log.save_to_file(session_dir / "operations.json")
        except Exception:
            pass  # Best-effort persistence

    def _maybe_extract_memories(self) -> None:
        """Trigger async memory extraction with full session context.

        Runs on a daemon thread using SMART_MODEL.
        Lock prevents concurrent extractions. The MemoryAgent sees
        memory-tagged events from the EventBus, curated into concise summaries.
        Also includes pipeline candidates for the LLM to curate.
        """
        # Check if there are new memory-relevant events since last extraction
        from .memory_agent import MEMORY_RELEVANT_TYPES

        memory_events = self._event_bus.get_events(
            types=MEMORY_RELEVANT_TYPES, since_index=self._last_memory_op_index
        )
        if not memory_events:
            return  # No new events since last extraction

        if not self._memory_lock.acquire(blocking=False):
            return  # Another extraction already running

        try:
            context = self._build_memory_context()
            context.pipeline_candidates = self._enumerate_pipeline_candidates()

            self._last_memory_op_index = len(self._event_bus._events)

            session_id = self._session_id or ""
            bus = self._event_bus  # capture before thread (ContextVar won't propagate)

            def _run():
                set_event_bus(bus)  # propagate session bus to daemon thread
                try:
                    bus.emit(
                        MEMORY_EXTRACTION_START,
                        agent="Memory",
                        level="info",
                        msg="[Memory] Extraction started",
                        data={
                            "curated_events": len(context.events),
                            "active_scopes": context.active_scopes,
                        },
                    )

                    # Dump memory feed for debugging
                    if session_id:
                        try:
                            from .memory_agent import CURATED_EVENTS_TOKEN_BUDGET
                            from datetime import datetime as _dt, timezone as _tz

                            feed_dir = config.get_data_dir() / "sessions" / session_id
                            feed_dir.mkdir(parents=True, exist_ok=True)
                            feed_payload = {
                                "timestamp": _dt.now(_tz.utc).isoformat(),
                                "active_scopes": context.active_scopes,
                                "token_budget": CURATED_EVENTS_TOKEN_BUDGET,
                                "curated_events_count": len(context.events),
                                "curated_events": context.events,
                                "active_memories_count": len(context.active_memories),
                                "pipeline_candidates_count": len(
                                    context.pipeline_candidates
                                ),
                            }
                            (feed_dir / "memory_feed.json").write_text(
                                json.dumps(feed_payload, indent=2, default=str)
                            )
                        except Exception:
                            pass  # Debug dump — never break extraction

                    agent = self._ensure_memory_agent(session_id=session_id, bus=bus)
                    executed = agent.run(context)

                    # Persist ops log so pipeline_status changes are saved to disk
                    self._persist_operations_log()

                    # Tally actions by type
                    counts = {}
                    for action in executed or []:
                        atype = action.get("action", "unknown")
                        counts[atype] = counts.get(atype, 0) + 1

                    bus.emit(
                        MEMORY_EXTRACTION_DONE,
                        agent="Memory",
                        level="info",
                        msg=f"[Memory] Extraction complete: {counts}"
                        if counts
                        else "[Memory] Extraction complete: no changes",
                        data={"actions": counts},
                    )
                except Exception as e:
                    bus.emit(
                        MEMORY_EXTRACTION_ERROR,
                        agent="Memory",
                        level="warning",
                        msg=f"[Memory] Extraction failed: {e}",
                    )
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
        turns = _extract_turns(
            history[-get_item_limit("items.follow_up_turns") :]
        )  # default 300 from registry

        if not turns:
            return []

        conversation_text = "\n".join(turns)

        # DataStore context
        store = self._store
        labels = [e["label"] for e in store.list_entries()]
        data_context = (
            f"Data in memory: {', '.join(labels)}"
            if labels
            else "No data in memory yet."
        )

        has_plot = self._renderer.get_figure() is not None
        plot_context = (
            "A plot is currently displayed." if has_plot else "No plot is displayed."
        )

        prompt = f"""Based on this conversation, suggest {max_suggestions} short follow-up questions the user might ask next.

{conversation_text}

{data_context}
{plot_context}

Respond with a JSON array of strings only (no markdown fencing). Each suggestion should be:
- A natural, conversational question (max 12 words)
- Actionable — something the agent can actually do
- Different from what was already asked
- Related to the current context (data, plots, missions)

Example: ["Compare this with solar wind speed", "Zoom in to January 10-15", "Export the plot as PDF"]"""

        try:
            response = self.adapter.generate(
                model=get_active_model(config.INLINE_MODEL),
                contents=prompt,
                temperature=0.7,
            )
            self._last_tool_context = "follow_up_suggestions"
            self._track_inline_usage(response)

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
            self._event_bus.emit(
                DEBUG, level="debug", msg=f"[FollowUp] Generation failed: {e}"
            )

        return []

    def generate_session_title(self) -> Optional[str]:
        """Generate a short title from the first exchange via INLINE_MODEL."""
        # Try chat history first (works for Chat API / Anthropic)
        try:
            history = self.chat.get_history()
        except Exception:
            history = []
        turns = _extract_turns(history[:4], max_text=500)

        # Fallback: EventBus (works for Interactions API / OpenAI Responses API)
        if not turns:
            events = self._event_bus.get_events(types={USER_MESSAGE, AGENT_RESPONSE})
            for ev in events[:4]:
                text = (ev.data or {}).get("text", ev.msg)
                if text:
                    label = "User" if ev.type == USER_MESSAGE else "Agent"
                    turns.append(f"{label}: {text[:500]}")

        if not turns:
            return None

        conversation_text = "\n".join(turns)
        prompt = (
            "Generate a concise title (3-7 words) for this conversation. "
            "Summarize the user's main intent. Use plain English.\n\n"
            f"{conversation_text}\n\n"
            "Respond with ONLY the title text, no quotes, no punctuation at the end."
        )
        try:
            response = self.adapter.generate(
                model=get_active_model(config.INLINE_MODEL),
                contents=prompt,
                temperature=0.3,
            )
            self._last_tool_context = "session_title"
            self._track_inline_usage(response)
            text = (response.text or "").strip().strip("\"'")
            if text and len(text) <= 100:
                return text
        except Exception as e:
            self._event_bus.emit(
                DEBUG, level="debug", msg=f"[SessionTitle] Generation failed: {e}"
            )
        return None

    def generate_inline_completions(
        self, partial: str, max_completions: int = 3
    ) -> list[str]:
        """Complete the user's partial input using the LLM.

        Returns full sentences that start with or continue from *partial*.
        Uses the cheapest model for low latency.

        Circuit breaker: after 5 consecutive parse failures, disables inline
        completions for 60 seconds to avoid burning API calls on models that
        consistently return malformed JSON.
        """
        # Circuit breaker: skip if disabled due to repeated failures
        if time.time() < self._inline_disabled_until:
            return []

        from knowledge.prompt_builder import build_inline_completion_prompt

        try:
            history = self.chat.get_history()
        except Exception:
            history = []

        # Last 4 turns for context
        turns = _extract_turns(
            (history or [])[-get_item_limit("items.inline_turns") :],
            max_text=get_limit("context.turn_text.inline"),
        )

        store = self._store
        labels = [e["label"] for e in store.list_entries()]

        prompt = build_inline_completion_prompt(
            partial,
            conversation_context="\n".join(turns),
            memory_section=self._memory_store.format_for_injection(
                scope="generic",
                include_summaries=True,
                include_review_instruction=False,
            ),
            data_labels=labels,
            max_completions=max_completions,
        )

        try:
            response = self.adapter.generate(
                model=get_active_model(config.INLINE_MODEL),
                contents=prompt,
                temperature=0.5,
                max_output_tokens=get_limit("output.inline_tokens"),
            )
            self._last_tool_context = "inline_completion"
            self._track_inline_usage(response)

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
                    if (
                        isinstance(c, str)
                        and c.startswith(partial)
                        and len(c) > len(partial)
                        and len(c) <= 120
                    ):
                        valid.append(c)
                if valid:
                    self._inline_fail_count = 0
                    return valid[:max_completions]

            # Parsed OK but no valid completions — not a parse failure
            return []
        except Exception as e:
            self._inline_fail_count += 1
            if self._inline_fail_count >= 5:
                self._inline_disabled_until = time.time() + 60
                self._event_bus.emit(
                    DEBUG,
                    level="warning",
                    msg=f"[InlineComplete] {self._inline_fail_count} consecutive failures, "
                    f"disabling for 60s",
                )
                self._inline_fail_count = 0
            else:
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[InlineComplete] Generation failed ({self._inline_fail_count}/5): {e}",
                )

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

        # Create disk-backed DataStore for this session
        session_dir = self._session_manager.base_dir / self._session_id
        self._store = DataStore(session_dir / "data")
        set_store(self._store)

        # Create a fresh OperationsLog scoped to this session
        self._ops_log = OperationsLog(session_id=self._session_id)
        set_operations_log(self._ops_log)

        # Start writing structured event log to disk
        self._start_event_log_writer()

        self._event_bus.emit(
            DEBUG, level="debug", msg=f"[Session] Started: {self._session_id}"
        )
        return self._session_id

    def save_session(self) -> None:
        """Persist the current chat history and DataStore to disk."""
        if not self._session_id:
            return
        try:
            history_dicts = self.chat.get_history()
        except Exception:
            history_dicts = []

        store = self._store
        usage = self.get_token_usage()

        # EventBus user messages — used for turn count and preview extraction
        # (original text, not augmented with injected context headers).
        bus_user_msgs = self._event_bus.get_events(types={USER_MESSAGE})

        # Turn count: prefer EventBus user messages (always available); fall
        # back to counting "user" roles in chat history for Chat API sessions.
        # Interactions API sessions don't expose full history client-side, so
        # the EventBus count is the primary source.
        turn_count = (
            len(bus_user_msgs)
            if bus_user_msgs
            else sum(1 for h in history_dicts if h.get("role") == "user")
        )

        # Round count: track orchestrator cycles (round_start → round_end pairs).
        # A cycle represents one complete processing round, which may include
        # multiple sub-agent delegations. This is distinct from turn_count which
        # counts user messages.
        # Note: _cycle_number is 1-indexed after first cycle completes.
        round_count = self._cycle_number

        # Don't persist empty sessions (no user messages, no data, no events)
        if turn_count == 0 and len(store) == 0:
            return

        # Preview from last user message (original, not augmented)
        last_preview = ""
        if bus_user_msgs:
            last_event = bus_user_msgs[-1]
            last_text = (
                (last_event.data or {}).get("text", "")
                if hasattr(last_event, "data")
                else ""
            )
            if last_text:
                last_preview = trunc(last_text, "history.error.brief")
        # Fallback to history if EventBus had no text
        if not last_preview:
            for h in reversed(history_dicts):
                if h.get("role") == "user":
                    parts = h.get("parts", [])
                    for p in parts:
                        text = p.get("text", "") if isinstance(p, dict) else ""
                        if text:
                            last_preview = trunc(text, "history.error.brief")
                            break
                    if last_preview:
                        break

        # Generate display_log from the EventBus (single source of truth)
        display_log = self._display_log_builder.entries

        metadata_updates = {
            "turn_count": turn_count,
            "round_count": round_count,
            "last_message_preview": last_preview,
            "token_usage": usage,
            "model": self.model_name,
        }
        # Persist Interactions API state for session resume
        iid = getattr(self.chat, "interaction_id", None)
        if iid:
            metadata_updates["interaction_id"] = iid

        # Persist loaded tool names for tool store resume
        if self._loaded_tool_names:
            metadata_updates["loaded_tool_names"] = sorted(self._loaded_tool_names)

        # Auto-generate a session title after the first round
        if round_count >= 1 and not getattr(self, "_session_title_generated", False):
            session_name = self.generate_session_title()
            if session_name:
                metadata_updates["name"] = session_name
                self._session_title_generated = True
                self._event_bus.emit(
                    SESSION_TITLE,
                    level="info",
                    msg=f"Session: {session_name}",
                    data={"name": session_name},
                )

        self._session_manager.save_session(
            session_id=self._session_id,
            chat_history=history_dicts,
            data_store=store,
            metadata_updates=metadata_updates,
            figure_state=self._renderer.save_state(),
            figure_obj=self._renderer.get_figure(),
            operations=self._ops_log.get_records(),
            display_log=display_log,
        )
        self._event_bus.emit(
            DEBUG,
            level="debug",
            msg=f"[Session] Saved ({turn_count} turns, {len(store)} data entries)",
        )

    def load_session(
        self, session_id: str
    ) -> tuple[dict, list[dict] | None, list[dict] | None]:
        """Restore chat history and DataStore from a saved session.

        Args:
            session_id: The session to load.

        Returns:
            Tuple of (metadata dict, display_log list or None, event_log list or None).
        """
        (
            history_dicts,
            data_dir,
            metadata,
            figure_state,
            operations,
            display_log,
            event_log,
        ) = self._session_manager.load_session(session_id)

        # Restore tool store state (loaded tool names) before chat creation
        saved_tool_names = metadata.get("loaded_tool_names")
        if saved_tool_names and self._use_tool_store:
            self._loaded_tool_names = set(saved_tool_names)
            loaded_schemas = [
                s
                for s in self._all_tool_schemas
                if s.name in self._loaded_tool_names and s.name not in META_TOOL_NAMES
            ]
            self._tool_schemas = list(self._meta_tool_schemas) + loaded_schemas
            self._event_bus.emit(
                DEBUG,
                level="debug",
                msg=f"[ToolStore] Restored {len(self._loaded_tool_names)} loaded tools from session",
            )

        # Restore chat — prefer Interactions API interaction_id if saved
        saved_interaction_id = metadata.get("interaction_id")
        if saved_interaction_id:
            try:
                self.chat = self.adapter.create_chat(
                    model=self.model_name,
                    system_prompt=self._system_prompt,
                    tools=self._tool_schemas,
                    thinking="high",
                    interaction_id=saved_interaction_id,
                )
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Session] Resumed via Interactions API (id={saved_interaction_id[:12]}...)",
                )
            except Exception as e:
                self._event_bus.emit(
                    DEBUG,
                    level="warning",
                    msg=f"[Session] Interactions resume failed: {e}. Falling back to history.",
                )
                saved_interaction_id = None  # fall through to history path

        if not saved_interaction_id:
            # Fall back to Chat API history restoration
            if history_dicts:
                try:
                    self.chat = self.adapter.create_chat(
                        model=self.model_name,
                        system_prompt=self._system_prompt,
                        tools=self._tool_schemas,
                        history=history_dicts,
                        thinking="high",
                    )
                except Exception as e:
                    self._event_bus.emit(
                        DEBUG,
                        level="warning",
                        msg=f"[Session] Could not restore chat history: {e}. "
                        "Starting fresh chat (data still restored).",
                    )
                    self.chat = self.adapter.create_chat(
                        model=self.model_name,
                        system_prompt=self._system_prompt,
                        tools=self._tool_schemas,
                        thinking="high",
                    )
            else:
                self.chat = self.adapter.create_chat(
                    model=self.model_name,
                    system_prompt=self._system_prompt,
                    tools=self._tool_schemas,
                    thinking="high",
                )

        # Restore DataStore — constructor auto-loads _labels.json (or migrates _index.json)
        self._store = DataStore(data_dir)
        set_store(self._store)
        self._event_bus.emit(
            DEBUG,
            level="debug",
            msg=f"[Session] DataStore opened at {data_dir} ({len(self._store)} entries)",
        )

        # Restore operations log with session-scoped IDs
        ops_log = OperationsLog(session_id=session_id)
        if operations:
            ops_log.load_from_records(operations)
            self._event_bus.emit(
                DEBUG,
                level="debug",
                msg=f"[Session] Restored {len(operations)} operation records",
            )
        self._ops_log = ops_log
        set_operations_log(ops_log)

        # Clear sub-agent caches (they'll be recreated on next use)
        with self._sub_agents_lock:
            for actor in self._sub_agents.values():
                actor.stop(timeout=2.0)
            self._sub_agents.clear()
        MISSION_TOOL_REGISTRY.clear_active()
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
            self._event_bus.emit(
                DEBUG,
                level="debug",
                msg="[Session] Figure restore deferred until first access",
            )
        else:
            self._deferred_figure_state = None

        self._session_id = session_id
        self._auto_save = True
        self._session_title_generated = bool(metadata.get("name"))
        set_session_id(session_id)
        attach_log_file(session_id)
        self._start_token_log_listener()
        self._event_bus.session_id = session_id

        # Restore cumulative token usage from previous session runs.
        # Sub-agents are cleared on resume, so seed the orchestrator's own
        # counters with the saved total — new API calls will accumulate on top.
        saved_usage = metadata.get("token_usage", {})
        if saved_usage:
            self._total_input_tokens = saved_usage.get("input_tokens", 0)
            self._total_output_tokens = saved_usage.get("output_tokens", 0)
            self._total_thinking_tokens = saved_usage.get("thinking_tokens", 0)
            self._total_cached_tokens = saved_usage.get("cached_tokens", 0)
            self._api_calls = saved_usage.get("api_calls", 0)

        # Restore cycle counter from previous session runs
        saved_round_count = metadata.get("round_count", 0)
        self._cycle_number = saved_round_count

        # Start writing structured event log (append mode — resumes keep adding)
        self._start_event_log_writer()

        self._event_bus.emit(
            DEBUG, level="debug", msg=f"[Session] Loaded: {session_id}"
        )
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
                store = self._store
                entries = {}
                for trace in last_fig_json.get("data", []):
                    label = trace.get("data_label")
                    if label and label not in entries:
                        entry, _ = self._resolve_entry(store, label)
                        if entry is not None:
                            entries[label] = entry
                # Only pass entries if we resolved all of them
                all_labels = {
                    t.get("data_label")
                    for t in last_fig_json.get("data", [])
                    if t.get("data_label")
                }
                if not all_labels.issubset(entries.keys()):
                    entries = None  # fall back to legacy path

            self._renderer.restore_state(figure_state, entries=entries)
            self._event_bus.emit(
                DEBUG, level="debug", msg="[Session] Deferred figure restore complete"
            )
        except Exception as e:
            self._event_bus.emit(
                DEBUG,
                level="warning",
                msg=f"[Session] Could not restore deferred figure: {e}",
            )

    def _get_plot_status(self) -> dict:
        """Return plot status dict: state, panel_count, traces."""
        figure = self._renderer.get_figure()
        if figure is not None:
            return {
                "state": "active",
                "panel_count": self._renderer._panel_count,
                "traces": list(self._renderer._trace_labels),
            }
        if self._deferred_figure_state is not None:
            return {
                "state": "restorable",
                "panel_count": self._deferred_figure_state.get("panel_count", 0),
                "traces": self._deferred_figure_state.get("trace_labels", []),
            }
        return {"state": "none", "panel_count": 0, "traces": []}

    def _get_data_status(self) -> dict:
        """Return data store status: total and cached counts."""
        store = self._store
        if store is None:
            return {"total_entries": 0, "loaded": 0, "deferred": 0}
        with store._lock:
            total = len(store._labels)
            cached = len(store._cache)
        return {
            "total_entries": total,
            "loaded": cached,
            "deferred": total - cached,
        }

    def _handle_get_session_assets(self) -> dict:
        """Handler for the get_session_assets tool."""
        plot_status = self._get_plot_status()
        data_status = self._get_data_status()
        return {
            "status": "success",
            "plot": plot_status,
            "data": data_status,
            "operations_count": len(self._ops_log) if self._ops_log else 0,
        }

    def _handle_restore_plot(self) -> dict:
        """Handler for the restore_plot tool."""
        # Already active — no-op
        if self._renderer.get_figure() is not None:
            return {
                "status": "success",
                "message": "Plot is already active.",
                "panel_count": self._renderer._panel_count,
                "traces": list(self._renderer._trace_labels),
            }
        # Attempt deferred restore
        if self._deferred_figure_state is not None:
            self._restore_deferred_figure()
            if self._renderer.get_figure() is not None:
                return {
                    "status": "success",
                    "message": "Plot restored from deferred session state.",
                    "panel_count": self._renderer._panel_count,
                    "traces": list(self._renderer._trace_labels),
                }
            return {
                "status": "error",
                "message": "Deferred figure state existed but restoration failed.",
            }
        return {
            "status": "error",
            "message": "No plot to restore — no active or deferred figure state.",
        }

    def _build_followup_context(self) -> str:
        """Build a compact context string from current data store state.

        Returns empty string on first turn or if store is empty and no plot exists.
        """
        store = self._store
        entries = store.list_entries()
        plot_status = self._get_plot_status()

        if not entries and plot_status["state"] == "none":
            return ""

        from knowledge.mission_prefixes import (
            match_dataset_to_mission,
            get_canonical_id,
        )

        missions: dict[str, set[str]] = {}  # stem -> set of dataset_ids
        for e in entries:
            label = e["label"]
            dataset_id = label.split(".")[0]
            stem, _ = match_dataset_to_mission(dataset_id)
            if stem:
                missions.setdefault(stem, set()).add(dataset_id)

        # Build compact context
        lines = ["[ACTIVE SESSION CONTEXT]"]

        if missions:
            mission_ids = [get_canonical_id(s) for s in sorted(missions)]
            lines.append(f"Active mission(s): {', '.join(mission_ids)}")

        # Data summary with loaded/deferred counts
        if entries:
            data_status = self._get_data_status()
            if data_status["deferred"] > 0:
                lines.append(
                    f"Data in memory: {data_status['total_entries']} entries "
                    f"({data_status['deferred']} deferred, {data_status['loaded']} loaded)"
                )
            else:
                lines.append(f"Data in memory: {data_status['total_entries']} entries")

            for stem in sorted(missions):
                mid = get_canonical_id(stem)
                labels = [
                    e["label"]
                    for e in entries
                    if e["label"].split(".")[0] in missions[stem]
                ]
                if labels:
                    _shown_labels, _total_labels = trunc_items(
                        labels, "items.data_labels"
                    )
                    lines.append(f"  {mid}: {', '.join(_shown_labels)}")
                    if _total_labels > len(_shown_labels):
                        lines.append(
                            f"    ... and {_total_labels - len(_shown_labels)} more"
                        )

        # Plot status
        if plot_status["state"] == "active":
            traces_str = (
                ", ".join(plot_status["traces"]) if plot_status["traces"] else "unknown"
            )
            lines.append(
                f"Plot: active ({plot_status['panel_count']} panels, traces: {traces_str})"
            )
        elif plot_status["state"] == "restorable":
            traces_str = (
                ", ".join(plot_status["traces"]) if plot_status["traces"] else "unknown"
            )
            lines.append(
                f"Plot: restorable ({plot_status['panel_count']} panels, "
                f"traces: {traces_str} — call restore_plot to activate)"
            )

        lines.append("")
        if missions:
            lines.append(
                "For follow-up requests on this data, delegate to the mission "
                "agent directly — do NOT re-search datasets or list parameters yourself."
            )
        lines.append("[END SESSION CONTEXT]")
        return "\n".join(lines)

    def _build_pipeline_context(self) -> str:
        """Build a compact context block listing saved pipelines.

        Injected into the orchestrator prompt so it can proactively suggest
        running a saved pipeline instead of rebuilding a workflow from scratch.

        Returns:
            Empty string if no saved pipelines exist, otherwise a compact listing.
        """
        entries = self._pipeline_store.get_for_injection(limit=15)
        if not entries:
            return ""

        lines = ["[SAVED PIPELINES]"]
        if config.PIPELINE_CONFIRMATION:
            lines.append("[PIPELINE CONFIRMATION REQUIRED]")

        for e in entries:
            _shown_ds, _total_ds = trunc_items(e.datasets, "items.datasets")
            datasets = ", ".join(_shown_ds)
            extra = (
                f" (+{_total_ds - len(_shown_ds)} more)"
                if _total_ds > len(_shown_ds)
                else ""
            )
            missions = ", ".join(e.missions)
            line = f'- {e.id}: "{e.name}" [{missions}] ({datasets}{extra})'
            if e.description:
                line += f" — {e.description}"
            lines.append(line)

        total = len(self._pipeline_store.get_active())
        if total > 15:
            lines.append(f"  ({total - 15} more — use search_pipelines to find)")

        if config.PIPELINE_CONFIRMATION:
            lines.append("")
            lines.append(
                "IMPORTANT: Before running any pipeline, present the top matches to the "
                "user with descriptions and reasoning, then use ask_clarification to get "
                "explicit permission. Do NOT call run_pipeline without user confirmation."
            )
        else:
            lines.append("")
            lines.append(
                "If the user's request matches a saved pipeline, run it with "
                "run_pipeline instead of building from scratch. "
                "Use search_pipelines to find by mission, dataset, or description."
            )
        lines.append("[END SAVED PIPELINES]")
        return "\n".join(lines)

    def process_message(self, user_message: str) -> str:
        """Process a user message and return the agent's response.

        All messages go through the orchestrator LLM, which decides whether to
        invoke the planner via the ``request_planning`` tool when multi-step
        coordination is needed.
        """
        self.clear_cancel()
        # Re-set ContextVar so executor threads inherit the correct OperationsLog
        if hasattr(self, "_ops_log"):
            set_operations_log(self._ops_log)
        self._memory_store._last_injected_ids.clear()
        self._event_bus.emit(
            USER_MESSAGE,
            level="info",
            msg=f"[User] {user_message}",
            data={"text": user_message},
        )

        # Inject long-term memory context (only when changed)
        memory_section = self._memory_store.build_prompt_section(
            include_review_instruction=self._planner_agent is not None,
        )
        if memory_section and self._ctx_tracker.is_changed(
            "orchestrator", "memory", memory_section
        ):
            augmented = f"{memory_section}\n\n{user_message}"
            self._ctx_tracker.record("orchestrator", memory=memory_section)
        else:
            augmented = user_message

        # Inject active session context for faster follow-up routing (only when changed)
        followup_ctx = self._build_followup_context()
        if followup_ctx and self._ctx_tracker.is_changed(
            "orchestrator", "session", followup_ctx
        ):
            augmented = f"{followup_ctx}\n\n{augmented}"
            self._ctx_tracker.record("orchestrator", session=followup_ctx)

        # Inject saved pipelines for proactive reuse (only when changed)
        pipeline_ctx = self._build_pipeline_context()
        if pipeline_ctx and self._ctx_tracker.is_changed(
            "orchestrator", "pipeline", pipeline_ctx
        ):
            augmented = f"{pipeline_ctx}\n\n{augmented}"
            self._ctx_tracker.record("orchestrator", pipeline=pipeline_ctx)

        from .llm_utils import _CancelledDuringLLM

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

        self._event_bus.emit(
            AGENT_RESPONSE,
            level="info",
            msg=f"[Agent] {result}",
            data={
                "text": result,
                "turn": self._turn_number,
                "cycle": self._cycle_number,
            },
        )

        # Auto-save after each turn
        if self._auto_save and self._session_id:
            try:
                self.save_session()
            except Exception as e:
                self._event_bus.emit(
                    DEBUG, level="warning", msg=f"Auto-save failed: {e}"
                )

        # Maybe trigger async memory extraction (first round, then every N more)
        self._memory_turn_counter += 1
        mem_interval = config.get("memory_extraction_interval", 2)
        if mem_interval > 0 and (self._memory_turn_counter - 1) % mem_interval == 0:
            self._maybe_extract_memories()

        # Eureka discovery (Step 2)
        if config.get("eureka_enabled", True):
            self._eureka_turn_counter += 1
            eureka_interval = config.get("eureka_extraction_interval", 2)
            if eureka_interval > 0 and (self._eureka_turn_counter - 1) % eureka_interval == 0:
                self._maybe_extract_eurekas()

        return result

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

    def reset(self):
        """Reset conversation history, mission agent cache, and sub-agents."""
        self._control_center.clear()  # Cancel in-flight work, remove all units
        self._cancel_event.clear()
        self.chat = self.adapter.create_chat(
            model=get_active_model(self.model_name),
            system_prompt=self._system_prompt,
            tools=self._tool_schemas,
            thinking="high",
        )
        self._current_plan = None
        self._plan_time_range = None
        with self._sub_agents_lock:
            for actor in self._sub_agents.values():
                actor.stop(timeout=2.0)
            self._sub_agents.clear()
        MISSION_TOOL_REGISTRY.clear_active()
        self._dataops_seq = 0
        self._mission_seq = 0
        self._planner_agent = None
        self._renderer.reset()

        # Reset memory turn counter and agent (do NOT clear memory store)
        self._memory_agent = None
        self._memory_turn_counter = 0
        self._last_memory_op_index = 0
        self._recent_custom_op_failures = {}
        self._inline_fail_count = 0
        self._inline_disabled_until = 0.0

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
            self._event_bus.emit(
                DEBUG,
                level="debug",
                msg=f"[Session] New session after reset: {self._session_id}",
            )

        # Create a fresh OperationsLog scoped to the (possibly new) session
        self._ops_log = OperationsLog(session_id=self._session_id or "")
        set_operations_log(self._ops_log)

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

    def resume_plan(self, plan: TaskPlan) -> str:
        """Resume an incomplete plan from storage."""
        self._current_plan = plan
        plan.status = PlanStatus.EXECUTING
        store = get_task_store()

        self._event_bus.emit(
            DEBUG,
            level="debug",
            msg=f"[Resume] Resuming plan: {trunc(plan.user_request, 'api.session_preview')}",
        )
        self._event_bus.emit(DEBUG, level="debug", msg=format_plan_for_display(plan))

        pending = plan.get_pending_tasks()
        if not pending:
            plan.status = (
                PlanStatus.COMPLETED
                if not plan.get_failed_tasks()
                else PlanStatus.FAILED
            )
            store.save(plan)
            return self._summarize_plan_execution(plan)

        for i, task in enumerate(plan.tasks):
            if task.status != TaskStatus.PENDING:
                continue

            plan.current_task_index = i
            store.save(plan)

            self._event_bus.emit(
                PLAN_TASK,
                level="debug",
                msg=f"[Plan] Resuming step {i + 1}/{len(plan.tasks)}: {task.description}",
            )

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
        return f"Discarded plan: {trunc(plan.user_request, 'api.session_preview')}"


def create_agent(
    verbose: bool = False,
    gui_mode: bool = False,
    model: str | None = None,
    defer_chat: bool = False,
) -> OrchestratorAgent:
    """Factory function to create a new agent instance.

    Args:
        verbose: If True, print debug info about tool calls.
        gui_mode: If True, launch with visible GUI window.
        model: Gemini model name (default: gemini-2.5-flash).
        defer_chat: If True, skip creating the initial chat session.

    Returns:
        Configured OrchestratorAgent instance.
    """
    return OrchestratorAgent(
        verbose=verbose, gui_mode=gui_mode, model=model, defer_chat=defer_chat
    )
