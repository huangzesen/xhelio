"""
Core agent logic - orchestrates Gemini calls and tool execution.

The OrchestratorAgent routes requests to:
- EnvoyAgent sub-agents for data operations (per mission)
- VizAgent[Plotly] / VizAgent[Mpl] sub-agents for visualization
"""

import contextvars
import json
import math
import queue
import time
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import config
from config import get_data_dir, get_api_key
from .llm import (
    LLMAdapter,
    LLMService,
    LLMResponse,
    FunctionSchema,
)
from .tools import (
    get_tool_schemas,
    get_function_schemas,
)
from .truncation import trunc, trunc_items, join_labels, get_limit, get_item_limit
from .prompts import get_system_prompt
from .time_utils import parse_time_range, TimeRangeError
from .tool_timing import ToolTimer, stamp_tool_result
from .tasks import Task, TaskPlan, TaskStatus, PlanStatus, get_task_store
from .planner import format_plan_for_display
from .turn_limits import get_limit as get_turn_limit
from .session import SessionManager
from .memory import MemoryStore, MEMORY_RELOAD_INTERVAL
from .token_counter import count_tokens as estimate_tokens, count_tool_tokens
from .sub_agent import SubAgent, AgentState, Message, _make_message
from .envoy_agent import EnvoyAgent
from .viz_plotly_agent import VizPlotlyAgent
from .viz_mpl_agent import VizMplAgent
from .viz_jsx_agent import VizJsxAgent
from .data_ops_agent import DataOpsAgent
from .data_io_agent import DataIOAgent
from .insight_agent import InsightAgent
from .logging import (
    setup_logging,
    attach_log_file,
    get_logger,
    log_error,
    log_tool_call,
    log_tool_result,
    set_session_id,
    LOG_DIR,
)
from .event_bus import (
    EventBus,
    SessionEvent,
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
    DELEGATION_ASYNC_COMPLETED,
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
    CONTEXT_COMPACTION,
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
    PERMISSION_REQUEST,
)
from .control_center import ControlCenter
from .memory_agent import MemoryAgent, MemoryContext
from .pipeline_store import PipelineStore
from .context_tracker import ContextTracker
from .loop_guard import LoopGuard, DupVerdict
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
    ENVOY_TOOL_REGISTRY,
)

# Roles that map to "user" or "assistant/model" across all LLM adapters
_USER_ROLES = {"user"}
_AGENT_ROLES = {"model", "assistant"}

# Context compaction prompt (shared with sub_agent.py)
_COMPACTION_PROMPT = (
    "Summarize the following conversation history concisely for an AI agent "
    "that needs to continue the session. Preserve:\n"
    "- ALL errors, failures, and their details verbatim\n"
    "- Key decisions and user preferences\n"
    "- Data that was fetched/computed and current state\n"
    "- Tool calls that produced important results\n\n"
    "Drop thinking blocks and routine acknowledgments. "
    "Output ONLY the summary, no commentary.\n\n"
    "Conversation history:\n"
)


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


def _create_llm_service() -> LLMService:
    """Create the LLM service based on config."""
    provider = config.LLM_PROVIDER.lower()
    api_key = get_api_key(provider)
    model = config.SMART_MODEL
    return LLMService(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=config.LLM_BASE_URL,
    )



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
        self._cancel_holdback = threading.Event()  # When set, hold fire-and-forget results
        self._held_results: list[str] = []  # Results held during cancel
        self._was_cancelled = False  # Set on cancel, cleared on next process_message
        self._held_results_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._text_already_streamed = False  # set by _send_message_streaming
        self._intermediate_text_streamed = False  # set by _send_message_streaming
        self._last_response_generated = False

        # Sub-agent inbox — primary coordination mechanism for the orchestrator
        # PriorityQueue: (priority, timestamp, message) - priority 0 = user, 1 = subagent
        self._inbox: queue.PriorityQueue[tuple[int, float, Message]] = queue.PriorityQueue()

        # Lifecycle state machine (mirrors SubAgent's AgentState)
        self._state = AgentState.SLEEPING
        self._cycle_number: int = 0
        self._rounds_since_last_reload: int = 0  # Memory hot reload counter
        self._turn_number: int = 0

        # Disk-backed data store (created at start_session / load_session)
        self._store: DataStore | None = None

        # Sub-agents: agent_id → SubAgent
        self._sub_agents: dict[str, SubAgent] = {}
        self._sub_agents_lock = threading.Lock()
        self._dataops_seq: int = 0  # Counter for ephemeral DataOps agents
        self._mission_seq: int = 0  # Counter for ephemeral Mission agents
        self._async_delegations: dict[str, float] = {}  # agent_id → start_time

        # Fire-and-forget async delegation tracking: agent_id → start_time
        # Written by delegation handler threads, read by run_loop thread.
        self._async_delegations: dict[str, float] = {}
        self._async_delegations_lock = threading.Lock()

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

        # Initialize LLM service (wraps all provider SDK calls)
        self.service: LLMService = _create_llm_service()
        self.adapter: LLMAdapter = self.service.adapter  # backward compat

        # Discover SPICE tools from MCP server (lazy, non-fatal).
        # Must happen before building tool schemas so the LLM sees them.
        self._ensure_spice_tools()

        # Build tool schemas for the orchestrator
        self._all_tool_schemas = get_function_schemas(names=ORCHESTRATOR_TOOLS)
        self._viz_backend = config.PREFER_VIZ_BACKEND

        # All tools active from the start (no browse+load gating)
        self._tool_schemas = list(self._all_tool_schemas)

        # Store model name and system prompt for chat creation
        self.model_name = model or config.SMART_MODEL
        self._system_prompt = get_system_prompt()

        if not defer_chat:
            # Create chat session
            self.chat = self.adapter.create_chat(
                model=self.model_name,
                system_prompt=self._system_prompt,
                tools=self._tool_schemas,
                thinking="high",
            )
        else:
            # Chat will be created by load_session() or _send_message()
            self.chat = None

        # Plotly renderer for visualization
        self._renderer = PlotlyRenderer(verbose=self.verbose, gui_mode=self.gui_mode)
        from rendering.plotly_renderer import set_renderer

        set_renderer(self._renderer)
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

        # Cycle state tracking
        self._responded_this_cycle = False  # Track if orchestrator responded to user

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

        # Counter for consecutive invalid-tool rejections per agent type.
        # Used to provide enhanced error messages listing available tools.
        self._invalid_tool_counts: dict[str, int] = {}

        # Inline model token usage (tracked separately for breakdown)
        self._inline_input_tokens = 0
        self._inline_output_tokens = 0
        self._inline_thinking_tokens = 0
        self._inline_cached_tokens = 0
        self._inline_api_calls = 0

        # Retired ephemeral agent token usage (preserved after cleanup)
        self._retired_agent_usage: list[dict] = []

        # Current plan being executed (if any) — either TaskPlan (old loop) or dict (new research-only)
        self._current_plan: Optional[TaskPlan | dict] = None

        # SSE event listener (subscribed by api/routes.py when streaming)
        self._sse_listener: SSEEventListener | None = None


        # Session persistence
        self._session_id: Optional[str] = None
        self._session_manager = SessionManager()
        self._auto_save: bool = False

        # Load persisted informed-tool overrides
        from .agent_registry import AGENT_INFORMED_REGISTRY as _informed_reg

        _informed_reg.load(get_data_dir() / "informed_tools.json")

        # Long-term memory — bake into core memory (system prompt)
        self._memory_store = MemoryStore()
        memory_section = self._memory_store.format_for_injection(
            scope="generic", include_review_instruction=False
        )
        if memory_section:
            self._system_prompt = f"{self._system_prompt}\n\n{memory_section}"
            # Update existing chat session if already created
            if self.chat is not None:
                self.chat.update_system_prompt(self._system_prompt)
        # Listen for memory mutations to refresh core memory
        self._event_bus.subscribe(self._on_memory_mutated)
        self._memory_agent: Optional[MemoryAgent] = None

        # Eureka discovery (Step 2) and Eureka Mode
        self._eureka_lock = threading.Lock()
        self._eureka_agent: Optional["EurekaAgent"] = None
        self._eureka_turn_counter: int = 0
        self._eureka_mode: bool = config.get("eureka_mode", False)
        self._eureka_round_counter: int = 0
        self._eureka_pending_suggestion = None

        # Pipeline template index (searchable metadata for saved templates)
        self._pipeline_store = PipelineStore()

        # Periodic memory extraction (mirrors discovery pattern)
        self._memory_turn_counter = 0
        self._last_memory_op_index = 0
        self._memory_lock = threading.Lock()

        # Inline completion circuit breaker: disable after repeated failures
        self._inline_fail_count: int = 0
        self._inline_disabled_until: float = 0.0

        # Insight review iteration counter (reset per user turn)
        self._insight_review_iter: int = 0

        # Latest rendered PNG bytes (set by Plotly and matplotlib handlers)
        self._latest_render_png: bytes | None = None

        # Cached pipeline DAG (invalidated when new ops are recorded)
        self._pipeline = None

        # Pull-based event feed for orchestrator (sub-agent feeds created on demand)
        from agent.event_feed import EventFeedBuffer

        self._event_feed = EventFeedBuffer(
            self._event_bus,
            "ctx:orchestrator",
        )

        # Thread pool for timeout-wrapped Gemini calls
        self._timeout_pool = ThreadPoolExecutor(max_workers=1)

        # Permission gate: request_id -> {"event": threading.Event, "approved": None}
        self._permission_gate: dict[str, dict] = {}
        self._permission_lock = threading.Lock()

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

    @property
    def _current_envoy_sandbox_imports(self) -> list[dict] | None:
        return getattr(self._tls, "envoy_sandbox_imports", None)

    @_current_envoy_sandbox_imports.setter
    def _current_envoy_sandbox_imports(self, value: list[dict] | None) -> None:
        self._tls.envoy_sandbox_imports = value

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

    _CANCEL_CONTEXT_PREFIX = (
        "[System] The user cancelled the previous operation. "
        "Some results may have been lost. Continue from here.\n\n"
    )

    def request_cancel(self):
        """Signal the agent to stop after the current atomic operation."""
        self._cancel_event.set()
        self._cancel_holdback.set()
        self._was_cancelled = True
        self._event_bus.emit(DEBUG, level="info", msg="[Cancel] Cancellation requested")

    def clear_cancel(self):
        """Clear the cancellation flag (called at start of process_message)."""
        self._cancel_event.clear()

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

    # Tools safe to run concurrently. Delegation tools block until the
    # sub-agent finishes, but run in parallel via ThreadPoolExecutor.
    _PARALLEL_SAFE_TOOLS = {
        "fetch_data",
        "delegate_to_envoy",
        "delegate_to_viz",
        "delegate_to_data_ops",
        "delegate_to_data_io",
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

    def _all_work_subagents_idle(self) -> bool:
        """Check if all work subagents (excluding Eureka) are idle."""
        with self._sub_agents_lock:
            for agent_id, agent in self._sub_agents.items():
                if agent_id in ("EurekaAgent", "MemoryAgent"):
                    continue  # Post-cycle agents, not part of work cycle
                if not agent.is_idle:
                    return False
        return True

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

    def _on_reset(self, chat, failed_message):
        """Rollback reset: new chat, drop failed turn, inject context about what happened.

        Called by send_with_timeout after 2 consecutive failures (~20s).
        The failed_message contains the tool results that were being sent
        when the 500 hit — the tools executed successfully but the LLM
        never saw the results.

        We drop the last assistant turn (tool calls), create a new chat,
        and tell the model what it tried and what the results were.
        """
        # Get interface from chat (single source of truth)
        iface = chat.interface

        # Summarize what the model tried (tool calls from the assistant turn)
        history = iface.to_dict()  # For summarization
        tool_summary = self._summarize_tool_calls_from_interface(iface)
        # Summarize the tool results that were ready but never delivered
        result_summary = self._summarize_tool_results(failed_message)

        # Drop failed turn: assistant with tool calls + orphaned tool results
        # Using interface.drop_trailing for clean rollback
        from agent.llm.interface import ToolResultBlock
        iface.drop_trailing(lambda e: e.role == "assistant")
        iface.drop_trailing(
            lambda e: e.role == "user" and all(
                isinstance(b, ToolResultBlock) for b in e.content
            )
        )

        self._event_bus.emit(
            LLM_CALL,
            agent="Orchestrator",
            level="warning",
            msg=f"[Orchestrator] Session rollback — new chat ({len(iface.entries)} entries kept)",
        )

        self.chat = self.adapter.create_chat(
            model=self.model_name,
            system_prompt=self._system_prompt,
            tools=self._tool_schemas,
            thinking="high",
            interface=iface,
        )

        rollback_msg = (
            "Your previous response was lost due to a server error. "
            "Here is what happened:\n\n"
            f"You called these tools:\n{tool_summary}\n\n"
            f"The tools executed successfully and returned:\n{result_summary}\n\n"
            "Data already fetched is still available in memory. "
            "Please continue based on these results."
        )
        return self.chat, rollback_msg

    @staticmethod
    def _summarize_tool_calls(history: list[dict]) -> str:
        """Extract tool call names and args from the last assistant turn."""
        parts = []
        for msg in reversed(history):
            if msg.get("role") != "assistant":
                break
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name", "?")
                    args = block.get("input", {})
                    args_str = ", ".join(f"{k}={repr(v)[:80]}" for k, v in args.items())
                    parts.append(f"- {name}({args_str})")
        return "\n".join(reversed(parts)) if parts else "(no tool calls found)"

    def _summarize_tool_calls_from_interface(self, iface) -> str:
        """Extract tool call names and args from the last assistant entry in interface."""
        from agent.llm.interface import ToolCallBlock
        parts = []
        last_asst = iface.last_assistant_entry()
        if last_asst:
            for block in last_asst.content:
                if isinstance(block, ToolCallBlock):
                    args_str = ", ".join(f"{k}={repr(v)[:80]}" for k, v in block.args.items())
                    parts.append(f"- {block.name}({args_str})")
        return "\n".join(parts) if parts else "(no tool calls found)"

    @staticmethod
    def _summarize_tool_results(message) -> str:
        """Extract tool result summaries from the message that failed to send."""
        if isinstance(message, str):
            return message[:500]
        parts = []
        items = message if isinstance(message, list) else [message]
        for item in items:
            if not isinstance(item, dict):
                continue
            content = item.get("content", item)
            if isinstance(content, str):
                parts.append(f"- {content[:300]}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_result":
                            tool_id = block.get("tool_use_id", "?")
                            text = block.get("content", "")
                            if isinstance(text, str):
                                parts.append(f"- [{tool_id}]: {text[:300]}")
                            elif isinstance(text, list):
                                # Content blocks inside tool_result
                                for sub in text:
                                    if (
                                        isinstance(sub, dict)
                                        and sub.get("type") == "text"
                                    ):
                                        parts.append(
                                            f"- [{tool_id}]: {sub.get('text', '')[:300]}"
                                        )
        return "\n".join(parts) if parts else "(tool results not available)"

    def _check_and_compact(self) -> None:
        """Check context usage and compact messages if nearing the limit."""
        if self.chat is None:
            return
        ctx_window = self.chat.context_window()
        if ctx_window <= 0:
            return
        ctx_tokens = self.chat.estimate_context_tokens()
        if ctx_tokens <= 0 or ctx_tokens < ctx_window * 0.8:
            return

        def summarizer(text: str) -> str:
            response = self.adapter.generate(
                model=self.model_name,
                contents=_COMPACTION_PROMPT + text,
                temperature=0.1,
                max_output_tokens=2048,
            )
            # Track token cost
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
                last_tool_context="context_compaction",
            )
            self._total_input_tokens = token_state["input"]
            self._total_output_tokens = token_state["output"]
            self._total_thinking_tokens = token_state["thinking"]
            self._total_cached_tokens = token_state["cached"]
            self._api_calls = token_state["api_calls"]

            return response.text.strip() if response and response.text else ""

        before = ctx_tokens
        if self.chat.compact(summarizer=summarizer):
            after = self.chat.estimate_context_tokens()
            self._event_bus.emit(
                CONTEXT_COMPACTION,
                agent="orchestrator",
                level="info",
                data={
                    "before_tokens": before,
                    "after_tokens": after,
                    "context_window": ctx_window,
                },
            )

    def _send_message(self, message) -> LLMResponse:
        """Send a message on self.chat with timeout/retry. 429 errors propagate."""
        self._check_and_compact()
        return self._send_with_timeout(self.chat, message)

    def _send_message_streaming(self, message) -> LLMResponse:
        """Send a message with streaming text deltas emitted via EventBus.

        Text tokens are emitted as ``TEXT_DELTA`` events as they arrive,
        giving the user near-instant first-token feedback.  The complete
        ``LLMResponse`` is returned at the end (same contract as
        ``_send_message``).

        Sets ``self._text_already_streamed = True`` so that ``run_loop()``
        knows to skip the final bulk ``TEXT_DELTA`` emission.
        """
        self._check_and_compact()

        def _on_chunk(text_delta: str) -> None:
            self._event_bus.emit(
                TEXT_DELTA,
                agent="orchestrator",
                level="info",
                msg=text_delta,
                data={"text": text_delta, "streaming": True},
            )

        response = send_with_timeout_stream(
            chat=self.chat,
            message=message,
            timeout_pool=self._timeout_pool,
            cancel_event=self._cancel_event,
            retry_timeout=_LLM_RETRY_TIMEOUT,
            agent_name="Orchestrator",
            logger=self.logger,
            on_chunk=_on_chunk,
            on_reset=self._on_reset,
        )

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
            on_reset=self._on_reset,
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

    def _web_search(self, query: str) -> dict:
        """Execute a web search query via the active LLM provider's native API."""
        try:
            response = self.adapter.web_search(
                query=query,
                model=self.model_name,
            )
        except Exception as e:
            return {"status": "error", "message": f"Web search failed: {e}"}

        if not response.text:
            # Distinguish "not configured" vs "call failed with empty response"
            provider = getattr(
                self.adapter, "provider_name", type(self.adapter).__name__
            )
            return {
                "status": "error",
                "message": (
                    f"Web search returned empty results (provider: {provider}). "
                    "This may indicate the web search API call failed silently, "
                    "or the provider doesn't support web search. "
                    "Check the provider's web_search configuration in config.json."
                ),
            }

        self._last_tool_context = "web_search"
        # Track web search tokens without decomposition — web search is a
        # separate LLM call with no system prompt or tool schemas.
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
            last_tool_context="web_search",
            system_tokens=0,
            tools_tokens=0,
        )
        self._total_input_tokens = token_state["input"]
        self._total_output_tokens = token_state["output"]
        self._total_thinking_tokens = token_state["thinking"]
        self._total_cached_tokens = token_state["cached"]
        self._api_calls = token_state["api_calls"]
        self._latest_input_tokens = response.usage.input_tokens

        sources_text = self._extract_grounding_sources(response)

        return {"status": "success", "answer": response.text + sources_text}

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

        # Note: MemoryAgent usage is included via _sub_agents iteration above

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
                "ctx_history_tokens": max(
                    0,
                    self._latest_input_tokens
                    - self._system_prompt_tokens
                    - self._tools_tokens,
                ),
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

        # Note: MemoryAgent usage is included via _sub_agents iteration above

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

            # Cache rendered PNG for InsightAgent (auto-review + manual delegation)
            figure = self._renderer.get_figure()
            if figure is not None:
                import io

                try:
                    buf = io.BytesIO()
                    figure.write_image(
                        buf, format="png", width=1100, height=600, scale=2
                    )
                    self._latest_render_png = buf.getvalue()
                except Exception as e:
                    get_logger().warning(
                        f"Failed to cache Plotly PNG for insight review: {e}"
                    )

            # Save figure JSON + thumbnail PNG to session for output verification
            import json as _json

            session_dir = self._session_manager.base_dir / self._session_id
            plotly_dir = session_dir / "plotly_outputs"
            plotly_dir.mkdir(parents=True, exist_ok=True)
            op_id = op_record["id"]

            json_path = plotly_dir / f"{op_id}.json"
            json_path.write_text(_json.dumps(fig_json, default=str))

            png_path = plotly_dir / f"{op_id}.png"
            if self._latest_render_png:
                png_path.write_bytes(self._latest_render_png)
            elif figure is not None:
                import io as _io

                try:
                    buf = _io.BytesIO()
                    figure.write_image(
                        buf, format="png", width=1100, height=600, scale=2
                    )
                    png_bytes = buf.getvalue()
                    png_path.write_bytes(png_bytes)
                    self._latest_render_png = png_bytes
                except Exception:
                    pass

            result["output_files"] = [str(json_path)]
            if png_path.exists():
                result["output_files"].append(str(png_path))

            if not json_path.is_file() or json_path.stat().st_size == 0:
                result["status"] = "error"
                result["message"] = (
                    f"Render completed but output file is missing or empty: {json_path}. "
                    "The figure JSON could not be saved to disk."
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

        # Try new _ids.json first, fall back to _labels.json for backward compat
        ids_path = data_dir / "_ids.json"
        labels_path = data_dir / "_labels.json"

        if not ids_path.exists() and not labels_path.exists():
            return {
                "status": "error",
                "message": "No data in session. Fetch data first before generating a plot.",
            }

        import json as _json

        if ids_path.exists():
            labels_index = _json.loads(ids_path.read_text())
        else:
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
            from rendering.mpl_sandbox import extract_data_labels as _extract_mpl_labels

            _mpl_labels_used = _extract_mpl_labels(code)
            self._event_bus.emit(
                MPL_RENDER_EXECUTED,
                agent="VizAgent[Mpl]",
                msg=f"[MplViz] Script executed: {description}",
                data={
                    "script_id": script_id,
                    "description": description,
                    "output_path": result.output_path,
                    "script_path": result.script_path,
                    "args": {"script": code, "description": description},
                    "inputs": _mpl_labels_used,
                    "outputs": [],
                    "status": "success",
                },
            )
            # Cache rendered PNG for InsightAgent (auto-review + manual delegation)
            if result.output_path:
                try:
                    self._latest_render_png = Path(result.output_path).read_bytes()
                except Exception as e:
                    get_logger().warning(
                        f"Failed to cache matplotlib PNG for insight review: {e}"
                    )

            response = {
                "status": "success",
                "script_id": script_id,
                "output_path": result.output_path,
                "message": f"Matplotlib plot saved successfully. Script ID: {script_id}",
            }

            # Verify output PNG exists and is non-empty
            if result.output_path:
                output_path = Path(result.output_path)
                response["output_files"] = [str(output_path)]
                if not output_path.is_file() or output_path.stat().st_size == 0:
                    response["status"] = "error"
                    response["message"] = (
                        f"Script executed but output PNG is missing or empty: {output_path}. "
                        "Check stderr for rendering errors."
                    )

            if result.stdout.strip():
                response["stdout"] = result.stdout

            # Auto-review via InsightAgent (same as Plotly render path)
            review = self._sync_insight_review()
            if review is not None and not review.get("failed", False):
                passed = review.get("passed", True)
                verdict = "PASS" if passed else "NEEDS_IMPROVEMENT"
                response["insight_review"] = {
                    "verdict": verdict,
                    "feedback": review.get("text", ""),
                    "suggestions": review.get("suggestions", []),
                }
                if not passed:
                    response["insight_review"]["action_needed"] = (
                        "The automatic figure review found issues. "
                        "Review the feedback and suggestions above, then "
                        "re-render with improvements."
                    )

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
            from rendering.mpl_sandbox import (
                _find_python,
                MplSandboxResult,
                extract_data_labels as _extract_mpl_labels,
            )
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
                        _rerun_code = script_file.read_text(encoding="utf-8")
                        _rerun_labels = _extract_mpl_labels(_rerun_code)
                        self._event_bus.emit(
                            MPL_RENDER_EXECUTED,
                            agent="VizAgent[Mpl]",
                            msg=f"[MplViz] Script re-executed: {script_id}",
                            data={
                                "script_id": script_id,
                                "description": f"Rerun of {script_id}",
                                "args": {
                                    "script": _rerun_code,
                                    "description": f"Rerun of {script_id}",
                                },
                                "inputs": _rerun_labels,
                                "outputs": [],
                                "status": "success",
                            },
                        )
                        # Cache rendered PNG for InsightAgent
                        try:
                            self._latest_render_png = output_path.read_bytes()
                        except Exception as e:
                            get_logger().warning(
                                f"Failed to cache matplotlib rerun PNG: {e}"
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
            from rendering.jsx_sandbox import extract_data_labels as _extract_jsx_labels

            _jsx_labels_used = _extract_jsx_labels(code)
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
                    "args": {"code": code, "description": description},
                    "inputs": _jsx_labels_used,
                    "outputs": [],
                    "status": "success",
                },
            )
            response = {
                "status": "success",
                "script_id": script_id,
                "output_path": result.output_path,
                "data_path": result.data_path,
                "message": f"JSX component compiled successfully. Script ID: {script_id}",
            }

            # Verify output bundle exists and is non-empty
            if result.output_path:
                output_path = Path(result.output_path)
                files = [str(output_path)]
                if result.data_path:
                    files.append(result.data_path)
                response["output_files"] = files
                if not output_path.is_file() or output_path.stat().st_size == 0:
                    response["status"] = "error"
                    response["message"] = (
                        f"JSX compiled but output bundle is missing or empty: {output_path}."
                    )

            return response
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
            from rendering.jsx_sandbox import (
                run_jsx_pipeline,
                extract_data_labels as _extract_jsx_labels,
            )
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
                _recompile_labels = _extract_jsx_labels(code)
                self._event_bus.emit(
                    JSX_RENDER_EXECUTED,
                    agent="VizAgent[JSX]",
                    msg=f"[JsxViz] Component recompiled: {script_id}",
                    data={
                        "script_id": script_id,
                        "description": f"Recompile of {script_id}",
                        "args": {
                            "code": code,
                            "description": f"Recompile of {script_id}",
                        },
                        "inputs": _recompile_labels,
                        "outputs": [],
                        "status": "success",
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
        from .tools import register_dynamic_tools
        from .agent_registry import register_spice_tools
        from .observations import register_spice_tool_names
        from .tool_handlers.spice import register_spice_handlers

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

        # Register into TOOL_REGISTRY (handlers)
        register_spice_handlers(names)

        OrchestratorAgent._spice_tools_ready = True
        self._event_bus.emit(
            DEBUG,
            level="info",
            msg=f"[SPICE] Discovered {len(names)} tools: {', '.join(names)}",
        )

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

        agent_ctx = f"ctx:{self._current_agent_type}"
        allowed = AGENT_CALL_REGISTRY.get(agent_ctx, frozenset())
        if tool_name not in allowed:
            # Track repeated invalid-tool rejections per agent type
            key = f"{self._current_agent_type}:{tool_name}"
            count = self._invalid_tool_counts.get(key, 0) + 1
            self._invalid_tool_counts[key] = count

            msg = f"Tool '{tool_name}' is not available for {self._current_agent_type}."
            if count >= 2:
                # After 2+ rejections of the same tool, list available tools
                available = sorted(allowed)
                msg += (
                    f" This tool has been rejected {count} times. "
                    f"STOP calling it. Your available tools are: {', '.join(available)}"
                )
            log_error(msg, context={"tool_name": tool_name, "tool_args": tool_args})
            return {"status": "error", "message": msg}

        # Reset invalid-tool counter on successful permission check
        reset_key = f"{self._current_agent_type}:{tool_name}"
        self._invalid_tool_counts.pop(reset_key, None)

        # ── Registry dispatch ──
        from agent.tool_handlers import TOOL_REGISTRY

        handler = TOOL_REGISTRY.get(tool_name)
        if handler:
            return handler(self, tool_args)

        result = {"status": "error", "message": f"Unknown tool: {tool_name}"}
        log_error(
            f"Unknown tool called: {tool_name}",
            context={"tool_name": tool_name, "tool_args": tool_args},
        )
        return result

    # ---- Delegation infrastructure ----

    def _build_envoy_request(self, mission_id: str, request: str, agent=None) -> str:
        """Build a full request string for an envoy delegation."""
        agent_id = agent.agent_id if agent else f"EnvoyAgent[{mission_id}]"
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
            # else: store unchanged — skip injection (agent already has it)
        if not (agent and agent._interaction_id):
            request += "\n\n[Tip: Call events(action='check') to see what happened earlier in this session.]"
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
            full_request += "\n\n[Tip: Call events(action='check') to see what happened earlier in this session.]"
        return full_request

    def _build_data_io_request(self, request: str, context: str) -> str:
        """Build a full request string for a DataIO delegation."""
        full_request = f"{request}\n\nContext: {context}" if context else request
        return full_request

    # ---- Permission gate (blocking approval from user) ----

    def request_permission(self, request_id: str, action: str, description: str, command: str, timeout: float = 300.0) -> dict:
        """Block until user approves/denies an action. Called from tool handler thread.

        Args:
            request_id: Unique ID for this permission request.
            action: Short action name (e.g., 'install_package').
            description: Human-readable description.
            command: Exact command to execute (shown to user).
            timeout: Max seconds to wait (default 5 minutes).

        Returns:
            {"approved": bool, "reason": str}
        """
        gate = {"event": threading.Event(), "approved": None, "reason": ""}
        with self._permission_lock:
            self._permission_gate[request_id] = gate

        # Emit SSE event to frontend
        self._event_bus.emit(
            PERMISSION_REQUEST,
            level="info",
            msg=f"[Permission] Requesting: {action}",
            data={
                "request_id": request_id,
                "action": action,
                "description": description,
                "command": command,
            },
        )

        # Block until user responds or timeout
        gate["event"].wait(timeout=timeout)

        with self._permission_lock:
            self._permission_gate.pop(request_id, None)

        if gate["approved"] is None:
            return {"approved": False, "reason": "timeout"}
        return {"approved": gate["approved"], "reason": gate.get("reason", "")}

    def set_permission_response(self, request_id: str, approved: bool) -> bool:
        """Unblock a pending permission request. Called from API layer.

        Returns True if the request_id was found and unblocked.
        """
        with self._permission_lock:
            gate = self._permission_gate.get(request_id)
        if gate is None:
            return False
        gate["approved"] = approved
        gate["reason"] = "approved" if approved else "denied"
        gate["event"].set()
        return True

    # ---- Turnless orchestrator: input queue and control center tools ----

    def push_input(self, message: str) -> None:
        """Push a user message into the input queue. Called by API layer.

        Non-blocking — always succeeds. The message will be processed as
        a separate turn after the current turn completes.

        Delivers the message to the agent inbox for processing.
        """
        self._put_message(_make_message("user_input", "user", message), priority=0)
        self._event_bus.emit(
            USER_AMENDMENT,
            level="info",
            msg=f"[Turnless] User input queued ({len(message)} chars)",
            data={"message_preview": trunc(message, "history.error.short")},
        )

    # ---- Persistent event loop (turnless orchestrator) ----

    def run_loop(self) -> None:
        """Persistent event loop — runs for the session's lifetime in a daemon thread.

        Uses the agent inbox (queue.Queue) for message delivery.
        """
        self._run_loop_actor()

    def _run_loop_actor(self) -> None:
        """Agent-based event loop: reads from inbox queue."""
        self._event_bus.emit(DEBUG, level="info", msg="[RunLoop] Started (agent mode)")
        self._set_state(AgentState.SLEEPING, reason="run loop started")

        while not self._shutdown_event.is_set():
            msg = self._get_message(timeout=1.0)
            if msg is None:
                continue

            if self._shutdown_event.is_set():
                break

            if msg.type == "user_input":
                # Track cycle: SLEEPING → ACTIVE starts a new cycle
                if self._state == AgentState.SLEEPING:
                    self._cycle_number += 1
                    self._responded_this_cycle = False  # Reset for new cycle
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
                    self._responded_this_cycle = True
                    if self._text_already_streamed:
                        # Text was already emitted incrementally via streaming
                        self._text_already_streamed = False
                    else:
                        self._event_bus.emit(
                            TEXT_DELTA,
                            level="info",
                            msg=response_text,
                            data={
                                "text": response_text,
                                "generated": getattr(
                                    self, "_last_response_generated", False
                                ),
                            },
                        )
                except Exception as e:
                    self._text_already_streamed = False
                    self.logger.error(f"[RunLoop] Error: {e}", exc_info=True)
                    self._event_bus.emit(
                        DEBUG,
                        level="error",
                        msg=f"[RunLoop] Error processing message: {e}",
                    )
                    # Emit CYCLE_END so the frontend clears isStreaming.
                    # Without this, the UI stays permanently disabled after an error.
                    self._event_bus.emit(
                        CYCLE_END,
                        level="info",
                        msg="Cycle complete (error recovery)",
                        data={
                            "cycle": self._cycle_number,
                            "turns_in_cycle": 0,
                            "token_usage": self.get_token_usage(),
                            "round_token_usage": {},
                        },
                    )
                    self._responded_this_cycle = True  # Prevent infinite wait

                # Drain any additional queued messages and merge user messages
                queued_user_messages = []
                while not self._shutdown_event.is_set():
                    extra = self._get_message_nowait()
                    if extra is None:
                        break
                    if extra.type == "user_input":
                        # Collect user messages to merge
                        content = extra.content if isinstance(extra.content, str) else str(extra.content)
                        queued_user_messages.append(content)
                    else:
                        # Non-user message (e.g. subagent_result) — put it back
                        # so the next iteration of the main loop processes it.
                        self._put_message(extra, priority=1)
                        break

                # Process merged user messages if any
                if queued_user_messages:
                    # Emit round boundary once for merged messages
                    mid_tokens = self.get_token_usage()
                    mid_start = self._round_start_tokens or {}
                    mid_delta = {
                        k: mid_tokens.get(k, 0) - mid_start.get(k, 0)
                        for k in mid_tokens
                    }
                    self._event_bus.emit(
                        CYCLE_END,
                        level="info",
                        msg="Round complete (merged queued messages)",
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
                        msg="Round started (merged queued messages)",
                        data={"cycle": self._cycle_number},
                    )

                    # Merge all user messages into one
                    merged_message = "\n\n---\n\n".join(queued_user_messages)
                    try:
                        response_text = self.process_message(merged_message)
                        self._turn_number += 1
                        turns_in_cycle += len(queued_user_messages)
                        # Mark that we responded (for cycle-end check) - Task 2 fix
                        self._responded_this_cycle = True
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
                        self._event_bus.emit(
                            CYCLE_END,
                            level="info",
                            msg="Cycle complete (error recovery)",
                            data={
                                "cycle": self._cycle_number,
                                "turns_in_cycle": 0,
                                "token_usage": self.get_token_usage(),
                                "round_token_usage": {},
                            },
                        )
                        self._responded_this_cycle = True

                # Check if cycle should end: responded + all work subagents idle
                if self._responded_this_cycle and self._all_work_subagents_idle():
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
                    # Check for memory hot reload
                    if MEMORY_RELOAD_INTERVAL > 0:
                        self._rounds_since_last_reload += 1
                        if self._rounds_since_last_reload >= MEMORY_RELOAD_INTERVAL:
                            self._trigger_memory_hot_reload()
                            self._rounds_since_last_reload = 0
                    # Trigger memory extraction at cycle end
                    self._maybe_extract_memories()
                    # Trigger Eureka discovery at cycle end
                    if config.get("eureka_enabled", True):
                        self._maybe_extract_eurekas()
                    self._set_state(
                        AgentState.SLEEPING,
                        reason=f"cycle {self._cycle_number} complete",
                    )
            elif msg.type == "subagent_result":
                # Hold fire-and-forget results during cancel
                if self._cancel_holdback.is_set():
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    with self._held_results_lock:
                        self._held_results.append(
                            f"[Subagent {msg.sender} completed]: {content}"
                        )
                    self._event_bus.emit(
                        DEBUG, level="info",
                        msg=f"[Cancel] Holding subagent result from {msg.sender}",
                    )
                    continue

                # Subagent sent result - process as new message
                self._set_state(AgentState.ACTIVE, reason="subagent result")

                # Initialize turns_in_cycle for cycle-end check (Task 1 fix)
                turns_in_cycle = 0

                # CYCLE START for subagent result round (Task 3 fix)
                self._round_start_tokens = self.get_token_usage()
                self._event_bus.emit(
                    CYCLE_START,
                    level="info",
                    msg="Cycle started (subagent result)",
                    data={"cycle": self._cycle_number},
                )

                # Extract content - subagent explains what was done + recommendations
                content = msg.content if isinstance(msg.content, str) else str(msg.content)

                # Process like any other message - triggers LLM call
                try:
                    response_text = self.process_message(
                        f"[Subagent {msg.sender} completed]: {content}"
                    )
                    self._responded_this_cycle = True
                    # Emit response
                    self._event_bus.emit(
                        TEXT_DELTA,
                        level="info",
                        msg=response_text,
                        data={"text": response_text, "source": "subagent"},
                    )
                    # Emit DELEGATION_ASYNC_COMPLETED only for async delegations
                    sender_id = msg.sender
                    if sender_id in self._async_delegations:
                        # Calculate duration from stored start time
                        start_time = self._async_delegations.get(sender_id)
                        duration = time.time() - start_time if start_time else 0
                        self._event_bus.emit(
                            DELEGATION_ASYNC_COMPLETED,
                            level="info",
                            msg=f"[Router] Background task completed: {sender_id}",
                            data={"agent": sender_id, "duration_seconds": round(duration, 2)},
                        )
                    # Mark any RUNNING WorkUnit for this agent as completed (C4 fix)
                    cc = self._control_center
                    for active in cc.list_active():
                        if active.get("agent_name") == sender_id:
                            cc.mark_completed(
                                active["id"],
                                {"status": "ok", "result": content[:500]},
                            )
                            break
                except Exception as e:
                    self.logger.error(f"Error processing subagent result: {e}", exc_info=True)

                # Check if cycle should end: responded + all work subagents idle
                if self._responded_this_cycle and self._all_work_subagents_idle():
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
                    # Check for memory hot reload
                    if MEMORY_RELOAD_INTERVAL > 0:
                        self._rounds_since_last_reload += 1
                        if self._rounds_since_last_reload >= MEMORY_RELOAD_INTERVAL:
                            self._trigger_memory_hot_reload()
                            self._rounds_since_last_reload = 0
                    # Trigger memory extraction at cycle end
                    self._maybe_extract_memories()
                    # Trigger Eureka discovery at cycle end
                    if config.get("eureka_enabled", True):
                        self._maybe_extract_eurekas()
                    self._set_state(
                        AgentState.SLEEPING,
                        reason=f"cycle {self._cycle_number} complete",
                    )

        self._event_bus.emit(DEBUG, level="info", msg="[RunLoop] Stopped (agent mode)")

    def shutdown_loop(self) -> None:
        """Signal the persistent event loop to stop."""
        self._shutdown_event.set()
        # Wake agent loop by putting a sentinel
        if hasattr(self, "_inbox"):
            self._put_message(_make_message("cancel", "system", "shutdown"), priority=0)

    def _trigger_memory_hot_reload(self) -> None:
        """Hot reload: restart chat session with fresh LTM injection.

        Called every N rounds to refresh the LLM context with latest memories.
        """
        if not self.chat:
            self.logger.debug("[Memory] No chat session to reload")
            return

        try:
            # Get canonical interface (works across all providers)
            interface = self.chat.interface
            self.logger.info(
                f"[Memory] Hot reload: preserving {len(interface.entries)} interface entries"
            )

            # Get fresh memory section
            memory_section = self._memory_store.format_for_injection(
                scope="generic", include_review_instruction=False
            )

            # Build new system prompt with fresh memory
            base_prompt = get_system_prompt()
            if memory_section:
                new_system_prompt = f"{base_prompt}\n\n{memory_section}"
            else:
                new_system_prompt = base_prompt

            # Update system prompt in interface before creating new session
            interface.add_system(new_system_prompt)

            # Create new chat session with canonical interface
            self.chat = self.adapter.create_chat(
                model=self.model_name,
                system_prompt=new_system_prompt,
                tools=self._tool_schemas,
                interface=interface,
                thinking="high",
            )

            self._event_bus.emit(
                DEBUG,
                level="info",
                msg=f"[Memory] Hot reload complete: {len(interface.entries)} entries, memory injected",
            )
        except Exception as e:
            self.logger.error(f"[Memory] Hot reload failed: {e}")
            self._event_bus.emit(
                DEBUG,
                level="error",
                msg=f"[Memory] Hot reload failed: {e}",
            )

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
            data={"tool_call_id": tool_call_id, "tool_name": tool_name, "tool_args": tool_args},
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
                    "tool_call_id": tool_call_id,
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

    def _execute_tool_for_agent(
        self,
        tool_name: str,
        tool_args: dict,
        tool_call_id: str | None = None,
        *,
        agent_type: str = "orchestrator",
    ) -> dict:
        """Event-free tool executor for agent sub-agents.

        Identical to _execute_tool_safe but emits NO events. Agents own their
        event lifecycle and emit SUB_AGENT_TOOL / SUB_AGENT_ERROR themselves.
        """
        prev_agent_type = self._current_agent_type
        self._current_agent_type = agent_type
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
        finally:
            self._current_agent_type = prev_agent_type

    def _execute_tool_for_agent_with_sandbox(
        self,
        tool_name: str,
        tool_args: dict,
        tool_call_id: str | None = None,
        *,
        agent_type: str = "",
        sandbox_imports: list[dict] | None = None,
    ) -> dict:
        """Execute a tool for an agent, with optional sandbox imports injection."""
        prev_imports = self._current_envoy_sandbox_imports
        self._current_envoy_sandbox_imports = sandbox_imports
        try:
            return self._execute_tool_for_agent(
                tool_name, tool_args, tool_call_id, agent_type=agent_type
            )
        finally:
            self._current_envoy_sandbox_imports = prev_imports

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
                model=self.model_name,
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

    def _put_message(self, msg: "Message", priority: int = 1):
        """Put message with priority. Lower number = higher priority."""
        self._inbox.put((priority, msg.timestamp, msg))

    def _get_message(self, timeout: float = 1.0):
        """Get message, respecting priority (user_input first)."""
        try:
            priority, timestamp, msg = self._inbox.get(timeout=timeout)
            return msg
        except queue.Empty:
            return None

    def _get_message_nowait(self):
        """Get message without blocking, respecting priority."""
        try:
            priority, timestamp, msg = self._inbox.get_nowait()
            return msg
        except queue.Empty:
            return None

    def _get_or_create_planner_agent(self):
        """Get the cached planner agent or create a new one. Thread-safe."""
        agent_id = "PlannerAgent"
        with self._sub_agents_lock:
            if agent_id not in self._sub_agents:
                from .planner import PlannerAgent
                agent = PlannerAgent(
                    adapter=self.adapter,
                    model_name=config.PLANNER_MODEL,
                    tool_executor=lambda name, args, tc_id=None: (
                        self._execute_tool_for_agent(
                            name, args, tc_id, agent_type="planner"
                        )
                    ),
                    event_bus=self._event_bus,
                    cancel_event=self._cancel_event,
                    memory_store=self._memory_store,
                    memory_scope="planner",
                    session_id=self._session_id,
                )
                agent._orchestrator_inbox = self._inbox
                agent.start()
                self._sub_agents[agent_id] = agent
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Router] Created PlannerAgent ({config.PLANNER_MODEL})",
                )
            return self._sub_agents[agent_id]

    def _process_single_message(self, user_message: str) -> str:
        """Process a single (non-complex) user message.

        Uses the Control Center for async delegation tracking.
        """
        self._insight_review_iter = 0
        self._latest_render_png = None
        self._event_bus.emit(
            DEBUG, level="debug", msg="[LLM] Sending message to model..."
        )
        self._last_tool_context = "initial_message"
        response = self._send_message_streaming(user_message)
        self._track_usage(response)
        self._emit_intermediate_text(response)

        self._event_bus.emit(DEBUG, level="debug", msg="[LLM] Response received.")
        self._log_grounding_queries(response)

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
                return "Request cancelled."

            if not response.tool_calls:
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

                if config.OBSERVATION_SUMMARIES and result.get("status") != "blocked":
                    from .observations import generate_observation

                    result["observation"] = generate_observation(
                        tool_name, tool_args, result
                    )

            # Append reflection hint when ALL tools in a round failed
            all_results = [r for _, _, _, r in tool_results]
            if config.SELF_REFLECTION and len(all_results) > 0:
                all_errors = all(r.get("status") == "error" for r in all_results)
                if all_errors:
                    last_result = all_results[-1]
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
            clarification_question = None  # Store question if clarification_needed
            for tc_id, tool_name, tool_args, result in tool_results:
                if result.get("status") == "clarification_needed":
                    # ALWAYS send tool result to LLM first (fixes history desync)
                    function_responses.append(
                        self.adapter.make_tool_result_message(
                            tool_name, result, tool_call_id=tc_id
                        )
                    )

                    # Build the question text for display
                    question = result["question"]
                    if result.get("context"):
                        question = f"{result['context']}\n\n{question}"
                    if result.get("options"):
                        question += "\n\nOptions:\n" + "\n".join(
                            f"  {i + 1}. {opt}"
                            for i, opt in enumerate(result["options"])
                        )
                    # Pair the tool_use with a tool_result to keep history in sync
                    clarification_response = self.adapter.make_tool_result_message(
                        tool_name, result, tool_call_id=tc_id
                    )
                    if self.chat and hasattr(self.chat, 'commit_tool_results'):
                        self.chat.commit_tool_results([clarification_response])
                    return question

                function_responses.append(
                    self.adapter.make_tool_result_message(
                        tool_name, result, tool_call_id=tc_id
                    )
                )

            guard.record_calls(len(function_calls))

            tool_names = [fc.name for fc in function_calls]
            self._last_tool_context = "+".join(tool_names)

            # Check cancel AFTER tool execution — strip the incomplete
            # assistant turn. Cancel context is prepended to the next user message.
            if self._cancel_event.is_set():
                if self._chat:
                    self._chat.rollback_last_turn()
                self._event_bus.emit(
                    DEBUG, level="info", msg="[Cancel] Stopping after tool execution"
                )
                return "Cancelled."

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

            # If this was a clarification_needed, display our built question
            # instead of the LLM's generic acknowledgment
            if clarification_question:
                return clarification_question

        # Extract text response
        text = response.text
        self._last_response_generated = not text
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

    def _get_active_envoy_ids(self) -> set[str]:
        """Return set of active mission IDs from current actors. Thread-safe."""
        with self._sub_agents_lock:
            return {
                k.removeprefix("EnvoyAgent[").rstrip("]")
                for k in self._sub_agents
                if k.startswith("EnvoyAgent[")
            }

    def _on_memory_mutated(self, event: SessionEvent) -> None:
        """Event bus listener: refresh core memory when long-term memory changes."""
        if event.type != MEMORY_EXTRACTION_DONE:
            return
        memory_section = self._memory_store.format_for_injection(
            scope="generic", include_review_instruction=False
        )
        base_prompt = get_system_prompt()
        if memory_section:
            self._system_prompt = f"{base_prompt}\n\n{memory_section}"
        else:
            self._system_prompt = base_prompt
        if self.chat is not None:
            self.chat.update_system_prompt(self._system_prompt)

    @staticmethod
    def _wrap_delegation_result(sub_result, store_snapshot=None) -> dict:
        """Convert an agent send result into a tool result dict.

        Success is determined by actual output, not by error heuristics.
        If the agent produced meaningful output (text or files), it's a success
        even if there were transient errors during retries. The LLM sees both
        the result and any errors in the text.

        Args:
            sub_result: Dict from agent's _handle_request ({text, failed, errors}).
            store_snapshot: Optional list of store entry summaries to include,
                so the orchestrator LLM sees concrete data state after delegation.
        """
        if isinstance(sub_result, dict):
            text = sub_result.get("text", "")
            failed = sub_result.get("failed", False)
            errors = sub_result.get("errors", [])
            output_files = sub_result.get("output_files", [])
        else:
            # Legacy: plain string (shouldn't happen, but be safe)
            text = str(sub_result)
            failed = False
            errors = []
            output_files = []

        # Check for actual success: has meaningful output (text or output_files)
        # Even if there were errors during retries, if we got output, it's a success
        has_output = bool(text.strip()) or bool(output_files)
        has_critical_errors = failed and errors

        if has_critical_errors and not has_output:
            # Failed with no output - true failure
            error_summary = "; ".join(errors[-get_item_limit("items.error_summary") :])
            result = {
                "status": "error",
                "message": f"Sub-agent failed. Errors: {error_summary}",
                "result": text,
            }
        else:
            # Has output (possibly with some errors during retries) - success
            # The LLM will see both the result and any errors in the text
            result = {"status": "success", "result": text}

        if output_files:
            result["output_files"] = output_files

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
        # Stop and clear agents
        with self._sub_agents_lock:
            for agent in self._sub_agents.values():
                agent.stop(timeout=2.0)
            self._sub_agents.clear()
        # Clear direct references so _ensure_*_agent() recreates them
        self._memory_agent = None
        self._eureka_agent = None
        ENVOY_TOOL_REGISTRY.clear_active()
        # Reset context tracker so recreated agents get full context on first use
        self._ctx_tracker.reset_all()
        self._event_bus.emit(
            DEBUG,
            level="debug",
            msg="[Config] Sub-agents invalidated after config reload",
        )

    # ---- Agent-based sub-agent management ----

    def _get_or_create_envoy_agent(self, mission_id: str) -> EnvoyAgent:
        """Get the persistent envoy agent, creating it on first use."""
        agent_id = f"EnvoyAgent[{mission_id}]"
        with self._sub_agents_lock:
            if agent_id not in self._sub_agents:
                # Load sandbox config for package envoys
                sandbox_config = None
                from knowledge.mission_loader import load_mission as _load_mission
                mission_data = _load_mission(mission_id.lower())
                if mission_data and mission_data.get("type") == "package":
                    sandbox_config = mission_data.get("sandbox")

                agent = EnvoyAgent(
                    mission_id=mission_id,
                    adapter=self.adapter,
                    model_name=config.SUB_AGENT_MODEL,
                    tool_executor=lambda name, args, tc_id=None, _sc=sandbox_config: (
                        self._execute_tool_for_agent_with_sandbox(
                            name, args, tc_id, agent_type="envoy",
                            sandbox_imports=_sc.get("imports") if _sc else None,
                        )
                    ),
                    event_bus=self._event_bus,
                    memory_store=self._memory_store,
                    memory_scope=f"envoy:{mission_id}",
                    cancel_event=self._cancel_event,
                    sandbox_config=sandbox_config,
                )
                agent._orchestrator_inbox = self._inbox
                agent.start()
                self._sub_agents[agent_id] = agent
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Router] Created {mission_id} envoy agent ({config.SUB_AGENT_MODEL})",
                )
                if ENVOY_TOOL_REGISTRY.mark_active(mission_id):
                    from .agent_registry import AGENT_INFORMED_REGISTRY

                    for tool_name in ENVOY_TOOL_REGISTRY.get_tools(mission_id):
                        AGENT_INFORMED_REGISTRY._registry.setdefault(
                            "ctx:orchestrator", set()
                        ).add(tool_name)
            return self._sub_agents[agent_id]

    def _create_ephemeral_envoy_agent(self, mission_id: str) -> EnvoyAgent:
        """Create an ephemeral overflow envoy agent for parallel delegation."""
        with self._sub_agents_lock:
            seq = self._mission_seq
            self._mission_seq = seq + 1
            ephemeral_id = f"EnvoyAgent[{mission_id}]#{seq}"

            # Load sandbox config for package envoys
            sandbox_config = None
            from knowledge.mission_loader import load_mission as _load_mission
            mission_data = _load_mission(mission_id.lower())
            if mission_data and mission_data.get("type") == "package":
                sandbox_config = mission_data.get("sandbox")

            agent = EnvoyAgent(
                mission_id=mission_id,
                adapter=self.adapter,
                model_name=config.SUB_AGENT_MODEL,
                tool_executor=lambda name, args, tc_id=None, _sc=sandbox_config: (
                    self._execute_tool_for_agent_with_sandbox(
                        name, args, tc_id, agent_type="envoy",
                        sandbox_imports=_sc.get("imports") if _sc else None,
                    )
                ),
                agent_id=ephemeral_id,
                event_bus=self._event_bus,
                memory_store=self._memory_store,
                memory_scope=f"envoy:{mission_id}",
                cancel_event=self._cancel_event,
                sandbox_config=sandbox_config,
            )
            agent._orchestrator_inbox = self._inbox
            agent.start()
            self._sub_agents[ephemeral_id] = agent
        self._event_bus.emit(
            DEBUG,
            level="debug",
            msg=f"[Router] Created ephemeral envoy agent {ephemeral_id}",
        )
        return agent

    def _get_or_create_viz_plotly_agent(self) -> VizPlotlyAgent:
        """Get the cached Plotly viz agent or create a new one. Thread-safe."""
        agent_id = "VizAgent[Plotly]"
        with self._sub_agents_lock:
            if agent_id not in self._sub_agents:
                agent = VizPlotlyAgent(
                    adapter=self.adapter,
                    model_name=config.SUB_AGENT_MODEL,
                    tool_executor=lambda name, args, tc_id=None: (
                        self._execute_tool_for_agent(
                            name, args, tc_id, agent_type="viz_plotly"
                        )
                    ),
                    gui_mode=self.gui_mode,
                    event_bus=self._event_bus,
                    memory_store=self._memory_store,
                    memory_scope="visualization",
                    cancel_event=self._cancel_event,
                )
                agent._orchestrator_inbox = self._inbox
                agent.start()
                self._sub_agents[agent_id] = agent
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Router] Created Plotly Visualization agent ({config.SUB_AGENT_MODEL})",
                )
            return self._sub_agents[agent_id]

    def _get_or_create_viz_mpl_agent(self) -> VizMplAgent:
        """Get the cached MPL viz agent or create a new one. Thread-safe."""
        agent_id = "VizAgent[Mpl]"
        with self._sub_agents_lock:
            if agent_id not in self._sub_agents:
                session_dir = self._session_manager.base_dir / self._session_id
                agent = VizMplAgent(
                    adapter=self.adapter,
                    model_name=config.SUB_AGENT_MODEL,
                    tool_executor=lambda name, args, tc_id=None: (
                        self._execute_tool_for_agent(
                            name, args, tc_id, agent_type="viz_mpl"
                        )
                    ),
                    gui_mode=self.gui_mode,
                    event_bus=self._event_bus,
                    memory_store=self._memory_store,
                    memory_scope="visualization",
                    session_dir=session_dir,
                    cancel_event=self._cancel_event,
                )
                agent._orchestrator_inbox = self._inbox
                agent.start()
                self._sub_agents[agent_id] = agent
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Router] Created MPL Visualization agent ({config.SUB_AGENT_MODEL})",
                )
            return self._sub_agents[agent_id]

    def _get_or_create_viz_jsx_agent(self) -> VizJsxAgent:
        """Get the cached JSX viz agent or create a new one. Thread-safe."""
        agent_id = "VizAgent[JSX]"
        with self._sub_agents_lock:
            if agent_id not in self._sub_agents:
                session_dir = self._session_manager.base_dir / self._session_id
                agent = VizJsxAgent(
                    adapter=self.adapter,
                    model_name=config.SUB_AGENT_MODEL,
                    tool_executor=lambda name, args, tc_id=None: (
                        self._execute_tool_for_agent(
                            name, args, tc_id, agent_type="viz_jsx"
                        )
                    ),
                    gui_mode=self.gui_mode,
                    event_bus=self._event_bus,
                    memory_store=self._memory_store,
                    memory_scope="visualization",
                    session_dir=session_dir,
                    cancel_event=self._cancel_event,
                )
                agent._orchestrator_inbox = self._inbox
                agent.start()
                self._sub_agents[agent_id] = agent
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Router] Created JSX Visualization agent ({config.SUB_AGENT_MODEL})",
                )
            return self._sub_agents[agent_id]

    def _get_available_dataops_agent(self) -> DataOpsAgent:
        """Get an idle DataOps agent or create a new ephemeral one.

        Priority: (1) idle primary agent, (2) create primary if it doesn't
        exist, (3) create an ephemeral overflow instance.  Ephemeral agents
        are cleaned up after their delegation completes.
        """
        primary_id = "DataOpsAgent"
        with self._sub_agents_lock:
            if primary_id in self._sub_agents:
                agent = self._sub_agents[primary_id]
                if agent.state == AgentState.SLEEPING and agent.inbox.qsize() == 0:
                    return agent
            else:
                # Create the primary (persistent) agent
                agent = DataOpsAgent(
                    adapter=self.adapter,
                    model_name=config.SUB_AGENT_MODEL,
                    tool_executor=lambda name, args, tc_id=None: (
                        self._execute_tool_for_agent(
                            name, args, tc_id, agent_type="dataops"
                        )
                    ),
                    event_bus=self._event_bus,
                    memory_store=self._memory_store,
                    memory_scope="data_ops",
                    active_missions_fn=self._get_active_envoy_ids,
                    cancel_event=self._cancel_event,
                )
                agent._orchestrator_inbox = self._inbox
                agent.start()
                self._sub_agents[primary_id] = agent
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg="[Router] Created DataOps agent",
                )
                return agent

            # Primary is busy — create an ephemeral overflow instance
            seq = self._dataops_seq
            self._dataops_seq = seq + 1
            ephemeral_id = f"DataOpsAgent#{seq}"
            agent = DataOpsAgent(
                adapter=self.adapter,
                model_name=config.SUB_AGENT_MODEL,
                tool_executor=lambda name, args, tc_id=None: (
                    self._execute_tool_for_agent(
                        name, args, tc_id, agent_type="dataops"
                    )
                ),
                agent_id=ephemeral_id,
                event_bus=self._event_bus,
                memory_store=self._memory_store,
                memory_scope="data_ops",
                active_missions_fn=self._get_active_envoy_ids,
                cancel_event=self._cancel_event,
            )
            agent._orchestrator_inbox = self._inbox
            agent.start()
            self._sub_agents[ephemeral_id] = agent
            self._event_bus.emit(
                DEBUG,
                level="debug",
                msg=f"[Router] Created ephemeral DataOps agent {ephemeral_id}",
            )
            return agent

    def _cleanup_ephemeral_agent(self, agent_id: str) -> None:
        """Shut down and remove an ephemeral agent from the registry.

        Preserves the agent's token usage in ``_retired_agent_usage`` so it
        is still included in ``get_token_usage()`` and
        ``get_token_usage_breakdown()`` after the agent is removed.
        """
        with self._sub_agents_lock:
            agent = self._sub_agents.pop(agent_id, None)
        if agent:
            # Preserve token usage before stopping the agent
            usage = agent.get_token_usage()
            if usage.get("api_calls", 0) > 0:
                self._retired_agent_usage.append(
                    {
                        "agent": agent_id,
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
            agent.stop()
            self._ctx_tracker.reset(agent_id)
            self._event_bus.emit(
                DEBUG,
                level="debug",
                msg=f"[Router] Cleaned up ephemeral agent {agent_id}",
            )

    def _get_or_create_data_io_agent(self) -> DataIOAgent:
        """Get the cached data I/O agent or create a new one. Thread-safe."""
        agent_id = "DataIOAgent"
        with self._sub_agents_lock:
            if agent_id not in self._sub_agents:
                agent = DataIOAgent(
                    adapter=self.adapter,
                    model_name=config.SUB_AGENT_MODEL,
                    tool_executor=lambda name, args, tc_id=None: (
                        self._execute_tool_for_agent(
                            name, args, tc_id, agent_type="data_io"
                        )
                    ),
                    event_bus=self._event_bus,
                    cancel_event=self._cancel_event,
                )
                agent._orchestrator_inbox = self._inbox
                agent.start()
                self._sub_agents[agent_id] = agent
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg="[Router] Created DataIO agent",
                )
            return self._sub_agents[agent_id]

    def _get_or_create_insight_agent(self) -> InsightAgent:
        """Get the cached insight agent or create a new one. Thread-safe."""
        agent_id = "InsightAgent"
        with self._sub_agents_lock:
            if agent_id not in self._sub_agents:
                agent = InsightAgent(
                    adapter=self.adapter,
                    model_name=config.INSIGHT_MODEL,
                    tool_executor=lambda name, args, tc_id=None: (
                        self._execute_tool_for_agent(
                            name, args, tc_id, agent_type="viz_plotly"
                        )
                    ),
                    event_bus=self._event_bus,
                    cancel_event=self._cancel_event,
                )
                agent._orchestrator_inbox = self._inbox
                agent.start()
                self._sub_agents[agent_id] = agent
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Router] Created Insight agent ({config.INSIGHT_MODEL})",
                )
            return self._sub_agents[agent_id]

    def _delegate_to_sub_agent(
        self,
        agent: SubAgent,
        request,
        timeout: float = 300.0,
        wait: bool = True,
        store_snapshot=None,
        tool_call_id: str | None = None,
        agent_type: str = "",
        agent_name: str = "",
        task_summary: str = "",
        post_process: Callable | None = None,
        post_complete: Callable | None = None,
    ) -> dict:
        """Dispatch delegation synchronously — blocks until the sub-agent finishes.

        Called from ``execute_tools_batch``'s ThreadPoolExecutor worker threads,
        so multiple delegations in the same LLM turn still run in parallel.
        Results are returned directly with proper tool_call_id pairing (no
        stale IDs, no pending_async).

        Args:
            agent: The target Agent instance.
            request: String or dict payload for the agent.
            timeout: Max seconds to wait.
            store_snapshot: Optional list of store entries to include in result.
            tool_call_id: LLM tool_call_id for result mapping (unused now but
                kept for API compatibility with callers).
            agent_type: Agent type for ControlCenter tracking.
            agent_name: Agent name for ControlCenter tracking.
            task_summary: Human-readable summary for ControlCenter.
            post_process: Optional callable(result) -> result to run after
                delegation completes.
            post_complete: Optional callable(result) to run after processing.
                For non-critical work (e.g. PNG export).
        """
        cc = self._control_center
        summary = task_summary or (
            request[:200] if isinstance(request, str) else str(request)[:200]
        )
        # Capture the full request for observability
        request_str = request if isinstance(request, str) else str(request)
        unit = cc.register(
            kind="delegation",
            agent_type=agent_type,
            agent_name=agent_name or agent.agent_id,
            task_summary=summary,
            request=request_str,
            tool_call_id=tool_call_id,
        )

        # Capture operation log index before the delegation starts so we can
        # collect only the operations produced by this delegation.
        ops_log = self._ops_log
        ops_start_index = len(ops_log.get_records())

        # Handle fire-and-forget delegation — start agent but don't wait
        if not wait:
            agent.send(request, sender="orchestrator", timeout=timeout, wait=False)

            # Track as async delegation for completion notification
            self._async_delegations[agent_name or agent.agent_id] = time.time()

            cc.mark_completed(
                unit.id,
                {
                    "status": "queued",
                    "message": f"Delegation to {agent_name or agent.agent_id} started (fire-and-forget)",
                },
            )
            return {
                "status": "queued",
                "message": f"Delegation to {agent_name or agent.agent_id} started (fire-and-forget)",
            }

        try:
            result = agent.send(
                request, sender="orchestrator", timeout=timeout, wait=wait
            )

            wrapped = self._wrap_delegation_result(
                result, store_snapshot=store_snapshot
            )
            if post_process is not None:
                wrapped = post_process(wrapped)

            # Build operation log from records added during this delegation
            all_records = ops_log.get_records()
            operation_log = all_records[ops_start_index:]

            cc.mark_completed(unit.id, wrapped, operation_log=operation_log)

            # Fire-and-forget callback after marking complete
            if post_complete is not None:
                try:
                    post_complete(wrapped)
                except Exception as pc_err:
                    get_logger().debug(f"post_complete callback failed: {pc_err}")

            return wrapped
        except Exception as e:
            error_msg = str(e)
            cc.mark_failed(unit.id, error_msg)
            return {
                "status": "error",
                "message": f"Delegation failed: {error_msg}",
            }

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
        current_provider = self.service.provider
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

        # 1. Viz Backend Hot-Reload (System Prompt Rebuild only)
        if viz_backend_changed:
            self._event_bus.emit(
                DEBUG,
                level="info",
                msg=f"[Config] Hot-reload: Switching viz backend to {target_viz_backend}",
            )
            # Rebuild system prompt (it might have backend-specific instructions)
            self._system_prompt = get_system_prompt()

            # Update the current chat session's prompt if it exists
            if self.chat is not None:
                self.chat.update_system_prompt(self._system_prompt)

            self._viz_backend = target_viz_backend

        # 2. Extract canonical interface (works across all providers)
        interface = None
        if self.chat is not None:
            try:
                interface = self.chat.interface
            except Exception:
                pass

        # 2b. Rebuild service+adapter if provider/base_url changed
        if adapter_needs_rebuild:
            self.service = _create_llm_service()
            self.adapter = self.service.adapter
            self._event_bus.emit(
                DEBUG,
                level="info",
                msg=f"[Config] Adapter rebuilt: {current_provider} → {target_provider}",
            )

        # 3. Update model name
        self.model_name = target_model

        # 4. Recreate chat session with preserved interface
        try:
            self.chat = self.adapter.create_chat(
                model=self.model_name,
                system_prompt=self._system_prompt,
                tools=self._tool_schemas,
                thinking="high",
                interface=interface,
            )
        except Exception as exc:
            # Fall back to fresh chat if interface transfer fails
            self._event_bus.emit(
                DEBUG,
                level="warning",
                msg=f"[Config] Chat recreation with interface failed ({exc}), starting fresh",
            )
            self.chat = self.adapter.create_chat(
                model=self.model_name,
                system_prompt=self._system_prompt,
                tools=self._tool_schemas,
                thinking="high",
            )

        # 5. Clear all cached sub-agents and memory agent
        self.reset_sub_agents()
        self._memory_agent = None


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
        if not state.get("traces") and self._latest_render_png is not None:
            lines.append("Visualization: matplotlib (static PNG)")

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
        a synchronous review via agent.send(). Blocks until the
        review completes, then returns the result.

        Returns:
            The review result dict, or None if the review was skipped
            (disabled, iteration cap, no figure, etc.).
        """
        if not config.INSIGHT_FEEDBACK:
            return None

        if self._insight_review_iter >= config.INSIGHT_FEEDBACK_MAX_ITERS:
            return None

        image_bytes = self.get_latest_figure_png()
        if image_bytes is None:
            return None

        agent = self._get_or_create_insight_agent()
        data_context = self._build_insight_context()

        user_msgs = self._event_bus.get_events(types={USER_MESSAGE})
        user_request = (
            user_msgs[-1].data.get("text", user_msgs[-1].msg) if user_msgs else ""
        )

        self._insight_review_iter += 1

        review_instruction = (
            f'Review this figure for correctness and quality against the user\'s original request: "{user_request}"\n\n'
            "Check: (1) Does it show the requested datasets/parameters and time range? "
            "(2) Are axis labels, units, title, and legend correct? "
            "(3) Are traces distinguishable and scales appropriate? "
            "(4) Do value ranges look physically reasonable?\n\n"
            "Start your response with VERDICT: PASS or VERDICT: NEEDS_IMPROVEMENT, "
            "then explain. If NEEDS_IMPROVEMENT, list specific actionable suggestions as bullet points (max 5)."
        )
        result = agent.send(
            {
                "action": "review",
                "image_bytes": image_bytes,
                "data_context": data_context,
                "user_request": review_instruction,
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
                if key.startswith("EnvoyAgent["):
                    mission_id = key.removeprefix("EnvoyAgent[").rstrip("]")
                    active_scopes.append(f"envoy:{mission_id}")

        # Collect all console-tagged events (same log the user sees)
        console_events = self._event_bus.get_events(tags={"console"})

        # MemoryAgent reads ALL memories directly from the store in its
        # system prompt — no need to build active_memories here.
        return MemoryContext(
            console_events=console_events,
            active_scopes=active_scopes,
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
            self._memory_agent.start()
            with self._sub_agents_lock:
                self._sub_agents["MemoryAgent"] = self._memory_agent
        return self._memory_agent

    def _ensure_eureka_agent(self) -> "EurekaAgent":
        """Lazily create or return the existing EurekaAgent (SubAgent)."""
        if self._eureka_agent is None:
            from .eureka_agent import EurekaAgent

            eureka_model = config.get("eureka_model", None) or config.SMART_MODEL
            self._eureka_agent = EurekaAgent(
                adapter=self.adapter,
                model_name=eureka_model,
                tool_executor=lambda name, args, tc_id=None: (
                    self._execute_tool_for_agent(name, args, tc_id, agent_type="eureka")
                ),
                event_bus=self._event_bus,
                memory_store=self._memory_store,
                memory_scope="eureka",
                orchestrator_ref=self,
            )
            self._eureka_agent.start()
            with self._sub_agents_lock:
                self._sub_agents["EurekaAgent"] = self._eureka_agent
        return self._eureka_agent

    def _build_eureka_context(self) -> dict:
        """Build context dict for Eureka discovery."""
        user_msgs = self._event_bus.get_events(types={USER_MESSAGE})
        return {
            "session_id": self._session_id or "unknown",
            "data_store_keys": [e["label"] for e in self._store.list_entries()]
            if self._store
            else [],
            "has_figure": self._renderer.get_figure() is not None,
            "recent_messages": [m.msg for m in user_msgs[-5:]],
        }

    def _format_eureka_suggestion_as_user_msg(self, suggestion) -> str:
        """Convert a EurekaSuggestion into a natural-language user message."""
        parts = [f"[Eureka Mode] {suggestion.description}"]
        if suggestion.rationale:
            parts.append(f"Rationale: {suggestion.rationale}")
        if suggestion.parameters:
            import json as _json

            parts.append(f"Parameters: {_json.dumps(suggestion.parameters)}")
        return "\n".join(parts)

    def _maybe_extract_eurekas(self) -> None:
        """Trigger async Eureka extraction on a daemon thread.

        Uses EurekaAgent.send() via the SubAgent inbox pattern.
        Lock prevents concurrent extractions. After the agent returns,
        if Eureka Mode is ON, queues the top suggestion for execution.
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
                # Build the context message for the SubAgent
                msg_content = agent.build_context_message(context)
                result = agent.send(
                    msg_content, sender="orchestrator", timeout=120.0
                )

                if result.get("failed"):
                    bus.emit(
                        EUREKA_EXTRACTION_ERROR,
                        agent="Eureka",
                        level="warning",
                        data={"error": result.get("text", "unknown error")},
                    )
                else:
                    # Count findings from the store for this session
                    session_id = context.get("session_id", "unknown")
                    findings = agent.eureka_store.list(session_id=session_id)
                    suggestions = agent.eureka_store.list_suggestions(
                        session_id=session_id, status="proposed"
                    )
                    bus.emit(
                        EUREKA_EXTRACTION_DONE,
                        agent="Eureka",
                        level="info",
                        data={
                            "n_findings": len(findings),
                            "n_suggestions": len(suggestions),
                        },
                    )

                    # If Eureka Mode is ON, inject the top suggestion as
                    # a synthetic user message into the orchestrator inbox
                    if self._eureka_mode and suggestions:
                        eureka_max_rounds = config.get("eureka_max_rounds", 5)
                        if self._eureka_round_counter >= eureka_max_rounds:
                            self._eureka_mode = False
                            self._eureka_round_counter = 0
                            bus.emit(
                                DEBUG,
                                agent="Eureka",
                                level="info",
                                msg=f"[Eureka Mode] Paused: reached max rounds ({eureka_max_rounds})",
                            )
                        else:
                            suggestion = suggestions[0]
                            self._eureka_pending_suggestion = suggestion
                            synthetic_msg = self._format_eureka_suggestion_as_user_msg(
                                suggestion
                            )
                            self._eureka_round_counter += 1
                            # Mark suggestion as approved
                            from .eureka_store import EurekaStore

                            EurekaStore().update_suggestion_status(
                                suggestion.id, "executed"
                            )
                            # Inject into orchestrator inbox if running in turnless mode
                            if hasattr(self, "_inbox") and self._inbox is not None:
                                from .sub_agent import _make_message

                                self._put_message(
                                    _make_message(
                                        "user_input", "eureka_mode", synthetic_msg
                                    ),
                                    priority=0,
                                )
                            else:
                                # Fallback: store for process_message to pick up
                                self._eureka_pending_suggestion = suggestion

            except Exception as e:
                self.logger.warning(f"Eureka extraction failed: {e}")
                bus.emit(
                    EUREKA_EXTRACTION_ERROR,
                    agent="Eureka",
                    level="warning",
                    data={"error": str(e)},
                )
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
        # Check if there are new console events since last extraction
        console_events = self._event_bus.get_events(
            tags={"console"}, since_index=self._last_memory_op_index
        )
        if not console_events:
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
                            "console_events": len(context.console_events),
                            "active_scopes": context.active_scopes,
                        },
                    )

                    # Dump memory feed for debugging
                    if session_id:
                        try:
                            from datetime import datetime as _dt, timezone as _tz

                            feed_dir = config.get_data_dir() / "sessions" / session_id
                            feed_dir.mkdir(parents=True, exist_ok=True)
                            feed_payload = {
                                "timestamp": _dt.now(_tz.utc).isoformat(),
                                "active_scopes": context.active_scopes,
                                "console_events_count": len(context.console_events),
                                "console_events": [
                                    {
                                        "index": i,
                                        "type": ev.type,
                                        "agent": ev.agent,
                                        "summary": ev.summary,
                                    }
                                    for i, ev in enumerate(context.console_events)
                                ],
                                "total_memory_tokens": context.total_memory_tokens,
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
                model=config.INLINE_MODEL,
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
                model=config.INLINE_MODEL,
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
                model=config.INLINE_MODEL,
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

        # Register user-defined package envoys into the 'package' group
        from .agent_registry import register_package_envoys
        register_package_envoys()

        self._event_bus.emit(
            DEBUG, level="debug", msg=f"[Session] Started: {self._session_id}"
        )
        return self._session_id

    def save_session(self) -> None:
        """Persist the current chat history and DataStore to disk."""
        if not self._session_id:
            return
        # Use interface as single source of truth
        try:
            interface_dict = self.chat.interface.to_dict()
        except Exception:
            interface_dict = []

        store = self._store
        usage = self.get_token_usage()

        # EventBus user messages — used for turn count and preview extraction
        # (original text, not augmented with injected context headers).
        bus_user_msgs = self._event_bus.get_events(types={USER_MESSAGE})

        # Turn count: prefer EventBus user messages (always available); fall
        # back to counting "user" roles in interface for sessions.
        # Interactions API sessions don't expose full history client-side, so
        # the EventBus count is the primary source.
        turn_count = (
            len(bus_user_msgs)
            if bus_user_msgs
            else sum(1 for e in interface_dict if e.get("role") == "user")
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
        # Fallback to interface history if EventBus had no text
        if not last_preview:
            for h in reversed(interface_dict):
                if h.get("role") == "user":
                    content = h.get("content", [])
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
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
            chat_history=interface_dict,
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
        self, session_id: str, *, skip_interaction_resume: bool = False
    ) -> tuple[dict, list[dict] | None, list[dict] | None]:
        """Restore chat history and DataStore from a saved session.

        Args:
            session_id: The session to load.
            skip_interaction_resume: Deprecated — kept for API compatibility.
                Sessions always create fresh LLM chats seeded with history.

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

        # Build session state dict for service.resume_session()
        if history_dicts:
            saved_state = {
                "session_id": session_id,
                "messages": history_dicts,
                "metadata": metadata,
            }
            try:
                self.chat = self.service.resume_session(saved_state)
                entry_count = len(self.chat.interface.entries) if hasattr(self.chat, 'interface') else 0
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Session] Resumed ({entry_count} entries)",
                )
            except Exception as e:
                self._event_bus.emit(
                    DEBUG,
                    level="warning",
                    msg=f"[Session] Resume failed: {e}. Starting fresh.",
                )
                self.chat = self.adapter.create_chat(
                    model=self.model_name,
                    system_prompt=self._system_prompt,
                    tools=self._tool_schemas,
                    thinking="high",
                )
        else:
            # No history — start fresh chat
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
            for agent in self._sub_agents.values():
                agent.stop(timeout=2.0)
            self._sub_agents.clear()
        ENVOY_TOOL_REGISTRY.clear_active()
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

        # Reset memory reload counter on session load
        self._rounds_since_last_reload = 0

        # Start writing structured event log (append mode — resumes keep adding)
        self._start_event_log_writer()

        # Register user-defined package envoys into the 'package' group
        from .agent_registry import register_package_envoys
        register_package_envoys()

        self._event_bus.emit(
            DEBUG, level="debug", msg=f"[Session] Loaded: {session_id}"
        )
        return metadata, display_log, event_log

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

    def get_latest_figure_png(self) -> bytes | None:
        """Return PNG bytes for the most recent figure, checking memory then disk.

        Lookup order:
        1. In-memory cache (_latest_render_png) -- fast path, works during active session
        2. Plotly renderer export -- works if figure is in memory (including deferred restore)
        3. Disk files (mpl_outputs/, plotly_outputs/) -- always works after reload
        """
        if self._latest_render_png is not None:
            return self._latest_render_png

        self._restore_deferred_figure()
        figure = self._renderer.get_figure()
        if figure is not None:
            import io

            try:
                buf = io.BytesIO()
                figure.write_image(buf, format="png", width=1100, height=600, scale=2)
                png_bytes = buf.getvalue()
                self._latest_render_png = png_bytes
                return png_bytes
            except Exception:
                pass

        session_dir = self._session_manager.base_dir / self._session_id
        latest_png = self._find_latest_png(session_dir)
        if latest_png is not None:
            try:
                png_bytes = latest_png.read_bytes()
                self._latest_render_png = png_bytes
                return png_bytes
            except Exception:
                pass

        return None

    @staticmethod
    def _find_latest_png(session_dir: Path) -> Path | None:
        """Find the most recently modified PNG file in the session output dirs."""
        png_dirs = ["mpl_outputs", "plotly_outputs"]
        latest: Path | None = None
        latest_mtime: float = 0
        for dirname in png_dirs:
            d = session_dir / dirname
            if not d.is_dir():
                continue
            for f in d.iterdir():
                if f.suffix == ".png" and f.stat().st_size > 0:
                    mtime = f.stat().st_mtime
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest = f
        return latest

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
            total = len(store._ids)
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

    def _handle_plan_check(self, tool_args: dict) -> dict:
        """Handler for the plan_check tool."""
        plan_file = tool_args.get("plan_file")
        if not plan_file:
            return {
                "status": "error",
                "message": "plan_file is required"
            }

        plan_path = Path(plan_file)
        if not plan_path.exists():
            return {
                "status": "error",
                "message": f"Plan file not found: {plan_file}"
            }

        with open(plan_path) as f:
            plan = json.load(f)

        return {
            "status": "success",
            "plan": plan,
            "task_count": len(plan.get("tasks", []))
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
            lines.append(
                f'  ({total - 15} more — use pipeline(action="search") to find)'
            )

        if config.PIPELINE_CONFIRMATION:
            lines.append("")
            lines.append(
                "IMPORTANT: Before running any pipeline, present the top matches to the "
                "user with descriptions and reasoning, then use ask_clarification to get "
                'explicit permission. Do NOT call pipeline(action="run") without user confirmation.'
            )
        else:
            lines.append("")
            lines.append(
                "If the user's request matches a saved pipeline, run it with "
                'pipeline(action="run") instead of building from scratch. '
                'Use pipeline(action="search") to find by mission, dataset, or description.'
            )
        lines.append("[END SAVED PIPELINES]")
        return "\n".join(lines)

    def process_message(self, user_message: str) -> str:
        """Process a user message and return the agent's response.

        All messages go through the orchestrator LLM, which decides whether to
        invoke the planner via the ``delegate_to_planner`` tool when multi-step
        coordination is needed.
        """
        self.clear_cancel()

        # Drain any held fire-and-forget results from the cancel period.
        # These are injected BEFORE the user's message so the LLM has context
        # but the user's intent takes priority (appears last).
        held_prefix = ""
        with self._held_results_lock:
            if self._held_results:
                logger.debug(
                    "Draining %d held results from cancel period",
                    len(self._held_results),
                )
                held_prefix = (
                    "[Background task results received during cancellation]\n"
                    + "\n".join(self._held_results)
                    + "\n\n"
                )
                self._held_results.clear()
            self._cancel_holdback.clear()

        # Prepend cancel context if the previous operation was cancelled
        cancel_prefix = ""
        if self._was_cancelled:
            cancel_prefix = self._CANCEL_CONTEXT_PREFIX
            self._was_cancelled = False

        # If a real user message arrives during Eureka Mode, reset the round counter
        # (eureka-driven synthetic messages are prefixed with "[Eureka Mode]")
        if self._eureka_mode and not user_message.startswith("[Eureka Mode]"):
            self._eureka_round_counter = 0
            self._eureka_pending_suggestion = None

        # Re-set ContextVar so executor threads inherit the correct OperationsLog
        if hasattr(self, "_ops_log"):
            set_operations_log(self._ops_log)
        self._memory_store._last_injected_ids.clear()
        self._event_bus.emit(
            USER_MESSAGE,
            level="info",
            target="orchestrator",
            msg=f"[User] {user_message}",
            data={"text": user_message},
        )

        # Memory is now baked into core memory (system prompt) — no per-turn injection
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

        # Prepend current time
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        augmented = f"[Current time: {current_time}]\n\n{augmented}"

        # Prepend cancel context and held results before user message
        prefix = cancel_prefix + held_prefix
        if prefix:
            augmented = f"{prefix}{augmented}"

        from .llm_utils import _CancelledDuringLLM

        try:
            result = self._process_single_message(augmented)
        except _CancelledDuringLLM:
            # Cancel fired during an LLM API call. Rollback incomplete turn.
            if self._chat:
                self._chat.rollback_last_turn()
            result = "Cancelled."
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
            target="user",
            msg=f"[Agent] {result}",
            data={
                "text": result,
                "turn": self._turn_number,
                "cycle": self._cycle_number,
                "generated": getattr(self, "_last_response_generated", False),
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
        self._cancel_holdback.clear()
        self._was_cancelled = False
        with self._held_results_lock:
            self._held_results.clear()
        self.chat = self.adapter.create_chat(
            model=self.model_name,
            system_prompt=self._system_prompt,
            tools=self._tool_schemas,
            thinking="high",
        )
        self._current_plan = None
        with self._sub_agents_lock:
            for agent in self._sub_agents.values():
                agent.stop(timeout=2.0)
            self._sub_agents.clear()
        ENVOY_TOOL_REGISTRY.clear_active()
        self._dataops_seq = 0
        self._mission_seq = 0
        self._renderer.reset()

        # Reset memory turn counter and agent (do NOT clear memory store)
        self._memory_agent = None
        self._memory_turn_counter = 0
        self._last_memory_op_index = 0
        self._inline_fail_count = 0
        self._inline_disabled_until = 0.0

        # Reset eureka agent and mode (agent already stopped via _sub_agents.clear())
        self._eureka_agent = None
        self._eureka_turn_counter = 0
        self._eureka_mode = config.get("eureka_mode", False)
        self._eureka_round_counter = 0
        self._eureka_pending_suggestion = None

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

    def get_current_plan(self) -> Optional["TaskPlan | dict"]:
        """Get the currently executing plan, if any.

        Returns a TaskPlan if one exists (from the old plan-execute loop),
        or a dict if one exists (from the research-only planner).
        """
        return self._current_plan

    def get_plan_status(self) -> Optional[str]:
        """Get a formatted status of the current plan.

        Falls back to the TaskStore for incomplete plans (CLI/MCP use case
        where the agent may have been reset but the plan is still on disk).
        The API endpoint uses get_current_plan() directly to avoid leaking
        plans from other sessions.
        """
        if self._current_plan is None:
            store = get_task_store()
            incomplete = store.get_incomplete_plans()
            if incomplete:
                plan = sorted(incomplete, key=lambda p: p.created_at, reverse=True)[0]
                return format_plan_for_display(plan)
            return None
        return format_plan_for_display(self._current_plan)


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
