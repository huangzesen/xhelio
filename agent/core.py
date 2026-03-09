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
from .memory_hooks import MemoryHooks, candidates_from_log
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
    build_outcome_summary,
)
from .token_tracker import TokenTracker
from .inline_completions import InlineCompletions
from .delegation import DelegationBus
from .eureka_hooks import EurekaHooks
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
)
from data_ops.fetch import fetch_data
from data_ops.operations_log import set_operations_log, OperationsLog
from data_ops.asset_registry import AssetRegistry

from .agent_registry import ORCHESTRATOR_TOOLS
from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY

# Roles that map to "user" or "assistant/model" across all LLM adapters
_USER_ROLES = {"user"}
_AGENT_ROLES = {"model", "assistant"}

# Context compaction prompt (shared with sub_agent.py via llm_session_mixin)
from agent.llm_session_mixin import COMPACTION_PROMPT as _COMPACTION_PROMPT


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
    """Create the LLM service based on config.

    Uses resolve_agent_model so the active workbench preset is respected.
    """
    provider, model, base_url = config.resolve_agent_model("orchestrator")
    api_key = get_api_key(provider)
    caps = config.resolve_capabilities()
    return LLMService(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url or config.LLM_BASE_URL,
        provider_config=caps,
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


_PIPELINE_INVALIDATING_TOOLS = frozenset({
    "fetch_data", "run_code",
    "render_plotly_json", "manage_plot",
})


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

        # Delegation subsystem (sub-agent lifecycle + dispatch)
        self._delegation = DelegationBus(ctx=self)
        # Compatibility aliases — existing code references these directly
        self._sub_agents = self._delegation._agents
        self._sub_agents_lock = self._delegation._lock

        # Fire-and-forget async delegation tracking: agent_id → start_time
        # Written by delegation handler threads, read by run_loop thread.
        self._async_delegations: dict[str, float] = {}

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
        # Discover SPICE tools from MCP server (lazy, non-fatal).
        # Registers tools for envoy agents only — orchestrator delegates.
        self._ensure_spice_tools()

        # Build tool schemas for the orchestrator
        self._all_tool_schemas = get_function_schemas(names=ORCHESTRATOR_TOOLS)
        self._viz_backend = config.PREFER_VIZ_BACKEND

        # All tools active from the start (no browse+load gating)
        self._tool_schemas = list(self._all_tool_schemas)

        # Store model name and system prompt for chat creation
        if not model:
            _, model, _ = config.resolve_agent_model("orchestrator")
        self.model_name = model
        self._system_prompt = get_system_prompt()

        if not defer_chat:
            # Create chat session
            self.chat = self.service.create_session(
                system_prompt=self._system_prompt,
                tools=self._tool_schemas,
                model=self.model_name,
                thinking="high",
                tracked=False,
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
        self._orch_tracker = TokenTracker("OrchestratorAgent")
        self._last_tool_context = "send_message"
        self._round_start_tokens: dict | None = None  # snapshot at CYCLE_START

        # Cycle state tracking
        self._responded_this_cycle = False  # Track if orchestrator responded to user

        # Token decomposition (system/tools/history breakdown)
        self._system_prompt_tokens = 0
        self._tools_tokens = 0
        self._token_decomp_dirty = True

        # Thread-local storage for per-thread agent identity (async delegation)
        self._tls = threading.local()
        self._tls.active_agent_name = "OrchestratorAgent"
        self._tls.current_agent_type = "orchestrator"
        self._tls.current_tool_call_id = None

        # Counter for consecutive invalid-tool rejections per agent type.
        # Used to provide enhanced error messages listing available tools.
        self._invalid_tool_counts: dict[str, int] = {}

        # Inline model token usage (tracked separately for breakdown)
        self._inline_tracker = TokenTracker("Inline")

        # Retired ephemeral agent token usage — alias into delegation bus
        self._retired_agent_usage = self._delegation._retired_usage

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
        # Memory hooks (extraction, hot reload, pipeline curation)
        self._memory_hooks = MemoryHooks(self)
        self._event_bus.subscribe(self._memory_hooks.on_memory_mutated)

        # Eureka discovery + insight review (extracted to EurekaHooks)
        self._eureka_hooks = EurekaHooks(
            ctx=self, eureka_mode=config.get("eureka_mode", False)
        )

        # Pipeline template index (searchable metadata for saved templates)
        self._pipeline_store = PipelineStore()

        # Inline completions (follow-ups, session titles, autocomplete)
        _inline_provider, _inline_model, _ = config.resolve_agent_model("inline")
        self._inline = InlineCompletions(
            service=self.service,
            inline_tracker=self._inline_tracker,
            event_bus=self._event_bus,
            provider=_inline_provider,
            model=_inline_model,
        )

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
        return self._delegation.all_work_subagents_idle()

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
        self._orch_tracker.track(
            response,
            last_tool_context=self._last_tool_context,
            system_tokens=self._system_prompt_tokens,
            tools_tokens=self._tools_tokens,
        )

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

        self.chat = self.service.create_session(
            system_prompt=self._system_prompt,
            tools=self._tool_schemas,
            model=self.model_name,
            thinking="high",
            tracked=False,
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
        from agent.llm_session_mixin import summarize_tool_calls
        return summarize_tool_calls(history)

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
        from agent.llm_session_mixin import summarize_tool_results
        return summarize_tool_results(message)

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
            response = self.service.generate(
                prompt=_COMPACTION_PROMPT + text,
                model=self.model_name,
                temperature=0.1,
                max_output_tokens=2048,
                tracked=False,
            )
            # Track token cost
            self._orch_tracker.track(
                response,
                last_tool_context="context_compaction",
            )

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
            response = self.service.web_search(
                query=query,
            )
        except Exception as e:
            return {"status": "error", "message": f"Web search failed: {e}"}

        if not response.text:
            # Distinguish "not configured" vs "call failed with empty response"
            provider = self.service.provider
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
        self._orch_tracker.track(
            response,
            last_tool_context="web_search",
            system_tokens=0,
            tools_tokens=0,
        )

        sources_text = self._extract_grounding_sources(response)

        return {"status": "success", "answer": response.text + sources_text}

    def get_token_usage(self) -> dict:
        """Return cumulative token usage for this session (including sub-agents)."""
        orch = self._orch_tracker.get_usage()
        inl = self._inline_tracker.get_usage()
        input_tokens = orch["input_tokens"] + inl["input_tokens"]
        output_tokens = orch["output_tokens"] + inl["output_tokens"]
        thinking_tokens = orch["thinking_tokens"] + inl["thinking_tokens"]
        cached_tokens = orch["cached_tokens"] + inl["cached_tokens"]
        api_calls = orch["api_calls"] + inl["api_calls"]

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
        orch = self._orch_tracker.get_usage()
        latest_input = self._orch_tracker.latest_input_tokens
        orch.update({
            "ctx_system_tokens": self._system_prompt_tokens,
            "ctx_tools_tokens": self._tools_tokens,
            "ctx_history_tokens": max(
                0,
                latest_input
                - self._system_prompt_tokens
                - self._tools_tokens,
            ),
            "ctx_total_tokens": latest_input,
        })
        _add("Orchestrator", orch)

        # Inline model usage (follow-ups, session titles, completions)
        _add("Inline", self._inline_tracker.get_usage())

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
                    self._eureka_hooks.latest_render_png = buf.getvalue()
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
            if self._eureka_hooks.latest_render_png:
                png_path.write_bytes(self._eureka_hooks.latest_render_png)
            elif figure is not None:
                import io as _io

                try:
                    buf = _io.BytesIO()
                    figure.write_image(
                        buf, format="png", width=1100, height=600, scale=2
                    )
                    png_bytes = buf.getvalue()
                    png_path.write_bytes(png_bytes)
                    self._eureka_hooks.latest_render_png = png_bytes
                except Exception:
                    pass

            result["output_files"] = [str(json_path)]
            if png_path.exists():
                result["output_files"].append(str(png_path))

            # Register figure in the asset registry
            if hasattr(self, "_asset_registry"):
                thumbnail = str(png_path) if png_path.exists() else None
                fig_asset = self._asset_registry.register_figure(
                    fig_json=fig_json,
                    trace_labels=list(entry_map.keys()),
                    panel_count=result.get("panels", 1),
                    op_id=op_id,
                    thumbnail_path=thumbnail,
                )
                result["asset_id"] = fig_asset.asset_id

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

        # 1. Extract data labels from user script
        from rendering.mpl_sandbox import extract_data_labels

        data_labels = extract_data_labels(code)

        # 2. Stage data + metadata into sandbox dir
        from agent.tool_handlers.sandbox import _stage_entry, _stage_meta

        session_dir = Path(self._session_dir)
        sandbox_dir = session_dir / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        for label in data_labels:
            entry = self._store.get(label)
            if entry is not None:
                _stage_entry(entry, sandbox_dir)
                _stage_meta(entry, sandbox_dir)

        # available_labels() returns all store labels (without staging their data)
        all_labels = [e["label"] for e in self._store.list_entries()]

        # 3. Generate script_id and output path
        from datetime import datetime as _dt
        import secrets

        script_id = _dt.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
        output_dir = session_dir / "mpl_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{script_id}.png"

        # 4. Build preamble + epilogue
        import json as _json
        labels_json = _json.dumps(all_labels)

        preamble = f'''import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

_OUTPUT_PATH = {repr(str(output_path))}
_STAGED_LABELS = json.loads({repr(labels_json)})

def load_data(label):
    """Load a DataFrame from staged data by label."""
    path = f"{{label}}.parquet"
    import os
    if not os.path.exists(path):
        raise KeyError(f"Label '{{label}}' not found. Available labels: {{available_labels()}}")
    return pd.read_parquet(path)

def load_meta(label):
    """Load metadata dict for a label."""
    path = f"{{label}}.meta.json"
    import os
    if not os.path.exists(path):
        return {{}}
    with open(path) as f:
        return json.load(f)

def available_labels():
    """Return all available data labels."""
    return _STAGED_LABELS

# === User script starts below ===
'''

        epilogue = f'''

# === Auto-generated epilogue ===
plt.savefig(_OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close("all")
'''

        wrapped_code = preamble + code + epilogue

        # 5. Validate via blocklist
        from data_ops.sandbox import validate_code_blocklist

        violations = validate_code_blocklist(code)
        if violations:
            return {
                "status": "error",
                "message": "Script validation failed:\n" + "\n".join(f"  - {v}" for v in violations),
            }

        # 6. Save wrapped script to mpl_scripts/
        scripts_dir = session_dir / "mpl_scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script_path = scripts_dir / f"{script_id}.py"
        script_path.write_text(wrapped_code, encoding="utf-8")

        # 7. Execute via sandbox
        from data_ops.sandbox import execute_sandboxed

        try:
            stdout_output, _ = execute_sandboxed(
                wrapped_code,
                work_dir=sandbox_dir,
                timeout=60,
            )
        except TimeoutError as e:
            return {
                "status": "error",
                "script_id": script_id,
                "message": f"Script execution timed out: {e}",
                "script_path": str(script_path),
            }

        # 8. Check if output PNG was created
        if output_path.exists() and output_path.stat().st_size > 0:
            # Success path
            _mpl_labels_used = data_labels
            self._event_bus.emit(
                MPL_RENDER_EXECUTED,
                agent="VizAgent[Mpl]",
                msg=f"[MplViz] Script executed: {description}",
                data={
                    "script_id": script_id,
                    "description": description,
                    "output_path": str(output_path),
                    "script_path": str(script_path),
                    "args": {"script": code, "description": description},
                    "inputs": _mpl_labels_used,
                    "outputs": [],
                    "status": "success",
                },
            )
            # Cache rendered PNG for InsightAgent
            try:
                self._eureka_hooks.latest_render_png = output_path.read_bytes()
            except Exception as e:
                get_logger().warning(
                    f"Failed to cache matplotlib PNG for insight review: {e}"
                )

            response = {
                "status": "success",
                "script_id": script_id,
                "output_path": str(output_path),
                "output_files": [str(output_path)],
                "message": f"Matplotlib plot saved successfully. Script ID: {script_id}",
            }

            if stdout_output and stdout_output.strip():
                response["stdout"] = stdout_output

            # Auto-review via InsightAgent
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
            # Failure path
            response = {
                "status": "error",
                "script_id": script_id,
                "message": "Script execution failed. See stderr for details.",
                "stderr": stdout_output or "",
            }
            if script_path:
                response["script_path"] = str(script_path)
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

            # Read the saved script (already has preamble/epilogue baked in)
            saved_script = script_file.read_text(encoding="utf-8")

            # Re-stage all data into sandbox
            from agent.tool_handlers.sandbox import _stage_entry, _stage_meta
            from rendering.mpl_sandbox import extract_data_labels as _extract_mpl_labels

            sandbox_dir = Path(self._session_dir) / "sandbox"
            sandbox_dir.mkdir(parents=True, exist_ok=True)

            _rerun_labels = _extract_mpl_labels(saved_script)
            for label in _rerun_labels:
                entry = self._store.get(label)
                if entry is not None:
                    _stage_entry(entry, sandbox_dir)
                    _stage_meta(entry, sandbox_dir)

            # Execute via sandbox
            from data_ops.sandbox import execute_sandboxed

            output_path = outputs_dir / f"{script_id}.png"
            try:
                stdout_output, _ = execute_sandboxed(
                    saved_script,
                    work_dir=sandbox_dir,
                    timeout=60,
                )
            except TimeoutError:
                return {"status": "error", "message": "Script timed out during rerun"}

            if output_path.exists() and output_path.stat().st_size > 0:
                self._event_bus.emit(
                    MPL_RENDER_EXECUTED,
                    agent="VizAgent[Mpl]",
                    msg=f"[MplViz] Script re-executed: {script_id}",
                    data={
                        "script_id": script_id,
                        "description": f"Rerun of {script_id}",
                        "args": {
                            "script": saved_script,
                            "description": f"Rerun of {script_id}",
                        },
                        "inputs": _rerun_labels,
                        "outputs": [],
                        "status": "success",
                    },
                )
                # Cache rendered PNG for InsightAgent
                try:
                    self._eureka_hooks.latest_render_png = output_path.read_bytes()
                except Exception as e:
                    get_logger().warning(
                        f"Failed to cache matplotlib rerun PNG: {e}"
                    )
                return {
                    "status": "success",
                    "script_id": script_id,
                    "output_path": str(output_path),
                    "output_files": [str(output_path)],
                    "message": "Script re-executed successfully",
                }
            return {
                "status": "error",
                "stderr": stdout_output or "",
                "message": "Script re-execution failed",
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
        """Lazily connect the SPICE MCP client and register tools for envoys.

        Discovers tool schemas from the MCP server and registers them for
        envoy agents. The orchestrator does not call SPICE tools directly —
        it delegates to envoys via delegate_to_envoy.
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

        # Register into all envoy kind modules (cdaweb, ppi, spice)
        from agent.tool_handlers import TOOL_REGISTRY
        spice_handlers = {n: TOOL_REGISTRY[n] for n in names if n in TOOL_REGISTRY}
        for kind in ("cdaweb", "ppi", "spice"):
            ENVOY_KIND_REGISTRY.add_tools_to_kind(kind, schemas, spice_handlers)

        # Generate SPICE envoy JSON for envoy_query discovery
        try:
            import heliospice
            from knowledge.generate_envoy_json import from_mcp
            from knowledge.mission_loader import _ENVOYS_DIR

            package_info = {
                "name": "heliospice",
                "version": heliospice.__version__,
                "doc": heliospice.__doc__ or "",
                "supported_missions": heliospice.list_supported_missions(),
                "coordinate_frames": heliospice.list_frames_with_descriptions(),
            }
            from_mcp(
                tool_schemas=schemas,
                package_info=package_info,
                envoy_id="SPICE",
                output_dir=_ENVOYS_DIR / "spice",
            )
            # Clear mission cache so load_mission("SPICE") picks up the new file
            from knowledge.mission_loader import clear_cache
            clear_cache()
        except Exception as e:
            logger.warning("Failed to generate SPICE envoy JSON: %s", e)

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

        # Per-mission permission gate for envoys
        if self._current_agent_type.startswith("envoy:"):
            mission_id = self._current_agent_type.split(":", 1)[1]
            from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY
            allowed = frozenset(ENVOY_KIND_REGISTRY.get_tool_names(mission_id))
        else:
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

    # ---- Delegation infrastructure (delegated to DelegationBus) ----

    def _build_envoy_request(self, mission_id: str, request: str, agent=None) -> str:
        return self._delegation.build_envoy_request(mission_id, request, agent=agent)

    def _build_dataops_request(self, request: str, context: str, agent=None) -> str:
        return self._delegation.build_dataops_request(request, context, agent=agent)

    def _build_data_io_request(self, request: str, context: str) -> str:
        return self._delegation.build_data_io_request(request, context)

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
                            self._memory_hooks.trigger_hot_reload()
                            self._rounds_since_last_reload = 0
                    # Trigger memory extraction at cycle end
                    self._memory_hooks.maybe_extract()
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
                            self._memory_hooks.trigger_hot_reload()
                            self._rounds_since_last_reload = 0
                    # Trigger memory extraction at cycle end
                    self._memory_hooks.maybe_extract()
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

        Forwarded to :pyattr:`_memory_hooks`.
        """
        self._memory_hooks.trigger_hot_reload()

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
            task_chat = self.service.create_session(
                system_prompt=self._system_prompt,
                tools=self._tool_schemas,
                model=self.model_name,
                force_tool_call=True,
                thinking="low",
                tracked=False,
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
                        self.service.make_tool_result(
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
        return self._delegation.get_or_create_planner_agent()

    def _process_single_message(self, user_message: str) -> str:
        """Process a single (non-complex) user message.

        Uses the Control Center for async delegation tracking.
        """
        self._eureka_hooks.reset_per_message()
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
                        self.service.make_tool_result(
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
                    clarification_response = self.service.make_tool_result(
                        tool_name, result, tool_call_id=tc_id
                    )
                    if self.chat and hasattr(self.chat, 'commit_tool_results'):
                        self.chat.commit_tool_results([clarification_response])
                    return question

                function_responses.append(
                    self.service.make_tool_result(
                        tool_name, result, tool_call_id=tc_id
                    )
                )

            guard.record_calls(len(function_calls))

            tool_names = [fc.name for fc in function_calls]
            self._last_tool_context = "+".join(tool_names)

            # Check cancel AFTER tool execution — strip the incomplete
            # assistant turn. Cancel context is prepended to the next user message.
            if self._cancel_event.is_set():
                if self.chat:
                    self.chat.rollback_last_turn()
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
        """Return set of active mission IDs from current agents. Thread-safe."""
        return self._delegation.get_active_envoy_ids()

    def _on_memory_mutated(self, event: SessionEvent) -> None:
        """Event bus listener: forwarded to :pyattr:`_memory_hooks`."""
        self._memory_hooks.on_memory_mutated(event)

    @staticmethod
    def _wrap_delegation_result(sub_result, store_snapshot=None) -> dict:
        return DelegationBus.wrap_delegation_result(sub_result, store_snapshot)

    def reset_sub_agents(self) -> None:
        """Invalidate all cached sub-agents so they are recreated with current config."""
        self._delegation.reset()
        # Clear direct references so _ensure_*_agent() recreates them
        self._memory_hooks._agent = None
        self._eureka_hooks.eureka_agent = None

    # ---- Agent-based sub-agent management (delegated to DelegationBus) ----

    def _get_or_create_envoy_agent(self, mission_id: str) -> EnvoyAgent:
        return self._delegation.get_or_create_envoy_agent(mission_id)

    def _create_ephemeral_envoy_agent(self, mission_id: str) -> EnvoyAgent:
        return self._delegation.create_ephemeral_envoy_agent(mission_id)

    def _get_or_create_viz_plotly_agent(self) -> VizPlotlyAgent:
        return self._delegation.get_or_create_viz_plotly_agent()

    def _get_or_create_viz_mpl_agent(self) -> VizMplAgent:
        return self._delegation.get_or_create_viz_mpl_agent()

    def _get_or_create_viz_jsx_agent(self) -> VizJsxAgent:
        return self._delegation.get_or_create_viz_jsx_agent()

    def _get_available_dataops_agent(self) -> DataOpsAgent:
        return self._delegation.get_available_dataops_agent()

    def _cleanup_ephemeral_agent(self, agent_id: str) -> None:
        self._delegation.cleanup_ephemeral(agent_id)

    def _get_or_create_data_io_agent(self) -> DataIOAgent:
        return self._delegation.get_or_create_data_io_agent()

    def _get_or_create_insight_agent(self) -> InsightAgent:
        return self._delegation.get_or_create_insight_agent()

    def _delegate_to_sub_agent(self, agent, request, **kwargs) -> dict:
        return self._delegation.delegate_to_sub_agent(agent, request, **kwargs)

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
        current_base_url = getattr(self.service.get_adapter(self.service.provider), "base_url", None)
        current_model = self.model_name

        # Use resolve_agent_model so the active preset is respected
        target_provider, target_model, target_base_url_resolved = config.resolve_agent_model("orchestrator")
        target_base_url = target_base_url_resolved or config.LLM_BASE_URL

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
            self._event_bus.emit(
                DEBUG,
                level="info",
                msg=f"[Config] Adapter rebuilt: {current_provider} → {target_provider}",
            )

        # 3. Update model name
        self.model_name = target_model

        # 4. Recreate chat session with preserved interface
        try:
            self.chat = self.service.create_session(
                system_prompt=self._system_prompt,
                tools=self._tool_schemas,
                model=self.model_name,
                thinking="high",
                tracked=False,
                interface=interface,
            )
        except Exception as exc:
            # Fall back to fresh chat if interface transfer fails
            self._event_bus.emit(
                DEBUG,
                level="warning",
                msg=f"[Config] Chat recreation with interface failed ({exc}), starting fresh",
            )
            self.chat = self.service.create_session(
                system_prompt=self._system_prompt,
                tools=self._tool_schemas,
                model=self.model_name,
                thinking="high",
                tracked=False,
            )

        # 5. Clear all cached sub-agents, memory agent, and eureka agent
        self.reset_sub_agents()
        self._memory_hooks._agent = None
        self._eureka_hooks._agent = None
        # Sync eureka mode flag from config (may have been toggled via UI)
        self._eureka_hooks.eureka_mode = config.get("eureka_mode", False)

        # 6. Recreate inline completions (picks up new service/adapter)
        _inline_provider, _inline_model, _ = config.resolve_agent_model("inline")
        self._inline = InlineCompletions(
            service=self.service,
            inline_tracker=self._inline_tracker,
            event_bus=self._event_bus,
            provider=_inline_provider,
            model=_inline_model,
        )

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
        """Build data context string for the InsightAgent (forwarded to EurekaHooks)."""
        return self._eureka_hooks.build_insight_context()

    def _sync_insight_review(self) -> dict | None:
        """Synchronous InsightAgent figure review (forwarded to EurekaHooks)."""
        return self._eureka_hooks.sync_insight_review()

    # ---- Long-term memory (end-of-session) ----

    def _build_memory_context(self) -> MemoryContext:
        """Forwarded to :pyattr:`_memory_hooks`."""
        return self._memory_hooks.build_context()

    def _enumerate_pipeline_candidates(self) -> list[dict]:
        """Forwarded to :pyattr:`_memory_hooks`."""
        return self._memory_hooks.enumerate_pipeline_candidates()

    @staticmethod
    def _candidates_from_log(ops_log) -> list[dict]:
        """Forwarded to :func:`memory_hooks.candidates_from_log`."""
        return candidates_from_log(ops_log)

    def _ensure_memory_agent(self, session_id: str = "", bus=None) -> MemoryAgent:
        """Forwarded to :pyattr:`_memory_hooks`."""
        return self._memory_hooks.ensure_agent(session_id=session_id, bus=bus)

    def _ensure_eureka_agent(self):
        """Lazily create or return the existing EurekaAgent (forwarded to EurekaHooks)."""
        return self._eureka_hooks.ensure_eureka_agent()

    def _build_eureka_context(self) -> dict:
        """Build context dict for Eureka discovery (forwarded to EurekaHooks)."""
        return self._eureka_hooks.build_eureka_context()

    def _format_eureka_suggestion_as_user_msg(self, suggestion) -> str:
        """Format eureka suggestion as user message (forwarded to EurekaHooks)."""
        return self._eureka_hooks.format_eureka_suggestion_as_user_msg(suggestion)

    def _maybe_extract_eurekas(self) -> None:
        """Trigger async Eureka extraction (forwarded to EurekaHooks)."""
        self._eureka_hooks.maybe_extract_eurekas()

    def _run_memory_agent_for_pipelines(self) -> list[dict]:
        """Forwarded to :pyattr:`_memory_hooks`."""
        return self._memory_hooks.run_for_pipelines()

    def _persist_operations_log(self) -> None:
        """Forwarded to :pyattr:`_memory_hooks`."""
        self._memory_hooks.persist_operations_log()

    def _maybe_extract_memories(self) -> None:
        """Forwarded to :pyattr:`_memory_hooks`."""
        self._memory_hooks.maybe_extract()

    def generate_follow_ups(self, max_suggestions: int = 3) -> list[str]:
        """Generate contextual follow-up suggestions based on the conversation.

        Delegates to InlineCompletions.
        """
        try:
            history = self.chat.get_history()
        except Exception:
            return []
        store = self._store
        labels = [e["label"] for e in store.list_entries()]
        has_plot = self._renderer.get_figure() is not None
        return self._inline.generate_follow_ups(
            chat_history=history,
            store_labels=labels,
            has_plot=has_plot,
            max_suggestions=max_suggestions,
        )

    def generate_session_title(self) -> Optional[str]:
        """Generate a short title from the first exchange. Delegates to InlineCompletions."""
        try:
            history = self.chat.get_history()
        except Exception:
            history = []
        events = self._event_bus.get_events(types={USER_MESSAGE, AGENT_RESPONSE})
        return self._inline.generate_session_title(
            chat_history=history,
            event_bus_events=events,
        )

    def generate_inline_completions(
        self, partial: str, max_completions: int = 3
    ) -> list[str]:
        """Complete the user's partial input. Delegates to InlineCompletions."""
        try:
            history = self.chat.get_history()
        except Exception:
            history = []
        store = self._store
        labels = [e["label"] for e in store.list_entries()]
        memory_section = self._memory_store.format_for_injection(
            scope="generic",
            include_summaries=True,
            include_review_instruction=False,
        )
        return self._inline.generate_inline_completions(
            partial=partial,
            chat_history=history,
            store_labels=labels,
            memory_section=memory_section,
            max_completions=max_completions,
        )

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
        self._session_dir = session_dir
        (session_dir / "sandbox").mkdir(exist_ok=True)
        self._store = DataStore(session_dir / "data")
        set_store(self._store)
        self._asset_registry = AssetRegistry(session_dir, self._store)

        # Create a fresh OperationsLog scoped to this session
        self._ops_log = OperationsLog(session_id=self._session_id)
        set_operations_log(self._ops_log)

        # Start writing structured event log to disk
        self._start_event_log_writer()

        # Push kind handlers into global TOOL_REGISTRY for dispatch
        ENVOY_KIND_REGISTRY.register_handlers_globally()

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

        if hasattr(self, "_asset_registry"):
            self._asset_registry.save()

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
                self.chat = self.service.create_session(
                    system_prompt=self._system_prompt,
                    tools=self._tool_schemas,
                    model=self.model_name,
                    thinking="high",
                    tracked=False,
                )
        else:
            # No history — start fresh chat
            self.chat = self.service.create_session(
                system_prompt=self._system_prompt,
                tools=self._tool_schemas,
                model=self.model_name,
                thinking="high",
                tracked=False,
            )

        # Restore DataStore — constructor auto-loads _labels.json (or migrates _index.json)
        self._session_dir = data_dir.parent
        (self._session_dir / "sandbox").mkdir(exist_ok=True)
        self._store = DataStore(data_dir)
        set_store(self._store)
        self._asset_registry = AssetRegistry(self._session_dir, self._store)
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
        ENVOY_KIND_REGISTRY.clear_active()
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
            self._orch_tracker.restore(saved_usage)

        # Restore cycle counter from previous session runs
        saved_round_count = metadata.get("round_count", 0)
        self._cycle_number = saved_round_count

        # Reset memory reload counter on session load
        self._rounds_since_last_reload = 0

        # Start writing structured event log (append mode — resumes keep adding)
        self._start_event_log_writer()

        # Push kind handlers into global TOOL_REGISTRY for dispatch
        ENVOY_KIND_REGISTRY.register_handlers_globally()

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
        if self._eureka_hooks.latest_render_png is not None:
            return self._eureka_hooks.latest_render_png

        self._restore_deferred_figure()
        figure = self._renderer.get_figure()
        if figure is not None:
            import io

            try:
                buf = io.BytesIO()
                figure.write_image(buf, format="png", width=1100, height=600, scale=2)
                png_bytes = buf.getvalue()
                self._eureka_hooks.latest_render_png = png_bytes
                return png_bytes
            except Exception:
                pass

        session_dir = self._session_manager.base_dir / self._session_id
        latest_png = self._find_latest_png(session_dir)
        if latest_png is not None:
            try:
                png_bytes = latest_png.read_bytes()
                self._eureka_hooks.latest_render_png = png_bytes
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
        """Handler for the plan_check tool.

        Loads the plan file for the current session.  The filename is
        deterministic: ``{session_id}_plan.json`` inside the plans dir.
        No arguments needed — the LLM no longer has to relay the path.
        """
        plan_dir = get_data_dir() / "plans"
        plan_path = plan_dir / f"{self._session_id}_plan.json"
        if not plan_path.exists():
            return {
                "status": "error",
                "message": f"No plan found for this session (looked for {plan_path.name})"
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
        self._eureka_hooks.reset_eureka_on_user_message(user_message)

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
            if self.chat:
                self.chat.rollback_last_turn()
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
        self.chat = self.service.create_session(
            system_prompt=self._system_prompt,
            tools=self._tool_schemas,
            model=self.model_name,
            thinking="high",
            tracked=False,
        )
        self._current_plan = None
        self._delegation.reset_full()
        self._renderer.reset()

        # Reset memory hooks (do NOT clear memory store)
        self._memory_hooks.reset()
        self._inline.reset()

        # Reset eureka agent and mode (agent already stopped via _sub_agents.clear())
        self._eureka_hooks.reset()

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
