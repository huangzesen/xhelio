"""
Base class for all sub-agents (Mission, DataOps, DataExtraction, Visualization).

Consolidates the shared logic: LLM chat setup, token tracking, tool-calling
loops (process_request and execute_task), and LoopGuard integration.

Sub-agents override:
- Constructor to provide agent_name, system_prompt, tool_categories, extra_tool_names
- Hook methods for agent-specific behavior (e.g., clarification interception)
"""

import contextvars
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Optional

from .llm import LLMAdapter, LLMResponse, FunctionSchema
from .tools import get_tool_schemas
from .tasks import Task, TaskStatus
from .logging import get_logger, log_error
from .event_bus import (
    EventBus, get_event_bus,
    DEBUG, TOOL_CALL, TOOL_RESULT, SUB_AGENT_TOOL, SUB_AGENT_ERROR,
    LLM_CALL, THINKING, TOKEN_USAGE,
)
from .loop_guard import LoopGuard, make_call_key
from .model_fallback import get_active_model
import config

# LLM API call timeout thresholds (seconds)
_LLM_WARN_INTERVAL = 20      # log a warning every N seconds while waiting
_LLM_RETRY_TIMEOUT = 180     # abandon call and retry after this
_LLM_MAX_RETRIES = 2         # max retries before giving up


class _CancelledDuringLLM(Exception):
    """Raised when user cancellation is detected during an LLM API wait."""


def send_with_timeout(
    chat,
    message,
    timeout_pool: ThreadPoolExecutor,
    cancel_event: threading.Event | None,
    retry_timeout: float,
    agent_name: str,
    logger,
) -> LLMResponse:
    """Send a message to the LLM with periodic warnings and retry on timeout.

    Shared implementation used by OrchestratorAgent, BaseSubAgent, and PlannerAgent.

    - Warns every _LLM_WARN_INTERVAL seconds while waiting.
    - After retry_timeout seconds, abandons the call and retries.
    - Retries up to _LLM_MAX_RETRIES times before raising.
    - Checks cancel_event between each poll interval so cancellation
      is responsive even during long LLM waits.
    """
    last_exc = None
    for attempt in range(1 + _LLM_MAX_RETRIES):
        future: Future = timeout_pool.submit(chat.send, message)
        t0 = time.monotonic()
        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    future.cancel()
                    get_event_bus().emit(DEBUG, agent=agent_name, level="info", msg=f"[{agent_name}] LLM call cancelled by user")
                    raise _CancelledDuringLLM()

                elapsed = time.monotonic() - t0
                remaining = retry_timeout - elapsed
                if remaining <= 0:
                    break
                wait = min(_LLM_WARN_INTERVAL, remaining)
                try:
                    return future.result(timeout=wait)
                except TimeoutError:
                    elapsed = time.monotonic() - t0
                    if elapsed >= retry_timeout:
                        break
                    get_event_bus().emit(LLM_CALL, agent=agent_name, level="warning", msg=f"[{agent_name}] LLM API not responding after {elapsed:.0f}s (attempt {attempt + 1})...")

            elapsed = time.monotonic() - t0
            future.cancel()
            last_exc = TimeoutError(
                f"LLM API call timed out after {elapsed:.0f}s"
            )
            if attempt < _LLM_MAX_RETRIES:
                get_event_bus().emit(LLM_CALL, agent=agent_name, level="warning", msg=f"[{agent_name}] LLM API timed out after {elapsed:.0f}s, retrying ({attempt + 1}/{_LLM_MAX_RETRIES})...")
            else:
                get_event_bus().emit(LLM_CALL, agent=agent_name, level="error", msg=f"[{agent_name}] LLM API timed out after {elapsed:.0f}s, no retries left")
        except _CancelledDuringLLM:
            raise
        except Exception:
            raise

    raise last_exc


def track_llm_usage(
    response: LLMResponse,
    token_state: dict,
    agent_name: str,
    last_tool_context: str,
):
    """Accumulate token usage from an LLMResponse and emit via EventBus.

    Shared implementation used by OrchestratorAgent, BaseSubAgent, and PlannerAgent.

    Args:
        response: The LLMResponse to extract usage from.
        token_state: Mutable dict with keys 'input', 'output', 'thinking',
            'cached', 'api_calls'. Updated in-place.
        agent_name: Label for log messages.
        last_tool_context: Tool context string for the token log.
    """
    usage = response.usage
    call_input = usage.input_tokens
    call_output = usage.output_tokens
    call_thinking = usage.thinking_tokens
    call_cached = usage.cached_tokens
    token_state["input"] += call_input
    token_state["output"] += call_output
    token_state["thinking"] += call_thinking
    token_state["cached"] += call_cached
    token_state["api_calls"] += 1
    get_event_bus().emit(
        TOKEN_USAGE,
        agent=agent_name,
        level="debug",
        msg=(
            f"[Tokens] {agent_name} in:{call_input} out:{call_output} "
            f"think:{call_thinking} | cum_in:{token_state['input']} "
            f"cum_out:{token_state['output']} calls:{token_state['api_calls']}"
        ),
        data={
            "agent_name": agent_name,
            "tool_context": last_tool_context[:60] if last_tool_context else "unknown",
            "input_tokens": call_input,
            "output_tokens": call_output,
            "thinking_tokens": call_thinking,
            "cached_tokens": call_cached,
            "cumulative_input": token_state["input"],
            "cumulative_output": token_state["output"],
            "cumulative_thinking": token_state["thinking"],
            "cumulative_cached": token_state["cached"],
            "api_calls": token_state["api_calls"],
        },
    )
    # Always emit thinking events so SSE listener can forward to frontend
    for thought in response.thoughts:
        get_event_bus().emit(THINKING, agent=agent_name, level="debug", msg=f"[Thinking] {thought}", data={"text": thought})


def execute_tools_batch(
    function_calls: list,
    tool_executor,
    parallel_safe_tools: set[str],
    parallel_enabled: bool,
    max_workers: int,
    agent_name: str,
    logger,
) -> list[tuple[str | None, str, dict, dict]]:
    """Execute tool calls, parallelizing when all are in the safe set.

    Shared implementation used by OrchestratorAgent and BaseSubAgent.

    Returns list of (tool_call_id, tool_name, tool_args, result) in original order.
    """
    parsed = [
        (
            getattr(fc, "id", None),
            fc.name,
            fc.args if isinstance(fc.args, dict) else (dict(fc.args) if fc.args else {}),
        )
        for fc in function_calls
    ]

    all_safe = (
        parallel_enabled
        and len(parsed) > 1
        and all(name in parallel_safe_tools for _, name, _ in parsed)
    )

    if not all_safe:
        return [
            (tc_id, name, args, tool_executor(name, args))
            for tc_id, name, args in parsed
        ]

    get_event_bus().emit(DEBUG, agent=agent_name, level="debug", msg=f"[{agent_name}] Parallel: {len(parsed)} tools concurrently")
    workers = min(len(parsed), max_workers)
    results_by_idx: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(contextvars.copy_context().run, tool_executor, name, args): idx
            for idx, (_, name, args) in enumerate(parsed)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results_by_idx[idx] = future.result()
            except Exception as e:
                results_by_idx[idx] = {
                    "status": "error",
                    "message": f"Parallel execution error: {e}",
                }

    return [
        (parsed[i][0], parsed[i][1], parsed[i][2], results_by_idx[i])
        for i in range(len(parsed))
    ]


class BaseSubAgent:
    """Base class with all shared sub-agent logic.

    Subclasses must call super().__init__() with appropriate parameters.
    """

    # Whether execute_task forces the LLM to produce a tool call on every turn.
    # Subclasses can set this to False when the tool requires complex arguments
    # that the LLM may emit as empty placeholders under forced-calling mode.
    _force_tool_call_in_tasks: bool = True

    def __init__(
        self,
        adapter: LLMAdapter,
        model_name: str,
        tool_executor,
        verbose: bool = False,
        agent_name: str = "SubAgent",
        system_prompt: str = "",
        tool_categories: list[str] | None = None,
        extra_tool_names: list[str] | None = None,
        cancel_event: threading.Event | None = None,
        llm_retry_timeout: int | None = None,
        event_bus: EventBus | None = None,
    ):
        self.adapter = adapter
        self.model_name = model_name
        self.tool_executor = tool_executor
        self.verbose = verbose
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self._cancel_event = cancel_event
        self.logger = get_logger()
        self._event_bus = event_bus or get_event_bus()

        # Build FunctionSchema list from categories
        categories = tool_categories or []
        extra = extra_tool_names or []
        self._tool_schemas: list[FunctionSchema] = []
        for tool_schema in get_tool_schemas(categories=categories, extra_names=extra):
            self._tool_schemas.append(FunctionSchema(
                name=tool_schema["name"],
                description=tool_schema["description"],
                parameters=tool_schema["parameters"],
            ))

        # Explicit context cache name (set by orchestrator for Gemini adapters)
        self._cache_name: str | None = None

        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_thinking_tokens = 0
        self._total_cached_tokens = 0
        self._api_calls = 0
        self._last_tool_context = "send_message"

        # Per-agent timeout (falls back to module default)
        self._llm_retry_timeout = llm_retry_timeout or _LLM_RETRY_TIMEOUT

        # Thread pool for timeout-wrapped LLM calls (1 worker — serial calls)
        self._timeout_pool = ThreadPoolExecutor(max_workers=1)

    # ---- Token tracking ----

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
            agent_name=self.agent_name,
            last_tool_context=self._last_tool_context,
        )
        self._total_input_tokens = token_state["input"]
        self._total_output_tokens = token_state["output"]
        self._total_thinking_tokens = token_state["thinking"]
        self._total_cached_tokens = token_state["cached"]
        self._api_calls = token_state["api_calls"]

    def get_token_usage(self) -> dict:
        """Return cumulative token usage for this agent."""
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "thinking_tokens": self._total_thinking_tokens,
            "cached_tokens": self._total_cached_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens + self._total_thinking_tokens,
            "api_calls": self._api_calls,
        }

    # ---- LLM send with timeout/retry ----

    def _send_with_timeout(self, chat, message) -> LLMResponse:
        """Send a message to the LLM with periodic warnings and retry on timeout."""
        return send_with_timeout(
            chat=chat,
            message=message,
            timeout_pool=self._timeout_pool,
            cancel_event=self._cancel_event,
            retry_timeout=self._llm_retry_timeout,
            agent_name=self.agent_name,
            logger=self.logger,
        )

    # ---- Hook methods for subclass customization ----

    def _on_tool_result(self, tool_name: str, tool_args: dict, result: dict) -> Optional[str]:
        """Hook called after each tool execution in process_request.

        If this returns a non-None string, process_request returns that string
        immediately (used by MissionAgent for clarification interception).
        """
        return None

    def _should_skip_function_call(self, function_calls: list) -> bool:
        """Hook called before executing function calls in execute_task.

        If True, the loop breaks without executing the calls
        (used by MissionAgent to skip ask_clarification in task execution).
        """
        return False

    def _get_task_prompt(self, task: Task) -> str:
        """Hook to customize the task execution prompt.

        Override to add agent-specific instructions (e.g., VisualizationAgent
        adds explicit tool-call guidance).
        """
        return (
            f"Execute this task: {task.instruction}\n\n"
            "CRITICAL: Do ONLY what the instruction says. Do NOT add extra steps.\n"
            "Return results as concise text when done."
        )

    def _get_error_context(self, **kwargs) -> dict:
        """Hook to add agent-specific context to error logs."""
        return kwargs

    # ---- Parallel tool execution ----

    _PARALLEL_SAFE_TOOLS = {"fetch_data"}

    def _execute_tools_batch(
        self, function_calls: list
    ) -> list[tuple[str | None, str, dict, dict]]:
        """Execute tool calls, parallelizing when all are fetch_data."""
        from config import PARALLEL_FETCH, PARALLEL_MAX_WORKERS
        return execute_tools_batch(
            function_calls=function_calls,
            tool_executor=self.tool_executor,
            parallel_safe_tools=self._PARALLEL_SAFE_TOOLS,
            parallel_enabled=PARALLEL_FETCH,
            max_workers=PARALLEL_MAX_WORKERS,
            agent_name=self.agent_name,
            logger=self.logger,
        )

    # ---- Shared loops ----

    def process_request(self, user_message: str) -> dict:
        """Process a user request conversationally (no forced function calling).

        Creates a fresh chat per request to avoid cross-request context pollution.

        Returns a dict with:
            text (str): The agent's text response.
            failed (bool): True if the agent stopped due to errors/loops.
            errors (list[str]): Error messages from failed tool calls.
        """
        self._event_bus.emit(DEBUG, agent=self.agent_name, level="debug", msg=f"[{self.agent_name}] Processing: {user_message}")

        try:
            # Create a fresh chat — conversational mode (no forced function calling)
            chat = self.adapter.create_chat(
                model=get_active_model(self.model_name),
                system_prompt=self.system_prompt,
                tools=self._tool_schemas,
                thinking="low",
                cached_content=self._cache_name,
            )
            self._last_tool_context = "initial_message"
            response = self._send_with_timeout(chat, user_message)
            self._track_usage(response)

            guard = LoopGuard(max_total_calls=20, max_iterations=8)
            consecutive_errors = 0
            collected_errors: list[str] = []
            had_successful_tool = False
            stop_reason = None

            while True:
                stop_reason = guard.check_iteration()
                if stop_reason:
                    self._event_bus.emit(DEBUG, agent=self.agent_name, level="debug", msg=f"[{self.agent_name}] Stopping: {stop_reason}")
                    break

                if self._cancel_event and self._cancel_event.is_set():
                    self._event_bus.emit(DEBUG, agent=self.agent_name, level="info", msg=f"[{self.agent_name}] Interrupted by user")
                    return {"text": "Interrupted by user.", "failed": True, "errors": ["Interrupted by user."]}

                if not response.tool_calls:
                    break

                # Check for loops/duplicates/cycling
                call_keys = set()
                for fc in response.tool_calls:
                    call_keys.add(make_call_key(fc.name, fc.args))
                stop_reason = guard.check_calls(call_keys)
                if stop_reason:
                    self._event_bus.emit(DEBUG, agent=self.agent_name, level="debug", msg=f"[{self.agent_name}] Stopping: {stop_reason}")
                    break

                # Execute tools — parallel when all are fetch_data, serial otherwise
                tool_results = self._execute_tools_batch(response.tool_calls)

                function_responses = []
                all_errors_this_round = True
                for tc_id, tool_name, tool_args, result in tool_results:
                    self._event_bus.emit(SUB_AGENT_TOOL, agent=self.agent_name, level="debug", msg=f"[{self.agent_name}] Tool: {tool_name}({tool_args})", data={"tool_name": tool_name, "tool_args": tool_args})

                    # Hook: let subclass intercept results (e.g., clarification)
                    intercept = self._on_tool_result(tool_name, tool_args, result)
                    if intercept is not None:
                        return intercept

                    if result.get("status") != "error":
                        all_errors_this_round = False
                        had_successful_tool = True

                    if result.get("status") == "error":
                        err_msg = result.get("message", "")
                        self._event_bus.emit(SUB_AGENT_ERROR, agent=self.agent_name, level="warning", msg=f"[{self.agent_name}] Tool error: {err_msg}", data={"tool_name": tool_name, "error": err_msg})
                        collected_errors.append(f"{tool_name}: {err_msg}")

                    # Inject structured observation summary
                    if config.OBSERVATION_SUMMARIES:
                        from .observations import generate_observation
                        result["observation"] = generate_observation(tool_name, tool_args, result)

                    function_responses.append(
                        self.adapter.make_tool_result_message(
                            tool_name, result, tool_call_id=tc_id
                        )
                    )

                guard.record_calls(call_keys)

                # Track consecutive error rounds
                if all_errors_this_round:
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0

                if consecutive_errors >= 2:
                    self._event_bus.emit(SUB_AGENT_ERROR, agent=self.agent_name, level="warning", msg=f"[{self.agent_name}] {consecutive_errors} consecutive error rounds, stopping")
                    stop_reason = "consecutive errors"
                    break

                self._event_bus.emit(DEBUG, agent=self.agent_name, level="debug", msg=f"[{self.agent_name}] Sending {len(function_responses)} tool result(s) back...")
                tool_names = [fc.name for fc in response.tool_calls]
                self._last_tool_context = "+".join(tool_names)
                response = self._send_with_timeout(chat, function_responses)
                self._track_usage(response)

            # Extract text response
            text = response.text or "Done."
            failed = stop_reason is not None and not had_successful_tool
            return {"text": text, "failed": failed, "errors": collected_errors}

        except _CancelledDuringLLM:
            return {"text": "Interrupted by user.", "failed": True, "errors": ["Interrupted by user."]}
        except Exception as e:
            ctx = self._get_error_context(request=user_message[:200])
            log_error(
                f"{self.agent_name} request failed",
                exc=e,
                context=ctx,
            )
            self._event_bus.emit(SUB_AGENT_ERROR, agent=self.agent_name, level="warning", msg=f"[{self.agent_name}] Failed: {e}")
            return {"text": f"Error processing request: {e}", "failed": True, "errors": [str(e)]}

    @staticmethod
    def _summarize_tool_outcome(tool_name: str, result: dict) -> dict:
        """Extract key fields from a tool result for structured tracking."""
        outcome = {"tool": tool_name, "status": result.get("status", "unknown")}
        # Capture the most informative fields from common tools
        for key in ("label", "num_points", "columns", "shape", "units",
                     "message", "nan_percentage", "quality_warning",
                     "time_range_note", "count"):
            if key in result:
                outcome[key] = result[key]
        return outcome

    @staticmethod
    def _build_outcome_summary(tool_outcomes: list[dict]) -> str:
        """Build a concise text summary from structured tool outcomes."""
        parts = []
        for o in tool_outcomes:
            s = f"{o['tool']}={o['status']}"
            if o.get("label"):
                s += f"(label={o['label']}"
                if o.get("num_points"):
                    s += f", {o['num_points']} pts"
                if o.get("units"):
                    s += f", {o['units']}"
                s += ")"
            elif o.get("message") and o["status"] == "error":
                msg = str(o["message"])[:100]
                s += f"({msg})"
            parts.append(s)
        return "Outcomes: " + "; ".join(parts)

    def execute_task(self, task: Task) -> str:
        """Execute a single task with forced function calling.

        Creates a fresh chat session for each task to avoid context pollution.
        Collects structured tool results in task.tool_results for downstream use.
        """
        task.status = TaskStatus.IN_PROGRESS
        task.tool_calls = []
        task.tool_results = []

        self._event_bus.emit(DEBUG, agent=self.agent_name, level="debug", msg=f"[{self.agent_name}] Executing: {task.description}")

        try:
            # Create a fresh chat for task execution
            chat = self.adapter.create_chat(
                model=get_active_model(self.model_name),
                system_prompt=self.system_prompt,
                tools=self._tool_schemas,
                force_tool_call=self._force_tool_call_in_tasks,
                thinking="low",
                cached_content=self._cache_name,
            )
            task_prompt = self._get_task_prompt(task)
            self._last_tool_context = "task:" + task.description[:50]
            response = self._send_with_timeout(chat, task_prompt)
            self._track_usage(response)

            guard = LoopGuard(max_total_calls=12, max_iterations=5)
            last_stop_reason = None
            had_successful_tool = False

            while True:
                stop_reason = guard.check_iteration()
                if stop_reason:
                    self._event_bus.emit(DEBUG, agent=self.agent_name, level="debug", msg=f"[{self.agent_name}] Stopping: {stop_reason}")
                    last_stop_reason = stop_reason
                    break

                if self._cancel_event and self._cancel_event.is_set():
                    self._event_bus.emit(DEBUG, agent=self.agent_name, level="info", msg=f"[{self.agent_name}] Task interrupted by user")
                    last_stop_reason = "cancelled by user"
                    break

                if not response.tool_calls and not response.text:
                    break

                if not response.tool_calls:
                    break

                # Hook: let subclass skip certain calls (e.g., ask_clarification)
                if self._should_skip_function_call(response.tool_calls):
                    self._event_bus.emit(DEBUG, agent=self.agent_name, level="debug", msg=f"[{self.agent_name}] Skipping function calls per hook")
                    break

                # Check for loops/duplicates/cycling
                call_keys = set()
                for fc in response.tool_calls:
                    call_keys.add(make_call_key(fc.name, fc.args))
                stop_reason = guard.check_calls(call_keys)
                if stop_reason:
                    self._event_bus.emit(DEBUG, agent=self.agent_name, level="debug", msg=f"[{self.agent_name}] Stopping: {stop_reason}")
                    last_stop_reason = stop_reason
                    break

                # Execute tools — parallel when all are fetch_data, serial otherwise
                tool_results = self._execute_tools_batch(response.tool_calls)

                function_responses = []
                for tc_id, tool_name, tool_args, result in tool_results:
                    task.tool_calls.append(tool_name)

                    # Collect structured outcome for downstream use
                    task.tool_results.append(self._summarize_tool_outcome(tool_name, result))

                    if result.get("status") == "success":
                        had_successful_tool = True
                    elif result.get("status") == "error":
                        self._event_bus.emit(SUB_AGENT_ERROR, agent=self.agent_name, level="warning", msg=f"[{self.agent_name}] Tool error: {result.get('message', '')}", data={"tool_name": tool_name, "error": result.get("message", "")})

                    # Inject structured observation summary
                    if config.OBSERVATION_SUMMARIES:
                        from .observations import generate_observation
                        result["observation"] = generate_observation(tool_name, tool_args, result)

                    function_responses.append(
                        self.adapter.make_tool_result_message(
                            tool_name, result, tool_call_id=tc_id
                        )
                    )

                guard.record_calls(call_keys)

                self._event_bus.emit(DEBUG, agent=self.agent_name, level="debug", msg=f"[{self.agent_name}] Sending {len(function_responses)} tool result(s) back...")
                tool_names = [fc.name for fc in response.tool_calls]
                self._last_tool_context = "+".join(tool_names)
                response = self._send_with_timeout(chat, function_responses)
                self._track_usage(response)

            # Warn if no tools were called
            if not task.tool_calls:
                log_error(
                    f"{self.agent_name} task completed without tool calls: {task.description}",
                    context=self._get_error_context(task_instruction=task.instruction),
                )
                self._event_bus.emit(SUB_AGENT_ERROR, agent=self.agent_name, level="warning", msg=f"[{self.agent_name}] No tools were called")

            # Extract text response
            result_text = response.text or ""

            # If LLM produced no text, build a structured summary from tool outcomes
            if not result_text and task.tool_results:
                result_text = self._build_outcome_summary(task.tool_results)

            if not result_text:
                result_text = "Done."

            if last_stop_reason:
                if last_stop_reason == "cancelled by user":
                    task.status = TaskStatus.FAILED
                    task.error = f"Task cancelled by user"
                    result_text += f" [CANCELLED]"
                elif had_successful_tool:
                    task.status = TaskStatus.COMPLETED
                    result_text += f" [loop guard stopped extra calls]"
                else:
                    task.status = TaskStatus.FAILED
                    task.error = f"Task stopped by loop guard: {last_stop_reason}"
                    result_text += f" [STOPPED: {last_stop_reason}]"
            else:
                task.status = TaskStatus.COMPLETED

            task.result = result_text

            self._event_bus.emit(DEBUG, agent=self.agent_name, level="debug", msg=f"[{self.agent_name}] {task.status.value}: {task.description}")

            return result_text

        except _CancelledDuringLLM:
            task.status = TaskStatus.FAILED
            task.error = "Task cancelled by user"
            task.result = "Interrupted by user. [CANCELLED]"
            return task.result
        except TimeoutError as e:
            # If tools already succeeded (data fetched / plot saved),
            # mark as completed — the timeout only lost the LLM's summary.
            if had_successful_tool:
                task.status = TaskStatus.COMPLETED
                if not task.result and task.tool_results:
                    task.result = self._build_outcome_summary(task.tool_results)
                task.result = (task.result or "Done.") + " [LLM follow-up timed out]"
                self._event_bus.emit(SUB_AGENT_ERROR, agent=self.agent_name, level="warning", msg=f"[{self.agent_name}] Completed with timeout: {task.description}")
                return task.result
            else:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                ctx = self._get_error_context(task=task.description)
                log_error(f"{self.agent_name} task failed", exc=e, context=ctx)
                self._event_bus.emit(SUB_AGENT_ERROR, agent=self.agent_name, level="warning", msg=f"[{self.agent_name}] Failed: {task.description} - {e}")
                return f"Error: {e}"
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            ctx = self._get_error_context(task=task.description)
            log_error(
                f"{self.agent_name} task failed",
                exc=e,
                context=ctx,
            )
            self._event_bus.emit(SUB_AGENT_ERROR, agent=self.agent_name, level="warning", msg=f"[{self.agent_name}] Failed: {task.description} - {e}")
            return f"Error: {e}"
