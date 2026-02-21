"""
Data operations sub-agent with two-phase compute.

Phase 1 (Think): Explore data structure and research function APIs via
an ephemeral chat with function_docs + data inspection tools.

Phase 2 (Execute): Write and run computation code via custom_operation,
enriched with the think phase's research findings.

The orchestrator delegates computation requests here, keeping fetching
in mission agents and visualization in the visualization agent.
"""

from .llm import LLMAdapter, FunctionSchema
from .base_agent import BaseSubAgent
from .tasks import Task
from .tools import get_tool_schemas
from .tool_loop import run_tool_loop, extract_text_from_response
from .event_bus import DEBUG, PROGRESS
from .model_fallback import get_active_model
from knowledge.prompt_builder import build_data_ops_prompt, build_data_ops_think_prompt

# DataOps agent gets compute tools + function docs + list_fetched_data
DATAOPS_TOOL_CATEGORIES = ["data_ops_compute", "conversation"]
DATAOPS_EXTRA_TOOLS = ["list_fetched_data", "search_function_docs", "get_function_docs", "review_memory"]

# Think phase: function docs + data inspection (no compute tools)
THINK_TOOL_CATEGORIES = ["function_docs"]
THINK_EXTRA_TOOLS = ["list_fetched_data", "preview_data", "describe_data"]


class DataOpsAgent(BaseSubAgent):
    """An LLM session specialized for data transformations and analysis.

    Implements a two-phase process:
    1. Think phase: explore data and research function APIs
    2. Execute phase: write computation code with enriched context
    """

    def __init__(
        self,
        adapter: LLMAdapter,
        model_name: str,
        tool_executor,
        verbose: bool = False,
        cancel_event=None,
        event_bus=None,
    ):
        super().__init__(
            adapter=adapter,
            model_name=model_name,
            tool_executor=tool_executor,
            verbose=verbose,
            agent_name="DataOps Agent",
            system_prompt=build_data_ops_prompt(),
            tool_categories=DATAOPS_TOOL_CATEGORIES,
            extra_tool_names=DATAOPS_EXTRA_TOOLS,
            cancel_event=cancel_event,
            event_bus=event_bus,
        )

        # Build think-phase tool schemas
        self._think_tool_schemas: list[FunctionSchema] = []
        for tool_schema in get_tool_schemas(
            categories=THINK_TOOL_CATEGORIES,
            extra_names=THINK_EXTRA_TOOLS,
        ):
            self._think_tool_schemas.append(FunctionSchema(
                name=tool_schema["name"],
                description=tool_schema["description"],
                parameters=tool_schema["parameters"],
            ))

    def _run_think_phase(self, user_request: str) -> str:
        """Phase 1: Research data structure and function APIs.

        Creates an ephemeral chat session with function doc + data inspection
        tools. Runs a tool-calling loop to explore the data and find relevant
        functions, then returns a text summary of findings.

        Args:
            user_request: The user's computation request.

        Returns:
            Text summary of research findings (data context, recommended
            functions, code hints).
        """
        think_prompt = build_data_ops_think_prompt()

        self._event_bus.emit(DEBUG, agent=self.agent_name, level="debug", msg="[DataOps] Think phase: researching data & functions...")

        chat = self.adapter.create_chat(
            model=get_active_model(self.model_name),
            system_prompt=think_prompt,
            tools=self._think_tool_schemas,
            thinking="high",
        )

        self._last_tool_context = "think_initial"
        response = self._send_with_timeout(chat, user_request)
        self._track_usage(response)

        response = run_tool_loop(
            chat=chat,
            response=response,
            tool_executor=self.tool_executor,
            agent_name="DataOps/Think",
            max_total_calls=15,
            max_iterations=6,
            track_usage=self._track_usage,
            cancel_event=self._cancel_event,
            send_fn=lambda msg: self._send_with_timeout(chat, msg),
            adapter=self.adapter,
        )

        text = extract_text_from_response(response)
        if self.verbose and text:
            self._event_bus.emit(DEBUG, agent=self.agent_name, level="debug", msg=f"[DataOps] Think result: {text[:500]}")

        self._event_bus.emit(PROGRESS, agent=self.agent_name, level="debug", msg="[DataOps] Think phase complete")
        return text or ""

    def process_request(self, user_message: str) -> dict:
        """Two-phase process: think then execute.

        Phase 1 (Think): Research data and functions via ephemeral chat.
        Phase 2 (Execute): Run the standard process_request with enriched context.

        Args:
            user_message: The user's computation request.

        Returns:
            Dict with text, failed, errors (same as BaseSubAgent.process_request).
        """
        # Phase 1: Think
        think_context = self._run_think_phase(user_message)

        # Phase 2: Execute with enriched message
        if think_context:
            enriched = (
                f"{user_message}\n\n"
                f"## Research Findings\n{think_context}\n\n"
                f"Now write the code using custom_operation."
            )
        else:
            enriched = user_message

        return super().process_request(enriched)

    def _get_task_prompt(self, task: Task) -> str:
        """Strict task prompt to prevent unnecessary post-compute tool calls."""
        return (
            f"Execute this task: {task.instruction}\n\n"
            "RULES:\n"
            "- Do ONLY what the instruction says. Do NOT add extra steps.\n"
            "- After a successful custom_operation, STOP. "
            "Do NOT call list_fetched_data, describe_data, or preview_data afterward.\n"
            "- If the operation fails due to wrong column names, call preview_data ONCE "
            "to check column names, then retry with corrected code.\n"
            "- Return the output label and point count as concise text."
        )
