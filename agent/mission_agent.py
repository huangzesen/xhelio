"""
Mission-specific sub-agent for executing tasks within a single mission's context.

Each MissionAgent gets a focused system prompt (via build_mission_prompt)
and its own Gemini chat session, so it has deep knowledge of one mission's
data products without context pollution from other missions.
"""

from typing import Optional

from .llm import LLMAdapter
from .base_agent import BaseSubAgent
from .tasks import Task
from .event_bus import DEBUG
from knowledge.prompt_builder import build_mission_prompt

# Mission sub-agents get discovery + fetch tools — compute is handled by DataOpsAgent
MISSION_TOOL_CATEGORIES = ["discovery", "data_ops_fetch", "conversation", "spice"]
MISSION_EXTRA_TOOLS = ["list_fetched_data", "review_memory"]


class MissionAgent(BaseSubAgent):
    """An LLM session specialized for one spacecraft mission.

    Attributes:
        mission_id: Spacecraft key (e.g., "PSP", "ACE")
    """

    def __init__(
        self,
        mission_id: str,
        adapter: LLMAdapter,
        model_name: str,
        tool_executor,
        verbose: bool = False,
        cancel_event=None,
        event_bus=None,
    ):
        self.mission_id = mission_id
        super().__init__(
            adapter=adapter,
            model_name=model_name,
            tool_executor=tool_executor,
            verbose=verbose,
            agent_name=f"{mission_id} Agent",
            system_prompt=build_mission_prompt(mission_id),
            tool_categories=MISSION_TOOL_CATEGORIES,
            extra_tool_names=MISSION_EXTRA_TOOLS,
            cancel_event=cancel_event,
            event_bus=event_bus,
        )

    # ---- Hook overrides ----

    def _on_tool_result(self, tool_name: str, tool_args: dict, result: dict) -> Optional[str]:
        """Intercept clarification_needed results and return formatted question."""
        if result.get("status") == "clarification_needed":
            question = result["question"]
            if result.get("context"):
                question = f"{result['context']}\n\n{question}"
            if result.get("options"):
                question += "\n\nOptions:\n" + "\n".join(
                    f"  {i+1}. {opt}" for i, opt in enumerate(result["options"])
                )
            return question
        return None

    def _should_skip_function_call(self, function_calls: list) -> bool:
        """Skip clarification requests in task execution mode."""
        if any(fc.name == "ask_clarification" for fc in function_calls):
            self._event_bus.emit(DEBUG, agent=self.agent_name, level="debug", msg=f"[{self.agent_name}] Skipping clarification request")
            return True
        return False

    def _get_task_prompt(self, task: Task) -> str:
        """Task prompt — allows dataset inspection when candidates are provided."""
        has_candidates = "Candidate datasets to inspect:" in task.instruction

        if has_candidates:
            return (
                f"Execute this task: {task.instruction}\n\n"
                "RULES:\n"
                "- Inspect the candidate datasets by calling list_parameters for each.\n"
                "- If list_parameters returns parameters marked NOT_IN_DATA_FILES, call "
                "inspect_dataset to see which variables actually exist in the data files.\n"
                "- Select the best dataset and parameters for the physical quantity requested.\n"
                "- If fetch_data returns an error about high NaN percentage (>25%), or returns "
                "an error about all-NaN data, skip that parameter and try a different candidate dataset.\n"
                "- Call fetch_data for each selected parameter.\n"
                "- After ALL fetch_data calls succeed, STOP IMMEDIATELY. Do NOT call "
                "list_fetched_data, get_data_availability, get_dataset_docs, or describe_data.\n"
                "- Return the stored label(s) and point count as concise text."
            )
        else:
            return (
                f"Execute this task: {task.instruction}\n\n"
                "RULES:\n"
                "- Do ONLY what the instruction says. Do NOT add extra steps.\n"
                "- After a successful fetch_data call, STOP. Do NOT call list_fetched_data, "
                "get_data_availability, list_parameters, describe_data, or get_dataset_docs.\n"
                "- Return the stored label and point count as concise text."
            )

    def _get_error_context(self, **kwargs) -> dict:
        """Add mission_id to error context."""
        kwargs["mission"] = self.mission_id
        return kwargs
