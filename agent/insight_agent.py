"""
InsightAgent — multimodal plot analysis agent.

Receives a rendered plot image (PNG) and data context, sends them to the LLM
via multimodal message, and returns a scientific interpretation.

Handles two message types via content dict:
    - {"action": "analyze", "image_bytes": ..., "data_context": ..., ...}
    - {"action": "review", "image_bytes": ..., "data_context": ..., ...}
"""

from __future__ import annotations

from .sub_agent import SubAgent, Message
from .llm_utils import _CancelledDuringLLM, send_with_timeout
from .event_bus import EventBus, INSIGHT_RESULT, INSIGHT_FEEDBACK
from .llm import LLMAdapter
from .logging import get_logger, log_error
from .model_fallback import get_active_model
from knowledge.prompt_builder import build_insight_prompt, build_insight_feedback_prompt


class InsightAgent(SubAgent):
    """A SubAgent specialized for multimodal plot analysis.

    Handles two message types via content dict:
        - {"action": "analyze", "image_bytes": ..., "data_context": ..., ...}
        - {"action": "review", "image_bytes": ..., "data_context": ..., ...}
    """

    def __init__(
        self,
        adapter: LLMAdapter,
        model_name: str,
        tool_executor,
        *,
        event_bus: EventBus | None = None,
    ):
        super().__init__(
            agent_id="InsightAgent",
            adapter=adapter,
            model_name=model_name,
            tool_executor=tool_executor,
            system_prompt=build_insight_prompt(),
            tool_schemas=[],  # No tools — pure text/image generation
            event_bus=event_bus,
        )

    def _handle_request(self, msg: Message) -> None:
        """Route to analyze or review based on content dict."""
        content = msg.content
        if not isinstance(content, dict):
            self._deliver_result(msg, {
                "text": "InsightAgent expects a dict with 'action' key.",
                "failed": True,
                "errors": ["Invalid message format"],
            })
            return

        action = content.get("action", "analyze")
        if action == "review":
            result = self._do_review(content)
        else:
            result = self._do_analyze(content)
        self._deliver_result(msg, result)

    def _do_analyze(self, content: dict) -> dict:
        """Send plot image + context to LLM, return analysis text."""
        image_bytes = content.get("image_bytes", b"")
        data_context = content.get("data_context", "")
        user_request = content.get("user_request", "")
        mime_type = content.get("mime_type", "image/png")

        self._event_bus.emit(
            INSIGHT_RESULT, agent=self.agent_id, level="debug",
            msg=f"[{self.agent_id}] Analyzing plot...",
            data={"text": "Analyzing plot..."},
        )

        try:
            prompt_parts = []
            if user_request:
                prompt_parts.append(f"User request: {user_request}")
            prompt_parts.append(f"\nData context:\n{data_context}")
            prompt_parts.append("\nAnalyze the attached plot image using the data context above.")
            prompt_text = "\n".join(prompt_parts)

            chat = self.adapter.create_chat(
                model=get_active_model(self.model_name),
                system_prompt=build_insight_prompt(),
                tools=None,
                thinking="insight",
            )

            multimodal_msg = self.adapter.make_multimodal_message(
                text=prompt_text,
                image_bytes=image_bytes,
                mime_type=mime_type,
            )

            response = send_with_timeout(
                chat=chat, message=multimodal_msg,
                timeout_pool=self._timeout_pool,
                cancel_event=self._cancel_event,
                retry_timeout=180,
                agent_name=self.agent_id,
                logger=get_logger(),
            )
            self._track_usage(response)

            analysis_text = response.text or "No analysis produced."
            self._event_bus.emit(
                INSIGHT_RESULT, agent=self.agent_id, level="info",
                msg=f"[{self.agent_id}] Analysis complete",
                data={"text": analysis_text},
            )
            return {"text": analysis_text, "failed": False, "errors": []}

        except _CancelledDuringLLM:
            return {"text": "Analysis interrupted.", "failed": True, "errors": ["Interrupted"]}
        except Exception as e:
            log_error(f"{self.agent_id} analysis failed", exc=e)
            return {"text": f"Analysis failed: {e}", "failed": True, "errors": [str(e)]}

    def _do_review(self, content: dict) -> dict:
        """Review a rendered figure for correctness and quality."""
        image_bytes = content.get("image_bytes", b"")
        data_context = content.get("data_context", "")
        user_request = content.get("user_request", "")
        conversation_history = content.get("conversation_history", "")
        mime_type = content.get("mime_type", "image/png")

        self._event_bus.emit(
            INSIGHT_FEEDBACK, agent=self.agent_id, level="debug",
            msg=f"[{self.agent_id}] Reviewing figure...",
            data={"text": "Reviewing figure..."},
        )

        try:
            prompt_parts = [f"User's original request: {user_request}"]
            if conversation_history:
                prompt_parts.append(f"\nConversation history:\n{conversation_history}")
            prompt_parts.append(f"\nData context:\n{data_context}")
            prompt_parts.append("\nReview the attached figure against the user's request.")
            prompt_text = "\n".join(prompt_parts)

            chat = self.adapter.create_chat(
                model=get_active_model(self.model_name),
                system_prompt=build_insight_feedback_prompt(),
                tools=None,
                thinking="insight",
            )

            multimodal_msg = self.adapter.make_multimodal_message(
                text=prompt_text,
                image_bytes=image_bytes,
                mime_type=mime_type,
            )

            response = send_with_timeout(
                chat=chat, message=multimodal_msg,
                timeout_pool=self._timeout_pool,
                cancel_event=self._cancel_event,
                retry_timeout=180,
                agent_name=self.agent_id,
                logger=get_logger(),
            )
            self._track_usage(response)

            review_text = response.text or "No review produced."

            # Parse verdict
            passed = True
            suggestions = []
            for line in review_text.splitlines():
                stripped = line.strip().upper()
                if stripped.startswith("VERDICT:"):
                    verdict_value = stripped.split(":", 1)[1].strip()
                    passed = verdict_value == "PASS"
                elif not passed and line.strip().startswith("- "):
                    suggestions.append(line.strip()[2:])

            self._event_bus.emit(
                INSIGHT_FEEDBACK, agent=self.agent_id, level="info",
                msg=f"[{self.agent_id}] Review: {'PASS' if passed else 'NEEDS_IMPROVEMENT'}",
                data={"text": review_text, "passed": passed},
            )
            return {
                "text": review_text, "passed": passed,
                "suggestions": suggestions, "failed": False, "errors": [],
            }

        except _CancelledDuringLLM:
            return {
                "text": "", "passed": True, "suggestions": [],
                "failed": True, "errors": ["Interrupted"],
            }
        except Exception as e:
            log_error(f"{self.agent_id} review failed", exc=e)
            return {
                "text": "", "passed": True, "suggestions": [],
                "failed": True, "errors": [str(e)],
            }
