"""
InsightAgent — multimodal plot analysis agent.

Receives a rendered plot image (PNG) and data context, sends them to the LLM
via multimodal message, and returns analysis based on the orchestrator's request.

Content dict:
    {"image_bytes": ..., "data_context": ..., "user_request": ..., "mime_type": ...}

The orchestrator controls what the agent analyzes via user_request — scientific
interpretation, quality check, issue detection, etc.  The system prompt is
generic and structured; the user_request steers the response.
"""

from __future__ import annotations

import threading

from .sub_agent import SubAgent, Message
from .llm_utils import _CancelledDuringLLM, send_with_timeout
from .event_bus import EventBus, INSIGHT_RESULT, INSIGHT_FEEDBACK
from .llm import LLMAdapter
from .llm.minimax_adapter import MiniMaxAdapter
from .logging import get_logger, log_error
from .model_fallback import get_active_model
from knowledge.prompt_builder import build_insight_prompt


class InsightAgent(SubAgent):
    """A SubAgent specialized for multimodal plot analysis."""

    def __init__(
        self,
        adapter: LLMAdapter,
        model_name: str,
        tool_executor,
        *,
        event_bus: EventBus | None = None,
        cancel_event: threading.Event | None = None,
    ):
        super().__init__(
            agent_id="InsightAgent",
            adapter=adapter,
            model_name=model_name,
            tool_executor=tool_executor,
            system_prompt=build_insight_prompt(),
            tool_schemas=[],  # No tools — pure text/image generation
            event_bus=event_bus,
            cancel_event=cancel_event,
        )

    def _is_minimax(self) -> bool:
        """Check if the active adapter is MiniMax."""
        return isinstance(self.adapter, MiniMaxAdapter)

    def _mcp_understand_image(
        self, prompt_text: str, image_bytes: bytes, mime_type: str = "image/png"
    ) -> str:
        """Analyze an image via MiniMax MCP understand_image tool."""
        import tempfile
        import os
        from .minimax_mcp_client import get_minimax_mcp_client

        ext = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
        }.get(mime_type, ".png")

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name

        try:
            client = get_minimax_mcp_client()
            result = client.call_tool(
                "understand_image",
                {
                    "prompt": prompt_text,
                    "image_source": temp_path,
                },
            )

            if result.get("status") == "error":
                logger = get_logger()
                logger.warning(
                    "MiniMax understand_image error: %s", result.get("message")
                )
                return f"Image analysis unavailable: {result.get('message', 'unknown error')}"

            return result.get("text", "") or result.get("answer", "") or str(result)
        except Exception as e:
            logger = get_logger()
            logger.warning("MiniMax understand_image failed: %s", e)
            return f"Image analysis failed: {e}"
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def _handle_request(self, msg: Message) -> None:
        """Analyze the plot image according to the user_request."""
        content = msg.content
        if not isinstance(content, dict):
            self._deliver_result(
                msg,
                {
                    "text": "InsightAgent expects a dict with image_bytes and user_request.",
                    "failed": True,
                    "errors": ["Invalid message format"],
                },
            )
            return

        result = self._do_analyze(content)

        # For backward compat with automated review callers: parse VERDICT if present
        action = content.get("action", "analyze")
        if action == "review":
            text = result.get("text", "")
            passed = True
            suggestions = []
            for line in text.splitlines():
                stripped = line.strip().upper()
                if stripped.startswith("VERDICT:"):
                    verdict_value = stripped.split(":", 1)[1].strip()
                    passed = verdict_value == "PASS"
                elif not passed and line.strip().startswith("- "):
                    suggestions.append(line.strip()[2:])
            result["passed"] = passed
            result["suggestions"] = suggestions
            self._event_bus.emit(
                INSIGHT_FEEDBACK,
                agent=self.agent_id,
                level="info",
                msg=f"[{self.agent_id}] Review: {'PASS' if passed else 'NEEDS_IMPROVEMENT'}",
                data={"text": text, "passed": passed},
            )

        self._deliver_result(msg, result)

    def _do_analyze(self, content: dict) -> dict:
        """Send plot image + context to LLM, return analysis text."""
        image_bytes = content.get("image_bytes", b"")
        data_context = content.get("data_context", "")
        user_request = content.get("user_request", "")
        mime_type = content.get("mime_type", "image/png")

        event_type = INSIGHT_RESULT

        self._event_bus.emit(
            event_type,
            agent=self.agent_id,
            level="debug",
            msg=f"[{self.agent_id}] Analyzing plot...",
            data={"text": "Analyzing plot..."},
        )

        try:
            prompt_parts = []
            if user_request:
                prompt_parts.append(f"User request: {user_request}")
            prompt_parts.append(f"\nData context:\n{data_context}")
            prompt_parts.append(
                "\nAnalyze the attached plot image based on the user request above."
            )
            prompt_text = "\n".join(prompt_parts)

            if self._is_minimax():
                analysis_text = self._mcp_understand_image(
                    prompt_text, image_bytes, mime_type
                )
                self._event_bus.emit(
                    event_type,
                    agent=self.agent_id,
                    level="info",
                    msg=f"[{self.agent_id}] Analysis complete",
                    data={"text": analysis_text},
                )
                return {"text": analysis_text, "failed": False, "errors": []}

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
                chat=chat,
                message=multimodal_msg,
                timeout_pool=self._timeout_pool,
                cancel_event=self._cancel_event,
                retry_timeout=180,
                agent_name=self.agent_id,
                logger=get_logger(),
            )
            self._track_usage(response)

            analysis_text = response.text or "No analysis produced."
            self._event_bus.emit(
                event_type,
                agent=self.agent_id,
                level="info",
                msg=f"[{self.agent_id}] Analysis complete",
                data={"text": analysis_text},
            )
            return {"text": analysis_text, "failed": False, "errors": []}

        except _CancelledDuringLLM:
            return {
                "text": "Analysis interrupted.",
                "failed": True,
                "errors": ["Interrupted"],
            }
        except Exception as e:
            log_error(f"{self.agent_id} analysis failed", exc=e)
            return {"text": f"Analysis failed: {e}", "failed": True, "errors": [str(e)]}
