"""DataIOAgent — structured data extraction and local file import (BaseAgent subclass)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from .llm import LLMService
    from .session_context import SessionContext


class DataIOAgent(BaseAgent):
    """Data I/O agent for structured extraction and file import."""

    agent_type = "data_io"

    def __init__(
        self,
        service: LLMService,
        session_ctx: SessionContext | None = None,
        **kwargs,
    ):
        system_prompt = "You are a data I/O specialist."
        try:
            from knowledge.prompt_builder import build_data_io_system_prompt
            system_prompt = build_data_io_system_prompt()
        except (ImportError, AttributeError):
            pass

        super().__init__(
            agent_id=f"data_io:{uuid4().hex[:6]}",
            service=service,
            system_prompt=system_prompt,
            session_ctx=session_ctx,
            **kwargs,
        )
