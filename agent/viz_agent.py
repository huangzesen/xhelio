"""VizAgent — thin BaseAgent subclass for visualization."""

from __future__ import annotations

from uuid import uuid4

from .base_agent import BaseAgent
from .llm import LLMService

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session_context import SessionContext


class VizAgent(BaseAgent):
    """Visualization agent backed by a specific rendering backend.

    The ``backend`` parameter (e.g. ``"plotly"``, ``"mpl"``, ``"jsx"``)
    determines the ``config_key`` used for model resolution and the
    tool/prompt set resolved automatically from the agent registry.
    """

    agent_type: str = "viz"

    _VALID_BACKENDS = {"plotly", "mpl", "jsx"}
    _BACKEND_ALIASES = {"matplotlib": "mpl"}

    def __init__(
        self,
        *,
        backend: str,
        service: LLMService,
        session_ctx: SessionContext | None = None,
        **kwargs,
    ):
        backend = self._BACKEND_ALIASES.get(backend, backend)
        if backend not in self._VALID_BACKENDS:
            raise ValueError(
                f"Unknown viz backend {backend!r}; "
                f"expected one of {sorted(self._VALID_BACKENDS)}"
            )

        # Must be set before super().__init__() because config_key reads it.
        self.backend = backend

        system_prompt = f"You are a {backend} visualization agent."
        try:
            from knowledge.prompt_builder import build_viz_system_prompt
            system_prompt = build_viz_system_prompt(backend=backend)
        except (ImportError, AttributeError):
            pass

        super().__init__(
            agent_id=f"viz:{backend}:{uuid4().hex[:6]}",
            service=service,
            system_prompt=system_prompt,
            session_ctx=session_ctx,
            **kwargs,
        )

    @property
    def config_key(self) -> str:
        """Return backend-specific config key (e.g. ``viz_plotly``)."""
        return f"viz_{self.backend}"
