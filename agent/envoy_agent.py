"""EnvoyAgent — per-mission specialist, parameterized by kind and instance."""
from __future__ import annotations

from uuid import uuid4

from .base_agent import BaseAgent


class EnvoyAgent(BaseAgent):
    agent_type = "envoy"

    def __init__(
        self,
        kind: str,
        instance_id: str,
        service,
        session_ctx,
        tool_schemas,
        system_prompt,
        **kwargs,
    ):
        self.kind = kind
        self.instance_id = instance_id

        super().__init__(
            agent_id=f"envoy:{kind}:{instance_id}:{uuid4().hex[:6]}",
            service=service,
            tool_schemas=tool_schemas,
            system_prompt=system_prompt,
            session_ctx=session_ctx,
            **kwargs,
        )

    @property
    def config_key(self) -> str:
        """Per-kind config resolution (e.g., 'envoy_cdaweb').

        Falls back to 'envoy' if no per-kind config exists.
        """
        return f"envoy_{self.kind}"
