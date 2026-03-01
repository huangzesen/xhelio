"""Visualization tool handlers."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent


def handle_render_plotly_json(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    return orch._handle_render_plotly_json(tool_args)


def handle_manage_plot(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    return orch._handle_manage_plot(tool_args)


def handle_generate_mpl_script(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    return orch._handle_generate_mpl_script(tool_args)


def handle_manage_mpl_output(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    return orch._handle_manage_mpl_output(tool_args)


def handle_generate_jsx_component(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    return orch._handle_generate_jsx_component(tool_args)


def handle_manage_jsx_output(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    return orch._handle_manage_jsx_output(tool_args)
