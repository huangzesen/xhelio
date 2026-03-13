"""Tool schemas for the memory agent's tool-calling loop.

Defines FunctionSchema objects for memory CRUD, event inspection, and
pipeline registration/discard.
"""

from __future__ import annotations

from .llm import FunctionSchema


def get_memory_tools() -> list[FunctionSchema]:
    """Return the list of tools available to the memory agent."""
    return [
        FunctionSchema(
            name="add_memory",
            description=(
                "Add a new long-term memory entry. Content format depends on "
                "the type:\n"
                "- preference: 1-2 concise sentences capturing the user's "
                "preference.\n"
                "- pitfall / reflection: Structured as "
                "Trigger / Problem / Fix.\n"
                "- summary: Structured as Data / Analysis / Finding."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["preference", "pitfall", "reflection", "summary"],
                        "description": "Category of the memory entry.",
                    },
                    "scopes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Scopes that this memory applies to. Valid values: "
                            '"generic", "visualization", "data_ops", '
                            '"envoy:<MISSION_ID>" (e.g. "envoy:ace").'
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "The memory content, formatted according to the type.",
                    },
                },
                "required": ["type", "scopes", "content"],
            },
        ),
        FunctionSchema(
            name="edit_memory",
            description=(
                "Edit an existing memory by ID. This creates a new version of "
                "the memory (supersede pattern) — the old version is preserved "
                "in the version history but the new content becomes the active "
                "version."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory entry to edit.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The new content that replaces the existing memory.",
                    },
                },
                "required": ["memory_id", "content"],
            },
        ),
        FunctionSchema(
            name="drop_memory",
            description=(
                "Archive a memory by ID, removing it from active use. Be "
                "conservative — only drop a memory when there is strong "
                "evidence that it is outdated, incorrect, or no longer "
                "relevant. When in doubt, prefer editing over dropping."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory entry to archive.",
                    },
                },
                "required": ["memory_id"],
            },
        ),
        FunctionSchema(
            name="register_pipeline",
            description=(
                "Register a pipeline candidate as a reusable workflow. Only "
                "register non-trivial pipelines that involve meaningful data "
                "transformations, multi-step processing, or custom analysis — "
                "not vanilla fetch-and-render sequences."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "render_op_id": {
                        "type": "string",
                        "description": "ID of the render operation that produced this pipeline.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Short, descriptive name for the pipeline.",
                    },
                    "description": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "Data source(s) used by this pipeline.",
                            },
                            "rationale": {
                                "type": "string",
                                "description": "Why this pipeline is worth saving.",
                            },
                            "use_cases": {
                                "type": "string",
                                "description": "When and how to reuse this pipeline.",
                            },
                        },
                        "required": ["source", "rationale", "use_cases"],
                    },
                    "scopes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Scopes this pipeline applies to.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorizing and searching pipelines.",
                    },
                },
                "required": ["render_op_id", "name", "description", "scopes", "tags"],
            },
        ),
        FunctionSchema(
            name="discard_pipeline",
            description=(
                "Discard a pipeline candidate, indicating it should not be "
                "saved. Use this for vanilla fetch-and-render pipelines or "
                "other trivial sequences that do not warrant reuse."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "render_op_id": {
                        "type": "string",
                        "description": "ID of the render operation to discard.",
                    },
                },
                "required": ["render_op_id"],
            },
        ),
    ]
