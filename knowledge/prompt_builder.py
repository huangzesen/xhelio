"""
Dynamic prompt generation.

Generates prompt sections for the agent system prompt.
Prompt prose lives in knowledge/prompts/*.md (markdown files).
Each build_*() function below is a *manifest* — it lists which sections to
load and in what order, with dynamic substitutions.
"""

import config
from .prompt_loader import assemble, load_section


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _preferred_viz_tool() -> str:
    """Return the config-preferred visualization delegation tool name."""
    return "delegate_to_viz"


# ---------------------------------------------------------------------------
# Reusable section lists
# ---------------------------------------------------------------------------

SHARED_DOMAIN = [
    "_shared/domain_rules.md",
    "_shared/error_recovery.md",
    "_shared/time_range_handling.md",
    "_shared/temporal_context.md",
    "_shared/commentary.md",
    "_shared/response_style.md",
]


# ---------------------------------------------------------------------------
# Section generators — each produces a markdown string
# ---------------------------------------------------------------------------


def _build_shared_domain_knowledge() -> str:
    """Build shared domain knowledge for the orchestrator.

    ALL domain rules, constraints, and knowledge go here.
    Agent-specific sections (delegation workflow, batching rules)
    go in the respective build_*() functions.

    Returns:
        Multi-section markdown string.
    """
    return assemble(SHARED_DOMAIN)


# ---------------------------------------------------------------------------
# Mission-specific prompt builder (for mission sub-agents)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Envoy prompt builder — three-layer assembly
# ---------------------------------------------------------------------------

def build_envoy_prompt(mission_id: str) -> str:
    """Generate the system prompt for an envoy agent.

    Stubbed — will be rebuilt when MCP-backed envoys are re-added.
    """
    raise NotImplementedError(
        f"No envoy prompt for '{mission_id}' — envoy system is being rebuilt"
    )




# ---------------------------------------------------------------------------
# DataOps sub-agent prompt builder
# ---------------------------------------------------------------------------


def build_data_ops_prompt() -> str:
    """Generate the system prompt for the DataOps sub-agent.

    Includes computation patterns, code guidelines, and workflow instructions
    for data transformation and analysis.

    Returns:
        System prompt string for the DataOpsAgent.
    """
    from data_ops.sandbox import build_sandbox_rules_prompt

    prompt = assemble(
        [
            "data_ops/full.md",
            "_shared/async_tools.md",
            "_shared/final_summary.md",
        ]
    )

    # Inject dynamic sandbox restrictions (built from THREAT_CATEGORIES)
    prompt = prompt + "\n\n" + build_sandbox_rules_prompt()

    # Inject saved operations library (if any entries exist)
    try:
        from data_ops.ops_library import get_ops_library

        library_section = get_ops_library().build_prompt_section()
        if library_section:
            prompt = prompt + "\n\n" + library_section
    except Exception:
        pass

    return prompt


# ---------------------------------------------------------------------------
# Data I/O sub-agent prompt builder
# ---------------------------------------------------------------------------


def build_data_io_prompt() -> str:
    """Generate the system prompt for the DataIO sub-agent.

    Returns:
        System prompt string for the DataIOAgent.
    """
    from data_ops.sandbox import build_sandbox_rules_prompt

    prompt = assemble(
        [
            "data_io/full.md",
            "_shared/async_tools.md",
            "_shared/final_summary.md",
        ]
    )

    # Inject dynamic sandbox restrictions (built from THREAT_CATEGORIES)
    prompt = prompt + "\n\n" + build_sandbox_rules_prompt()

    return prompt


# ---------------------------------------------------------------------------
# Visualization sub-agent prompt builder
# ---------------------------------------------------------------------------


def build_viz_prompt(prompt_dir: str, *, gui_mode: bool = False) -> str:
    """Generate the system prompt for any visualization sub-agent.

    Args:
        prompt_dir: Subdirectory under knowledge/prompts/ (e.g., "viz_plotly").
        gui_mode: If True, append GUI-mode specific instructions.

    Returns:
        System prompt string for the visualization agent.
    """
    sections = [f"{prompt_dir}/full.md"]
    if gui_mode:
        sections.append(f"{prompt_dir}/gui_mode.md")
    sections.append("_shared/async_tools.md")
    sections.append("_shared/final_summary.md")
    prompt = assemble(sections)

    # Inject sandbox rules for backends that execute Python code
    if prompt_dir == "viz_mpl":
        from data_ops.sandbox import build_sandbox_rules_prompt
        prompt = prompt + "\n\n" + build_sandbox_rules_prompt()

    return prompt


# ---------------------------------------------------------------------------
# Full prompt assemblers
# ---------------------------------------------------------------------------


def build_system_prompt_agent_specific() -> str:
    """Return only the orchestrator-specific prompt sections (no shared domain knowledge)."""
    viz_tool = _preferred_viz_tool()
    viz_backend = config.PREFER_VIZ_BACKEND
    other_backends = ", ".join(
        f"`{b}`" for b in ("plotly", "matplotlib", "jsx") if b != viz_backend
    )

    role_and_questions = assemble(
        [
            "orchestrator/role.md",
            "orchestrator/answering_questions.md",
        ],
        viz_backend=viz_backend,
        other_backends=other_backends,
    )

    body_sections = assemble(
        [
            "orchestrator/tool_store.md",
            "orchestrator/workflow.md",
            "orchestrator/after_delegation.md",
            "orchestrator/delegation_rule.md",
            "orchestrator/clarification.md",
            "orchestrator/planning_and_delegation.md",
            "orchestrator/permissions.md",
            "orchestrator/examples.md",
            "orchestrator/follow_up_routing.md",
            "_shared/creating_datasets.md",
            "_shared/pipeline_confirmation.md",
        ],
        viz_tool=viz_tool,
    )

    return "\n\n".join([role_and_questions, body_sections])


def build_system_prompt() -> str:
    """Assemble the complete system prompt — slim orchestrator version."""
    viz_tool = _preferred_viz_tool()
    viz_backend = config.PREFER_VIZ_BACKEND
    other_backends = ", ".join(
        f"`{b}`" for b in ("plotly", "matplotlib", "jsx") if b != viz_backend
    )
    shared = _build_shared_domain_knowledge()

    role_and_questions = assemble(
        [
            "orchestrator/role.md",
            "orchestrator/answering_questions.md",
        ],
        viz_backend=viz_backend,
        other_backends=other_backends,
    )

    body_sections = assemble(
        [
            "orchestrator/tool_store.md",
            "orchestrator/workflow.md",
            "orchestrator/after_delegation.md",
            "orchestrator/delegation_rule.md",
            "orchestrator/clarification.md",
            "orchestrator/planning_and_delegation.md",
            "orchestrator/permissions.md",
            "orchestrator/examples.md",
            "orchestrator/follow_up_routing.md",
            "_shared/creating_datasets.md",
            "_shared/pipeline_confirmation.md",
        ],
        viz_tool=viz_tool,
    )

    return "\n\n".join([role_and_questions, shared, body_sections])




# ---------------------------------------------------------------------------
# Inline completion prompt builder
# ---------------------------------------------------------------------------


def _build_inline_static_context() -> str:
    """Build the static (cacheable) prefix for inline completion prompts."""
    return load_section("inline/static_context.md").format(
        mission_ref="(No missions currently registered — use xhelio__envoy_query to check)",
    )


def build_inline_completion_prompt(
    partial: str,
    *,
    conversation_context: str = "",
    memory_section: str = "",
    data_labels: list[str] | None = None,
    max_completions: int = 3,
) -> str:
    """Build the prompt for Copilot-style inline input completion.

    Structured as [static context] + [session context] + [dynamic query]
    to maximize prompt cache hits. The static prefix (~800-1000 tokens)
    stays identical across calls; only the tail changes.

    Args:
        partial: The text the user has typed so far.
        conversation_context: Recent conversation turns (formatted).
        memory_section: Long-term memory section (from MemoryStore).
        data_labels: Labels of data currently in the DataStore.
        max_completions: Number of completions to request.

    Returns:
        Prompt string for the inline completion LLM call.
    """
    # --- Static prefix (cacheable) ---
    parts = [_build_inline_static_context()]

    # Memory is semi-static (changes rarely within a session)
    if memory_section:
        parts.append(f"\n{memory_section}")

    # --- Dynamic suffix (changes per keystroke) ---
    if conversation_context:
        parts.append(f"\nRecent conversation:\n{conversation_context}")

    if data_labels:
        # Lazy import to avoid circular dependency (knowledge → agent)
        from agent.truncation import trunc_items

        shown_labels, _ = trunc_items(data_labels, "items.data_labels")
        parts.append(f"\nData in memory: {', '.join(shown_labels)}")

    if partial:
        parts.append(f'\nThe user is currently typing: "{partial}"')
        parts.append(f"""
Suggest {max_completions} possible complete messages. Each must:
- Start with exactly "{partial}" (case-sensitive)
- Be a single short sentence (max 80 characters total)
- Never combine multiple questions or sentences

Respond with a JSON array of strings only.""")
    else:
        parts.append(f"""
The user has not started typing yet. Suggest {max_completions} example queries
they might want to ask, based on the conversation context and available data.
Each must:
- Be a single short sentence (max 80 characters total)
- Be a natural question or command the user might type

Respond with a JSON array of strings only.""")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Memory agent prompt
# ---------------------------------------------------------------------------


def build_memory_prompt(memory_store=None) -> str:
    """Build the system prompt for the MemoryAgent.

    Injects current memories into the prompt so the agent can see what
    exists and avoid duplicates.
    """
    template = load_section("memory/full.md")
    current_memories = "No memories stored yet."
    if memory_store is not None:
        try:
            memories = memory_store.get_all()
            if memories:
                lines = []
                for m in memories:
                    scopes = ", ".join(m.scopes) if m.scopes else "generic"
                    lines.append(
                        f"- [{m.id}] ({m.type}, scopes={scopes}) {m.content[:200]}"
                    )
                current_memories = "\n".join(lines)
        except Exception:
            pass
    return template.format(current_memories=current_memories)


# ---------------------------------------------------------------------------
# Eureka agent prompt
# ---------------------------------------------------------------------------


def build_eureka_prompt(max_per_cycle: int = 3) -> str:
    """Build the system prompt for the EurekaAgent.

    Returns the full system prompt string.
    """
    return load_section("eureka/full.md").format(max_per_cycle=max_per_cycle)


# ---------------------------------------------------------------------------
# Agent-facing aliases — thin wrappers matching the names agents import
# ---------------------------------------------------------------------------


def build_orchestrator_system_prompt() -> str:
    """Alias for ``build_system_prompt()`` — used by OrchestratorAgent."""
    return build_system_prompt()


def build_viz_system_prompt(backend: str) -> str:
    """Build viz agent prompt from the backend's prompt_dir.

    Maps backend name → prompt directory:
      "plotly" → "viz_plotly", "mpl" → "viz_mpl", "jsx" → "viz_jsx"
    """
    prompt_dir = f"viz_{backend}"
    return build_viz_prompt(prompt_dir)


def build_data_ops_system_prompt() -> str:
    """Alias for ``build_data_ops_prompt()`` — used by DataOpsAgent."""
    return build_data_ops_prompt()


def build_data_io_system_prompt() -> str:
    """Alias for ``build_data_io_prompt()`` — used by DataIOAgent."""
    return build_data_io_prompt()


def build_memory_system_prompt(memory_store=None) -> str:
    """Alias for ``build_memory_prompt()`` — used by MemoryAgent."""
    return build_memory_prompt(memory_store=memory_store)


def build_eureka_system_prompt(max_per_cycle: int = 3) -> str:
    """Alias for ``build_eureka_prompt()`` — used by EurekaAgent."""
    return build_eureka_prompt(max_per_cycle=max_per_cycle)
