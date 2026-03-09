"""
Dynamic prompt generation from the mission catalog.

Generates prompt sections for the agent system prompt and planner prompt
from the single source of truth in catalog.py and per-mission JSON files.

The main agent gets a slim routing table (no dataset IDs or analysis tips).
Mission sub-agents get rich, focused prompts with full domain knowledge.

Architecture: Prompt prose lives in knowledge/prompts/*.md (markdown files).
Each build_*() function below is a *manifest* — it lists which sections to
load and in what order, with dynamic substitutions. Only dynamic content
generators (catalog tables, mission profiles, inline completions) and
conditional logic remain as Python code.
"""

import config
from .catalog import MISSIONS, classify_instrument_type
from .mission_loader import (
    load_mission,
    load_all_missions,
    get_mission_datasets,
)
from .prompt_loader import assemble, load_section


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _preferred_viz_tool() -> str:
    """Return the config-preferred visualization delegation tool name."""
    return "delegate_to_viz"


def _viz_tool_for_planner() -> dict:
    """Return viz tool metadata for planner prompts."""
    pref = config.PREFER_VIZ_BACKEND
    if pref == "matplotlib":
        return {
            "tool_name": "generate_mpl_script",
            "tool_line": "generate_mpl_script(script, ...): Plot data from memory via a matplotlib Python script",
            "instruction_prefix": "Use generate_mpl_script to plot",
        }
    return {
        "tool_name": "render_plotly_json",
        "tool_line": "render_plotly_json(figure_json): Plot data from memory via Plotly figure JSON with data_label placeholders",
        "instruction_prefix": "Use render_plotly_json to plot",
    }


# ---------------------------------------------------------------------------
# Reusable section lists
# ---------------------------------------------------------------------------

SHARED_DOMAIN = [
    "_shared/supported_missions.md",
    "_shared/domain_rules.md",
    "_shared/error_recovery.md",
    "_shared/time_range_handling.md",
    "_shared/temporal_context.md",
    "_shared/commentary.md",
    "_shared/creating_datasets.md",
    "_shared/pipeline_confirmation.md",
    "_shared/response_style.md",
]


# ---------------------------------------------------------------------------
# Section generators — each produces a markdown string
# ---------------------------------------------------------------------------


def generate_mission_overview() -> str:
    """Generate the mission/instruments/example-data table for the system prompt.

    Kept for backward compatibility but now only used in the slim system prompt.
    """
    lines = [
        "| Mission | Instruments | Example Data |",
        "|---------|-------------|--------------|",
    ]
    for sc_id, sc in MISSIONS.items():
        name = sc["name"]
        instruments = ", ".join(inst["name"] for inst in sc["instruments"].values())
        # Summarise from profile if available, else from instrument keywords
        profile = sc.get("profile", {})
        example = profile.get("description", "")
        if not example:
            # Fallback: first two instrument keywords
            all_kw = []
            for inst in sc["instruments"].values():
                from agent.truncation import get_item_limit

                all_kw.extend(
                    inst["keywords"][: get_item_limit("items.mission_keywords")]
                )
            example = ", ".join(dict.fromkeys(all_kw))  # unique, ordered
        # Truncate to keep table readable
        from agent.truncation import trunc

        example = trunc(example, "context.mission_example")
        lines.append(f"| {name} ({sc_id}) | {instruments} | {example} |")
    return "\n".join(lines)


# Backward-compatible alias
generate_spacecraft_overview = generate_mission_overview


def generate_dataset_quick_reference() -> str:
    """Generate the known-dataset-ID table for the system prompt.

    Lists dataset IDs and types. Parameter details come from
    list_parameters at runtime — not hardcoded here.
    """
    lines = [
        "| Mission | Dataset ID | Type | Notes |",
        "|---------|------------|------|-------|",
    ]
    for sc_id, sc in MISSIONS.items():
        name = sc["name"]
        for inst_id, inst in sc["instruments"].items():
            dtype = classify_instrument_type(inst["keywords"]).capitalize()
            for ds in inst["datasets"]:
                lines.append(f"| {name} | {ds} | {dtype} | use envoy_query to explore |")
    return "\n".join(lines)


def generate_planner_dataset_reference() -> str:
    """Generate the dataset reference block for the planner prompt.

    Lists all instrument-level datasets from JSON files.
    """
    missions = load_all_missions()
    lines = []
    for mission_id, mission in missions.items():
        parts = []
        for inst_id, inst in mission.get("instruments", {}).items():
            kind = classify_instrument_type(inst.get("keywords", []))
            for ds_id, ds_info in inst.get("datasets", {}).items():
                parts.append(f"dataset={ds_id} ({kind})")
        lines.append(f"- {mission['name']}: {'; '.join(parts)}")
    return "\n".join(lines)


def generate_mission_profiles() -> str:
    """Generate detailed per-mission context sections.

    Provides domain knowledge (analysis tips, caveats, coordinate systems).
    Parameter-level metadata (units, descriptions) comes from
    list_parameters at runtime via Master CDF.
    """
    sections = []
    for mission_id, sc in MISSIONS.items():
        profile = sc.get("profile")
        if not profile:
            continue
        lines = [f"### {sc['name']} ({mission_id})"]
        lines.append(f"{profile['description']}")
        coords = profile.get('coordinate_systems', [])
        if coords:
            lines.append(f"- Coordinates: {', '.join(coords)}")
        cadence = profile.get('typical_cadence')
        if cadence:
            lines.append(f"- Typical cadence: {cadence}")
        if profile.get("data_caveats"):
            lines.append("- Caveats: " + "; ".join(profile["data_caveats"]))
        if profile.get("analysis_patterns"):
            lines.append("- Analysis tips:")
            for tip in profile["analysis_patterns"]:
                lines.append(f"  - {tip}")
        # List instruments and datasets
        for inst_id, inst in sc["instruments"].items():
            ds_list = ", ".join(inst["datasets"])
            lines.append(f"  **{inst['name']}** ({inst_id}): {ds_list}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _build_shared_domain_knowledge() -> str:
    """Build domain knowledge shared by orchestrator AND planner.

    ALL domain rules, constraints, and knowledge that both agents need
    MUST go here. Agent-specific sections (delegation workflow, JSON
    format, batching rules) go in the respective build_*() functions.

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

# Per-kind prompt configuration. Controls which template directory to use
# and how to render the mission JSON.
_KIND_PROMPT_CONFIG: dict[str, dict] = {
    "cdaweb": {
        "template_dir": "envoy_cdaweb",
        "compress_json": False,
    },
    "ppi": {
        "template_dir": "envoy_ppi",
        "compress_json": False,
    },
    "spice": {
        "template_dir": "envoy_spice",
        "compress_json": True,
    },
}
_DEFAULT_PROMPT_CONFIG: dict = {
    "template_dir": None,  # No kind-specific template
    "compress_json": False,
}


def build_envoy_prompt(mission_id: str) -> str:
    """Generate the system prompt for an envoy agent.

    Three layers assembled in order:
    1. Generic envoy role (knowledge/prompts/envoy/generic_role.md)
    2. Kind-specific prompt (knowledge/prompts/envoy_{kind}/role.md)
    3. Mission JSON rendered as markdown (profile + dataset catalog)

    Plus shared sections (async tools, final summary) at the end.

    Args:
        mission_id: Mission key (e.g., "PSP", "ACE", "SPICE")

    Returns:
        Complete system prompt string.

    Raises:
        KeyError: If mission_id has no JSON data.
    """
    from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY

    kind = ENVOY_KIND_REGISTRY.get_kind(mission_id)
    kind_config = _KIND_PROMPT_CONFIG.get(kind)
    if kind_config is None:
        # Auto-detect runtime kind template
        from .prompt_loader import _PROMPTS_DIR
        if (_PROMPTS_DIR / f"envoy_{kind}").exists():
            kind_config = {"template_dir": f"envoy_{kind}", "compress_json": False}
        else:
            kind_config = _DEFAULT_PROMPT_CONFIG

    # Layer 1: Generic envoy role
    generic_role = load_section("envoy/generic_role.md")

    # Layer 2: Kind-specific prompt
    template_dir = kind_config.get("template_dir")
    if template_dir:
        kind_prompt = load_section(f"{template_dir}/role.md")
    else:
        kind_prompt = ""

    # Layer 3: Mission JSON as markdown
    # Validate mission exists (backward compat for KeyError)
    if mission_id not in MISSIONS and mission_id != "SPICE":
        raise KeyError(mission_id)

    try:
        mission = load_mission(mission_id)
    except Exception:
        # SPICE or other kinds may not have mission JSON
        mission = {"name": mission_id, "profile": {}, "instruments": {}}

    profile = mission.get("profile", {})

    overview_lines = []
    if profile:
        overview_lines.append(f"## Mission: {mission.get('name', mission_id)}")
        if profile.get("description"):
            overview_lines.append(profile["description"])
        coords = profile.get("coordinate_systems", [])
        if coords:
            overview_lines.append(f"- Coordinate system(s): {', '.join(coords)}")
        cadence = profile.get("typical_cadence")
        if cadence:
            overview_lines.append(f"- Typical cadence: {cadence}")
        if profile.get("data_caveats"):
            overview_lines.append("- Data caveats: " + "; ".join(profile["data_caveats"]))
        overview_lines.append("")
    mission_overview = "\n".join(overview_lines)

    compress = kind_config.get("compress_json", False)
    dataset_catalog = _mission_to_markdown(mission, simplified=compress)

    # Shared tail sections
    tail = assemble([
        "_shared/async_tools.md",
        "_shared/final_summary.md",
    ])

    # Assemble all layers
    parts = [generic_role]
    if kind_prompt:
        parts.append(kind_prompt)
    if mission_overview:
        parts.append(mission_overview)
    parts.append(dataset_catalog)
    parts.append(tail)

    return "\n\n".join(parts)


def _mission_to_markdown(mission: dict, *, simplified: bool = False) -> str:
    """Convert a mission JSON dict to a readable markdown dataset catalog.

    Args:
        mission: Full mission dict from load_mission().
        simplified: If True, omit PI, DOI, and truncate long descriptions.
            Use for orchestrator tool results where brevity matters.
    """
    lines = ["## Dataset Catalog", ""]
    for inst_name, inst_data in sorted(mission.get("instruments", {}).items()):
        lines.append(f"### {inst_name}")
        if inst_data.get("keywords"):
            lines.append(f"Keywords: {', '.join(inst_data['keywords'])}")
        lines.append("")
        for ds_id, ds_info in sorted(inst_data.get("datasets", {}).items()):
            desc = ds_info.get("description", "")
            start = ds_info.get("start_date", "?")
            stop = ds_info.get("stop_date", "?")
            lines.append(f"- **{ds_id}**: {desc}")
            lines.append(f"  Coverage: {start} to {stop}")
            if not simplified:
                if ds_info.get("pi_name"):
                    lines.append(f"  PI: {ds_info['pi_name']}")
                if ds_info.get("doi"):
                    lines.append(f"  DOI: {ds_info['doi']}")
        lines.append("")
    return "\n".join(lines)




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
    prompt = assemble(
        [
            "data_ops/full.md",
            "_shared/async_tools.md",
            "_shared/final_summary.md",
        ]
    )

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
# Insight agent prompt builders
# ---------------------------------------------------------------------------


def build_insight_prompt() -> str:
    """Generate the system prompt for the InsightAgent (multimodal plot analysis).

    Returns:
        System prompt string for the InsightAgent.
    """
    return load_section("insight/full.md")


def build_insight_feedback_prompt() -> str:
    """Deprecated — now returns the same prompt as build_insight_prompt().

    The insight agent uses one generic prompt; the orchestrator/caller
    controls the task via user_request.
    """
    return build_insight_prompt()


# ---------------------------------------------------------------------------
# Data I/O sub-agent prompt builder
# ---------------------------------------------------------------------------


def build_data_io_prompt() -> str:
    """Generate the system prompt for the DataIO sub-agent.

    Returns:
        System prompt string for the DataIOAgent.
    """
    return assemble(
        [
            "data_io/full.md",
            "_shared/async_tools.md",
            "_shared/final_summary.md",
        ]
    )


# ---------------------------------------------------------------------------
# Visualization sub-agent prompt builder
# ---------------------------------------------------------------------------


def build_viz_plotly_prompt(gui_mode: bool = False) -> str:
    """Generate the system prompt for the Plotly visualization sub-agent.

    Args:
        gui_mode: If True, append GUI-mode specific instructions.

    Returns:
        System prompt string for the VizAgent[Plotly].
    """
    sections = ["viz_plotly/full.md"]
    if gui_mode:
        sections.append("viz_plotly/gui_mode.md")
    sections.append("_shared/async_tools.md")
    sections.append("_shared/final_summary.md")
    return assemble(sections)


# ---------------------------------------------------------------------------
# Matplotlib visualization prompt builders
# ---------------------------------------------------------------------------


def build_viz_mpl_prompt(gui_mode: bool = False) -> str:
    """Generate the system prompt for the matplotlib visualization sub-agent.

    Args:
        gui_mode: If True, append GUI-mode specific instructions.

    Returns:
        System prompt string for the VizAgent[Mpl].
    """
    sections = ["viz_mpl/full.md"]
    if gui_mode:
        sections.append("viz_mpl/gui_mode.md")
    sections.append("_shared/async_tools.md")
    sections.append("_shared/final_summary.md")
    return assemble(sections)


def build_viz_jsx_prompt(gui_mode: bool = False) -> str:
    """Generate the system prompt for the JSX/Recharts visualization sub-agent.

    Args:
        gui_mode: If True, append GUI-mode specific instructions.

    Returns:
        System prompt string for the VizAgent[JSX].
    """
    sections = ["viz_jsx/full.md"]
    if gui_mode:
        sections.append("viz_jsx/gui_mode.md")
    sections.append("_shared/async_tools.md")
    sections.append("_shared/final_summary.md")
    return assemble(sections)


# ---------------------------------------------------------------------------
# Full prompt assemblers
# ---------------------------------------------------------------------------


def _build_catalog_section(include_catalog: bool) -> str:
    """Build the optional full mission catalog section."""
    if not include_catalog:
        return ""
    return f"""
## Full Mission Catalog

The following catalog lists every dataset available for each mission, grouped by
instrument. Use this to route requests without calling envoy_query — you already
know what exists. Delegate to the envoy agent with the appropriate envoy name.

{generate_mission_profiles()}
"""


def build_system_prompt_agent_specific(include_catalog: bool = False) -> str:
    """Return only the orchestrator-specific prompt sections (no shared domain knowledge).

    Useful for prompt decomposition and testing — separates the
    orchestrator-specific instructions from the shared domain knowledge
    that ``build_system_prompt()`` combines.

    Args:
        include_catalog: If True, include the full mission catalog.

    Returns:
        Orchestrator-specific prompt template.
    """
    viz_tool = _preferred_viz_tool()
    viz_backend = config.PREFER_VIZ_BACKEND
    other_backends = ", ".join(
        f"`{b}`" for b in ("plotly", "matplotlib", "jsx") if b != viz_backend
    )
    catalog_section = _build_catalog_section(include_catalog)

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
        ],
        viz_tool=viz_tool,
    )

    parts = [role_and_questions]
    if catalog_section:
        parts.append(catalog_section)
    parts.append(body_sections)
    return "\n\n".join(parts)


def build_system_prompt(include_catalog: bool = False) -> str:
    """Assemble the complete system prompt — slim orchestrator version.

    The main agent routes requests to mission sub-agents. It does NOT need
    dataset IDs, analysis tips, or detailed mission profiles.

    Composed from: shared domain knowledge +
    orchestrator-specific sections (role, workflow, delegation rules, examples).

    Args:
        include_catalog: If True, include the full mission catalog with all
            dataset IDs and descriptions.

    Returns a template string.
    """
    viz_tool = _preferred_viz_tool()
    viz_backend = config.PREFER_VIZ_BACKEND
    other_backends = ", ".join(
        f"`{b}`" for b in ("plotly", "matplotlib", "jsx") if b != viz_backend
    )
    shared = _build_shared_domain_knowledge()
    catalog_section = _build_catalog_section(include_catalog)

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
        ],
        viz_tool=viz_tool,
    )

    parts = [role_and_questions, shared]
    if catalog_section:
        parts.append(catalog_section)
    parts.append(body_sections)
    return "\n\n".join(parts)


def build_planner_prompt_agent_specific() -> str:
    """Return only the planner-specific prompt sections (no shared domain knowledge).

    Useful for prompt decomposition and testing — separates the
    planner-specific instructions from the shared domain knowledge
    that ``build_planner_agent_prompt()`` combines.

    Returns:
        Planner-specific prompt string (no placeholders).
    """
    viz = _viz_tool_for_planner()

    # Sections that need substitution
    role_and_format = assemble(
        [
            "planner/role.md",
            "planner/response_format.md",
            "planner/available_tools.md",
        ],
        viz_tool_line=viz["tool_line"],
    )

    middle_sections = assemble(
        [
            "planner/mission_tagging.md",
            "planner/batching_rules.md",
            "planner/when_to_stop.md",
            "planner/planning_guidelines.md",
            "planner/dataset_selection.md",
            "planner/task_instruction_format.md",
        ],
        viz_instruction_prefix=viz["instruction_prefix"],
    )

    # Multi-round examples contain JSON with {{ }} — need substitution for viz prefix
    examples = load_section("planner/multi_round_examples.md").format(
        viz_instruction_prefix=viz["instruction_prefix"],
    )

    return "\n\n".join([role_and_format, middle_sections, examples])


def build_planner_agent_prompt() -> str:
    """Assemble the system prompt for the PlannerAgent (chat-based, multi-round).

    Composed from: role description + shared domain knowledge +
    planner-only sections (response format, tools, mission tagging,
    batching, planning guidelines, examples).

    Returns:
        System prompt string (no placeholders — user request comes via chat).
    """
    viz = _viz_tool_for_planner()
    shared = _build_shared_domain_knowledge()

    role_and_format = assemble(
        [
            "planner/role.md",
            "planner/response_format.md",
            "planner/available_tools.md",
        ],
        viz_tool_line=viz["tool_line"],
    )

    middle_sections = assemble(
        [
            "planner/mission_tagging.md",
            "planner/batching_rules.md",
            "planner/when_to_stop.md",
            "planner/planning_guidelines.md",
            "planner/dataset_selection.md",
            "planner/task_instruction_format.md",
        ],
        viz_instruction_prefix=viz["instruction_prefix"],
    )

    examples = load_section("planner/multi_round_examples.md").format(
        viz_instruction_prefix=viz["instruction_prefix"],
    )

    return "\n\n".join([role_and_format, shared, middle_sections, examples])


# ---------------------------------------------------------------------------
# Inline completion prompt builder
# ---------------------------------------------------------------------------


def _build_inline_static_context() -> str:
    """Build the static (cacheable) prefix for inline completion prompts.

    This section is identical across all autocomplete calls in a session,
    enabling Gemini implicit/explicit prompt caching (1,024 token minimum).
    """
    # Slim mission list — just names and key instrument types
    mission_lines = []
    for mission_id, m in MISSIONS.items():
        name = m["name"]
        instruments = ", ".join(inst["name"] for inst in m["instruments"].values())
        mission_lines.append(f"- {name}: {instruments}")
    mission_ref = "\n".join(mission_lines)

    return load_section("inline/static_context.md").format(
        mission_ref=mission_ref,
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
# Eureka agent prompt
# ---------------------------------------------------------------------------


def build_eureka_prompt(max_per_cycle: int = 3) -> str:
    """Build the system prompt for the EurekaAgent.

    Returns the full system prompt string.
    """
    return load_section("eureka/full.md").format(max_per_cycle=max_per_cycle)
