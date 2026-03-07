"""
Deterministic prompt quality checks for CI.

Catches regressions in tool references, cross-prompt consistency, forbidden
patterns, token budgets, structural consistency, and tool assignment correctness.

Does NOT duplicate the structural presence tests in test_prompt_builder.py.

Run with: python -m pytest tests/test_prompt_quality.py -v
"""

import json
import re
from pathlib import Path

import pytest

from knowledge.prompt_builder import (
    _build_shared_domain_knowledge,
    build_system_prompt,
    build_planner_agent_prompt,
    build_envoy_prompt,
    build_data_ops_prompt,
    build_viz_plotly_prompt,
    build_insight_prompt,
    build_insight_feedback_prompt,
    build_data_io_prompt,
    build_inline_completion_prompt,
    build_system_prompt_agent_specific,
    build_planner_prompt_agent_specific,
)
from agent.tools import TOOLS
from agent.agent_registry import (
    ORCHESTRATOR_TOOLS,
    ENVOY_TOOLS,
    VIZ_PLOTLY_TOOLS,
    DATAOPS_TOOLS,
    PLANNER_TOOLS,
    DATA_IO_TOOLS,
)

# ---------------------------------------------------------------------------
# All tool names from the TOOLS list in agent/tools.py
# ---------------------------------------------------------------------------
ALL_TOOL_NAMES = frozenset(t["name"] for t in TOOLS)


# ---------------------------------------------------------------------------
# Shared fixture: block network calls (same pattern as test_prompt_builder.py)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Block network calls so prompt builders use only local cache."""
    monkeypatch.setattr(
        "knowledge.metadata_client.requests.get",
        lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("blocked")),
    )


# ---------------------------------------------------------------------------
# Helper: build all prompts once per test session
# ---------------------------------------------------------------------------
_PROMPT_CACHE: dict[str, str] = {}


def _get_all_prompts() -> dict[str, str]:
    """Build all prompts and return a dict keyed by prompt name.

    Cached for the test session — builders are deterministic given the same
    catalog data.
    """
    if _PROMPT_CACHE:
        return _PROMPT_CACHE

    _PROMPT_CACHE["orchestrator"] = build_system_prompt()
    _PROMPT_CACHE["orchestrator_agent_specific"] = build_system_prompt_agent_specific()
    _PROMPT_CACHE["planner"] = build_planner_agent_prompt()
    _PROMPT_CACHE["planner_agent_specific"] = build_planner_prompt_agent_specific()
    _PROMPT_CACHE["mission_psp"] = build_envoy_prompt("PSP")
    _PROMPT_CACHE["dataops"] = build_data_ops_prompt()
    _PROMPT_CACHE["viz"] = build_viz_plotly_prompt()
    _PROMPT_CACHE["insight"] = build_insight_prompt()
    _PROMPT_CACHE["insight_feedback"] = build_insight_feedback_prompt()
    _PROMPT_CACHE["data_io"] = build_data_io_prompt()
    _PROMPT_CACHE["inline"] = build_inline_completion_prompt("Show me")

    return _PROMPT_CACHE


# ---------------------------------------------------------------------------
# Helper: extract backtick-quoted tool-like identifiers from a prompt
# ---------------------------------------------------------------------------
def _extract_backtick_tool_refs(text: str) -> set[str]:
    """Extract backtick-quoted identifiers that look like tool names.

    Filters to names that are snake_case and exist in the known tool set OR
    agent registry lists. Excludes things like `df`, `result`, `pd`, `np`,
    `xr`, short variables, and Plotly property names.
    """
    # Find all `backtick_quoted` identifiers
    candidates = set(re.findall(r"`([a-z][a-z0-9_]+)`", text))

    # Exclude known non-tool identifiers
    non_tools = {
        "df",
        "da",
        "result",
        "pd",
        "np",
        "xr",
        "scipy",
        "pywt",
        "df_SUFFIX",
        "da_SUFFIX",
        "df_BR",
        "df_BT",
        "df_BN",
        "df_Bmag",
        "df_density",
        "df_BGSEc",
        "df_EFLUX_VS_PA_E",
        "da_EFLUX_VS_PA_E",
        "df_PITCHANGLE",
        "df_ENERGY_VALS",
        "data_label",
        "figure_json",
        "source_labels",
        "output_label",
        "time_start",
        "time_end",
        "dataset_id",
        "parameter_id",
        "mission_id",
        "is_timeseries",
        "include_velocity",
        "commentary",
        "skipna",
        "storage_type",
        "median_cadence",
        "time_min",
        "time_max",
        "candidate_datasets",
    }

    return candidates & ALL_TOOL_NAMES - non_tools


# ===================================================================
# Test Class 1: Tool Reference Integrity
# ===================================================================


class TestToolReferenceIntegrity:
    """Verify that tool names referenced in prompts actually exist in TOOLS."""

    def test_orchestrator_tool_refs_exist(self):
        prompts = _get_all_prompts()
        refs = _extract_backtick_tool_refs(prompts["orchestrator"])
        for ref in refs:
            assert ref in ALL_TOOL_NAMES, (
                f"Orchestrator prompt references `{ref}` but it's not in TOOLS"
            )

    def test_planner_tool_refs_exist(self):
        prompts = _get_all_prompts()
        refs = _extract_backtick_tool_refs(prompts["planner"])
        for ref in refs:
            assert ref in ALL_TOOL_NAMES, (
                f"Planner prompt references `{ref}` but it's not in TOOLS"
            )

    def test_mission_tool_refs_exist(self):
        prompts = _get_all_prompts()
        refs = _extract_backtick_tool_refs(prompts["mission_psp"])
        for ref in refs:
            assert ref in ALL_TOOL_NAMES, (
                f"Mission (PSP) prompt references `{ref}` but it's not in TOOLS"
            )

    def test_viz_tool_refs_exist(self):
        prompts = _get_all_prompts()
        refs = _extract_backtick_tool_refs(prompts["viz"])
        for ref in refs:
            assert ref in ALL_TOOL_NAMES, (
                f"Viz prompt references `{ref}` but it's not in TOOLS"
            )

    def test_dataops_tool_refs_exist(self):
        prompts = _get_all_prompts()
        refs = _extract_backtick_tool_refs(prompts["dataops"])
        for ref in refs:
            assert ref in ALL_TOOL_NAMES, (
                f"DataOps prompt references `{ref}` but it's not in TOOLS"
            )

    def test_data_io_tool_refs_exist(self):
        prompts = _get_all_prompts()
        refs = _extract_backtick_tool_refs(prompts["data_io"])
        for ref in refs:
            assert ref in ALL_TOOL_NAMES, (
                f"Data I/O prompt references `{ref}` but it's not in TOOLS"
            )

    def test_registry_tools_exist_in_tools_list(self):
        """All tools in agent_registry must exist in TOOLS."""
        all_registry_tools = set()
        for tool_list in [
            ORCHESTRATOR_TOOLS,
            ENVOY_TOOLS,
            VIZ_PLOTLY_TOOLS,
            DATAOPS_TOOLS,
            PLANNER_TOOLS,
            DATA_IO_TOOLS,
        ]:
            all_registry_tools.update(tool_list)

        # SPICE tools are registered dynamically — exclude them
        for name in all_registry_tools:
            # SPICE tools (get_spacecraft_ephemeris, etc.) are MCP tools, not in TOOLS
            if (
                name.startswith("get_spacecraft")
                or name.startswith("compute_distance")
                or name.startswith("transform_coord")
                or name.startswith("list_spice")
                or name.startswith("list_coordinate")
                or name.startswith("manage_kernels")
            ):
                continue
            assert name in ALL_TOOL_NAMES, (
                f"Registry tool `{name}` not found in agent/tools.py TOOLS list"
            )

    def test_no_deleted_tool_refs_in_prompts(self):
        """Prompts should not reference tools that were previously deleted."""
        deleted_tools = {
            "plot_data",
            "plot_computed_data",
            "set_render_type",
            "set_color_table",
            "save_session",
            "load_session",
            "compute_magnitude",
            "compute_average",
            "add_panel",
            "remove_panel",
        }
        prompts = _get_all_prompts()
        for name, text in prompts.items():
            for deleted in deleted_tools:
                assert f"`{deleted}`" not in text, (
                    f"Prompt '{name}' references deleted tool `{deleted}`"
                )


# ===================================================================
# Test Class 2: Cross-Prompt Consistency
# ===================================================================


class TestCrossPromptConsistency:
    """Verify shared domain knowledge is consistent across orchestrator and planner."""

    def test_shared_domain_knowledge_no_today(self):
        """Shared domain knowledge should not contain {today} placeholder.

        The date is now injected via ephemeral context at each turn, not baked
        into the static system prompt.
        """
        shared = _build_shared_domain_knowledge()
        assert "{today}" not in shared

    def test_both_have_domain_rules(self):
        prompts = _get_all_prompts()
        assert "## Domain Rules" in prompts["orchestrator"]
        assert "## Domain Rules" in prompts["planner"]

    def test_both_have_error_recovery(self):
        prompts = _get_all_prompts()
        assert "## Error Recovery" in prompts["orchestrator"]
        assert "## Error Recovery" in prompts["planner"]

    def test_critical_viz_rules_in_domain_rules(self):
        """Key visualization constraints must exist in Domain Rules (shared knowledge)."""
        prompts = _get_all_prompts()
        for name in ("orchestrator", "planner"):
            text = prompts[name]
            assert "## Domain Rules" in text, f"{name} missing Domain Rules section"
            # Units on same panel rule is implicit in viz agent; log transform rule is critical
            assert "log transform" in text.lower() or "Log scale" in text, (
                f"{name} missing log transform rule in Domain Rules"
            )
            assert "30 shape" in text or "Shape/annotation limit" in text, (
                f"{name} missing shape/annotation limit in Domain Rules"
            )

    def test_mission_boundary_complementarity(self):
        """Mission prompts should forbid plotting; viz prompts should forbid fetching."""
        prompts = _get_all_prompts()
        # Mission forbids plotting
        assert "Do NOT attempt to plot" in prompts["mission_psp"]
        # Viz prompt should not reference fetch_data as something it can call
        viz_text = prompts["viz"]
        assert "fetch_data" not in viz_text or "Do NOT" in viz_text


# ===================================================================
# Test Class 3: Forbidden Patterns
# ===================================================================


class TestForbiddenPatterns:
    """Detect stale references, old project names, and forbidden tool mentions."""

    def test_no_base_sub_agent_reference(self):
        """BaseSubAgent class was deleted — should not appear in prompts."""
        prompts = _get_all_prompts()
        for name, text in prompts.items():
            assert "BaseSubAgent" not in text, (
                f"Prompt '{name}' references deleted class BaseSubAgent"
            )

    def test_no_use_agents_flag(self):
        """USE_AGENTS config flag was removed."""
        prompts = _get_all_prompts()
        for name, text in prompts.items():
            assert "USE_AGENTS" not in text, (
                f"Prompt '{name}' references removed config flag USE_AGENTS"
            )

    def test_no_old_project_name(self):
        """The project was renamed from Helion to xhelio."""
        prompts = _get_all_prompts()
        for name, text in prompts.items():
            # Case-insensitive check but exclude legitimate references
            # like "heliophysics" or "heliospice"
            lower = text.lower()
            # Find "helion" that's not part of "heliophysics", "heliospice", "helio"
            matches = list(re.finditer(r"\bhelion\b", lower))
            assert not matches, f"Prompt '{name}' contains old project name 'Helion'"

    def test_no_custom_operation_in_mission_prompt(self):
        """Mission prompts should not reference custom_operation (moved to DataOps)."""
        prompts = _get_all_prompts()
        assert "custom_operation" not in prompts["mission_psp"], (
            "Mission (PSP) prompt references custom_operation — should be DataOps only"
        )

    def test_no_dataset_ids_in_routing_table(self):
        """The orchestrator's routing table should NOT contain dataset IDs."""
        prompts = _get_all_prompts()
        orch = prompts["orchestrator"]
        if "## Supported Missions" in orch:
            routing_start = orch.index("## Supported Missions")
            # Find next ## section
            next_section = orch.find("\n## ", routing_start + 1)
            routing_section = (
                orch[routing_start:next_section]
                if next_section != -1
                else orch[routing_start:]
            )
            # These are well-known dataset IDs that should NOT appear in routing table
            forbidden_ids = [
                "PSP_FLD_L2_MAG_RTN_1MIN",
                "AC_H2_MFI",
                "WI_H2_MFI",
                "OMNI_HRO_1MIN",
                "SO_MAG",
            ]
            for ds_id in forbidden_ids:
                assert ds_id not in routing_section, (
                    f"Routing table contains dataset ID '{ds_id}' — should be slim"
                )

    def test_no_hardcoded_dates_in_prompts(self):
        """Prompts should use {today} placeholder or relative expressions, not hardcoded years
        as the current date. Example dates in documentation are fine."""
        prompts = _get_all_prompts()
        for name, text in prompts.items():
            # Check for patterns like "Today is 2025" or "Current date: 2024"
            assert not re.search(
                r"(?:today(?:'s date)? is|current date[:\s]+)\s*20\d{2}",
                text,
                re.IGNORECASE,
            ), f"Prompt '{name}' has a hardcoded current date"

    def test_no_base_agent_py_reference(self):
        """base_agent.py was deleted — should not appear in prompts."""
        prompts = _get_all_prompts()
        for name, text in prompts.items():
            assert "base_agent.py" not in text, (
                f"Prompt '{name}' references deleted file base_agent.py"
            )


# ===================================================================
# Test Class 4: Token Budgets
# ===================================================================


class TestTokenBudgets:
    """Verify prompts stay within token budget thresholds."""

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Count tokens using the project's token counter."""
        from agent.token_counter import count_tokens

        return count_tokens(text)

    def test_non_mission_sub_agent_under_5k_tokens(self):
        """Non-mission sub-agent prompts should be under 5000 tokens.

        Mission prompts include per-mission dataset catalogs and are larger.
        """
        prompts = _get_all_prompts()
        sub_agents = [
            "dataops",
            "viz",
            "data_io",
            "insight",
            "insight_feedback",
        ]
        for name in sub_agents:
            tokens = self._count_tokens(prompts[name])
            assert tokens < 5000, (
                f"Sub-agent prompt '{name}' is {tokens} tokens (limit: 5000)"
            )

    def test_mission_sub_agent_under_20k_tokens(self):
        """Mission sub-agent prompts include dataset catalogs — budget is higher."""
        prompts = _get_all_prompts()
        for name in ["mission_psp"]:
            tokens = self._count_tokens(prompts[name])
            assert tokens < 20000, (
                f"Mission prompt '{name}' is {tokens} tokens (limit: 20000)"
            )

    def test_orchestrator_under_15k_tokens(self):
        """Orchestrator prompt (without catalog) should be under 15000 tokens."""
        prompts = _get_all_prompts()
        tokens = self._count_tokens(prompts["orchestrator"])
        assert tokens < 15000, f"Orchestrator prompt is {tokens} tokens (limit: 15000)"

    def test_planner_under_15k_tokens(self):
        """Planner prompt should be under 15000 tokens."""
        prompts = _get_all_prompts()
        tokens = self._count_tokens(prompts["planner"])
        assert tokens < 15000, f"Planner prompt is {tokens} tokens (limit: 15000)"

    def test_inline_under_3k_tokens(self):
        """Inline completion prompt should be under 3000 tokens.

        The inline prompt includes a mission list that scales with the catalog.
        """
        prompts = _get_all_prompts()
        tokens = self._count_tokens(prompts["inline"])
        assert tokens < 3000, f"Inline prompt is {tokens} tokens (limit: 3000)"

    def test_against_baseline(self):
        """If baselines exist, no prompt should grow more than 20% over baseline."""
        baseline_path = Path(__file__).parent.parent / "docs" / "prompt-baselines.json"
        if not baseline_path.exists():
            pytest.skip(
                "No prompt-baselines.json found — run /check-prompts to generate"
            )

        baselines = json.loads(baseline_path.read_text())
        prompts = _get_all_prompts()

        regressions = []
        for name, text in prompts.items():
            if name not in baselines:
                continue
            current = self._count_tokens(text)
            baseline = baselines[name]
            if baseline == 0:
                continue
            growth = (current - baseline) / baseline
            if growth > 0.20:
                regressions.append(
                    f"  {name}: {baseline} -> {current} tokens (+{growth:.0%})"
                )

        assert not regressions, (
            "Prompt token count regressions (>20% over baseline):\n"
            + "\n".join(regressions)
        )


# ===================================================================
# Test Class 5: Structural Consistency
# ===================================================================


class TestStructuralConsistency:
    """Verify structural properties of prompts (headings, placeholders)."""

    def test_no_duplicate_h2_headings(self):
        """No prompt should have duplicate ## headings."""
        prompts = _get_all_prompts()
        for name, text in prompts.items():
            headings = re.findall(r"^## (.+)$", text, re.MULTILINE)
            seen = set()
            dupes = []
            for h in headings:
                if h in seen:
                    dupes.append(h)
                seen.add(h)
            assert not dupes, f"Prompt '{name}' has duplicate ## headings: {dupes}"

    def test_no_unresolved_placeholders(self):
        """No prompt should have unresolved {placeholders} except known ones.

        Excludes {{...}} (JSON template syntax in planner examples),
        {today} (runtime placeholder), and documentation template variables
        like {SPACECRAFT} and {suffix} in SPICE labeling conventions.
        """
        prompts = _get_all_prompts()
        # Allowed: runtime placeholders + documentation template variables
        allowed = {"today", "SPACECRAFT", "suffix"}

        for name, text in prompts.items():
            # Remove {{...}} patterns first (JSON template syntax)
            cleaned = re.sub(r"\{\{[^}]*\}\}", "", text)
            # Find remaining {word} patterns
            placeholders = set(re.findall(r"\{(\w+)\}", cleaned))
            unexpected = placeholders - allowed
            assert not unexpected, (
                f"Prompt '{name}' has unresolved placeholders: {unexpected}"
            )

    def test_all_prompts_have_identity_statement(self):
        """Every agent prompt should start with 'You are...' identity statement."""
        prompts = _get_all_prompts()
        identity_prompts = [
            "orchestrator",
            "planner",
            "mission_psp",
            "dataops",
            "viz",
            "insight",
            "data_io",
        ]
        for name in identity_prompts:
            text = prompts[name]
            assert text.strip().startswith("You are"), (
                f"Prompt '{name}' doesn't start with 'You are...' identity statement"
            )


# ===================================================================
# Test Class 6: Tool Assignment Consistency
# ===================================================================


class TestToolAssignmentConsistency:
    """Verify prompts don't reference tools outside their agent's assignment."""

    def test_mission_prompt_no_viz_or_delegation_tools(self):
        """Mission prompt should not reference viz or delegation tools."""
        prompts = _get_all_prompts()
        mission = prompts["mission_psp"]
        forbidden = [
            "render_plotly_json",
            "manage_plot",
            "delegate_to_envoy",
            "delegate_to_viz",
            "delegate_to_data_ops",
            "delegate_to_data_io",
            "delegate_to_insight",
            "delegate_to_planner",
        ]
        for tool in forbidden:
            assert f"`{tool}`" not in mission, (
                f"Mission prompt references `{tool}` — not in mission tool set"
            )

    def test_viz_prompt_no_fetch_or_discovery_tools(self):
        """Viz prompt should not reference fetch or discovery tools."""
        prompts = _get_all_prompts()
        viz = prompts["viz"]
        forbidden = [
            "fetch_data",
            "search_datasets",
            "browse_datasets",
            "get_dataset_docs",
        ]
        for tool in forbidden:
            assert f"`{tool}`" not in viz, (
                f"Viz prompt references `{tool}` — not in viz tool set"
            )

    def test_dataops_prompt_no_fetch_or_plot_tools(self):
        """DataOps prompt should not reference fetch or plot tools."""
        prompts = _get_all_prompts()
        dataops = prompts["dataops"]
        forbidden = [
            "fetch_data",
            "render_plotly_json",
            "manage_plot",
            "search_datasets",
            "browse_datasets",
        ]
        for tool in forbidden:
            assert f"`{tool}`" not in dataops, (
                f"DataOps prompt references `{tool}` — not in dataops tool set"
            )
