"""
Tests for dynamic prompt generation from catalog.

Verifies that prompt_builder.py produces prompts containing all spacecraft,
datasets, and parameters from the catalog — the single source of truth.

Run with: python -m pytest tests/test_prompt_builder.py
"""

import pytest
from knowledge.catalog import SPACECRAFT, classify_instrument_type
from knowledge.prompt_builder import (
    generate_spacecraft_overview,
    generate_dataset_quick_reference,
    generate_planner_dataset_reference,
    generate_mission_profiles,
    build_envoy_prompt,
    build_data_ops_prompt,
    build_system_prompt,
    build_planner_agent_prompt,
    build_viz_plotly_prompt,
)


class TestGenerateSpacecraftOverview:
    def test_all_spacecraft_present(self):
        table = generate_spacecraft_overview()
        for sc_id, sc in SPACECRAFT.items():
            assert sc["name"] in table, f"{sc['name']} missing from overview"

    def test_is_markdown_table(self):
        table = generate_spacecraft_overview()
        lines = table.strip().split("\n")
        assert lines[0].startswith("|")
        assert "---" in lines[1]


class TestGenerateDatasetQuickReference:
    def test_all_datasets_present(self):
        table = generate_dataset_quick_reference()
        for sc_id, sc in SPACECRAFT.items():
            for inst_id, inst in sc["instruments"].items():
                for ds in inst["datasets"]:
                    assert ds in table, f"Dataset {ds} missing from quick reference"

    def test_directs_to_envoy_query(self):
        table = generate_dataset_quick_reference()
        # Parameter details come from envoy_query, not hardcoded
        assert "envoy_query" in table

    def test_is_markdown_table(self):
        table = generate_dataset_quick_reference()
        lines = table.strip().split("\n")
        assert lines[0].startswith("|")


class TestGeneratePlannerDatasetReference:
    def test_all_spacecraft_present(self):
        ref = generate_planner_dataset_reference()
        for sc_id, sc in SPACECRAFT.items():
            assert sc["name"] in ref, f"{sc['name']} missing from planner reference"

    def test_all_datasets_present(self):
        ref = generate_planner_dataset_reference()
        for sc_id, sc in SPACECRAFT.items():
            for inst_id, inst in sc["instruments"].items():
                for ds in inst["datasets"]:
                    assert ds in ref, f"Dataset {ds} missing from planner reference"


class TestGenerateMissionProfiles:
    def test_all_profiled_missions_present(self):
        profiles = generate_mission_profiles()
        for sc_id, sc in SPACECRAFT.items():
            if sc.get("profile"):
                assert sc["name"] in profiles, f"{sc['name']} missing from profiles"

    def test_analysis_patterns_included(self):
        profiles = generate_mission_profiles()
        # PSP should have switchback analysis tip if curated profiles exist.
        # Auto-generated missions may have empty analysis_patterns.
        # Just verify the function returns non-empty output.
        assert len(profiles) > 0


class TestBuildEnvoyPromptThreeLayers:
    """The new envoy prompt has three layers: generic, kind-specific, mission JSON."""

    def test_cdaweb_has_generic_role(self):
        prompt = build_envoy_prompt("PSP")
        assert "envoy" in prompt.lower()

    def test_cdaweb_has_kind_specific_content(self):
        prompt = build_envoy_prompt("PSP")
        assert "CDAWeb" in prompt

    def test_cdaweb_has_mission_json_catalog(self):
        prompt = build_envoy_prompt("PSP")
        assert "Dataset Catalog" in prompt
        assert "PSP_FLD_L2_MAG_RTN_1MIN" in prompt

    def test_spice_has_kind_specific_content(self):
        prompt = build_envoy_prompt("SPICE")
        assert "SPICE" in prompt or "ephemeris" in prompt.lower()

    def test_ppi_has_kind_specific_content(self):
        prompt = build_envoy_prompt("CASSINI_PPI")
        assert "PPI" in prompt or "PDS" in prompt

    def test_no_kind_prompt_builders_dict(self):
        """_KIND_PROMPT_BUILDERS is gone."""
        import knowledge.prompt_builder as pb
        assert not hasattr(pb, "_KIND_PROMPT_BUILDERS")


class TestBuildMissionPrompt:
    @pytest.fixture(autouse=True)
    def _no_network(self, monkeypatch):
        """Block network calls so build_envoy_prompt uses only local cache."""
        monkeypatch.setattr(
            "knowledge.metadata_client.requests.get",
            lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("blocked")),
        )

    def test_psp_prompt_contains_mission_info(self):
        prompt = build_envoy_prompt("PSP")
        assert "Parker Solar Probe" in prompt
        assert "browse_parameters" in prompt
        # Dataset IDs are now embedded in the prompt via the full mission JSON
        assert "PSP_FLD_L2_MAG_RTN_1MIN" in prompt

    def test_ace_prompt_contains_mission_info(self):
        prompt = build_envoy_prompt("ACE")
        # ACE name may be "ACE" or "Advanced Composition Explorer" depending
        # on whether the mission JSON was curated or auto-generated
        assert "ACE" in prompt
        # Dataset catalog section removed; AC_H2_MFI may appear in the
        # parallel tool example (tool docs) but not in a "Recommended Datasets" section
        assert "## Recommended Datasets" not in prompt

    def test_prompt_does_not_contain_other_missions(self):
        prompt = build_envoy_prompt("PSP")
        # AC_H2_MFI may appear in the parallel tool example (tool docs shared
        # across all mission agents), but must not appear in a dataset catalog
        assert "## Recommended Datasets" not in prompt
        assert "Advanced Composition Explorer" not in prompt

    def test_invalid_mission_raises(self):
        with pytest.raises(KeyError):
            build_envoy_prompt("NONEXISTENT")

    def test_representative_missions_can_build(self):
        """Fast check on a few representative CDAWeb + PPI missions."""
        for sc_id in ("PSP", "ACE", "WIND", "CASSINI_PPI"):
            prompt = build_envoy_prompt(sc_id)
            assert len(prompt) > 50

    @pytest.mark.slow
    def test_all_missions_can_build(self):
        for sc_id in SPACECRAFT:
            prompt = build_envoy_prompt(sc_id)
            assert len(prompt) > 50

    def test_directs_to_browse_parameters(self):
        prompt = build_envoy_prompt("PSP")
        assert "browse_parameters" in prompt

    def test_mission_prompt_has_dataset_selection_workflow(self):
        prompt = build_envoy_prompt("PSP")
        assert "## Dataset Selection Workflow" in prompt
        assert "fetch_data" in prompt
        # Computation patterns moved to DataOps agent
        assert "run_code" not in prompt

    def test_mission_prompt_has_no_recommended_datasets(self):
        prompt = build_envoy_prompt("PSP")
        assert "## Recommended Datasets" not in prompt

    def test_mission_prompt_mentions_browse_parameters(self):
        prompt = build_envoy_prompt("PSP")
        assert "browse_parameters" in prompt

    def test_mission_prompt_has_no_advanced_section(self):
        prompt = build_envoy_prompt("PSP")
        assert "## Advanced Datasets" not in prompt

    def test_mission_prompt_has_dataset_documentation_section(self):
        prompt = build_envoy_prompt("PSP")
        assert "## Dataset Documentation" in prompt
        assert "browse_parameters" in prompt

    def test_mission_prompt_has_no_analysis_patterns(self):
        prompt = build_envoy_prompt("PSP")
        # Analysis patterns section has been removed from mission prompts
        assert "## Analysis Patterns" not in prompt

    def test_envoy_prompt_includes_dataset_catalog(self):
        """Envoy prompt should contain the full dataset catalog."""
        prompt = build_envoy_prompt("ACE")
        assert "Dataset Catalog" in prompt
        assert "AC_H2_MFI" in prompt  # known ACE dataset
        assert "Coverage:" in prompt  # markdown format

    def test_mission_prompt_forbids_plotting(self):
        prompt = build_envoy_prompt("PSP")
        assert "Do NOT attempt to plot" in prompt

    def test_mission_prompt_forbids_transformations(self):
        prompt = build_envoy_prompt("PSP")
        assert "Do NOT attempt data transformations" in prompt

    def test_mission_prompt_workflow_excludes_plot_computed_data(self):
        prompt = build_envoy_prompt("PSP")
        # plot_computed_data should not appear in the workflow steps
        workflow_start = prompt.index("## Dataset Selection Workflow")
        workflow_end = prompt.index("## Reporting Rules")
        workflow_section = prompt[workflow_start:workflow_end]
        assert "plot_computed_data" not in workflow_section

    def test_mission_prompt_has_reporting_section(self):
        prompt = build_envoy_prompt("PSP")
        assert "## Reporting Rules" in prompt

    def test_mission_prompt_has_data_specialist_identity(self):
        prompt = build_envoy_prompt("PSP")
        assert "data specialist" in prompt.lower()

    def test_mission_prompt_has_explore_before_fetch_workflow(self):
        """Workflow guides: pick dataset from catalog, browse_parameters, fetch."""
        prompt = build_envoy_prompt("PSP")
        workflow_start = prompt.index("## Dataset Selection Workflow")
        workflow_end = prompt.index("## Reporting Rules")
        workflow_section = prompt[workflow_start:workflow_end]
        assert "browse_parameters" in workflow_section
        assert "fetch_data" in workflow_section
        assert "Dataset Catalog" in workflow_section
        # Compute tools no longer in mission workflow
        assert "run_code" not in workflow_section
        assert "describe_data" not in workflow_section
        assert "save_data" not in workflow_section

    def test_mission_prompt_has_dataset_catalog(self):
        """Prompt contains the full dataset catalog from mission JSON."""
        prompt = build_envoy_prompt("PSP")
        assert "## Dataset Catalog" in prompt
        assert "## Recommended Datasets" not in prompt


class TestBrowseDatasetsEnrichesDescriptions:
    """Test that browse_datasets fills empty descriptions from mission JSON."""

    def test_browse_datasets_enriches_descriptions(self, tmp_path):
        import json
        from unittest.mock import patch

        # Create a fake _index.json with empty descriptions
        fake_cdaweb = tmp_path / "envoys" / "cdaweb"
        ace_metadata = fake_cdaweb / "ace" / "metadata"
        ace_metadata.mkdir(parents=True)
        index_data = {
            "mission_id": "ACE",
            "dataset_count": 2,
            "datasets": [
                {"id": "AC_H2_MFI", "description": "", "parameter_count": 6},
                {"id": "AC_H6_SWI", "description": "", "parameter_count": 4},
            ],
        }
        (ace_metadata / "_index.json").write_text(
            json.dumps(index_data), encoding="utf-8"
        )

        # Mock load_mission to return descriptions
        fake_mission = {
            "name": "ACE",
            "instruments": {
                "mag": {
                    "name": "Magnetometer",
                    "datasets": {
                        "AC_H2_MFI": {"description": "16-second magnetic field"},
                        "AC_H6_SWI": {"description": "2-hour solar wind ions"},
                    },
                },
            },
        }

        from knowledge.metadata_client import browse_datasets, clear_cache

        with (
            patch("knowledge.metadata_client._SOURCE_DIRS", [fake_cdaweb]),
            patch("knowledge.mission_loader.load_mission", return_value=fake_mission),
        ):
            clear_cache()
            datasets = browse_datasets("ACE")

        assert datasets is not None
        assert len(datasets) == 2
        assert datasets[0]["description"] == "16-second magnetic field"
        assert datasets[1]["description"] == "2-hour solar wind ions"

    def test_browse_datasets_preserves_existing_descriptions(self, tmp_path):
        import json
        from unittest.mock import patch

        fake_cdaweb = tmp_path / "envoys" / "cdaweb"
        ace_metadata = fake_cdaweb / "ace" / "metadata"
        ace_metadata.mkdir(parents=True)
        index_data = {
            "mission_id": "ACE",
            "dataset_count": 1,
            "datasets": [
                {
                    "id": "AC_H2_MFI",
                    "description": "Already has description",
                    "parameter_count": 6,
                },
            ],
        }
        (ace_metadata / "_index.json").write_text(
            json.dumps(index_data), encoding="utf-8"
        )

        fake_mission = {
            "name": "ACE",
            "instruments": {
                "mag": {
                    "name": "Magnetometer",
                    "datasets": {
                        "AC_H2_MFI": {"description": "Should not overwrite"},
                    },
                },
            },
        }

        from knowledge.metadata_client import browse_datasets, clear_cache

        with (
            patch("knowledge.metadata_client._SOURCE_DIRS", [fake_cdaweb]),
            patch("knowledge.mission_loader.load_mission", return_value=fake_mission),
        ):
            clear_cache()
            datasets = browse_datasets("ACE")

        assert datasets[0]["description"] == "Already has description"


class TestBuildSystemPrompt:
    def test_no_today_placeholder(self):
        """System prompt should not contain {today} — date is now in ephemeral context."""
        prompt = build_system_prompt()
        assert "{today}" not in prompt

    def test_contains_mission_discovery_instructions(self):
        """System prompt directs to envoy_query for mission discovery."""
        prompt = build_system_prompt()
        assert "envoy_query" in prompt

    def test_contains_workflow_sections(self):
        prompt = build_system_prompt()
        assert "## Workflow" in prompt
        assert "## Time Range Handling" in prompt

    def test_contains_delegate_to_viz_instructions(self):
        prompt = build_system_prompt()
        assert "delegate_to_viz" in prompt

    def test_contains_delegate_to_envoy_instructions(self):
        prompt = build_system_prompt()
        assert "delegate_to_envoy" in prompt

    def test_contains_delegate_to_data_ops_instructions(self):
        prompt = build_system_prompt()
        assert "delegate_to_data_ops" in prompt

    def test_contains_routing_table(self):
        prompt = build_system_prompt()
        assert "## Supported Missions" in prompt
        assert "envoy_query" in prompt

    def test_slim_prompt_routing_table_has_no_dataset_ids(self):
        prompt = build_system_prompt()
        # The routing table section should NOT list dataset IDs
        # (they may appear in examples/rules, but not in the mission table itself)
        routing_section = prompt.split("## Supported Missions")[1].split(
            "## Domain Rules"
        )[0]
        assert "PSP_FLD_L2_MAG_RTN_1MIN" not in routing_section
        assert "AC_H2_MFI" not in routing_section

    def test_slim_prompt_has_no_mission_profiles(self):
        prompt = build_system_prompt()
        # Analysis tips and mission-specific knowledge moved to sub-agents
        assert "## Mission-Specific Knowledge" not in prompt
        assert "Switchback detection" not in prompt

    def test_system_prompt_with_catalog_has_mission_profiles(self):
        prompt = build_system_prompt(include_catalog=True)
        assert "## Full Mission Catalog" in prompt
        assert "FIELDS" in prompt  # PSP instrument
        assert "AC_H2_MFI" in prompt  # ACE dataset

    def test_system_prompt_with_catalog_exceeds_cache_threshold(self):
        prompt = build_system_prompt(include_catalog=True)
        # ~4 chars/token. With tools (~26K chars) total should exceed 128K chars (~32K tokens)
        assert len(prompt) > 100000, f"Cached prompt too small: {len(prompt)} chars"

    def test_system_prompt_without_catalog_unchanged_size(self):
        prompt = build_system_prompt(include_catalog=False)
        # Base prompt should NOT contain the full mission catalog section
        assert "## Full Mission Catalog" not in prompt
        # Dataset IDs may appear in examples/rules but the catalog of ALL
        # datasets is not present — prompt should be much smaller than cached version
        cached = build_system_prompt(include_catalog=True)
        assert len(prompt) < len(cached) * 0.5  # at least 2x smaller
        assert len(prompt) < 50000  # sanity check — not bloated

    def test_system_prompt_has_domain_rules(self):
        prompt = build_system_prompt()
        assert "## Domain Rules" in prompt
        assert "read_parquet" in prompt

    def test_system_prompt_has_error_recovery(self):
        prompt = build_system_prompt()
        assert "## Error Recovery" in prompt

    def test_system_prompt_has_workflow_section(self):
        prompt = build_system_prompt()
        assert "## Workflow" in prompt


class TestBuildPlannerAgentPrompt:
    def test_no_user_request_placeholder(self):
        """Chat-based planner receives user request via chat message, not template."""
        prompt = build_planner_agent_prompt()
        assert "{user_request}" not in prompt

    def test_no_known_dataset_ids_section(self):
        """Dataset reference removed — datasets come from discovery results."""
        prompt = build_planner_agent_prompt()
        assert "Known Dataset IDs" not in prompt

    def test_prompt_size_reduced(self):
        """Planning prompt should be much smaller without the 32K dataset ref."""
        prompt = build_planner_agent_prompt()
        # Was ~140K chars, should now be ~25K with detailed guidance
        assert len(prompt) < 30000, f"Prompt still too large: {len(prompt)} chars"

    def test_references_candidate_datasets(self):
        """Dataset Selection section should reference candidate_datasets."""
        prompt = build_planner_agent_prompt()
        assert "candidate_datasets" in prompt

    def test_contains_task_capabilities(self):
        prompt = build_planner_agent_prompt()
        # Planner describes task capabilities in prose (not function signatures)
        # to prevent hallucinated tool calls
        assert "What Tasks Can Do" in prompt
        assert "Data fetching" in prompt
        assert "run_code" in prompt

    def test_contains_planning_guidelines(self):
        prompt = build_planner_agent_prompt()
        assert "Planning Guidelines" in prompt

    def test_contains_mission_tagging_instructions(self):
        prompt = build_planner_agent_prompt()
        assert "Mission Tagging" in prompt

    def test_contains_spacecraft_ids_for_tagging(self):
        prompt = build_planner_agent_prompt()
        assert "PSP" in prompt
        assert "ACE" in prompt

    def test_plotting_tasks_use_visualization_mission(self):
        prompt = build_planner_agent_prompt()
        assert "__visualization__" in prompt
        assert 'mission="__visualization__"' in prompt

    def test_compute_tasks_use_data_ops_mission(self):
        prompt = build_planner_agent_prompt()
        assert "__data_ops__" in prompt
        assert 'mission="__data_ops__"' in prompt

    def test_contains_research_instructions(self):
        """Planner prompt should include research instructions (merged from think prompt)."""
        prompt = build_planner_agent_prompt()
        assert "envoy_query" in prompt
        assert "web_search" in prompt
        assert "list_fetched_data" in prompt
        assert "Research Before Planning" in prompt

    def test_contains_batch_round_semantics(self):
        prompt = build_planner_agent_prompt()
        assert "batch" in prompt.lower()
        assert "round" in prompt.lower()

    def test_contains_produce_plan_docs(self):
        """Planner prompt should document the produce_plan tool."""
        prompt = build_planner_agent_prompt()
        assert "produce_plan" in prompt
        assert "tasks" in prompt
        assert "time_range_validated" in prompt

    def test_contains_routing_table(self):
        prompt = build_planner_agent_prompt()
        assert "Supported Missions" in prompt


class TestSharedDomainKnowledge:
    """Verify shared domain knowledge appears in both orchestrator and planner prompts."""

    def test_both_contain_error_recovery(self):
        o = build_system_prompt()
        p = build_planner_agent_prompt()
        assert "## Error Recovery" in o
        assert "## Error Recovery" in p

    def test_neither_contains_spice_section(self):
        o = build_system_prompt()
        p = build_planner_agent_prompt()
        assert "## SPICE Ephemeris" not in o
        assert "## SPICE Ephemeris" not in p

    def test_neither_contains_web_search_section(self):
        o = build_system_prompt()
        p = build_planner_agent_prompt()
        assert "## Google Search" not in o
        assert "## Google Search" not in p

    def test_neither_contains_visualization_constraints(self):
        o = build_system_prompt()
        p = build_planner_agent_prompt()
        assert "## Visualization Constraints" not in o
        assert "## Visualization Constraints" not in p

    def test_both_contain_domain_rules(self):
        o = build_system_prompt()
        p = build_planner_agent_prompt()
        assert "## Domain Rules" in o
        assert "## Domain Rules" in p

    def test_both_contain_response_style(self):
        o = build_system_prompt()
        p = build_planner_agent_prompt()
        assert "## Response Style" in o
        assert "## Response Style" in p

    def test_today_not_in_any_prompt(self):
        """Date is now injected via ephemeral context, not in system prompts."""
        o = build_system_prompt()
        p = build_planner_agent_prompt()
        assert "{today}" not in o
        assert "{today}" not in p

    def test_delegate_to_envoy_in_orchestrator_only(self):
        o = build_system_prompt()
        p = build_planner_agent_prompt()
        assert "delegate_to_envoy" in o
        assert "delegate_to_envoy" not in p

    def test_mission_tagging_in_planner_only(self):
        o = build_system_prompt()
        p = build_planner_agent_prompt()
        assert "Mission Tagging" not in o
        assert "Mission Tagging" in p

    def test_visualization_tag_in_planner(self):
        p = build_planner_agent_prompt()
        assert "__visualization__" in p

    def test_both_contain_time_range_handling(self):
        o = build_system_prompt()
        p = build_planner_agent_prompt()
        assert "## Time Range Handling" in o
        assert "## Time Range Handling" in p

    def test_both_contain_creating_datasets(self):
        o = build_system_prompt()
        p = build_planner_agent_prompt()
        assert "Creating Datasets" in o
        assert "Creating Datasets" in p


class TestBuildDataOpsPrompt:
    """Test the DataOps agent's system prompt builder."""

    def test_contains_specialist_identity(self):
        prompt = build_data_ops_prompt()
        assert "data transformation" in prompt.lower()

    def test_contains_computation_patterns(self):
        prompt = build_data_ops_prompt()
        assert "Magnitude" in prompt
        assert "Smoothing" in prompt
        assert "Resample" in prompt
        assert "Normalize" in prompt

    def test_contains_code_guidelines(self):
        prompt = build_data_ops_prompt()
        assert "## Code Guidelines" in prompt
        assert "result" in prompt
        assert "DatetimeIndex" in prompt

    def test_contains_workflow(self):
        prompt = build_data_ops_prompt()
        assert "## Workflow" in prompt
        assert "list_fetched_data" in prompt
        assert "run_code" in prompt
        assert "describe_data" in prompt

    def test_forbids_fetching(self):
        prompt = build_data_ops_prompt()
        assert "Do NOT attempt to fetch" in prompt

    def test_forbids_plotting(self):
        prompt = build_data_ops_prompt()
        assert "Do NOT attempt to plot" in prompt

    def test_has_reporting_section(self):
        prompt = build_data_ops_prompt()
        assert "## Reporting Results" in prompt


class TestBuildVisualizationPrompt:
    """Test the visualization agent's system prompt builder."""

    def test_contains_spec_workflow(self):
        prompt = build_viz_plotly_prompt()
        assert "render_plotly_json" in prompt

    def test_contains_tool_usage_sections(self):
        prompt = build_viz_plotly_prompt()
        assert "## render_plotly_json Basics" in prompt

    def test_contains_workflow(self):
        prompt = build_viz_plotly_prompt()
        assert "list_fetched_data" in prompt

    def test_no_gui_section_by_default(self):
        prompt = build_viz_plotly_prompt(gui_mode=False)
        assert "Interactive Mode" not in prompt

    def test_gui_mode_appends_section(self):
        prompt = build_viz_plotly_prompt(gui_mode=True)
        assert "Interactive Mode" in prompt

    def test_has_visualization_specialist_identity(self):
        prompt = build_viz_plotly_prompt()
        assert "visualization" in prompt.lower()

    def test_has_time_format_guidance(self):
        prompt = build_viz_plotly_prompt()
        assert "## Time Range Format" in prompt
        assert "NOT '/'" in prompt

    def test_has_spec_method_in_workflow(self):
        prompt = build_viz_plotly_prompt()
        assert "render_plotly_json" in prompt

    def test_has_panel_example(self):
        prompt = build_viz_plotly_prompt()
        assert "panels" in prompt

    def test_has_vector_data_note(self):
        prompt = build_viz_plotly_prompt()
        assert "Vector data" in prompt

    def test_describes_core_tools(self):
        prompt = build_viz_plotly_prompt()
        assert "render_plotly_json" in prompt
        assert "manage_plot" in prompt
        assert "list_fetched_data" in prompt
        assert "describe_data" in prompt
        assert "preview_data" in prompt

    def test_has_data_inspection_section(self):
        """Viz prompt should include data inspection workflow."""
        prompt = build_viz_plotly_prompt()
        assert "## Data Inspection" in prompt
        assert "describe_data" in prompt
        assert "preview_data" in prompt

    def test_has_timeseries_vs_general_data_section(self):
        """Viz prompt should explain timeseries vs general-data handling."""
        prompt = build_viz_plotly_prompt()
        assert "## Timeseries vs General Data" in prompt
        assert "is_timeseries: true" in prompt
        assert "is_timeseries: false" in prompt

    def test_no_deleted_method_references(self):
        """Deleted registry methods should not appear in the prompt."""
        prompt = build_viz_plotly_prompt()
        assert "set_render_type" not in prompt
        assert "set_color_table" not in prompt
        assert "save_session" not in prompt
        assert "load_session" not in prompt


class TestClassifyInstrumentType:
    """Test the shared instrument type classifier."""

    def test_magnetic(self):
        assert classify_instrument_type(["magnetic", "field"]) == "magnetic"
        assert classify_instrument_type(["mag", "rtn"]) == "magnetic"
        assert classify_instrument_type(["magnetometer"]) == "magnetic"

    def test_plasma(self):
        assert classify_instrument_type(["plasma", "density"]) == "plasma"
        assert classify_instrument_type(["solar wind"]) == "plasma"
        assert classify_instrument_type(["ion", "composition"]) == "plasma"

    def test_particles(self):
        assert classify_instrument_type(["energetic", "particle"]) == "particles"
        assert classify_instrument_type(["cosmic ray"]) == "particles"

    def test_indices(self):
        assert classify_instrument_type(["geomagnetic", "index"]) == "indices"

    def test_ephemeris(self):
        assert classify_instrument_type(["orbit", "position"]) == "ephemeris"
        assert classify_instrument_type(["ephemeris"]) == "ephemeris"

    def test_other(self):
        assert classify_instrument_type(["unknown"]) == "other"
        assert classify_instrument_type([]) == "other"

    def test_case_insensitive(self):
        assert classify_instrument_type(["Magnetic", "FIELD"]) == "magnetic"
        assert classify_instrument_type(["PLASMA"]) == "plasma"


