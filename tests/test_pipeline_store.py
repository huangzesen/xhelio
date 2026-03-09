"""
Tests for agent.pipeline_store — PipelineStore, PipelineEntry, search.

Run with: python -m pytest tests/test_pipeline_store.py -v
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from agent.pipeline_store import PipelineEntry, PipelineStore, SCHEMA_VERSION
from agent.memory_agent import MemoryAgent, MemoryContext
from agent.tools import get_tool_schemas
from data_ops.pipeline import is_vanilla, appropriation_fingerprint


@pytest.fixture
def tmp_path_file():
    """Provide a temporary file path for pipeline store storage."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "pipeline_store.json"


@pytest.fixture
def store(tmp_path_file):
    """Provide a PipelineStore backed by a temp file."""
    return PipelineStore(path=tmp_path_file)


def _make_pipeline(
    pipeline_id="pl_test1234",
    name="ACE Solar Wind Overview",
    description="Fetch ACE mag + plasma, compute magnitude, plot overview",
    tags=None,
    steps=None,
    source_session_id="session_abc",
):
    """Create a mock SavedPipeline-like object for register()."""
    if tags is None:
        tags = ["ace", "solar-wind", "overview"]
    if steps is None:
        steps = [
            {
                "step_id": "s001",
                "tool": "fetch_data",
                "params": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"},
                "phase": "appropriation",
            },
            {
                "step_id": "s002",
                "tool": "run_code",
                "params": {"code": "result = df.pow(2).sum(axis=1).pow(0.5).to_frame('magnitude')"},
                "phase": "appropriation",
            },
            {
                "step_id": "s003",
                "tool": "render_plotly_json",
                "params": {"figure_json": {}},
                "phase": "presentation",
            },
        ]
    return SimpleNamespace(
        id=pipeline_id,
        name=name,
        description=description,
        tags=tags,
        steps=steps,
        source_session_id=source_session_id,
    )


# ---- PipelineEntry defaults ----

class TestPipelineEntryDefaults:
    def test_default_id(self):
        e = PipelineEntry()
        assert e.id.startswith("pl_")
        assert len(e.id) == 20  # "pl_" + 8 date chars + "_" + 8 hex chars

    def test_default_fields(self):
        e = PipelineEntry()
        assert e.name == ""
        assert e.description == ""
        assert e.tags == []
        assert e.datasets == []
        assert e.missions == []
        assert e.step_count == 0
        assert e.source_session_id == ""
        assert e.pipeline_file == ""
        assert e.created_at  # non-empty
        assert e.updated_at  # non-empty
        assert e.version == 1
        assert e.supersedes == ""
        assert e.archived is False


# ---- Registration ----

class TestRegister:
    def test_register_creates_entry(self, store):
        pipeline = _make_pipeline()
        entry = store.register(pipeline)
        assert entry.id == "pl_test1234"
        assert entry.name == "ACE Solar Wind Overview"
        assert entry.step_count == 3
        assert "AC_H2_MFI.BGSEc" in entry.datasets
        assert "ACE" in entry.missions
        assert entry.source_session_id == "session_abc"
        assert entry.pipeline_file == "pl_test1234.json"

    def test_register_persists(self, tmp_path_file):
        store = PipelineStore(path=tmp_path_file)
        pipeline = _make_pipeline()
        store.register(pipeline)

        # Reload
        store2 = PipelineStore(path=tmp_path_file)
        active = store2.get_active()
        assert len(active) == 1
        assert active[0].id == "pl_test1234"

    def test_register_auto_generates_tags(self, store):
        pipeline = _make_pipeline(tags=[])
        entry = store.register(pipeline)
        assert len(entry.tags) > 0  # auto-generated from name/description/datasets

    def test_register_uses_provided_tags(self, store):
        pipeline = _make_pipeline(tags=["custom-tag", "my-workflow"])
        entry = store.register(pipeline)
        assert entry.tags == ["custom-tag", "my-workflow"]

    def test_register_extracts_missions(self, store):
        steps = [
            {
                "step_id": "s001",
                "tool": "fetch_data",
                "params": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"},
                "inputs": [],
                "phase": "appropriation",
            },
            {
                "step_id": "s002",
                "tool": "fetch_data",
                "params": {"dataset_id": "WI_H2_MFI", "parameter_id": "BGSE"},
                "inputs": [],
                "phase": "appropriation",
            },
            {
                "step_id": "s003",
                "tool": "run_code",
                "params": {"code": "result = df.mean()"},
                "inputs": ["s001", "s002"],
                "phase": "appropriation",
            },
            {
                "step_id": "s004",
                "tool": "render_plotly_json",
                "params": {"figure_json": {}},
                "inputs": ["s003"],
                "phase": "presentation",
            },
        ]
        pipeline = _make_pipeline(steps=steps)
        entry = store.register(pipeline)
        assert entry is not None
        assert "ACE" in entry.missions
        assert "WIND" in entry.missions

    def test_register_replaces_existing(self, store):
        pipeline = _make_pipeline()
        store.register(pipeline)
        assert len(store.get_active()) == 1

        # Re-register same ID with different appropriation (different code)
        different_steps = [
            {
                "step_id": "s001",
                "tool": "fetch_data",
                "params": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"},
                "inputs": [],
                "phase": "appropriation",
            },
            {
                "step_id": "s002",
                "tool": "run_code",
                "params": {"code": "result = df.std()"},
                "inputs": ["s001"],
                "phase": "appropriation",
            },
            {
                "step_id": "s003",
                "tool": "render_plotly_json",
                "params": {"figure_json": {}},
                "inputs": ["s002"],
                "phase": "presentation",
            },
        ]
        pipeline2 = _make_pipeline(name="Updated Name", steps=different_steps)
        entry = store.register(pipeline2)
        assert entry.name == "Updated Name"
        active = store.get_active()
        assert len(active) == 1
        assert active[0].name == "Updated Name"

        # Old one should be archived
        archived = store.get_archived()
        assert len(archived) == 1


# ---- Search ----

class TestSearch:
    def test_search_empty_store(self, store):
        assert store.search("anything") == []

    def test_search_by_mission(self, store):
        store.register(_make_pipeline(pipeline_id="pl_ace10001", name="ACE overview"))
        steps_psp = [
            {
                "step_id": "s001",
                "tool": "fetch_data",
                "params": {"dataset_id": "PSP_FLD_L2_MAG_RTN_1MIN", "parameter_id": "psp_fld"},
                "phase": "appropriation",
            },
        ]
        store.register(_make_pipeline(
            pipeline_id="pl_psp10001", name="PSP mag", steps=steps_psp,
        ))

        results = store.search(mission="ACE")
        assert len(results) == 1
        assert results[0].id == "pl_ace10001"

    def test_search_by_dataset(self, store):
        store.register(_make_pipeline())
        results = store.search(dataset="AC_H2_MFI")
        assert len(results) == 1
        assert results[0].id == "pl_test1234"

    def test_search_by_dataset_no_match(self, store):
        store.register(_make_pipeline())
        results = store.search(dataset="NONEXISTENT")
        assert len(results) == 0

    def test_search_no_query_returns_all(self, store):
        steps_a = [
            {"step_id": "s001", "tool": "fetch_data", "params": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"}, "inputs": [], "phase": "appropriation"},
            {"step_id": "s002", "tool": "run_code", "params": {"code": "result = df.mean()"}, "inputs": ["s001"], "phase": "appropriation"},
            {"step_id": "s003", "tool": "render_plotly_json", "params": {"figure_json": {}}, "inputs": ["s002"], "phase": "presentation"},
        ]
        steps_b = [
            {"step_id": "s001", "tool": "fetch_data", "params": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"}, "inputs": [], "phase": "appropriation"},
            {"step_id": "s002", "tool": "run_code", "params": {"code": "result = df.std()"}, "inputs": ["s001"], "phase": "appropriation"},
            {"step_id": "s003", "tool": "render_plotly_json", "params": {"figure_json": {}}, "inputs": ["s002"], "phase": "presentation"},
        ]
        store.register(_make_pipeline(pipeline_id="pl_aaa00001", name="Pipeline A", steps=steps_a))
        store.register(_make_pipeline(pipeline_id="pl_bbb00001", name="Pipeline B", steps=steps_b))
        results = store.search()
        assert len(results) == 2

    def test_search_tag_fallback(self, store):
        store.embeddings._available = False
        store.register(_make_pipeline(
            pipeline_id="pl_ace10001", name="ACE mag",
            tags=["ace", "magnetic", "overview"],
        ))
        store.register(_make_pipeline(
            pipeline_id="pl_psp10001", name="PSP plasma",
            tags=["psp", "plasma", "speed"],
        ))
        results = store.search("ace magnetic")
        assert len(results) >= 1
        assert results[0].id == "pl_ace10001"

    def test_search_limit(self, store):
        store.embeddings._available = False
        for i in range(10):
            steps = [
                {"step_id": "s001", "tool": "fetch_data", "params": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"}, "inputs": [], "phase": "appropriation"},
                {"step_id": "s002", "tool": "run_code", "params": {"code": f"result = df + {i}"}, "inputs": ["s001"], "phase": "appropriation"},
                {"step_id": "s003", "tool": "render_plotly_json", "params": {"figure_json": {}}, "inputs": ["s002"], "phase": "presentation"},
            ]
            store.register(_make_pipeline(
                pipeline_id=f"pl_{i:08x}",
                name=f"Pipeline {i} for magnetic data",
                tags=["magnetic"],
                steps=steps,
            ))
        results = store.search("magnetic", limit=3)
        assert len(results) == 3

    def test_search_mission_case_insensitive(self, store):
        store.register(_make_pipeline())
        results = store.search(mission="ace")
        assert len(results) == 1


# ---- Context injection ----

class TestContextInjection:
    def test_get_for_injection_empty(self, store):
        assert store.get_for_injection() == []

    def test_get_for_injection_returns_entries(self, store):
        store.register(_make_pipeline())
        entries = store.get_for_injection()
        assert len(entries) == 1
        assert entries[0].id == "pl_test1234"

    def test_get_for_injection_respects_limit(self, store):
        for i in range(20):
            steps = [
                {"step_id": "s001", "tool": "fetch_data", "params": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"}, "inputs": [], "phase": "appropriation"},
                {"step_id": "s002", "tool": "run_code", "params": {"code": f"result = df + {i}"}, "inputs": ["s001"], "phase": "appropriation"},
                {"step_id": "s003", "tool": "render_plotly_json", "params": {"figure_json": {}}, "inputs": ["s002"], "phase": "presentation"},
            ]
            store.register(_make_pipeline(
                pipeline_id=f"pl_{i:08x}", name=f"Pipeline {i}",
                steps=steps,
            ))
        entries = store.get_for_injection(limit=5)
        assert len(entries) == 5


# ---- Persistence ----

class TestPersistence:
    def test_save_and_reload(self, tmp_path_file):
        store1 = PipelineStore(path=tmp_path_file)
        store1.register(_make_pipeline())

        store2 = PipelineStore(path=tmp_path_file)
        assert len(store2.get_active()) == 1
        assert store2.get_active()[0].name == "ACE Solar Wind Overview"

    def test_corrupt_file_handled(self, tmp_path_file):
        tmp_path_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path_file.write_text("not valid json")
        store = PipelineStore(path=tmp_path_file)
        assert len(store.get_active()) == 0

    def test_missing_file_ok(self, tmp_path_file):
        store = PipelineStore(path=tmp_path_file)
        assert len(store.get_active()) == 0

    def test_atomic_write(self, tmp_path_file):
        store = PipelineStore(path=tmp_path_file)
        store.register(_make_pipeline())
        assert not tmp_path_file.with_suffix(".tmp").exists()
        assert tmp_path_file.exists()
        data = json.loads(tmp_path_file.read_text())
        assert data["schema_version"] == SCHEMA_VERSION


# ---- Tool availability ----

class TestToolAvailability:
    def test_pipeline_tools_in_orchestrator(self):
        """Verify all pipeline_ops tools are returned for orchestrator tools."""
        from agent.agent_registry import ORCHESTRATOR_TOOLS
        schemas = get_tool_schemas(names=ORCHESTRATOR_TOOLS)
        tool_names = {s["name"] for s in schemas}
        assert "pipeline" in tool_names

    def test_pipeline_schema(self):
        """Verify consolidated pipeline tool schema is well-formed."""
        schemas = get_tool_schemas(names=["pipeline"])
        pipeline_schema = next(
            s for s in schemas if s["name"] == "pipeline"
        )
        params = pipeline_schema["parameters"]["properties"]
        assert "action" in params
        assert set(params["action"]["enum"]) == {"info", "modify", "execute", "save", "run", "search"}


# ---- is_vanilla ----

class TestIsVanilla:
    def test_empty_steps(self):
        assert is_vanilla([]) is True

    def test_one_fetch_and_render(self):
        steps = [
            {"step_id": "s001", "tool": "fetch_data", "params": {}, "phase": "appropriation"},
            {"step_id": "s002", "tool": "render_plotly_json", "params": {}, "phase": "presentation"},
        ]
        assert is_vanilla(steps) is True

    def test_two_fetches_and_render(self):
        steps = [
            {"step_id": "s001", "tool": "fetch_data", "params": {}, "phase": "appropriation"},
            {"step_id": "s002", "tool": "fetch_data", "params": {}, "phase": "appropriation"},
            {"step_id": "s003", "tool": "render_plotly_json", "params": {}, "phase": "presentation"},
        ]
        assert is_vanilla(steps) is True

    def test_three_fetches_not_vanilla(self):
        steps = [
            {"step_id": "s001", "tool": "fetch_data", "params": {}, "phase": "appropriation"},
            {"step_id": "s002", "tool": "fetch_data", "params": {}, "phase": "appropriation"},
            {"step_id": "s003", "tool": "fetch_data", "params": {}, "phase": "appropriation"},
            {"step_id": "s004", "tool": "render_plotly_json", "params": {}, "phase": "presentation"},
        ]
        assert is_vanilla(steps) is False

    def test_fetch_plus_compute_not_vanilla(self):
        steps = [
            {"step_id": "s001", "tool": "fetch_data", "params": {}, "phase": "appropriation"},
            {"step_id": "s002", "tool": "run_code", "params": {"code": "x=1"}, "phase": "appropriation"},
            {"step_id": "s003", "tool": "render_plotly_json", "params": {}, "phase": "presentation"},
        ]
        assert is_vanilla(steps) is False

    def test_fetch_plus_run_code_not_vanilla(self):
        steps = [
            {"step_id": "s001", "tool": "fetch_data", "params": {}, "phase": "appropriation"},
            {"step_id": "s002", "tool": "run_code", "params": {"code": "x=1"}, "phase": "appropriation"},
            {"step_id": "s003", "tool": "render_plotly_json", "params": {}, "phase": "presentation"},
        ]
        assert is_vanilla(steps) is False


# ---- appropriation_fingerprint ----

class TestAppropriationFingerprint:
    def test_deterministic(self):
        steps = [
            {"step_id": "s001", "tool": "fetch_data", "params": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"}, "inputs": [], "phase": "appropriation"},
            {"step_id": "s002", "tool": "run_code", "params": {"code": "result = df.mean()"}, "inputs": ["s001"], "phase": "appropriation"},
        ]
        fp1 = appropriation_fingerprint(steps)
        fp2 = appropriation_fingerprint(steps)
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex digest

    def test_different_step_ids_same_structure(self):
        steps_a = [
            {"step_id": "s001", "tool": "fetch_data", "params": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"}, "inputs": [], "phase": "appropriation"},
            {"step_id": "s002", "tool": "run_code", "params": {"code": "result = df.mean()"}, "inputs": ["s001"], "phase": "appropriation"},
        ]
        steps_b = [
            {"step_id": "s010", "tool": "fetch_data", "params": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"}, "inputs": [], "phase": "appropriation"},
            {"step_id": "s011", "tool": "run_code", "params": {"code": "result = df.mean()"}, "inputs": ["s010"], "phase": "appropriation"},
        ]
        assert appropriation_fingerprint(steps_a) == appropriation_fingerprint(steps_b)

    def test_different_code_different_hash(self):
        steps_a = [
            {"step_id": "s001", "tool": "fetch_data", "params": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"}, "inputs": [], "phase": "appropriation"},
            {"step_id": "s002", "tool": "run_code", "params": {"code": "result = df.mean()"}, "inputs": ["s001"], "phase": "appropriation"},
        ]
        steps_b = [
            {"step_id": "s001", "tool": "fetch_data", "params": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"}, "inputs": [], "phase": "appropriation"},
            {"step_id": "s002", "tool": "run_code", "params": {"code": "result = df.std()"}, "inputs": ["s001"], "phase": "appropriation"},
        ]
        assert appropriation_fingerprint(steps_a) != appropriation_fingerprint(steps_b)

    def test_presentation_steps_excluded(self):
        base = [
            {"step_id": "s001", "tool": "fetch_data", "params": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"}, "inputs": [], "phase": "appropriation"},
            {"step_id": "s002", "tool": "run_code", "params": {"code": "result = df.mean()"}, "inputs": ["s001"], "phase": "appropriation"},
        ]
        with_render = base + [
            {"step_id": "s003", "tool": "render_plotly_json", "params": {"figure_json": {"data": []}}, "inputs": ["s002"], "phase": "presentation"},
        ]
        with_different_render = base + [
            {"step_id": "s003", "tool": "render_plotly_json", "params": {"figure_json": {"data": [{"type": "bar"}]}}, "inputs": ["s002"], "phase": "presentation"},
        ]
        assert appropriation_fingerprint(with_render) == appropriation_fingerprint(with_different_render)
        assert appropriation_fingerprint(with_render) == appropriation_fingerprint(base)


# ---- Vanilla filter behavior (no hard filter in store) ----

class TestVanillaFilter:
    """The store no longer has a hard vanilla filter.  Registration is driven
    by the MemoryAgent.  If a pipeline is passed to register(), it gets
    registered regardless of vanilla status.  The is_vanilla() signal is now
    an LLM input, not a store-level gate."""

    def test_vanilla_pipeline_registered_when_passed_to_store(self, store):
        """Vanilla pipeline → register() returns an entry (no hard filter)."""
        vanilla_steps = [
            {"step_id": "s001", "tool": "fetch_data", "params": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"}, "phase": "appropriation"},
            {"step_id": "s002", "tool": "render_plotly_json", "params": {"figure_json": {}}, "phase": "presentation"},
        ]
        pipeline = _make_pipeline(steps=vanilla_steps)
        result = store.register(pipeline)
        assert result is not None
        assert len(store.get_active()) == 1

    def test_non_vanilla_pipeline_registered(self, store):
        """Non-vanilla pipeline → registered, store has 1 active entry."""
        pipeline = _make_pipeline()  # default has run_code
        result = store.register(pipeline)
        assert result is not None
        assert len(store.get_active()) == 1


# ---- Family dedup ----

class TestFamilyDedup:
    def _make_appro_steps(self, render_data=None):
        """Helper: same appropriation, variable presentation."""
        steps = [
            {"step_id": "s001", "tool": "fetch_data",
             "params": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"},
             "inputs": [], "phase": "appropriation"},
            {"step_id": "s002", "tool": "run_code",
             "params": {"code": "result = df.pow(2).sum(axis=1).pow(0.5).to_frame('magnitude')"},
             "inputs": ["s001"], "phase": "appropriation"},
            {"step_id": "s003", "tool": "render_plotly_json",
             "params": {"figure_json": render_data or {}},
             "inputs": ["s002"], "phase": "presentation"},
        ]
        return steps

    def test_same_appropriation_different_render_one_entry(self, store):
        """Same appropriation + different render → 1 active entry, 2 variant_ids."""
        steps_a = self._make_appro_steps(render_data={"data": [{"type": "scatter"}]})
        steps_b = self._make_appro_steps(render_data={"data": [{"type": "bar"}]})

        pipeline_a = _make_pipeline(pipeline_id="pl_aaa00001", steps=steps_a)
        pipeline_b = _make_pipeline(pipeline_id="pl_bbb00001", steps=steps_b)

        entry_a = store.register(pipeline_a)
        entry_b = store.register(pipeline_b)

        assert entry_a is not None
        assert entry_b is not None
        assert entry_a.id == entry_b.id  # same family entry
        assert len(store.get_active()) == 1
        assert len(entry_b.variant_ids) == 2
        assert "pl_aaa00001" in entry_b.variant_ids
        assert "pl_bbb00001" in entry_b.variant_ids

    def test_different_appropriation_separate_entries(self, store):
        """Different appropriation → 2 separate entries."""
        steps_a = [
            {"step_id": "s001", "tool": "fetch_data",
             "params": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"},
             "inputs": [], "phase": "appropriation"},
            {"step_id": "s002", "tool": "run_code",
             "params": {"code": "result = df.mean()"},
             "inputs": ["s001"], "phase": "appropriation"},
            {"step_id": "s003", "tool": "render_plotly_json",
             "params": {"figure_json": {}},
             "inputs": ["s002"], "phase": "presentation"},
        ]
        steps_b = [
            {"step_id": "s001", "tool": "fetch_data",
             "params": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"},
             "inputs": [], "phase": "appropriation"},
            {"step_id": "s002", "tool": "run_code",
             "params": {"code": "result = df.std()"},
             "inputs": ["s001"], "phase": "appropriation"},
            {"step_id": "s003", "tool": "render_plotly_json",
             "params": {"figure_json": {}},
             "inputs": ["s002"], "phase": "presentation"},
        ]

        pipeline_a = _make_pipeline(pipeline_id="pl_aaa00001", steps=steps_a)
        pipeline_b = _make_pipeline(pipeline_id="pl_bbb00001", steps=steps_b)

        store.register(pipeline_a)
        store.register(pipeline_b)

        assert len(store.get_active()) == 2

    def test_duplicate_variant_not_added_twice(self, store):
        """Re-register same pl_ ID → variant_ids doesn't duplicate."""
        steps = self._make_appro_steps()
        pipeline = _make_pipeline(pipeline_id="pl_aaa00001", steps=steps)

        store.register(pipeline)
        entry = store.register(pipeline)

        assert entry is not None
        assert len(entry.variant_ids) == 1
        assert entry.variant_ids == ["pl_aaa00001"]

    def test_richer_description_updates_entry(self, store):
        """Richer description from a new variant updates existing entry's description."""
        steps_a = self._make_appro_steps()
        steps_b = self._make_appro_steps()

        pipeline_a = _make_pipeline(
            pipeline_id="pl_aaa00001", steps=steps_a,
            description="Short desc",
        )
        pipeline_b = _make_pipeline(
            pipeline_id="pl_bbb00001", steps=steps_b,
            description="A much longer and more detailed description of the pipeline",
        )

        store.register(pipeline_a)
        entry = store.register(pipeline_b)

        assert entry is not None
        assert entry.description == "A much longer and more detailed description of the pipeline"


# ---- Memory Agent pipeline curation ----

class TestMemoryAgentPipelineCuration:
    """Tests for MemoryAgent._execute_register_pipeline()."""

    def _make_memory_agent(self, pipeline_store, session_id="test_session"):
        """Create a MemoryAgent with mocked service and memory store."""
        svc = MagicMock()
        svc.get_adapter.return_value = MagicMock()
        svc.provider = "gemini"
        memory_store = MagicMock()
        return MemoryAgent(
            service=svc,
            memory_store=memory_store,
            pipeline_store=pipeline_store,
            session_id=session_id,
        )

    @patch("agent.memory_agent.SavedPipeline", create=True)
    def test_register_pipeline_action_extracts_and_saves(self, mock_sp_cls, tmp_path_file):
        """Mock from_session + store.register, verify pipeline saved + registered."""
        store = PipelineStore(path=tmp_path_file)
        agent = self._make_memory_agent(store)

        # Mock SavedPipeline.from_session to return a mock pipeline
        mock_pipeline = _make_pipeline(pipeline_id="pl_test0001")
        with patch("data_ops.pipeline.SavedPipeline") as MockSP:
            MockSP.from_session.return_value = mock_pipeline
            mock_pipeline.save = MagicMock()
            mock_pipeline.validate = MagicMock(return_value=[])
            mock_pipeline.time_range_original = ["", ""]

            action = {
                "action": "register_pipeline",
                "render_op_id": "op_042",
                "name": "ACE Mag Overview",
                "description": "Fetch ACE mag, compute magnitude, plot",
                "tags": ["ace", "magnetic"],
            }
            result = agent._execute_register_pipeline(action)

        assert result is not None
        assert result["pipeline_id"] == "pl_test0001"
        assert result["registered"] is True
        mock_pipeline.save.assert_called_once()

    def test_register_pipeline_action_returns_none_on_missing_render_op(self, tmp_path_file):
        """Empty render_op_id → returns None."""
        store = PipelineStore(path=tmp_path_file)
        agent = self._make_memory_agent(store)

        action = {
            "action": "register_pipeline",
            "render_op_id": "",
            "name": "Test",
        }
        result = agent._execute_register_pipeline(action)
        assert result is None

    def test_register_pipeline_action_returns_none_without_session(self, tmp_path_file):
        """No session_id → returns None."""
        store = PipelineStore(path=tmp_path_file)
        agent = self._make_memory_agent(store, session_id="")

        action = {
            "action": "register_pipeline",
            "render_op_id": "op_042",
            "name": "Test",
        }
        result = agent._execute_register_pipeline(action)
        assert result is None

    def test_register_pipeline_action_returns_none_without_store(self, tmp_path_file):
        """No pipeline_store → returns None."""
        agent = self._make_memory_agent(pipeline_store=None)

        action = {
            "action": "register_pipeline",
            "render_op_id": "op_042",
            "name": "Test",
        }
        result = agent._execute_register_pipeline(action)
        assert result is None

    def test_register_pipeline_action_returns_family_info(self, tmp_path_file):
        """When store.register returns existing family, verify variant count returned."""
        store = PipelineStore(path=tmp_path_file)
        agent = self._make_memory_agent(store)

        # Pre-register a pipeline to create the family
        existing = _make_pipeline(pipeline_id="pl_existing1")
        store.register(existing)

        # Now register a variant with same appropriation fingerprint
        mock_pipeline = _make_pipeline(pipeline_id="pl_variant01")
        with patch("data_ops.pipeline.SavedPipeline") as MockSP:
            MockSP.from_session.return_value = mock_pipeline
            mock_pipeline.save = MagicMock()
            mock_pipeline.validate = MagicMock(return_value=[])
            mock_pipeline.time_range_original = ["", ""]

            action = {
                "action": "register_pipeline",
                "render_op_id": "op_099",
                "name": "ACE Mag Variant",
                "description": "Same pipeline, different render",
                "tags": ["ace"],
            }
            result = agent._execute_register_pipeline(action)

        assert result is not None
        assert result["registered"] is True
        assert result["family_variants"] == 2  # existing + variant


# ---- Enumerate pipeline candidates ----

class TestEnumeratePipelineCandidates:
    """Tests for OrchestratorAgent._enumerate_pipeline_candidates().

    Uses a mock OperationsLog to avoid needing a full agent setup.
    Scans all sessions (current + past) for fresh pipelines.
    """

    def _make_records(self, records):
        """Create a mock ops_log with given records."""
        mock_log = MagicMock()
        mock_log.get_records.return_value = records
        # get_state_pipeline returns a filtered sub-DAG
        def _get_state_pipeline(render_id, all_labels):
            # Simple: return all records up to and including the render
            result = []
            for r in records:
                result.append(r)
                if r["id"] == render_id:
                    break
            return result
        mock_log.get_state_pipeline = _get_state_pipeline
        return mock_log

    @patch("config.get_data_dir")
    @patch("data_ops.operations_log.get_operations_log")
    def test_enumerates_render_ops(self, mock_get_log, mock_data_dir, tmp_path):
        """Log with 2 render ops → 2 candidates with steps and scopes."""
        mock_data_dir.return_value = tmp_path
        records = [
            {"id": "op_001", "tool": "fetch_data", "status": "success",
             "args": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"}, "outputs": ["AC_H2_MFI.BGSEc"]},
            {"id": "op_002", "tool": "render_plotly_json", "status": "success",
             "args": {}, "outputs": [], "inputs": ["AC_H2_MFI.BGSEc"]},
            {"id": "op_003", "tool": "fetch_data", "status": "success",
             "args": {"dataset_id": "WI_H2_MFI", "parameter_id": "BGSE"}, "outputs": ["WI_H2_MFI.BGSE"]},
            {"id": "op_004", "tool": "render_plotly_json", "status": "success",
             "args": {}, "outputs": [], "inputs": ["WI_H2_MFI.BGSE"]},
        ]
        mock_get_log.return_value = self._make_records(records)

        from agent.core import OrchestratorAgent
        agent = OrchestratorAgent.__new__(OrchestratorAgent)
        agent._session_id = "test_session"
        candidates = agent._enumerate_pipeline_candidates()

        assert len(candidates) == 2
        assert candidates[0]["render_op_id"] == "op_002"
        assert candidates[1]["render_op_id"] == "op_004"

        # New format: steps list and scopes
        assert "steps" in candidates[0]
        assert "scopes" in candidates[0]
        assert candidates[0]["scopes"] == ["mission:ACE"]
        # The mock _get_state_pipeline returns all records up to render,
        # so the second candidate's sub-DAG includes the first fetch too.
        assert "mission:WIND" in candidates[1]["scopes"]

    @patch("config.get_data_dir")
    @patch("data_ops.operations_log.get_operations_log")
    def test_rich_step_fields(self, mock_get_log, mock_data_dir, tmp_path):
        """Pipeline with run_code → step has code, description fields."""
        mock_data_dir.return_value = tmp_path
        records = [
            {"id": "op_001", "tool": "fetch_data", "status": "success",
             "args": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"}, "outputs": ["AC_H2_MFI.BGSEc"]},
            {"id": "op_002", "tool": "run_code", "status": "success",
             "args": {"description": "compute magnitude", "code": "df['Bmag'] = np.sqrt(df['Bx']**2)"}, "outputs": ["magnitude"]},
            {"id": "op_003", "tool": "render_plotly_json", "status": "success",
             "args": {}, "outputs": [], "inputs": ["AC_H2_MFI.BGSEc", "magnitude"]},
        ]
        mock_get_log.return_value = self._make_records(records)

        from agent.core import OrchestratorAgent
        agent = OrchestratorAgent.__new__(OrchestratorAgent)
        agent._session_id = "test_session"
        candidates = agent._enumerate_pipeline_candidates()

        assert len(candidates) == 1
        cand = candidates[0]
        assert cand["step_count"] == 3
        assert cand["scopes"] == ["mission:ACE"]

        # Check step details
        fetch_step = cand["steps"][0]
        assert fetch_step["tool"] == "fetch_data"
        assert fetch_step["dataset_id"] == "AC_H2_MFI"
        assert fetch_step["parameter_id"] == "BGSEc"
        assert fetch_step["output_label"] == "AC_H2_MFI.BGSEc"

        custom_step = cand["steps"][1]
        assert custom_step["tool"] == "run_code"
        assert custom_step["description"] == "compute magnitude"
        assert custom_step["code"] == "df['Bmag'] = np.sqrt(df['Bx']**2)"
        assert custom_step["output_label"] == "magnitude"

        render_step = cand["steps"][2]
        assert render_step["tool"] == "render_plotly_json"
        assert render_step["inputs"] == ["AC_H2_MFI.BGSEc", "magnitude"]
        assert "figure_json" not in render_step  # intentionally omitted

    @patch("config.get_data_dir")
    @patch("data_ops.operations_log.get_operations_log")
    def test_skips_failed_renders(self, mock_get_log, mock_data_dir, tmp_path):
        """Failed render_plotly_json → not in candidates."""
        mock_data_dir.return_value = tmp_path
        records = [
            {"id": "op_001", "tool": "fetch_data", "status": "success",
             "args": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"}, "outputs": ["AC_H2_MFI.BGSEc"]},
            {"id": "op_002", "tool": "render_plotly_json", "status": "error",
             "args": {}, "outputs": [], "error": "Invalid figure JSON"},
            {"id": "op_003", "tool": "render_plotly_json", "status": "success",
             "args": {}, "outputs": []},
        ]
        mock_get_log.return_value = self._make_records(records)

        from agent.core import OrchestratorAgent
        agent = OrchestratorAgent.__new__(OrchestratorAgent)
        agent._session_id = "test_session"
        candidates = agent._enumerate_pipeline_candidates()

        assert len(candidates) == 1
        assert candidates[0]["render_op_id"] == "op_003"

    @patch("config.get_data_dir")
    @patch("data_ops.operations_log.get_operations_log")
    def test_only_fresh_render_ops(self, mock_get_log, mock_data_dir, tmp_path):
        """Only render ops with pipeline_status='fresh' (or absent) are candidates."""
        mock_data_dir.return_value = tmp_path
        records = [
            {"id": "op_001", "tool": "fetch_data", "status": "success",
             "args": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"}, "outputs": ["AC_H2_MFI.BGSEc"]},
            {"id": "op_002", "tool": "render_plotly_json", "status": "success",
             "args": {}, "outputs": [], "pipeline_status": "registered"},
            {"id": "op_003", "tool": "render_plotly_json", "status": "success",
             "args": {}, "outputs": [], "pipeline_status": "discarded"},
            {"id": "op_004", "tool": "render_plotly_json", "status": "success",
             "args": {}, "outputs": []},  # no pipeline_status → fresh
        ]
        mock_get_log.return_value = self._make_records(records)

        from agent.core import OrchestratorAgent
        agent = OrchestratorAgent.__new__(OrchestratorAgent)
        agent._session_id = "test_session"
        candidates = agent._enumerate_pipeline_candidates()

        assert len(candidates) == 1
        assert candidates[0]["render_op_id"] == "op_004"


class TestCandidatesFromLog:
    """Tests for OrchestratorAgent._candidates_from_log() static method."""

    def test_basic_extraction(self):
        """Extract candidates with steps and scopes from a simple log."""
        from data_ops.operations_log import OperationsLog
        from agent.core import OrchestratorAgent

        log = OperationsLog(session_id="s1")
        log.record(tool="fetch_data", args={"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"}, outputs=["AC_H2_MFI.BGSEc"])
        log.record(tool="render_plotly_json", args={}, inputs=["AC_H2_MFI.BGSEc"], outputs=[])

        candidates = OrchestratorAgent._candidates_from_log(log)
        assert len(candidates) == 1
        cand = candidates[0]
        assert cand["render_op_id"] == "s1:op_002"
        assert cand["scopes"] == ["mission:ACE"]
        assert len(cand["steps"]) == 2
        assert cand["steps"][0]["tool"] == "fetch_data"
        assert cand["steps"][0]["dataset_id"] == "AC_H2_MFI"
        assert cand["steps"][1]["tool"] == "render_plotly_json"

    def test_mission_auto_extraction(self):
        """Multi-mission pipeline extracts multiple scopes."""
        from data_ops.operations_log import OperationsLog
        from agent.core import OrchestratorAgent

        log = OperationsLog(session_id="s1")
        log.record(tool="fetch_data", args={"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"}, outputs=["AC_H2_MFI.BGSEc"])
        log.record(tool="fetch_data", args={"dataset_id": "WI_H2_MFI", "parameter_id": "BGSE"}, outputs=["WI_H2_MFI.BGSE"])
        log.record(tool="render_plotly_json", args={}, inputs=["AC_H2_MFI.BGSEc", "WI_H2_MFI.BGSE"], outputs=[])

        candidates = OrchestratorAgent._candidates_from_log(log)
        assert len(candidates) == 1
        assert "mission:ACE" in candidates[0]["scopes"]
        assert "mission:WIND" in candidates[0]["scopes"]

    def test_skips_registered_and_discarded(self):
        """Registered/discarded ops are not returned as candidates."""
        from data_ops.operations_log import OperationsLog
        from agent.core import OrchestratorAgent

        log = OperationsLog(session_id="s1")
        log.record(tool="fetch_data", args={"dataset_id": "X"}, outputs=["X"])
        log.record(tool="render_plotly_json", args={}, inputs=["X"], outputs=[])
        log.record(tool="render_plotly_json", args={}, inputs=["X"], outputs=[])
        log.record(tool="render_plotly_json", args={}, inputs=["X"], outputs=[])

        log.set_pipeline_status("s1:op_002", "registered")
        log.set_pipeline_status("s1:op_003", "discarded")

        candidates = OrchestratorAgent._candidates_from_log(log)
        assert len(candidates) == 1
        assert candidates[0]["render_op_id"] == "s1:op_004"


class TestRegisterWithLLMScopesAndTags:
    """Tests for PipelineStore.register() with llm_missions and llm_tags params."""

    def test_register_with_llm_missions_and_tags(self, store):
        """register() with llm_missions uses provided missions, llm_tags are merged."""
        pipeline = _make_pipeline(tags=["existing-tag"])
        entry = store.register(pipeline, llm_missions=["ACE"], llm_tags=["magnitude"])

        assert entry is not None
        assert "ACE" in entry.missions
        assert "existing-tag" in entry.tags
        assert "magnitude" in entry.tags

    def test_register_validates_missions_fallback(self, store):
        """Without llm_missions, falls back to auto-extraction from dataset IDs."""
        pipeline = _make_pipeline(tags=[])
        entry = store.register(pipeline)

        assert entry is not None
        # Auto-extracted from AC_H2_MFI in the default steps
        assert "ACE" in entry.missions

    def test_register_llm_tags_merged_with_pipeline_tags(self, store):
        """llm_tags are merged with pipeline.tags, no duplicates."""
        pipeline = _make_pipeline(tags=["solar-wind"])
        entry = store.register(
            pipeline,
            llm_tags=["solar-wind", "magnetic-field"],
        )

        assert entry is not None
        assert entry.tags.count("solar-wind") == 1  # no duplicate
        assert "magnetic-field" in entry.tags

    def test_register_llm_missions_overrides_auto(self, store):
        """llm_missions completely overrides auto-extraction."""
        pipeline = _make_pipeline(tags=["test"])
        entry = store.register(
            pipeline,
            llm_missions=["WIND"],
            llm_tags=["cross-mission"],
        )

        assert entry is not None
        assert entry.missions == ["WIND"]  # uses LLM value, not auto-extracted ACE


class TestDiscardPipelineAction:
    """Tests for the discard_pipeline action in MemoryAgent."""

    def _make_memory_agent(self, session_id="test_session"):
        svc = MagicMock()
        svc.get_adapter.return_value = MagicMock()
        svc.provider = "gemini"
        memory_store = MagicMock()
        return MemoryAgent(
            service=svc,
            memory_store=memory_store,
            pipeline_store=None,
            session_id=session_id,
        )

    @patch("data_ops.operations_log.get_operations_log")
    def test_discard_pipeline_marks_status(self, mock_get_log):
        """discard_pipeline action calls set_pipeline_status('discarded')."""
        mock_log = MagicMock()
        mock_log.set_pipeline_status.return_value = True
        mock_get_log.return_value = mock_log

        agent = self._make_memory_agent()
        result = agent._execute_discard_pipeline({"render_op_id": "s1:op_002"})

        assert result is True
        mock_log.set_pipeline_status.assert_called_once_with("s1:op_002", "discarded")

    @patch("data_ops.operations_log.get_operations_log")
    def test_discard_pipeline_empty_op_id(self, mock_get_log):
        """discard_pipeline with empty render_op_id returns False."""
        agent = self._make_memory_agent()
        result = agent._execute_discard_pipeline({"render_op_id": ""})
        assert result is False

    @patch("data_ops.operations_log.get_operations_log")
    def test_discard_pipeline_via_tool(self, mock_get_log):
        """discard_pipeline tool call routes through _tool_discard_pipeline."""
        mock_log = MagicMock()
        mock_log.set_pipeline_status.return_value = True
        mock_get_log.return_value = mock_log

        agent = self._make_memory_agent()
        agent._executed_actions = []
        result = agent._tool_discard_pipeline(
            {"render_op_id": "s1:op_003"}, agent._executed_actions
        )

        assert result["status"] == "ok"
        assert len(agent._executed_actions) == 1
        assert agent._executed_actions[0]["action"] == "discard_pipeline"
