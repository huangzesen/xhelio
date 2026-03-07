"""
Tests for data_ops.saved_pipeline — SavedPipeline extraction,
validation, execution, mutation, and persistence.

Run with: python -m pytest tests/test_saved_pipeline.py -v
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from data_ops.pipeline import (
    SavedPipeline,
    _extract_time_range_from_fetch,
    _scrub_xaxis_ranges,
    _pipelines_dir,
    topological_sort_steps,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(monkeypatch, tmp_path):
    """Redirect get_data_dir() to a temp directory for test isolation."""
    monkeypatch.setattr(
        "data_ops.pipeline.get_data_dir",
        lambda: tmp_path,
        raising=False,
    )
    # Also patch config.get_data_dir in case it's imported elsewhere
    try:
        import config
        monkeypatch.setattr(config, "get_data_dir", lambda: tmp_path)
    except (ImportError, AttributeError):
        pass
    return tmp_path


@pytest.fixture
def mock_session(tmp_data_dir):
    """Create a mock session with operations.json for extraction tests."""
    session_id = "test_session_001"
    session_dir = tmp_data_dir / "sessions" / session_id
    session_dir.mkdir(parents=True)

    records = [
        {
            "id": "op_001",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "tool": "fetch_data",
            "status": "success",
            "inputs": [],
            "outputs": ["AC_H2_MFI.BGSEc"],
            "args": {
                "dataset_id": "AC_H2_MFI",
                "parameter_id": "BGSEc",
                "time_start": "2024-01-01",
                "time_end": "2024-01-07",
                "time_range_resolved": ["2024-01-01", "2024-01-07"],
            },
            "error": None,
            "input_producers": {},
        },
        {
            "id": "op_002",
            "timestamp": "2026-01-01T00:00:01+00:00",
            "tool": "custom_operation",
            "status": "success",
            "inputs": ["AC_H2_MFI.BGSEc"],
            "outputs": ["Bmag"],
            "args": {
                "source_labels": ["AC_H2_MFI.BGSEc"],
                "code": "result = (df_BGSEc ** 2).sum(axis=1) ** 0.5",
                "output_label": "Bmag",
                "description": "Compute B magnitude",
                "units": "nT",
            },
            "error": None,
            "input_producers": {"AC_H2_MFI.BGSEc": "op_001"},
        },
        {
            "id": "op_003",
            "timestamp": "2026-01-01T00:00:02+00:00",
            "tool": "render_plotly_json",
            "status": "success",
            "inputs": ["AC_H2_MFI.BGSEc", "Bmag"],
            "outputs": [],
            "args": {
                "figure_json": {
                    "data": [
                        {"type": "scatter", "data_label": "AC_H2_MFI.BGSEc"},
                        {"type": "scatter", "data_label": "Bmag"},
                    ],
                    "layout": {
                        "title": {"text": "ACE Magnetic Field"},
                        "xaxis": {
                            "range": ["2024-01-01", "2024-01-07"],
                            "domain": [0, 1],
                        },
                        "yaxis": {"title": {"text": "nT"}},
                    },
                },
            },
            "error": None,
            "input_producers": {
                "AC_H2_MFI.BGSEc": "op_001",
                "Bmag": "op_002",
            },
        },
    ]

    ops_path = session_dir / "operations.json"
    with open(ops_path, "w") as f:
        json.dump(records, f)

    return session_id


@pytest.fixture
def sample_pipeline_data():
    """Return a minimal valid pipeline dict."""
    return {
        "version": 1,
        "id": "pl_testtest",
        "name": "Test Pipeline",
        "description": "A test pipeline",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "source_session_id": "test_session_001",
        "source_render_op_id": None,
        "tags": ["test"],
        "time_range_original": ["2024-01-01", "2024-01-07"],
        "steps": [
            {
                "step_id": "s001",
                "phase": "appropriation",
                "tool": "fetch_data",
                "params": {
                    "dataset_id": "AC_H2_MFI",
                    "parameter_id": "BGSEc",
                },
                "inputs": [],
                "output_label": "AC_H2_MFI.BGSEc",
                "description": "Fetch ACE magnetic field vector",
            },
            {
                "step_id": "s002",
                "phase": "appropriation",
                "tool": "custom_operation",
                "params": {
                    "code": "result = (df_BGSEc ** 2).sum(axis=1) ** 0.5",
                    "description": "Compute B magnitude",
                    "units": "nT",
                },
                "inputs": ["s001"],
                "output_label": "Bmag",
                "description": "Compute magnetic field magnitude",
            },
            {
                "step_id": "s003",
                "phase": "presentation",
                "tool": "render_plotly_json",
                "params": {
                    "figure_json": {
                        "data": [
                            {"type": "scatter", "data_label": "AC_H2_MFI.BGSEc"},
                            {"type": "scatter", "data_label": "Bmag"},
                        ],
                        "layout": {
                            "title": {"text": "ACE Magnetic Field"},
                            "yaxis": {"title": {"text": "nT"}},
                        },
                    },
                },
                "inputs": ["s001", "s002"],
                "output_label": None,
                "description": "2-trace timeseries plot",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Tests: Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_extract_time_range_resolved_list(self):
        args = {"time_range_resolved": ["2024-01-01", "2024-01-07"]}
        assert _extract_time_range_from_fetch(args) == ("2024-01-01", "2024-01-07")

    def test_extract_time_range_resolved_string(self):
        args = {"time_range_resolved": "2024-01-01 to 2024-01-07"}
        assert _extract_time_range_from_fetch(args) == ("2024-01-01", "2024-01-07")

    def test_extract_time_range_start_end(self):
        args = {"time_start": "2024-01-01", "time_end": "2024-01-07"}
        assert _extract_time_range_from_fetch(args) == ("2024-01-01", "2024-01-07")

    def test_extract_time_range_raw(self):
        args = {"time_range": "2024-01-01 to 2024-01-07"}
        assert _extract_time_range_from_fetch(args) == ("2024-01-01", "2024-01-07")

    def test_extract_time_range_none(self):
        assert _extract_time_range_from_fetch({}) is None

    def test_scrub_xaxis_ranges(self):
        layout = {
            "xaxis": {"range": ["2024-01-01", "2024-01-07"], "domain": [0, 1]},
            "xaxis2": {"range": ["2024-01-01", "2024-01-07"], "matches": "x"},
            "yaxis": {"title": {"text": "nT"}, "range": [0, 100]},
        }
        scrubbed = _scrub_xaxis_ranges(layout)
        # xaxis ranges removed
        assert "range" not in scrubbed["xaxis"]
        assert "range" not in scrubbed["xaxis2"]
        # xaxis domain preserved
        assert scrubbed["xaxis"]["domain"] == [0, 1]
        # yaxis range preserved
        assert scrubbed["yaxis"]["range"] == [0, 100]

    def test_topological_sort_basic(self):
        steps = [
            {"step_id": "s002", "inputs": ["s001"]},
            {"step_id": "s001", "inputs": []},
            {"step_id": "s003", "inputs": ["s001", "s002"]},
        ]
        sorted_steps = topological_sort_steps(steps)
        ids = [s["step_id"] for s in sorted_steps]
        assert ids.index("s001") < ids.index("s002")
        assert ids.index("s001") < ids.index("s003")
        assert ids.index("s002") < ids.index("s003")

    def test_topological_sort_cycle(self):
        steps = [
            {"step_id": "s001", "inputs": ["s002"]},
            {"step_id": "s002", "inputs": ["s001"]},
        ]
        with pytest.raises(ValueError, match="Cycle"):
            topological_sort_steps(steps)

    def test_topological_sort_independent(self):
        steps = [
            {"step_id": "s001", "inputs": []},
            {"step_id": "s002", "inputs": []},
            {"step_id": "s003", "inputs": ["s001", "s002"]},
        ]
        sorted_steps = topological_sort_steps(steps)
        ids = [s["step_id"] for s in sorted_steps]
        assert ids.index("s001") < ids.index("s003")
        assert ids.index("s002") < ids.index("s003")


# ---------------------------------------------------------------------------
# Tests: Extraction
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_from_session_basic(self, tmp_data_dir, mock_session):
        pipeline = SavedPipeline.from_session(
            mock_session,
            name="Test Pipeline",
            description="Test description",
            tags=["ace"],
        )

        assert pipeline.name == "Test Pipeline"
        assert pipeline.description == "Test description"
        assert pipeline.tags == ["ace"]
        assert pipeline.id.startswith("pl_")
        assert len(pipeline.steps) == 3

    def test_from_session_step_structure(self, tmp_data_dir, mock_session):
        pipeline = SavedPipeline.from_session(mock_session, name="Test")

        # Step 1: fetch
        s1 = pipeline.steps[0]
        assert s1["tool"] == "fetch_data"
        assert s1["phase"] == "appropriation"
        assert s1["params"]["dataset_id"] == "AC_H2_MFI"
        assert s1["params"]["parameter_id"] == "BGSEc"
        assert s1["output_label"] == "AC_H2_MFI.BGSEc"
        assert s1["inputs"] == []
        # Time keys stripped
        assert "time_start" not in s1["params"]
        assert "time_end" not in s1["params"]
        assert "time_range_resolved" not in s1["params"]

        # Step 2: compute
        s2 = pipeline.steps[1]
        assert s2["tool"] == "custom_operation"
        assert s2["phase"] == "appropriation"
        assert "code" in s2["params"]
        assert s2["output_label"] == "Bmag"
        assert s2["inputs"] == ["s001"]

        # Step 3: render
        s3 = pipeline.steps[2]
        assert s3["tool"] == "render_plotly_json"
        assert s3["phase"] == "presentation"
        assert s3["output_label"] is None
        assert set(s3["inputs"]) == {"s001", "s002"}

    def test_from_session_time_range_original(self, tmp_data_dir, mock_session):
        pipeline = SavedPipeline.from_session(mock_session, name="Test")
        assert pipeline.time_range_original == ["2024-01-01", "2024-01-07"]

    def test_from_session_xaxis_range_scrubbed(self, tmp_data_dir, mock_session):
        pipeline = SavedPipeline.from_session(mock_session, name="Test")
        render_step = pipeline.steps[2]
        layout = render_step["params"]["figure_json"]["layout"]
        # xaxis.range should be scrubbed
        assert "range" not in layout.get("xaxis", {})

    def test_from_session_with_render_op_id(self, tmp_data_dir, mock_session):
        pipeline = SavedPipeline.from_session(
            mock_session, render_op_id="op_003", name="Single Render"
        )
        assert len(pipeline.steps) == 3

    def test_from_session_not_found(self, tmp_data_dir):
        with pytest.raises(FileNotFoundError):
            SavedPipeline.from_session("nonexistent", name="Test")

    def test_from_session_skips_already_loaded(self, tmp_data_dir):
        """Fetch with already_loaded=True should be skipped."""
        session_id = "test_skip_session"
        session_dir = tmp_data_dir / "sessions" / session_id
        session_dir.mkdir(parents=True)

        records = [
            {
                "id": "op_001",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "tool": "fetch_data",
                "status": "success",
                "inputs": [],
                "outputs": ["AC_H2_MFI.BGSEc"],
                "args": {
                    "dataset_id": "AC_H2_MFI",
                    "parameter_id": "BGSEc",
                    "time_range_resolved": ["2024-01-01", "2024-01-07"],
                },
                "error": None,
                "input_producers": {},
            },
            {
                "id": "op_002",
                "timestamp": "2026-01-01T00:00:01+00:00",
                "tool": "fetch_data",
                "status": "success",
                "inputs": [],
                "outputs": ["AC_H2_MFI.BGSEc"],
                "args": {
                    "dataset_id": "AC_H2_MFI",
                    "parameter_id": "BGSEc",
                    "already_loaded": True,
                },
                "error": None,
                "input_producers": {},
            },
            {
                "id": "op_003",
                "timestamp": "2026-01-01T00:00:02+00:00",
                "tool": "render_plotly_json",
                "status": "success",
                "inputs": ["AC_H2_MFI.BGSEc"],
                "outputs": [],
                "args": {
                    "figure_json": {
                        "data": [{"type": "scatter", "data_label": "AC_H2_MFI.BGSEc"}],
                        "layout": {},
                    },
                },
                "error": None,
                "input_producers": {"AC_H2_MFI.BGSEc": "op_001"},
            },
        ]

        ops_path = session_dir / "operations.json"
        with open(ops_path, "w") as f:
            json.dump(records, f)

        pipeline = SavedPipeline.from_session(session_id, name="Skip Test")
        # Should have 2 steps (fetch + render), not 3 (skip the already_loaded)
        fetch_steps = [s for s in pipeline.steps if s["tool"] == "fetch_data"]
        assert len(fetch_steps) == 1

    def test_from_session_skips_manage_plot(self, tmp_data_dir):
        """manage_plot operations should be skipped."""
        session_id = "test_manage_plot_session"
        session_dir = tmp_data_dir / "sessions" / session_id
        session_dir.mkdir(parents=True)

        records = [
            {
                "id": "op_001",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "tool": "fetch_data",
                "status": "success",
                "inputs": [],
                "outputs": ["AC_H2_MFI.BGSEc"],
                "args": {
                    "dataset_id": "AC_H2_MFI",
                    "parameter_id": "BGSEc",
                    "time_range_resolved": ["2024-01-01", "2024-01-07"],
                },
                "error": None,
                "input_producers": {},
            },
            {
                "id": "op_002",
                "timestamp": "2026-01-01T00:00:01+00:00",
                "tool": "manage_plot",
                "status": "success",
                "inputs": [],
                "outputs": [],
                "args": {"action": "reset"},
                "error": None,
                "input_producers": {},
            },
            {
                "id": "op_003",
                "timestamp": "2026-01-01T00:00:02+00:00",
                "tool": "render_plotly_json",
                "status": "success",
                "inputs": ["AC_H2_MFI.BGSEc"],
                "outputs": [],
                "args": {
                    "figure_json": {
                        "data": [{"type": "scatter", "data_label": "AC_H2_MFI.BGSEc"}],
                        "layout": {},
                    },
                },
                "error": None,
                "input_producers": {"AC_H2_MFI.BGSEc": "op_001"},
            },
        ]

        ops_path = session_dir / "operations.json"
        with open(ops_path, "w") as f:
            json.dump(records, f)

        pipeline = SavedPipeline.from_session(session_id, name="No Manage Plot")
        tools = [s["tool"] for s in pipeline.steps]
        assert "manage_plot" not in tools


# ---------------------------------------------------------------------------
# Tests: Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_template(self, sample_pipeline_data):
        pipeline = SavedPipeline(sample_pipeline_data)
        issues = pipeline.validate()
        assert issues == []

    def test_missing_version(self, sample_pipeline_data):
        del sample_pipeline_data["version"]
        pipeline = SavedPipeline(sample_pipeline_data)
        issues = pipeline.validate()
        assert any("version" in i.lower() or "Missing" in i for i in issues)

    def test_wrong_version(self, sample_pipeline_data):
        sample_pipeline_data["version"] = 99
        pipeline = SavedPipeline(sample_pipeline_data)
        issues = pipeline.validate()
        assert any("version" in i.lower() for i in issues)

    def test_missing_dataset_id(self, sample_pipeline_data):
        sample_pipeline_data["steps"][0]["params"]["dataset_id"] = ""
        pipeline = SavedPipeline(sample_pipeline_data)
        issues = pipeline.validate()
        assert any("dataset_id" in i for i in issues)

    def test_missing_parameter_id(self, sample_pipeline_data):
        sample_pipeline_data["steps"][0]["params"]["parameter_id"] = ""
        pipeline = SavedPipeline(sample_pipeline_data)
        issues = pipeline.validate()
        assert any("parameter_id" in i for i in issues)

    def test_bad_code(self, sample_pipeline_data):
        sample_pipeline_data["steps"][1]["params"]["code"] = "import os; os.system('rm -rf /')"
        pipeline = SavedPipeline(sample_pipeline_data)
        issues = pipeline.validate()
        assert any("code violation" in i or "Import" in i for i in issues)

    def test_dangling_input(self, sample_pipeline_data):
        sample_pipeline_data["steps"][1]["inputs"] = ["s999"]
        pipeline = SavedPipeline(sample_pipeline_data)
        issues = pipeline.validate()
        assert any("non-existent" in i for i in issues)

    def test_cycle_detection(self, sample_pipeline_data):
        # Make s001 depend on s002 (cycle: s001 → s002 → s001)
        sample_pipeline_data["steps"][0]["inputs"] = ["s002"]
        sample_pipeline_data["steps"][1]["inputs"] = ["s001"]
        pipeline = SavedPipeline(sample_pipeline_data)
        issues = pipeline.validate()
        assert any("Cycle" in i or "cycle" in i for i in issues)

    def test_presentation_not_terminal(self, sample_pipeline_data):
        # Add a step that references the render (presentation) step
        sample_pipeline_data["steps"].append({
            "step_id": "s004",
            "phase": "appropriation",
            "tool": "custom_operation",
            "params": {"code": "result = df", "description": "bad"},
            "inputs": ["s003"],  # s003 is presentation
            "output_label": "bad_output",
            "description": "references presentation step",
        })
        pipeline = SavedPipeline(sample_pipeline_data)
        issues = pipeline.validate()
        assert any("presentation" in i.lower() for i in issues)

    def test_orphan_detection(self, sample_pipeline_data):
        # Add an appropriation step that's not consumed by anything
        sample_pipeline_data["steps"].append({
            "step_id": "s004",
            "phase": "appropriation",
            "tool": "custom_operation",
            "params": {
                "code": "result = df_BGSEc * 2",
                "description": "orphan op",
            },
            "inputs": ["s001"],
            "output_label": "orphan_data",
            "description": "orphan step",
        })
        pipeline = SavedPipeline(sample_pipeline_data)
        issues = pipeline.validate()
        assert any("orphan" in i.lower() for i in issues)

    def test_duplicate_labels(self, sample_pipeline_data):
        # Make two steps produce the same label
        sample_pipeline_data["steps"][1]["output_label"] = "AC_H2_MFI.BGSEc"
        pipeline = SavedPipeline(sample_pipeline_data)
        issues = pipeline.validate()
        assert any("Duplicate" in i or "duplicate" in i for i in issues)


# ---------------------------------------------------------------------------
# Tests: Mutation
# ---------------------------------------------------------------------------

class TestMutation:
    def test_add_step_at_end(self, sample_pipeline_data):
        pipeline = SavedPipeline(sample_pipeline_data)
        new_step = {
            "phase": "appropriation",
            "tool": "custom_operation",
            "params": {
                "code": "result = df_BGSEc.rolling(10).mean()",
                "description": "Smoothed B",
            },
            "inputs": ["s001"],
            "output_label": "B_smooth",
            "description": "10-point smoothing",
        }
        step_id = pipeline.add_step(new_step)
        assert step_id == "s004"
        assert len(pipeline.steps) == 4
        assert pipeline.steps[-1]["step_id"] == "s004"

    def test_add_step_after(self, sample_pipeline_data):
        pipeline = SavedPipeline(sample_pipeline_data)
        new_step = {
            "phase": "appropriation",
            "tool": "custom_operation",
            "params": {
                "code": "result = df_BGSEc.rolling(10).mean()",
                "description": "Smoothed B",
            },
            "inputs": ["s001"],
            "output_label": "B_smooth",
            "description": "10-point smoothing",
        }
        step_id = pipeline.add_step(new_step, after_step_id="s001")
        assert step_id == "s004"
        # Should be inserted at index 1 (after s001)
        assert pipeline.steps[1]["step_id"] == "s004"

    def test_add_step_after_not_found(self, sample_pipeline_data):
        pipeline = SavedPipeline(sample_pipeline_data)
        with pytest.raises(KeyError, match="s999"):
            pipeline.add_step({"tool": "fetch_data"}, after_step_id="s999")

    def test_remove_step(self, sample_pipeline_data):
        pipeline = SavedPipeline(sample_pipeline_data)
        removed = pipeline.remove_step("s002")
        assert removed["step_id"] == "s002"
        assert len(pipeline.steps) == 2
        # References to s002 should be cleaned up from s003's inputs
        s3 = next(s for s in pipeline.steps if s["step_id"] == "s003")
        assert "s002" not in s3["inputs"]

    def test_remove_step_not_found(self, sample_pipeline_data):
        pipeline = SavedPipeline(sample_pipeline_data)
        with pytest.raises(KeyError, match="s999"):
            pipeline.remove_step("s999")

    def test_update_step_params(self, sample_pipeline_data):
        pipeline = SavedPipeline(sample_pipeline_data)
        pipeline.update_step_params("s002", {
            "code": "result = (df_BGSEc ** 2).sum(axis=1).apply(np.sqrt)",
            "description": "Updated Bmag computation",
        })
        s2 = pipeline.steps[1]
        assert "apply(np.sqrt)" in s2["params"]["code"]
        assert s2["params"]["description"] == "Updated Bmag computation"

    def test_update_step_params_not_found(self, sample_pipeline_data):
        pipeline = SavedPipeline(sample_pipeline_data)
        with pytest.raises(KeyError, match="s999"):
            pipeline.update_step_params("s999", {"code": "x"})


# ---------------------------------------------------------------------------
# Tests: Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, tmp_data_dir, sample_pipeline_data):
        pipeline = SavedPipeline(sample_pipeline_data)
        pipeline.save()

        loaded = SavedPipeline.load("pl_testtest")
        assert loaded.id == "pl_testtest"
        assert loaded.name == "Test Pipeline"
        assert len(loaded.steps) == 3

    def test_load_not_found(self, tmp_data_dir):
        with pytest.raises(FileNotFoundError):
            SavedPipeline.load("pl_nonexistent")

    def test_list_all_empty(self, tmp_data_dir):
        assert SavedPipeline.list_all() == []

    def test_list_all_after_save(self, tmp_data_dir, sample_pipeline_data):
        pipeline = SavedPipeline(sample_pipeline_data)
        pipeline.save()

        items = SavedPipeline.list_all()
        assert len(items) == 1
        assert items[0]["id"] == "pl_testtest"
        assert items[0]["name"] == "Test Pipeline"
        assert items[0]["step_count"] == 3
        assert "AC_H2_MFI.BGSEc" in items[0]["datasets"]

    def test_delete(self, tmp_data_dir, sample_pipeline_data):
        pipeline = SavedPipeline(sample_pipeline_data)
        pipeline.save()

        assert SavedPipeline.delete("pl_testtest") is True
        assert SavedPipeline.list_all() == []

    def test_delete_not_found(self, tmp_data_dir):
        assert SavedPipeline.delete("pl_nonexistent") is False

    def test_save_updates_index(self, tmp_data_dir, sample_pipeline_data):
        # Save first pipeline
        t1 = SavedPipeline(sample_pipeline_data)
        t1.save()

        # Save second pipeline
        data2 = dict(sample_pipeline_data)
        data2["id"] = "pl_second"
        data2["name"] = "Second Pipeline"
        t2 = SavedPipeline(data2)
        t2.save()

        items = SavedPipeline.list_all()
        assert len(items) == 2
        ids = {it["id"] for it in items}
        assert ids == {"pl_testtest", "pl_second"}

    def test_save_overwrites_existing(self, tmp_data_dir, sample_pipeline_data):
        pipeline = SavedPipeline(sample_pipeline_data)
        pipeline.save()

        # Modify and re-save
        pipeline.name = "Updated Name"
        pipeline.save()

        loaded = SavedPipeline.load("pl_testtest")
        assert loaded.name == "Updated Name"

        # Index should still have only one entry
        items = SavedPipeline.list_all()
        assert len(items) == 1
        assert items[0]["name"] == "Updated Name"


# ---------------------------------------------------------------------------
# Tests: Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict(self, sample_pipeline_data):
        pipeline = SavedPipeline(sample_pipeline_data)
        d = pipeline.to_dict()
        assert d["id"] == "pl_testtest"
        assert d is not pipeline._data  # deep copy

    def test_from_dict(self, sample_pipeline_data):
        pipeline = SavedPipeline.from_dict(sample_pipeline_data)
        assert pipeline.id == "pl_testtest"
        assert pipeline._data is not sample_pipeline_data  # deep copy

    def test_round_trip_dict(self, sample_pipeline_data):
        original = SavedPipeline(sample_pipeline_data)
        d = original.to_dict()
        restored = SavedPipeline.from_dict(d)
        assert restored.to_dict() == original.to_dict()


# ---------------------------------------------------------------------------
# Tests: Extraction from session → save → load round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_extract_save_load(self, tmp_data_dir, mock_session):
        # Extract
        pipeline = SavedPipeline.from_session(
            mock_session, name="Round Trip Test", tags=["roundtrip"]
        )

        # Validate
        issues = pipeline.validate()
        assert issues == []

        # Save
        pipeline.save()
        pipeline_id = pipeline.id

        # Load
        loaded = SavedPipeline.load(pipeline_id)
        assert loaded.name == "Round Trip Test"
        assert loaded.tags == ["roundtrip"]
        assert len(loaded.steps) == 3

        # Validate loaded
        issues = loaded.validate()
        assert issues == []

    def test_extract_save_list_delete(self, tmp_data_dir, mock_session):
        pipeline = SavedPipeline.from_session(mock_session, name="CRUD Test")
        pipeline.save()

        # List
        items = SavedPipeline.list_all()
        assert len(items) == 1

        # Delete
        assert SavedPipeline.delete(pipeline.id) is True
        assert SavedPipeline.list_all() == []


# ---------------------------------------------------------------------------
# Tests: Build replay record
# ---------------------------------------------------------------------------

class TestBuildReplayRecord:
    def test_fetch_record(self, sample_pipeline_data):
        pipeline = SavedPipeline(sample_pipeline_data)
        step = pipeline.steps[0]  # fetch_data
        step_to_label = {"s001": "AC_H2_MFI.BGSEc", "s002": "Bmag"}

        record = pipeline._build_replay_record(
            step, step_to_label, "2025-06-01", "2025-06-07"
        )
        assert record["tool"] == "fetch_data"
        assert record["args"]["time_range_resolved"] == ["2025-06-01", "2025-06-07"]
        assert record["outputs"] == ["AC_H2_MFI.BGSEc"]

    def test_compute_record(self, sample_pipeline_data):
        pipeline = SavedPipeline(sample_pipeline_data)
        step = pipeline.steps[1]  # custom_operation
        step_to_label = {"s001": "AC_H2_MFI.BGSEc", "s002": "Bmag"}

        record = pipeline._build_replay_record(
            step, step_to_label, "2025-06-01", "2025-06-07"
        )
        assert record["tool"] == "custom_operation"
        assert record["inputs"] == ["AC_H2_MFI.BGSEc"]
        assert record["outputs"] == ["Bmag"]

    def test_render_record(self, sample_pipeline_data):
        pipeline = SavedPipeline(sample_pipeline_data)
        step = pipeline.steps[2]  # render_plotly_json
        step_to_label = {"s001": "AC_H2_MFI.BGSEc", "s002": "Bmag"}

        record = pipeline._build_replay_record(
            step, step_to_label, "2025-06-01", "2025-06-07"
        )
        assert record["tool"] == "render_plotly_json"
        assert set(record["inputs"]) == {"AC_H2_MFI.BGSEc", "Bmag"}
        assert record["outputs"] == []
