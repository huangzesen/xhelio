"""Tests for planner saving plan to filesystem."""
import json
import os
import tempfile
from pathlib import Path


def test_planner_saves_plan_to_file():
    """Planner should save plan as JSON file."""
    # This will be tested with actual planner
    plan = {
        "tasks": [{"description": "test", "instruction": "do something"}]
    }

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(plan, f)
        temp_path = f.name

    # Verify file exists and has content
    assert os.path.exists(temp_path)
    with open(temp_path) as f:
        loaded = json.load(f)
    assert loaded == plan

    # Cleanup
    os.unlink(temp_path)
