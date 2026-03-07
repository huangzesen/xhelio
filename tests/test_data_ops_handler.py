"""
Tests for agent.tool_handlers.data_ops — handler-level tests.

Run with: python -m pytest tests/test_data_ops_handler.py -v
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from data_ops.store import DataEntry, DataStore


def _make_store_with_timeseries(tmp_path: Path) -> DataStore:
    """Create a DataStore with one timeseries entry."""
    store = DataStore(tmp_path / "data")
    idx = pd.date_range("2024-01-01", periods=256, freq="60s")
    entry = DataEntry(
        label="TEST.MAG",
        data=pd.DataFrame({"Bx": np.random.randn(256)}, index=idx),
        units="nT",
        description="Test magnetic field",
        source="cdf",
        is_timeseries=True,
    )
    store.put(entry)
    return store


def _make_orch(store: DataStore) -> MagicMock:
    """Create a minimal mock OrchestratorAgent with the given store."""
    orch = MagicMock()
    orch._store = store
    orch._event_bus = MagicMock()
    return orch


class TestHandleCustomOperationForceTimeseries:
    """Tests for the force_timeseries opt-out in handle_custom_operation."""

    def test_force_timeseries_false_accepts_frequency_index(self, tmp_path):
        """force_timeseries=false allows PSD-like results with frequency index."""
        from agent.tool_handlers.data_ops import handle_custom_operation

        store = _make_store_with_timeseries(tmp_path)
        orch = _make_orch(store)

        result = handle_custom_operation(orch, {
            "source_labels": ["TEST.MAG"],
            "code": (
                "freqs = np.fft.rfftfreq(len(df), d=60.0)\n"
                "power = np.abs(np.fft.rfft(df['Bx'].values))**2\n"
                "result = pd.DataFrame({'power': power}, index=freqs)"
            ),
            "output_label": "TEST.PSD",
            "description": "Power spectral density of Bx",
            "units": "nT^2/Hz",
            "force_timeseries": False,
        })

        assert result["status"] == "success"
        assert result["label"] == "TEST.PSD"
        # Verify stored entry is non-timeseries
        stored = store.get("TEST.PSD")
        assert stored is not None
        assert stored.is_timeseries is False
        assert not isinstance(stored.data.index, pd.DatetimeIndex)

    def test_force_timeseries_default_rejects_frequency_index(self, tmp_path):
        """Without force_timeseries (default=true), PSD-like results are rejected."""
        from agent.tool_handlers.data_ops import handle_custom_operation

        store = _make_store_with_timeseries(tmp_path)
        orch = _make_orch(store)

        result = handle_custom_operation(orch, {
            "source_labels": ["TEST.MAG"],
            "code": (
                "freqs = np.fft.rfftfreq(len(df), d=60.0)\n"
                "power = np.abs(np.fft.rfft(df['Bx'].values))**2\n"
                "result = pd.DataFrame({'power': power}, index=freqs)"
            ),
            "output_label": "TEST.PSD",
            "description": "Power spectral density of Bx",
        })

        assert result["status"] == "error"
        assert "DatetimeIndex" in result["message"]

    def test_force_timeseries_true_preserves_timeseries(self, tmp_path):
        """force_timeseries=true (explicit) still accepts valid timeseries results."""
        from agent.tool_handlers.data_ops import handle_custom_operation

        store = _make_store_with_timeseries(tmp_path)
        orch = _make_orch(store)

        result = handle_custom_operation(orch, {
            "source_labels": ["TEST.MAG"],
            "code": "result = df * 2",
            "output_label": "TEST.MAG_x2",
            "description": "Double the field",
            "force_timeseries": True,
        })

        assert result["status"] == "success"
        stored = store.get("TEST.MAG_x2")
        assert stored is not None
        assert stored.is_timeseries is True
