"""
Capability boundary tests for custom_operation.

Probes what kinds of natural-language-to-pandas translations are feasible
and where the approach breaks down. Each test represents a plausible user
request and the pandas code the LLM would need to generate.

Run with: python -m pytest tests/test_capability_boundary.py -v
"""

import numpy as np
import pandas as pd
import pytest

from data_ops.custom_ops import run_custom_operation, execute_multi_source_operation


def _make_time(n=100, start="2024-01-01", cadence_s=60):
    return pd.date_range(start, periods=n, freq=f"{cadence_s}s")


def _make_df(values, index, columns=None):
    if isinstance(values, np.ndarray) and values.ndim == 1:
        return pd.DataFrame(values, index=index, columns=columns or ["value"])
    return pd.DataFrame(values, index=index, columns=columns)


# ═══════════════════════════════════════════════════════════════════════════
# TIER 1: Basic operations — should all pass trivially
# ═══════════════════════════════════════════════════════════════════════════

class TestTier1Basic:
    """Simple one-liner pandas operations."""

    def test_scale_by_constant(self):
        """'Convert nT to Gauss (multiply by 1e-5)'"""
        idx = _make_time(5)
        df = _make_df(np.array([100.0, 200.0, 300.0, 400.0, 500.0]), idx)
        result = run_custom_operation(df, "result = df * 1e-5")
        np.testing.assert_allclose(result.values.squeeze(), [1e-3, 2e-3, 3e-3, 4e-3, 5e-3])

    def test_absolute_value(self):
        """'Show me absolute values'"""
        idx = _make_time(4)
        df = _make_df(np.array([-3.0, 2.0, -1.0, 4.0]), idx)
        result = run_custom_operation(df, "result = df.abs()")
        np.testing.assert_allclose(result.values.squeeze(), [3.0, 2.0, 1.0, 4.0])

    def test_fill_gaps(self):
        """'Fill in the data gaps with linear interpolation'"""
        idx = _make_time(5)
        df = _make_df(np.array([1.0, np.nan, np.nan, 4.0, 5.0]), idx)
        result = run_custom_operation(df, "result = df.interpolate(method='linear')")
        np.testing.assert_allclose(result.values.squeeze(), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_remove_outliers_by_threshold(self):
        """'Remove values above 100'"""
        idx = _make_time(5)
        df = _make_df(np.array([10.0, 50.0, 200.0, 30.0, 150.0]), idx)
        result = run_custom_operation(df, "result = df.where(df <= 100)")
        assert np.isnan(result.iloc[2, 0])
        assert np.isnan(result.iloc[4, 0])
        np.testing.assert_allclose(result.iloc[0, 0], 10.0)


# ═══════════════════════════════════════════════════════════════════════════
# TIER 2: Statistical / windowed operations
# ═══════════════════════════════════════════════════════════════════════════

class TestTier2Statistical:
    """Operations requiring statistical thinking."""

    def test_z_score_outlier_removal(self):
        """'Remove outliers beyond 2 sigma'"""
        idx = _make_time(100)
        values = np.random.randn(100)
        values[50] = 100.0  # obvious outlier
        df = _make_df(values, idx)
        code = "z = (df - df.mean()) / df.std()\nresult = df[z.abs() < 2].reindex(df.index)"
        result = run_custom_operation(df, code)
        assert np.isnan(result.iloc[50, 0])  # outlier should be NaN

    def test_percentile_clipping(self):
        """'Clip to the 5th-95th percentile range'"""
        idx = _make_time(1000)
        df = _make_df(np.random.randn(1000), idx)
        code = "lo = df.quantile(0.05).iloc[0]\nhi = df.quantile(0.95).iloc[0]\nresult = df.clip(lower=lo, upper=hi)"
        result = run_custom_operation(df, code)
        assert result.values.min() >= df.quantile(0.05).iloc[0]
        assert result.values.max() <= df.quantile(0.95).iloc[0]

    def test_rolling_std(self):
        """'Show me the rolling standard deviation with a 10-point window'"""
        idx = _make_time(50)
        df = _make_df(np.random.randn(50), idx)
        result = run_custom_operation(df, "result = df.rolling(10, center=True, min_periods=1).std()")
        assert len(result) == 50
        assert result.values.squeeze()[25] > 0  # std should be positive

    def test_ewm_smoothing(self):
        """'Apply exponential smoothing with span 10'"""
        idx = _make_time(20)
        df = _make_df(np.random.randn(20), idx)
        result = run_custom_operation(df, "result = df.ewm(span=10).mean()")
        assert len(result) == 20

    def test_cumulative_max(self):
        """'Show the running maximum'"""
        idx = _make_time(5)
        df = _make_df(np.array([3.0, 1.0, 4.0, 1.0, 5.0]), idx)
        result = run_custom_operation(df, "result = df.cummax()")
        np.testing.assert_allclose(result.values.squeeze(), [3.0, 3.0, 4.0, 4.0, 5.0])

    def test_rank_percentile(self):
        """'Convert to percentile rank'"""
        idx = _make_time(5)
        df = _make_df(np.array([10.0, 30.0, 20.0, 50.0, 40.0]), idx)
        result = run_custom_operation(df, "result = df.rank(pct=True)")
        np.testing.assert_allclose(result.iloc[3, 0], 1.0)  # 50 is highest → 100th pctile


# ═══════════════════════════════════════════════════════════════════════════
# TIER 3: Multi-column / vector operations
# ═══════════════════════════════════════════════════════════════════════════

class TestTier3Vector:
    """Operations on multi-column (vector) data."""

    def test_select_single_component(self):
        """'Show me just the Bz component'"""
        idx = _make_time(5)
        df = pd.DataFrame({"Bx": [1.0]*5, "By": [2.0]*5, "Bz": [3.0]*5}, index=idx)
        result = run_custom_operation(df, "result = df[['Bz']]")
        assert list(result.columns) == ["Bz"]
        np.testing.assert_allclose(result.values.squeeze(), [3.0]*5)

    def test_cross_component_ratio(self):
        """'What's the ratio of Bx to Bz?'"""
        idx = _make_time(3)
        df = pd.DataFrame({"Bx": [6.0, 8.0, 10.0], "Bz": [2.0, 4.0, 5.0]}, index=idx)
        code = "result = (df['Bx'] / df['Bz']).to_frame('Bx_over_Bz')"
        result = run_custom_operation(df, code)
        np.testing.assert_allclose(result.values.squeeze(), [3.0, 2.0, 2.0])

    def test_cone_angle(self):
        """'Calculate the cone angle of the magnetic field' (angle from x-axis)"""
        idx = _make_time(3)
        df = pd.DataFrame(
            {"Bx": [1.0, 0.0, 1.0], "By": [0.0, 1.0, 1.0], "Bz": [0.0, 0.0, 0.0]},
            index=idx,
        )
        code = "Bmag = df.pow(2).sum(axis=1).pow(0.5)\nresult = np.degrees(np.arccos(df['Bx'].abs() / Bmag)).to_frame('cone_angle')"
        result = run_custom_operation(df, code)
        np.testing.assert_allclose(result.iloc[0, 0], 0.0, atol=1e-10)   # purely radial
        np.testing.assert_allclose(result.iloc[1, 0], 90.0, atol=1e-10)  # purely tangential
        np.testing.assert_allclose(result.iloc[2, 0], 45.0, atol=1e-10)  # 45 degrees

    def test_rotate_vector_2d(self):
        """'Rotate the magnetic field by 45 degrees in the xy plane'"""
        idx = _make_time(1)
        df = pd.DataFrame({"Bx": [1.0], "By": [0.0], "Bz": [0.0]}, index=idx)
        code = (
            "angle = np.radians(45)\n"
            "Bx_new = df['Bx'] * np.cos(angle) - df['By'] * np.sin(angle)\n"
            "By_new = df['Bx'] * np.sin(angle) + df['By'] * np.cos(angle)\n"
            "result = pd.DataFrame({'Bx': Bx_new, 'By': By_new, 'Bz': df['Bz']}, index=df.index)"
        )
        result = run_custom_operation(df, code)
        np.testing.assert_allclose(result["Bx"].iloc[0], np.sqrt(2)/2, atol=1e-10)
        np.testing.assert_allclose(result["By"].iloc[0], np.sqrt(2)/2, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# TIER 4: Time-aware operations
# ═══════════════════════════════════════════════════════════════════════════

class TestTier4TimeAware:
    """Operations that depend on the time axis."""

    def test_select_time_window(self):
        """'Show me only data from the first 30 minutes'"""
        idx = _make_time(100, cadence_s=60)  # 100 minutes
        df = _make_df(np.arange(100, dtype=float), idx)
        code = "result = df.iloc[:30]"
        result = run_custom_operation(df, code)
        assert len(result) == 30

    def test_time_since_start_in_hours(self):
        """'Add a column showing hours since the start'"""
        idx = _make_time(5, cadence_s=3600)  # hourly
        df = _make_df(np.ones(5), idx)
        code = "hours = (df.index - df.index[0]).total_seconds() / 3600\nresult = pd.DataFrame({'hours': hours}, index=df.index)"
        result = run_custom_operation(df, code)
        np.testing.assert_allclose(result.values.squeeze(), [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_day_night_flag(self):
        """'Flag daytime (6-18h) vs nighttime'"""
        idx = pd.date_range("2024-01-01", periods=24, freq="1h")
        df = _make_df(np.ones(24), idx)
        code = "hour = df.index.hour\nresult = pd.DataFrame({'is_day': ((hour >= 6) & (hour < 18)).astype(float)}, index=df.index)"
        result = run_custom_operation(df, code)
        assert result.iloc[3, 0] == 0.0   # 3am = night
        assert result.iloc[12, 0] == 1.0  # noon = day

    def test_resample_to_hourly_max(self):
        """'Give me hourly maximum values'"""
        idx = _make_time(120, cadence_s=60)  # 2 hours at 1-min cadence
        df = _make_df(np.arange(120, dtype=float), idx)
        code = "result = df.resample('1h').max().dropna(how='all')"
        result = run_custom_operation(df, code)
        assert len(result) == 2
        np.testing.assert_allclose(result.iloc[0, 0], 59.0)
        np.testing.assert_allclose(result.iloc[1, 0], 119.0)


# ═══════════════════════════════════════════════════════════════════════════
# TIER 5: Complex / multi-step operations
# ═══════════════════════════════════════════════════════════════════════════

class TestTier5Complex:
    """Multi-step operations that push pandas capabilities."""

    def test_detrend_linear(self):
        """'Remove the linear trend from the data'"""
        idx = _make_time(100, cadence_s=1)
        trend = np.linspace(0, 10, 100)
        noise = np.random.randn(100) * 0.1
        df = _make_df(trend + noise, idx)
        code = (
            "x = np.arange(len(df), dtype=float)\n"
            "coeffs = np.polyfit(x, df.iloc[:, 0].values, 1)\n"
            "trend = np.polyval(coeffs, x)\n"
            "result = df - pd.DataFrame(trend, index=df.index, columns=df.columns)"
        )
        result = run_custom_operation(df, code)
        # Detrended data should have mean close to 0
        assert abs(result.values.mean()) < 0.5

    def test_forward_fill_then_diff(self):
        """'Fill gaps then compute differences'"""
        idx = _make_time(5)
        df = _make_df(np.array([1.0, np.nan, 3.0, np.nan, 5.0]), idx)
        code = "filled = df.ffill()\nresult = filled.diff().iloc[1:]"
        result = run_custom_operation(df, code)
        assert len(result) == 4

    def test_rolling_correlation_two_columns(self):
        """'Show rolling correlation between Bx and By over 20-point window'"""
        idx = _make_time(50)
        df = pd.DataFrame({
            "Bx": np.random.randn(50),
            "By": np.random.randn(50),
        }, index=idx)
        code = "result = df['Bx'].rolling(20).corr(df['By']).to_frame('correlation')"
        result = run_custom_operation(df, code)
        assert len(result) == 50
        # First 19 should be NaN (not enough data for window)
        assert np.isnan(result.iloc[0, 0])
        # Values should be between -1 and 1
        valid = result.dropna()
        assert valid.values.min() >= -1.0 - 1e-10
        assert valid.values.max() <= 1.0 + 1e-10

    def test_piecewise_function(self):
        """'Set values to 1 where B > 0, -1 where B < 0, 0 where B = 0'"""
        idx = _make_time(5)
        df = _make_df(np.array([-2.0, 0.0, 3.0, -1.0, 5.0]), idx)
        code = "result = np.sign(df)"
        result = run_custom_operation(df, code)
        np.testing.assert_allclose(result.values.squeeze(), [-1.0, 0.0, 1.0, -1.0, 1.0])

    def test_power_spectral_density_prep(self):
        """'Prepare data for FFT: interpolate, detrend, apply Hanning window'"""
        idx = _make_time(64, cadence_s=1)
        df = _make_df(np.sin(np.linspace(0, 4*np.pi, 64)) + np.random.randn(64)*0.1, idx)
        code = (
            "clean = df.interpolate().ffill().bfill()\n"
            "detrended = clean - clean.mean()\n"
            "window = np.hanning(len(detrended))\n"
            "result = detrended.mul(window, axis=0)"
        )
        result = run_custom_operation(df, code)
        assert len(result) == 64
        # Hanning window zeroes edges
        np.testing.assert_allclose(result.iloc[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.iloc[-1, 0], 0.0, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# TIER 6: Known limitations — things that CANNOT work
# ═══════════════════════════════════════════════════════════════════════════

class TestTier6Limitations:
    """Operations that hit fundamental limits of the sandbox."""

    def test_cannot_access_second_dataset(self):
        """Two-dataset operations require both in df, but custom_operation only gets one.
        The LLM must fetch both and embed the second as a literal or use a workaround."""
        idx = _make_time(3)
        df = _make_df(np.array([1.0, 2.0, 3.0]), idx)
        # Trying to reference a non-existent variable fails
        with pytest.raises(RuntimeError, match="Execution error"):
            run_custom_operation(df, "result = df - other_dataset")

    def test_cannot_import_scipy(self):
        """No access to scipy for advanced signal processing."""
        idx = _make_time(10)
        df = _make_df(np.ones(10), idx)
        with pytest.raises(ValueError, match="validation failed"):
            run_custom_operation(df, "from scipy import signal\nresult = signal.detrend(df)")

    def test_cannot_do_file_io(self):
        """Cannot read/write files."""
        idx = _make_time(3)
        df = _make_df(np.ones(3), idx)
        with pytest.raises(ValueError, match="validation failed"):
            run_custom_operation(df, "open('test.txt', 'w')\nresult = df")

    def test_cannot_use_sklearn(self):
        """No access to machine learning libraries."""
        idx = _make_time(10)
        df = _make_df(np.ones(10), idx)
        with pytest.raises(ValueError, match="validation failed"):
            run_custom_operation(df, "from sklearn.linear_model import LinearRegression\nresult = df")

    def test_result_with_numeric_index_accepted_legacy(self):
        """Legacy single-source path accepts numeric index (no enforcement)."""
        idx = _make_time(5)
        df = _make_df(np.arange(5, dtype=float), idx)
        result = run_custom_operation(df, "result = pd.DataFrame({'a': df.values.squeeze()})")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_timeseries_mode_rejects_lost_index(self):
        """When source_timeseries flags all sources as timeseries, losing
        DatetimeIndex raises ValueError."""
        idx = _make_time(5)
        sources = {"df_A": _make_df(np.arange(5, dtype=float), idx)}
        with pytest.raises(ValueError, match="DatetimeIndex"):
            execute_multi_source_operation(
                sources,
                "result = pd.DataFrame({'a': df.values.squeeze()})",
                source_timeseries={"df_A": True},
            )

    def test_cannot_return_scalar(self):
        """Cannot return a single number — must be DataFrame/Series."""
        idx = _make_time(5)
        df = _make_df(np.arange(5, dtype=float), idx)
        with pytest.raises(ValueError, match="DataFrame, Series, or xarray DataArray"):
            run_custom_operation(df, "result = df.mean().iloc[0]")

    def test_workaround_embed_second_dataset(self):
        """The LLM CAN do two-dataset ops by embedding the second as a literal."""
        idx = _make_time(3)
        df_a = _make_df(np.array([10.0, 20.0, 30.0]), idx)
        # LLM embeds dataset B's values directly in the code
        code = "b = pd.DataFrame([1.0, 2.0, 3.0], index=df.index, columns=df.columns)\nresult = df - b"
        result = run_custom_operation(df_a, code)
        np.testing.assert_allclose(result.values.squeeze(), [9.0, 18.0, 27.0])
