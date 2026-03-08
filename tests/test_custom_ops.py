"""
Tests for data_ops.custom_ops — AST validator and sandboxed executor.

Run with: python -m pytest tests/test_custom_ops.py
"""

import numpy as np
import pandas as pd
import pytest

from data_ops.custom_ops import (
    validate_code,
    execute_custom_operation,
    run_custom_operation,
    execute_multi_source_operation,
    run_multi_source_operation,
    validate_result,
    execute_dataframe_creation,
    run_dataframe_creation,
)


class TestReloadSandboxRegistry:
    def test_reload_sandbox_registry(self):
        """reload_sandbox_registry() rebuilds derived constants."""
        from data_ops import custom_ops

        old_module_names = frozenset(custom_ops._MODULE_NAMES)
        custom_ops.reload_sandbox_registry()
        assert custom_ops._MODULE_NAMES == old_module_names


def _make_time(n=100, start="2024-01-01", cadence_s=60):
    """Create a DatetimeIndex with fixed cadence."""
    return pd.date_range(start, periods=n, freq=f"{cadence_s}s")


def _make_df(values, index, columns=None):
    """Create a DataFrame from values and DatetimeIndex."""
    if isinstance(values, np.ndarray) and values.ndim == 1:
        return pd.DataFrame(values, index=index, columns=columns or ["value"])
    return pd.DataFrame(values, index=index, columns=columns)


# ── Validator Tests ──────────────────────────────────────────────────────────


class TestValidatePandasCode:
    def test_valid_simple_operation(self):
        assert validate_code("result = df * 2") == []

    def test_valid_multiline(self):
        code = "mean = df.mean()\nresult = df - mean"
        assert validate_code(code) == []

    def test_valid_numpy_operation(self):
        assert validate_code("result = np.log10(df.abs())") == []

    def test_valid_rolling(self):
        assert validate_code("result = df.rolling(10, center=True, min_periods=1).mean()") == []

    def test_valid_interpolate(self):
        assert validate_code("result = df.interpolate(method='linear')") == []

    def test_valid_clip(self):
        assert validate_code("result = df.clip(lower=-50, upper=50)") == []

    def test_valid_complex_multiline(self):
        code = "z = (df - df.mean()) / df.std()\nmask = z.abs() < 3\nresult = df[mask].reindex(df.index)"
        assert validate_code(code) == []

    def test_reject_no_result_assignment(self):
        violations = validate_code("x = df * 2")
        assert any("result" in v for v in violations)

    def test_reject_import(self):
        violations = validate_code("import os\nresult = df")
        assert any("Import" in v for v in violations)

    def test_reject_from_import(self):
        violations = validate_code("from os import path\nresult = df")
        assert any("Import" in v for v in violations)

    def test_reject_exec(self):
        violations = validate_code("exec('x=1')\nresult = df")
        assert any("exec" in v for v in violations)

    def test_reject_eval(self):
        violations = validate_code("result = eval('df * 2')")
        assert any("eval" in v for v in violations)

    def test_reject_open(self):
        violations = validate_code("open('test.txt')\nresult = df")
        assert any("open" in v for v in violations)

    def test_reject_dunder_access(self):
        violations = validate_code("result = df.__class__")
        assert any("__class__" in v for v in violations)

    def test_reject_global(self):
        violations = validate_code("global x\nresult = df")
        assert any("global" in v for v in violations)

    def test_reject_nonlocal(self):
        violations = validate_code("nonlocal x\nresult = df")
        assert any("global/nonlocal" in v.lower() or "nonlocal" in v.lower() for v in violations)

    def test_reject_syntax_error(self):
        violations = validate_code("result = df +")
        assert any("Syntax" in v for v in violations)

    def test_reject_async(self):
        violations = validate_code("async def f(): pass\nresult = df")
        assert any("Async" in v or "async" in v for v in violations)

    def test_require_result_false_allows_no_assignment(self):
        violations = validate_code("x = 42", require_result=False)
        assert violations == []

    def test_require_result_false_still_blocks_imports(self):
        violations = validate_code("import os", require_result=False)
        assert any("Import" in v for v in violations)

    def test_require_result_false_still_blocks_exec(self):
        violations = validate_code("exec('x=1')", require_result=False)
        assert any("exec" in v for v in violations)

    def test_require_result_false_still_blocks_dunder(self):
        violations = validate_code("x = obj.__class__", require_result=False)
        assert any("__class__" in v for v in violations)

    def test_require_result_default_is_true(self):
        violations = validate_code("x = 42")
        assert any("result" in v for v in violations)


# ── Executor Tests ───────────────────────────────────────────────────────────


class TestExecuteCustomOperation:
    def test_multiply(self):
        idx = _make_time(5)
        df = _make_df(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), idx)
        result = execute_custom_operation(df, "result = df * 2")
        np.testing.assert_allclose(result.values.squeeze(), [2.0, 4.0, 6.0, 8.0, 10.0])

    def test_normalize(self):
        idx = _make_time(5)
        df = _make_df(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), idx)
        result = execute_custom_operation(df, "result = (df - df.mean()) / df.std()")
        # Normalized data should have mean ~0 and std ~1 (using ddof=1 to match pandas)
        np.testing.assert_allclose(result.values.mean(), 0.0, atol=1e-10)
        np.testing.assert_allclose(result.values.std(ddof=1), 1.0, atol=1e-10)

    def test_clip(self):
        idx = _make_time(5)
        df = _make_df(np.array([-100.0, -5.0, 0.0, 5.0, 100.0]), idx)
        result = execute_custom_operation(df, "result = df.clip(lower=-10, upper=10)")
        np.testing.assert_allclose(result.values.squeeze(), [-10.0, -5.0, 0.0, 5.0, 10.0])

    def test_numpy_function(self):
        idx = _make_time(3)
        df = _make_df(np.array([1.0, 10.0, 100.0]), idx)
        result = execute_custom_operation(df, "result = np.log10(df)")
        np.testing.assert_allclose(result.values.squeeze(), [0.0, 1.0, 2.0])

    def test_series_to_dataframe_conversion(self):
        idx = _make_time(3)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}, index=idx)
        result = execute_custom_operation(df, "result = df['a']")
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 1

    def test_datetime_index_preserved(self):
        idx = _make_time(5)
        df = _make_df(np.arange(5, dtype=float), idx)
        result = execute_custom_operation(df, "result = df * 2")
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result.index) == 5

    def test_source_not_mutated(self):
        idx = _make_time(3)
        df = _make_df(np.array([1.0, 2.0, 3.0]), idx)
        original_values = df.values.copy()
        execute_custom_operation(df, "result = df * 0")
        np.testing.assert_allclose(df.values, original_values)

    def test_no_result_error(self):
        idx = _make_time(3)
        df = _make_df(np.ones(3), idx)
        with pytest.raises(ValueError, match="did not assign"):
            execute_custom_operation(df, "x = df * 2")

    def test_runtime_error(self):
        idx = _make_time(3)
        df = _make_df(np.ones(3), idx)
        with pytest.raises(RuntimeError, match="Execution error"):
            execute_custom_operation(df, "result = df / undefined_var")

    def test_non_dataframe_error(self):
        idx = _make_time(3)
        df = _make_df(np.ones(3), idx)
        with pytest.raises(ValueError, match="DataFrame, Series, or xarray DataArray"):
            execute_custom_operation(df, "result = 42")

    def test_numeric_index_accepted(self):
        """A result with numeric index (not DatetimeIndex) is accepted.

        execute_custom_operation uses the legacy single-source path which
        does NOT enforce DatetimeIndex on the result (require_timeseries=False).
        """
        idx = _make_time(3)
        df = _make_df(np.ones(3), idx)
        result = execute_custom_operation(df, "result = pd.DataFrame({'a': [1.0, 2.0, 3.0]})")
        assert isinstance(result, pd.DataFrame)
        assert list(result["a"]) == [1.0, 2.0, 3.0]

    def test_multiline_code(self):
        idx = _make_time(5)
        df = _make_df(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), idx)
        code = "mean = df.mean()\nstd = df.std()\nresult = (df - mean) / std"
        result = execute_custom_operation(df, code)
        assert len(result) == 5

    def test_vector_dataframe(self):
        idx = _make_time(5)
        df = pd.DataFrame(
            np.ones((5, 3)), index=idx, columns=["x", "y", "z"]
        )
        result = execute_custom_operation(df, "result = df * 3")
        np.testing.assert_allclose(result.values, np.ones((5, 3)) * 3)


# ── Integration Tests (run_custom_operation) ─────────────────────────────────


class TestRunCustomOperation:
    def test_end_to_end_success(self):
        idx = _make_time(5)
        df = _make_df(np.array([1.0, 4.0, 9.0, 16.0, 25.0]), idx)
        result = run_custom_operation(df, "result = np.sqrt(df)")
        np.testing.assert_allclose(result.values.squeeze(), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_validation_rejection(self):
        idx = _make_time(3)
        df = _make_df(np.ones(3), idx)
        with pytest.raises(ValueError, match="validation failed"):
            run_custom_operation(df, "import os\nresult = df")

    def test_execution_error_propagation(self):
        idx = _make_time(3)
        df = _make_df(np.ones(3), idx)
        with pytest.raises(RuntimeError, match="Execution error"):
            run_custom_operation(df, "result = df.nonexistent_method()")


# ── Replacement Tests: Can custom_operation replicate the 5 hardcoded ops? ──


class TestReplaceHardcodedOps:
    """Verify that custom_operation can replicate each of the 5 dedicated tools."""

    def test_replaces_compute_magnitude(self):
        """magnitude = sqrt(x^2 + y^2 + z^2)"""
        idx = _make_time(3)
        df = pd.DataFrame(
            {"x": [3.0, 0.0, 1.0], "y": [4.0, 0.0, 2.0], "z": [0.0, 5.0, 2.0]},
            index=idx,
        )
        code = "result = df.pow(2).sum(axis=1, skipna=False).pow(0.5).to_frame('magnitude')"
        result = run_custom_operation(df, code)
        np.testing.assert_allclose(result.values.squeeze(), [5.0, 5.0, 3.0])

    def test_replaces_compute_arithmetic_add(self):
        """Element-wise addition of two aligned DataFrames."""
        idx = _make_time(3)
        df = _make_df(np.array([1.0, 2.0, 3.0]), idx)
        # Simulate: df is "a", we embed "b" values in code
        code = "b = pd.DataFrame([10.0, 20.0, 30.0], index=df.index, columns=df.columns)\nresult = df + b"
        result = run_custom_operation(df, code)
        np.testing.assert_allclose(result.values.squeeze(), [11.0, 22.0, 33.0])

    def test_replaces_compute_arithmetic_divide_with_nan(self):
        """Division with zero handling."""
        idx = _make_time(3)
        df = _make_df(np.array([10.0, 20.0, 30.0]), idx)
        code = "divisor = pd.DataFrame([2.0, 0.0, 5.0], index=df.index, columns=df.columns)\nresult = (df / divisor).replace([np.inf, -np.inf], np.nan)"
        result = run_custom_operation(df, code)
        np.testing.assert_allclose(result.iloc[0, 0], 5.0)
        assert np.isnan(result.iloc[1, 0])
        np.testing.assert_allclose(result.iloc[2, 0], 6.0)

    def test_replaces_compute_running_average(self):
        """Centered rolling mean with min_periods=1."""
        idx = _make_time(5)
        df = _make_df(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), idx)
        code = "result = df.rolling(3, center=True, min_periods=1).mean()"
        result = run_custom_operation(df, code)
        np.testing.assert_allclose(result.iloc[0, 0], 15.0)   # avg(10,20)
        np.testing.assert_allclose(result.iloc[2, 0], 30.0)   # avg(20,30,40)
        np.testing.assert_allclose(result.iloc[4, 0], 45.0)   # avg(40,50)

    def test_replaces_compute_resample(self):
        """Downsample by bin-averaging at fixed cadence."""
        idx = _make_time(100, cadence_s=1)
        df = _make_df(np.arange(100, dtype=np.float64), idx)
        code = "result = df.resample('10s').mean().dropna(how='all')"
        result = run_custom_operation(df, code)
        assert len(result) == 10
        np.testing.assert_allclose(result.iloc[0, 0], 4.5)  # mean(0..9)

    def test_replaces_compute_delta_difference(self):
        """Differences: df.diff().iloc[1:]"""
        idx = _make_time(5, cadence_s=60)
        df = _make_df(np.array([10.0, 12.0, 15.0, 11.0, 14.0]), idx)
        code = "result = df.diff().iloc[1:]"
        result = run_custom_operation(df, code)
        assert len(result) == 4
        np.testing.assert_allclose(result.values.squeeze(), [2.0, 3.0, -4.0, 3.0])

    def test_replaces_compute_delta_derivative(self):
        """Time derivative: dv/dt in units per second."""
        idx = _make_time(3, cadence_s=60)
        df = _make_df(np.array([0.0, 60.0, 180.0]), idx)
        code = "dv = df.diff().iloc[1:]\ndt_s = df.index.to_series().diff().dt.total_seconds().iloc[1:]\nresult = dv.div(dt_s, axis=0)"
        result = run_custom_operation(df, code)
        np.testing.assert_allclose(result.values.squeeze(), [1.0, 2.0])


# ── DataFrame Creation Tests ─────────────────────────────────────────────────


class TestExecuteDataframeCreation:
    def test_simple_creation_with_date_range(self):
        code = "result = pd.DataFrame({'value': [1.0, 2.0, 3.0]}, index=pd.date_range('2024-01-01', periods=3, freq='D'))"
        result = execute_dataframe_creation(code)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 3
        np.testing.assert_allclose(result["value"].values, [1.0, 2.0, 3.0])

    def test_event_catalog_from_string_dates(self):
        code = (
            "dates = pd.to_datetime(['2024-01-01', '2024-02-15', '2024-05-10'])\n"
            "result = pd.DataFrame({'flux': [5.2, 7.8, 6.1]}, index=dates)"
        )
        result = execute_dataframe_creation(code)
        assert len(result) == 3
        assert "flux" in result.columns
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_no_result_assignment_error(self):
        with pytest.raises(ValueError, match="did not assign"):
            execute_dataframe_creation("x = pd.DataFrame({'a': [1]})")

    def test_non_dataframe_result_error(self):
        with pytest.raises(ValueError, match="DataFrame, Series, or xarray DataArray"):
            execute_dataframe_creation("result = 42")

    def test_numeric_index_accepted(self):
        """DataFrame with default numeric index is accepted."""
        result = execute_dataframe_creation("result = pd.DataFrame({'a': [1.0, 2.0, 3.0]})")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_series_auto_converted_to_dataframe(self):
        code = "result = pd.Series([1.0, 2.0], index=pd.date_range('2024-01-01', periods=2, freq='D'))"
        result = execute_dataframe_creation(code)
        assert isinstance(result, pd.DataFrame)
        assert "value" in result.columns

    def test_no_df_in_namespace(self):
        """Attempting to use `df` should raise a RuntimeError since it's not provided."""
        with pytest.raises(RuntimeError, match="Execution error"):
            execute_dataframe_creation("result = df * 2")

    def test_string_columns_preserved(self):
        code = (
            "dates = pd.to_datetime(['2024-01-10', '2024-03-22'])\n"
            "result = pd.DataFrame({'class': ['X1.5', 'X2.1'], 'region': ['AR3555', 'AR3590']}, index=dates)"
        )
        result = execute_dataframe_creation(code)
        assert list(result.columns) == ["class", "region"]
        assert result.iloc[0]["class"] == "X1.5"
        assert result.iloc[1]["region"] == "AR3590"

    def test_numpy_operations(self):
        code = "result = pd.DataFrame({'sin': np.sin(np.linspace(0, 2*np.pi, 10))}, index=pd.date_range('2024-01-01', periods=10, freq='h'))"
        result = execute_dataframe_creation(code)
        assert len(result) == 10

    def test_runtime_error(self):
        with pytest.raises(RuntimeError, match="Execution error"):
            execute_dataframe_creation("result = pd.DataFrame(undefined_var)")


class TestRunDataframeCreation:
    def test_end_to_end_success(self):
        code = "result = pd.DataFrame({'v': [10.0, 20.0]}, index=pd.date_range('2024-06-01', periods=2, freq='D'))"
        result = run_dataframe_creation(code)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_validation_rejects_imports(self):
        with pytest.raises(ValueError, match="validation failed"):
            run_dataframe_creation("import os\nresult = pd.DataFrame()")

    def test_validation_rejects_no_result(self):
        with pytest.raises(ValueError, match="validation failed"):
            run_dataframe_creation("x = pd.DataFrame()")

    def test_numeric_index_accepted(self):
        """DataFrame with numeric index passes through run_dataframe_creation."""
        result = run_dataframe_creation("result = pd.DataFrame({'a': [1.0]})")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1


# ── Multi-Source Operation Tests ──────────────────────────────────────────────


class TestExecuteMultiSourceOperation:
    def test_magnitude_from_three_sources(self):
        """Merge 3 scalar dfs, compute magnitude with skipna=False."""
        idx = _make_time(5)
        sources = {
            "df_BR": _make_df(np.array([3.0, 0.0, 1.0, 0.0, 0.0]), idx),
            "df_BT": _make_df(np.array([4.0, 0.0, 2.0, 0.0, 0.0]), idx),
            "df_BN": _make_df(np.array([0.0, 5.0, 2.0, 0.0, 0.0]), idx),
        }
        code = (
            "merged = pd.concat([df_BR, df_BT, df_BN], axis=1)\n"
            "result = merged.pow(2).sum(axis=1, skipna=False).pow(0.5).to_frame('magnitude')"
        )
        result = execute_multi_source_operation(sources, code)
        np.testing.assert_allclose(result.values.squeeze(), [5.0, 5.0, 3.0, 0.0, 0.0])

    def test_nan_preserved_with_skipna_false(self):
        """NaN in a source + skipna=False → NaN in result."""
        idx = _make_time(4)
        sources = {
            "df_BR": _make_df(np.array([3.0, np.nan, 1.0, 0.0]), idx),
            "df_BT": _make_df(np.array([4.0, 5.0, np.nan, 0.0]), idx),
        }
        code = (
            "merged = pd.concat([df_BR, df_BT], axis=1)\n"
            "result = merged.pow(2).sum(axis=1, skipna=False).pow(0.5).to_frame('magnitude')"
        )
        result = execute_multi_source_operation(sources, code)
        assert result.iloc[0, 0] == pytest.approx(5.0)
        assert np.isnan(result.iloc[1, 0])  # NaN from df_BR propagates
        assert np.isnan(result.iloc[2, 0])  # NaN from df_BT propagates
        assert result.iloc[3, 0] == pytest.approx(0.0)

    def test_backward_compat_single_source(self):
        """Single source, code uses `df` — should work."""
        idx = _make_time(3)
        sources = {"df_val": _make_df(np.array([1.0, 4.0, 9.0]), idx)}
        result = execute_multi_source_operation(sources, "result = df * 2")
        np.testing.assert_allclose(result.values.squeeze(), [2.0, 8.0, 18.0])

    def test_cross_cadence_resample(self):
        """Different cadences, code resamples before merging."""
        idx_fast = pd.date_range("2024-01-01", periods=6, freq="1min")
        idx_slow = pd.date_range("2024-01-01", periods=3, freq="2min")
        sources = {
            "df_fast": _make_df(np.arange(6, dtype=float), idx_fast),
            "df_slow": _make_df(np.array([10.0, 20.0, 30.0]), idx_slow),
        }
        code = (
            "slow_resampled = df_slow.reindex(df_fast.index).interpolate()\n"
            "merged = pd.concat([df_fast, slow_resampled], axis=1)\n"
            "result = merged.dropna()"
        )
        result = execute_multi_source_operation(sources, code)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) > 0


class TestRunMultiSourceOperation:
    def test_validates_and_executes(self):
        idx = _make_time(3)
        sources = {"df_A": _make_df(np.array([1.0, 2.0, 3.0]), idx)}
        result_df, warnings = run_multi_source_operation(sources, "result = df * 3")
        np.testing.assert_allclose(result_df.values.squeeze(), [3.0, 6.0, 9.0])
        assert isinstance(warnings, list)

    def test_rejects_invalid_code(self):
        idx = _make_time(3)
        sources = {"df_A": _make_df(np.ones(3), idx)}
        with pytest.raises(ValueError, match="validation failed"):
            run_multi_source_operation(sources, "import os\nresult = df")


class TestValidateResult:
    def test_nan_to_zero_warning(self):
        """NaN in source + skipna=True → zeros in result → warning."""
        idx = _make_time(5)
        src = _make_df(np.array([1.0, np.nan, 3.0, np.nan, 5.0]), idx)
        # Simulate skipna=True: sum treats NaN as 0
        result_df = pd.DataFrame(
            {"mag": [1.0, 0.0, 3.0, 0.0, 5.0]}, index=idx
        )
        warnings = validate_result(result_df, {"df_A": src})
        assert len(warnings) >= 1
        assert any("skipna" in w for w in warnings)

    def test_no_warning_when_clean(self):
        """No NaN, no issues → no warnings."""
        idx = _make_time(5)
        src = _make_df(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), idx)
        result_df = pd.DataFrame({"out": [2.0, 4.0, 6.0, 8.0, 10.0]}, index=idx)
        warnings = validate_result(result_df, {"df_A": src})
        assert warnings == []

    def test_constant_output_warning(self):
        """Constant result from non-constant source → warning."""
        idx = _make_time(5)
        src = _make_df(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), idx)
        result_df = pd.DataFrame({"out": [7.0, 7.0, 7.0, 7.0, 7.0]}, index=idx)
        warnings = validate_result(result_df, {"df_A": src})
        assert any("constant" in w for w in warnings)

    def test_row_count_warning(self):
        """Result has significantly more rows than source → warning."""
        idx_small = _make_time(10)
        src = _make_df(np.ones(10), idx_small)
        idx_big = _make_time(20)
        result_df = pd.DataFrame({"out": np.ones(20)}, index=idx_big)
        warnings = validate_result(result_df, {"df_A": src})
        assert any("unexpected expansion" in w for w in warnings)

    def test_no_row_count_warning_for_small_expansion(self):
        """Result only slightly larger than source → no warning."""
        idx = _make_time(100)
        src = _make_df(np.ones(100), idx)
        idx_big = _make_time(105)
        result_df = pd.DataFrame({"out": np.ones(105)}, index=idx_big)
        warnings = validate_result(result_df, {"df_A": src})
        assert not any("unexpected expansion" in w for w in warnings)


# ── Sandbox Namespace Tests (scipy + pywt) ────────────────────────────────────


class TestSandboxScipy:
    def test_scipy_signal_detrend(self):
        """scipy.signal.detrend available in sandbox."""
        idx = _make_time(100)
        df = _make_df(np.arange(100, dtype=float) + np.random.randn(100) * 0.1, idx)
        code = (
            "vals = np.array(df.iloc[:,0].values, copy=True)\n"
            "detrended = scipy.signal.detrend(vals)\n"
            "result = pd.DataFrame({'detrended': detrended}, index=df.index)"
        )
        result = execute_custom_operation(df, code)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
        # Detrended data should have near-zero mean
        assert abs(result["detrended"].mean()) < 1.0

    def test_scipy_fft_rfft(self):
        """scipy.fft available in sandbox."""
        idx = _make_time(64)
        vals = np.sin(2 * np.pi * np.arange(64) / 16)
        df = _make_df(vals, idx)
        code = (
            "vals = np.array(df.iloc[:,0].values, copy=True)\n"
            "fft_vals = scipy.fft.rfft(vals)\n"
            "n_freq = len(fft_vals)\n"
            "result = pd.DataFrame({'amplitude': np.abs(fft_vals)}, "
            "index=df.index[:n_freq])"
        )
        result = execute_custom_operation(df, code)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 33  # rfft of N points returns N//2+1

    def test_scipy_in_multi_source(self):
        """scipy available in multi-source operations."""
        idx = _make_time(50)
        sources = {"df_A": _make_df(np.random.randn(50), idx)}
        code = (
            "vals = np.array(df.iloc[:,0].values, copy=True)\n"
            "detrended = scipy.signal.detrend(vals)\n"
            "result = pd.DataFrame({'x': detrended}, index=df.index)"
        )
        result = execute_multi_source_operation(sources, code)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 50


class TestSandboxPywt:
    def test_pywt_dwt(self):
        """pywt.dwt available in sandbox."""
        idx = _make_time(100)
        df = _make_df(np.sin(2 * np.pi * np.arange(100) / 20), idx)
        code = (
            "vals = np.array(df.iloc[:,0].values, copy=True)\n"
            "coeffs = pywt.dwt(vals, 'db1')\n"
            "approx = coeffs[0]\n"
            "result = pd.DataFrame({'approx': approx[:len(df)]}, index=df.index[:len(approx)])"
        )
        result = execute_custom_operation(df, code)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_pywt_wavedec(self):
        """pywt.wavedec available in sandbox."""
        idx = _make_time(128)
        df = _make_df(np.random.randn(128), idx)
        code = (
            "vals = np.array(df.iloc[:,0].values, copy=True)\n"
            "coeffs = pywt.wavedec(vals, 'db4', level=3)\n"
            "approx = coeffs[0]\n"
            "result = pd.DataFrame({'approx': np.interp("
            "np.linspace(0, 1, len(df)), np.linspace(0, 1, len(approx)), approx"
            ")}, index=df.index)"
        )
        result = execute_custom_operation(df, code)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 128

    def test_pywt_in_multi_source(self):
        """pywt available in multi-source operations."""
        idx = _make_time(64)
        sources = {"df_A": _make_df(np.random.randn(64), idx)}
        code = (
            "vals = np.array(df.iloc[:,0].values, copy=True)\n"
            "coeffs = pywt.dwt(vals, 'haar')\n"
            "approx = coeffs[0]\n"
            "result = pd.DataFrame({'approx': approx[:len(df)]}, index=df.index[:len(approx)])"
        )
        result = execute_multi_source_operation(sources, code)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# ── Timeseries Mode Tests ─────────────────────────────────────────────────


class TestTimeseriesMode:
    """Tests for the per-DataEntry timeseries vs general-data mode."""

    def test_require_timeseries_rejects_numeric_index(self):
        """When all sources are timeseries, result must have DatetimeIndex."""
        idx = _make_time(3)
        sources = {"df_A": _make_df(np.ones(3), idx)}
        source_ts = {"df_A": True}
        with pytest.raises(ValueError, match="DatetimeIndex"):
            execute_multi_source_operation(
                sources,
                "result = pd.DataFrame({'a': [1.0, 2.0, 3.0]})",
                source_timeseries=source_ts,
            )

    def test_non_timeseries_source_accepts_numeric_index(self):
        """When source is non-timeseries, numeric index is accepted."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        sources = {"df_A": df}
        source_ts = {"df_A": False}
        result = execute_multi_source_operation(
            sources,
            "result = df * 2",
            source_timeseries=source_ts,
        )
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result.index, pd.DatetimeIndex)
        np.testing.assert_allclose(result["a"].values, [2.0, 4.0, 6.0])

    def test_mixed_sources_accepts_any_index(self):
        """Mixed timeseries + non-timeseries sources → no DatetimeIndex required."""
        idx = _make_time(3)
        ts_df = _make_df(np.array([1.0, 2.0, 3.0]), idx)
        non_ts_df = pd.DataFrame({"b": [10.0, 20.0, 30.0]})
        sources = {"df_ts": ts_df, "df_nonts": non_ts_df}
        source_ts = {"df_ts": True, "df_nonts": False}
        result = execute_multi_source_operation(
            sources,
            "result = pd.DataFrame({'sum': df_ts.values.squeeze() + df_nonts['b'].values})",
            source_timeseries=source_ts,
        )
        assert isinstance(result, pd.DataFrame)

    def test_all_timeseries_preserves_datetime(self):
        """When all sources are timeseries, result with DatetimeIndex is accepted."""
        idx = _make_time(3)
        sources = {"df_A": _make_df(np.array([1.0, 2.0, 3.0]), idx)}
        source_ts = {"df_A": True}
        result = execute_multi_source_operation(
            sources,
            "result = df * 2",
            source_timeseries=source_ts,
        )
        assert isinstance(result.index, pd.DatetimeIndex)
        np.testing.assert_allclose(result.values.squeeze(), [2.0, 4.0, 6.0])

    def test_run_multi_source_passes_source_timeseries(self):
        """run_multi_source_operation propagates source_timeseries correctly."""
        df = pd.DataFrame({"a": [1.0, 2.0]})
        sources = {"df_A": df}
        source_ts = {"df_A": False}
        result_df, warnings = run_multi_source_operation(
            sources, "result = df * 3", source_timeseries=source_ts,
        )
        assert isinstance(result_df, pd.DataFrame)
        np.testing.assert_allclose(result_df["a"].values, [3.0, 6.0])

    def test_no_datetime_coercion_for_non_timeseries(self):
        """Non-timeseries sources should NOT have pd.to_datetime() applied."""
        # String index that would fail pd.to_datetime()
        df = pd.DataFrame({"val": [1.0, 2.0]}, index=["event_a", "event_b"])
        sources = {"df_A": df}
        source_ts = {"df_A": False}
        result = execute_multi_source_operation(
            sources,
            "result = df * 10",
            source_timeseries=source_ts,
        )
        assert list(result.index) == ["event_a", "event_b"]

    def test_force_timeseries_false_allows_frequency_index(self):
        """Simulates force_timeseries=false: timeseries source marked as
        non-timeseries in source_ts allows a frequency-indexed result (PSD)."""
        idx = _make_time(256)
        sources = {"df_A": _make_df(np.random.randn(256), idx)}
        # force_timeseries=false is implemented by overriding source_ts to all-False
        source_ts = {"df_A": False}
        result = execute_multi_source_operation(
            sources,
            "freqs = np.fft.rfftfreq(len(df), d=60.0)\n"
            "power = np.abs(np.fft.rfft(df['value'].values))**2\n"
            "result = pd.DataFrame({'power': power}, index=freqs)",
            source_timeseries=source_ts,
        )
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result.index, pd.DatetimeIndex)
        assert "power" in result.columns

    def test_force_timeseries_true_rejects_frequency_index(self):
        """With default force_timeseries=true, a frequency-indexed result
        from timeseries sources is rejected."""
        idx = _make_time(256)
        sources = {"df_A": _make_df(np.random.randn(256), idx)}
        source_ts = {"df_A": True}
        with pytest.raises(ValueError, match="DatetimeIndex"):
            execute_multi_source_operation(
                sources,
                "freqs = np.fft.rfftfreq(len(df), d=60.0)\n"
                "power = np.abs(np.fft.rfft(df['value'].values))**2\n"
                "result = pd.DataFrame({'power': power}, index=freqs)",
                source_timeseries=source_ts,
            )
