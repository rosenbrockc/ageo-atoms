"""Tests for tempo_jl."""

import pytest
import numpy as np
import icontract
from ageoa.tempo_jl.atoms import graph_time_scale_management, high_precision_duration
from ageoa.tempo_jl.offsets.atoms import offset_tt2tdb, offset_tt2tdbh, tt2tdb_offset


class TestGraphTimeScaleManagement:
    def test_returns_result(self):
        result = graph_time_scale_management(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, np.ndarray)

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            graph_time_scale_management(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            graph_time_scale_management(np.array([]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            graph_time_scale_management(np.array([np.nan]))


class TestHighPrecisionDuration:
    def test_returns_result(self):
        result = high_precision_duration(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, np.ndarray)

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            high_precision_duration(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            high_precision_duration(np.array([]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            high_precision_duration(np.array([np.inf]))


class TestTempoOffsets:
    def test_scalar_offsets_return_finite_scalars(self):
        seconds = 12345.0
        assert np.isfinite(offset_tt2tdb(seconds))
        assert np.isfinite(offset_tt2tdbh(seconds))

    def test_vectorized_offset_preserves_shape(self):
        seconds = np.array([0.0, 100.0, 200.0], dtype=float)
        result = tt2tdb_offset(seconds)
        assert isinstance(result, np.ndarray)
        assert result.shape == seconds.shape
