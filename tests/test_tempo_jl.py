"""Tests for tempo_jl."""

import pytest
import numpy as np
import icontract
from ageoa.tempo_jl.atoms import graph_time_scale_management, high_precision_duration


class TestGraphTimeScaleManagement:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            graph_time_scale_management(np.array([1.0, 2.0, 3.0]))

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
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            high_precision_duration(np.array([1.0, 2.0, 3.0]))

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            high_precision_duration(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            high_precision_duration(np.array([]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            high_precision_duration(np.array([np.inf]))
