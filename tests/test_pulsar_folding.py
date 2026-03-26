"""Tests for pulsar_folding."""

import pytest
import numpy as np
import icontract
from ageoa.pulsar_folding.atoms import dm_can_brute_force, spline_bandpass_correction


class TestDMCanBruteForce:
    def test_returns_result(self):
        result = dm_can_brute_force(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, np.ndarray)

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            dm_can_brute_force(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            dm_can_brute_force(np.array([]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            dm_can_brute_force(np.array([np.nan]))


class TestSplineBandpassCorrection:
    def test_returns_result(self):
        result = spline_bandpass_correction(np.array([1.0, 2.0, 3.0, 2.5, 1.5]))
        assert isinstance(result, np.ndarray)

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            spline_bandpass_correction(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            spline_bandpass_correction(np.array([]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            spline_bandpass_correction(np.array([np.inf]))
