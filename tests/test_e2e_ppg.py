"""Tests for e2e_ppg."""

import pytest
import numpy as np
import icontract
from ageoa.e2e_ppg.atoms import kazemi_peak_detection, ppg_reconstruction, ppg_sqa


class TestKazemiPeakDetection:
    def test_returns_result(self):
        result = kazemi_peak_detection(np.array([1.0, 3.0, 1.0, 3.0, 1.0]), 10, 4, 1, 1)
        assert isinstance(result, np.ndarray)

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            kazemi_peak_detection(None, 10, 4, 1, 1)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            kazemi_peak_detection(np.array([]), 10, 4, 1, 1)

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            kazemi_peak_detection(np.array([1.0, np.nan, 3.0]), 10, 4, 1, 1)


class TestPPGReconstruction:
    def test_returns_result(self):
        result = ppg_reconstruction(np.array([1.0, 2.0, 3.0]), [[0]], [[1]], 20)
        assert isinstance(result, tuple)
        assert isinstance(result[0], np.ndarray)

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            ppg_reconstruction(None, [], [], 20)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            ppg_reconstruction(np.array([]), [], [], 20)


class TestPPGSQA:
    def test_returns_result(self):
        signal = np.sin(np.linspace(0.0, 60.0, 1000))
        result = ppg_sqa(signal, 20)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            ppg_sqa(None, 20)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            ppg_sqa(np.array([]), 20)

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            ppg_sqa(np.array([np.inf]), 20)
