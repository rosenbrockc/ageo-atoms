"""Tests for e2e_ppg."""

import pytest
import numpy as np
import icontract
from ageoa.e2e_ppg.atoms import kazemi_peak_detection, ppg_reconstruction, ppg_sqa


class TestKazemiPeakDetection:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            kazemi_peak_detection(np.array([1.0, 2.0, 3.0]))

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            kazemi_peak_detection(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            kazemi_peak_detection(np.array([]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            kazemi_peak_detection(np.array([1.0, np.nan, 3.0]))


class TestPPGReconstruction:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ppg_reconstruction(np.array([1.0, 2.0, 3.0]))

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            ppg_reconstruction(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            ppg_reconstruction(np.array([]))


class TestPPGSQA:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ppg_sqa(np.array([1.0, 2.0, 3.0]))

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            ppg_sqa(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            ppg_sqa(np.array([]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            ppg_sqa(np.array([np.inf]))
