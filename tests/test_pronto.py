"""Tests for pronto."""

import pytest
import numpy as np
import icontract
from ageoa.pronto.atoms import rbis_state_estimation


class TestRBISStateEstimation:
    def test_returns_result(self):
        result = rbis_state_estimation(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, np.ndarray)

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            rbis_state_estimation(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            rbis_state_estimation(np.array([]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            rbis_state_estimation(np.array([np.nan, np.inf]))
