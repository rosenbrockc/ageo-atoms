"""Tests for quantfin."""

import pytest
import numpy as np
import icontract
from ageoa.quantfin.atoms import functional_monte_carlo, volatility_surface_modeling


class TestFunctionalMonteCarlo:
    def test_returns_array(self):
        result = functional_monte_carlo(np.array([0.05, 0.2, 1.0]))
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 1

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            functional_monte_carlo(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            functional_monte_carlo(np.array([]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            functional_monte_carlo(np.array([np.nan, 1.0]))


class TestVolatilitySurfaceModeling:
    def test_returns_array(self):
        result = volatility_surface_modeling(np.array([0.2, 0.25, 0.3]))
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 1

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            volatility_surface_modeling(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            volatility_surface_modeling(np.array([]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            volatility_surface_modeling(np.array([np.inf]))
