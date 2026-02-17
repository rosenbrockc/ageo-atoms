"""Tests for ageoa.scipy.integrate atoms."""

import numpy as np
import pytest
import icontract
import scipy.integrate
from ageoa.scipy import integrate as ag_integrate

class TestQuad:
    """Tests for the quad atom."""

    def test_positive_basic(self):
        res, err = ag_integrate.quad(lambda x: np.sin(x), 0, np.pi)
        assert np.allclose(res, 2.0)

    def test_require_non_null(self):
        with pytest.raises(icontract.ViolationError, match="not be None"):
            ag_integrate.quad(None, 0, 1)

class TestSolveIvp:
    """Tests for the solve_ivp atom."""

    def test_positive_basic(self):
        def f(t, y): return -2 * y
        res = ag_integrate.solve_ivp(f, (0, 5), [1.0])
        assert res.success
        assert np.allclose(res.y[0][-1], np.exp(-10), atol=1e-3)

    def test_require_non_null(self):
        with pytest.raises(icontract.ViolationError, match="not be None"):
            ag_integrate.solve_ivp(None, (0, 1), [1])

class TestSimpson:
    """Tests for the simpson atom."""

    def test_positive_basic(self):
        x = np.linspace(0, 1, 11)
        y = x**2
        res = ag_integrate.simpson(y, x)
        # Integral of x^2 from 0 to 1 is 1/3
        assert np.allclose(res, 1/3, atol=1e-2)

    def test_require_non_empty(self):
        with pytest.raises(icontract.ViolationError, match="not be empty"):
            ag_integrate.simpson([])
