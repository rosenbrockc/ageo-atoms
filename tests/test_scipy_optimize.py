"""Tests for ageoa.scipy.optimize atoms."""

import numpy as np
import pytest
import icontract
import scipy.optimize
from ageoa.scipy import optimize as ag_optimize

class TestMinimize:
    """Tests for the minimize atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        def rosen(x):
            return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
        x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
        res = ag_optimize.minimize(rosen, x0, method='nelder-mead')
        assert res.success
        assert np.allclose(res.x, [1, 1, 1, 1, 1], atol=1e-1)

    # -- Category 2: Precondition violations --
    def test_require_non_null(self):
        with pytest.raises(icontract.ViolationError, match="not be None"):
            ag_optimize.minimize(None, [0, 0])

    # -- Category 3: Postcondition verification --
    def test_postcondition_result(self):
        res = ag_optimize.minimize(lambda x: x**2, [1.0])
        assert isinstance(res, scipy.optimize.OptimizeResult)

    # -- Category 4: Edge cases --
    def test_scalar_minimization(self):
        res = ag_optimize.minimize(lambda x: (x[0]-2)**2, [0.0])
        assert res.success
        assert np.allclose(res.x, [2.0])

    # -- Category 5: Upstream parity --
    def test_matches_upstream(self):
        def f(x): return (x[0]-1)**2 + (x[1]-2.5)**2
        x0 = [2, 0]
        res_atom = ag_optimize.minimize(f, x0)
        res_raw = scipy.optimize.minimize(f, x0)
        np.testing.assert_array_equal(res_atom.x, res_raw.x)

class TestRoot:
    """Tests for the root atom."""

    def test_positive_basic(self):
        def f(x): return [x[0] + 0.5 * (x[0] - x[1])**3 - 1.0,
                         0.5 * (x[1] - x[0])**3 + x[1]]
        res = ag_optimize.root(f, [0, 0])
        assert res.success
        assert np.allclose(f(res.x), [0, 0])

    def test_require_non_null(self):
        with pytest.raises(icontract.ViolationError, match="not be None"):
            ag_optimize.root(None, [0])

    def test_matches_upstream(self):
        def f(x): return x + 2 * np.cos(x)
        res_atom = ag_optimize.root(f, 0.3)
        res_raw = scipy.optimize.root(f, 0.3)
        np.testing.assert_array_equal(res_atom.x, res_raw.x)

class TestLinprog:
    """Tests for the linprog atom."""

    def test_positive_basic(self):
        c = [-1, 4]
        A = [[-3, 1], [1, 2]]
        b = [6, 4]
        x0_bounds = (None, None)
        x1_bounds = (-3, None)
        res = ag_optimize.linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])
        assert res.success
        # Optimal solution x=[10, -3], fun=-22
        assert np.allclose(res.x, [10, -3])

    def test_require_non_null_c(self):
        with pytest.raises(icontract.ViolationError, match="must not be None"):
            ag_optimize.linprog(None)

class TestCurveFit:
    """Tests for the curve_fit atom."""

    def test_positive_basic(self):
        def f(x, a, b): return a * np.exp(-b * x)
        xdata = np.linspace(0, 4, 50)
        y = f(xdata, 2.5, 1.3)
        rng = np.random.default_rng()
        y_noise = 0.2 * rng.normal(size=xdata.size)
        ydata = y + y_noise
        popt, pcov = ag_optimize.curve_fit(f, xdata, ydata)
        assert len(popt) == 2

    def test_require_same_length(self):
        with pytest.raises(icontract.ViolationError, match="same length"):
            ag_optimize.curve_fit(lambda x, a: a*x, [1, 2], [1])
