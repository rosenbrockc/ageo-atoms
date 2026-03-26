"""Tests for ageoa.numpy.emath atoms."""

import numpy as np
import pytest
import icontract
from ageoa.numpy import emath as ag_emath

class TestSqrt:
    """Tests for the sqrt atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        assert ag_emath.sqrt(4) == 2.0
        assert ag_emath.sqrt(-1) == 1j

    # -- Category 2: Precondition violations --
    def test_require_non_null(self):
        with pytest.raises(icontract.ViolationError, match="not be None"):
            ag_emath.sqrt(None)

    # -- Category 3: Postcondition verification --
    def test_postcondition_squared(self):
        x = -4
        res = ag_emath.sqrt(x)
        assert np.allclose(np.square(res), x)

    # -- Category 4: Edge cases --
    def test_zero(self):
        assert ag_emath.sqrt(0) == 0

    # -- Category 5: Upstream parity --
    def test_matches_upstream(self):
        x = [-1, 0, 1]
        res_atom = ag_emath.sqrt(x)
        res_raw = np.emath.sqrt(x)
        np.testing.assert_array_equal(res_atom, res_raw)

class TestLog:
    """Tests for the log atom."""

    def test_positive_basic(self):
        assert np.allclose(ag_emath.log(np.e), 1.0)
        assert ag_emath.log(-1).imag == np.pi

    def test_require_non_zero(self):
        with pytest.raises(icontract.ViolationError, match="undefined"):
            ag_emath.log(0)

    def test_postcondition_exp(self):
        x = 10.0
        res = ag_emath.log(x)
        assert np.allclose(np.exp(res), x)

    def test_matches_upstream(self):
        x = [1, 2, 3]
        res_atom = ag_emath.log(x)
        res_raw = np.emath.log(x)
        np.testing.assert_array_equal(res_atom, res_raw)

class TestLogn:
    """Tests for the logn atom."""

    def test_positive_basic(self):
        assert ag_emath.logn(2, 8) == 3.0
        assert ag_emath.logn(10, 100) == 2.0

    def test_require_base_constraints(self):
        with pytest.raises(icontract.ViolationError, match="positive"):
            ag_emath.logn(-2, 4)
        with pytest.raises(icontract.ViolationError, match="not equal to 1"):
            ag_emath.logn(1, 4)

    def test_matches_upstream(self):
        res_atom = ag_emath.logn(2, [2, 4, 8])
        res_raw = np.emath.logn(2, [2, 4, 8])
        np.testing.assert_array_equal(res_atom, res_raw)

class TestPower:
    """Tests for the power atom."""

    def test_positive_basic(self):
        assert ag_emath.power(2, 3) == 8
        assert np.allclose(ag_emath.power(-2, 2), 4)

    def test_complex_power(self):
        # (-1)^0.5 = j
        res = ag_emath.power(-1, 0.5)
        assert np.allclose(res, 1j)

    def test_matches_upstream(self):
        x = [-2, 2]
        p = 2
        res_atom = ag_emath.power(x, p)
        res_raw = np.emath.power(x, p)
        np.testing.assert_array_equal(res_atom, res_raw)
