"""Tests for ageoa.numpy.polynomial atoms."""

import numpy as np
import pytest
import icontract
from ageoa.numpy import polynomial as ag_poly

class TestPolyval:
    """Tests for the polyval atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        # 1 + 2x + 3x^2 at x=2 -> 1 + 4 + 12 = 17
        res = ag_poly.polyval(2, [1, 2, 3])
        assert res == 17.0

    # -- Category 2: Precondition violations --
    def test_require_non_null(self):
        with pytest.raises(icontract.ViolationError, match="not be None"):
            ag_poly.polyval(None, [1, 2])

    def test_require_non_empty_c(self):
        with pytest.raises(icontract.ViolationError, match="not be empty"):
            ag_poly.polyval(1, [])

    # -- Category 3: Postcondition verification --
    def test_postcondition_shape(self):
        x = [1, 2, 3]
        res = ag_poly.polyval(x, [1, 1])
        assert res.shape == (3,)

    # -- Category 4: Edge cases --
    def test_scalar_coeff(self):
        res = ag_poly.polyval(10, [5])
        assert res == 5.0

    # -- Category 5: Upstream parity --
    def test_matches_upstream(self):
        x = np.linspace(-1, 1, 10)
        c = [1, 0, -1, 2]
        res_atom = ag_poly.polyval(x, c)
        res_raw = np.polynomial.polynomial.polyval(x, c)
        np.testing.assert_array_equal(res_atom, res_raw)

class TestPolyfit:
    """Tests for the polyfit atom."""

    def test_positive_basic(self):
        x = [0, 1, 2]
        y = [1, 3, 5] # y = 1 + 2x
        res = ag_poly.polyfit(x, y, 1)
        assert np.allclose(res, [1, 2])

    def test_require_same_length(self):
        with pytest.raises(icontract.ViolationError, match="same length"):
            ag_poly.polyfit([1, 2], [1], 1)

    def test_postcondition_deg(self):
        res = ag_poly.polyfit([0, 1, 2], [1, 2, 3], 2)
        assert len(res) == 3

    def test_matches_upstream(self):
        x = [0, 1, 2, 3]
        y = [0, 1, 4, 9]
        res_atom = ag_poly.polyfit(x, y, 2)
        res_raw = np.polynomial.polynomial.polyfit(x, y, 2)
        np.testing.assert_array_equal(res_atom, res_raw)

class TestPolyder:
    """Tests for the polyder atom."""

    def test_positive_basic(self):
        # 1 + 2x + 3x^2 -> 2 + 6x
        res = ag_poly.polyder([1, 2, 3])
        assert np.allclose(res, [2, 6])

    def test_postcondition_len(self):
        res = ag_poly.polyder([1, 2, 3, 4], m=2)
        assert len(res) == 2 # 3x^2 + 6x + 9 -> 6x + 6

    def test_matches_upstream(self):
        c = [1, 2, 3, 4, 5]
        res_atom = ag_poly.polyder(c, m=2)
        res_raw = np.polynomial.polynomial.polyder(c, m=2)
        np.testing.assert_array_equal(res_atom, res_raw)

class TestPolyint:
    """Tests for the polyint atom."""

    def test_positive_basic(self):
        # 2 + 6x -> 2x + 3x^2 + k
        res = ag_poly.polyint([2, 6], k=5)
        assert np.allclose(res, [5, 2, 3])

    def test_postcondition_len(self):
        res = ag_poly.polyint([1, 1], m=3)
        assert len(res) == 5

    def test_matches_upstream(self):
        c = [1, 2, 3]
        res_atom = ag_poly.polyint(c, m=2, k=[1, 2])
        res_raw = np.polynomial.polynomial.polyint(c, m=2, k=[1, 2])
        np.testing.assert_array_equal(res_atom, res_raw)

class TestPolyroots:
    """Tests for the polyroots atom."""

    def test_positive_basic(self):
        # x^2 - 1 = 0 -> roots [-1, 1]
        res = ag_poly.polyroots([-1, 0, 1])
        assert np.allclose(sorted(res), [-1, 1])

    def test_require_degree_1(self):
        with pytest.raises(icontract.ViolationError, match="at least degree 1"):
            ag_poly.polyroots([1])

    def test_postcondition_len(self):
        res = ag_poly.polyroots([1, 2, 1]) # (x+1)^2
        assert len(res) == 2

    def test_matches_upstream(self):
        c = [6, -5, 1] # (x-2)(x-3) = x^2 - 5x + 6
        res_atom = ag_poly.polyroots(c)
        res_raw = np.polynomial.polynomial.polyroots(c)
        np.testing.assert_allclose(sorted(res_atom), sorted(res_raw))
