"""Tests for ageoa.scipy.linalg atoms."""

import numpy as np
import pytest
import icontract
import scipy.linalg
from ageoa.scipy import linalg as ag_slinalg

class TestSolve:
    """Tests for the solve atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        a = np.array([[3, 2], [1, 2]])
        b = np.array([1, 0])
        x = ag_slinalg.solve(a, b)
        assert np.allclose(a @ x, b)

    # -- Category 2: Precondition violations --
    def test_require_2d(self):
        with pytest.raises(icontract.ViolationError, match="(2D matrix|square)"):
            ag_slinalg.solve([1, 2], [1, 2])

    def test_require_square(self):
        with pytest.raises(icontract.ViolationError, match="square"):
            ag_slinalg.solve([[1, 2, 3], [4, 5, 6]], [1, 2])

    def test_require_dim_match(self):
        with pytest.raises(icontract.ViolationError, match="match"):
            ag_slinalg.solve([[1, 2], [3, 4]], [1, 2, 3])

    # -- Category 3: Postcondition verification --
    def test_postcondition_shape(self):
        a = np.eye(2)
        b = np.array([1, 2])
        x = ag_slinalg.solve(a, b)
        assert x.shape == b.shape

    # -- Category 4: Edge cases --
    def test_1x1(self):
        a = np.array([[5.0]])
        b = np.array([10.0])
        x = ag_slinalg.solve(a, b)
        assert x[0] == 2.0

    # -- Category 5: Upstream parity --
    def test_matches_upstream(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal((4, 4))
        b = rng.standard_normal((4, 2))
        x_atom = ag_slinalg.solve(a, b)
        x_raw = scipy.linalg.solve(a, b)
        np.testing.assert_array_equal(x_atom, x_raw)

class TestInv:
    """Tests for the inv atom."""

    def test_positive_basic(self):
        a = np.array([[1, 2], [3, 4]])
        a_inv = ag_slinalg.inv(a)
        assert np.allclose(a @ a_inv, np.eye(2))

    def test_require_square_2d(self):
        with pytest.raises(icontract.ViolationError, match="square 2D matrix"):
            ag_slinalg.inv([1, 2, 3, 4])

    def test_postcondition_shape(self):
        a = np.eye(3)
        a_inv = ag_slinalg.inv(a)
        assert a_inv.shape == a.shape

    def test_matches_upstream(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal((5, 5))
        a_inv_atom = ag_slinalg.inv(a)
        a_inv_raw = scipy.linalg.inv(a)
        np.testing.assert_array_equal(a_inv_atom, a_inv_raw)

class TestLuFactorSolve:
    """Tests for lu_factor and lu_solve atoms."""

    def test_positive_basic(self):
        a = np.array([[2, 5, 8], [5, 2, 2], [7, 5, 6]])
        b = np.array([1, 1, 1])
        lu, piv = ag_slinalg.lu_factor(a)
        x = ag_slinalg.lu_solve((lu, piv), b)
        assert np.allclose(a @ x, b)

    def test_require_lu_tuple(self):
        with pytest.raises(icontract.ViolationError, match="tuple"):
            ag_slinalg.lu_solve([np.eye(2)], [1, 2])

    def test_postcondition_lu_shapes(self):
        a = np.random.rand(4, 4)
        lu, piv = ag_slinalg.lu_factor(a)
        assert lu.shape == a.shape
        assert piv.shape == (a.shape[0],)

    def test_matches_upstream(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal((4, 4))
        b = rng.standard_normal(4)
        lu_atom, piv_atom = ag_slinalg.lu_factor(a)
        x_atom = ag_slinalg.lu_solve((lu_atom, piv_atom), b)
        lu_raw, piv_raw = scipy.linalg.lu_factor(a)
        x_raw = scipy.linalg.lu_solve((lu_raw, piv_raw), b)
        np.testing.assert_array_equal(x_atom, x_raw)
