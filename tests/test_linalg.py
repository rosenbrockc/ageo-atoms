"""Tests for ageoa.numpy.linalg atoms."""

import numpy as np
import pytest
import icontract
from ageoa.numpy import linalg as ag_linalg

class TestSolve:
    """Tests for the solve atom."""

    def test_positive_basic(self):
        a = np.array([[3, 1], [1, 2]])
        b = np.array([9, 8])
        x = ag_linalg.solve(a, b)
        assert np.allclose(a @ x, b)

    def test_require_2d(self):
        with pytest.raises(icontract.ViolationError, match="(2D matrix|square)"):
            ag_linalg.solve([1, 2], [1, 2])

    def test_require_square(self):
        with pytest.raises(icontract.ViolationError, match="square"):
            ag_linalg.solve([[1, 2, 3], [4, 5, 6]], [1, 2])

    def test_require_dim_match(self):
        with pytest.raises(icontract.ViolationError, match="match"):
            ag_linalg.solve([[1, 2], [3, 4]], [1, 2, 3])

    def test_postcondition_shape(self):
        a = np.eye(2)
        b = np.array([1, 2])
        x = ag_linalg.solve(a, b)
        assert x.shape == b.shape

    def test_1x1(self):
        a = np.array([[5.0]])
        b = np.array([10.0])
        x = ag_linalg.solve(a, b)
        assert x[0] == 2.0

    def test_matches_upstream(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal((4, 4))
        b = rng.standard_normal((4, 2))
        x_atom = ag_linalg.solve(a, b)
        x_raw = np.linalg.solve(a, b)
        np.testing.assert_array_equal(x_atom, x_raw)

class TestInv:
    """Tests for the inv atom."""

    def test_positive_basic(self):
        a = np.array([[1, 2], [3, 4]])
        a_inv = ag_linalg.inv(a)
        assert np.allclose(a @ a_inv, np.eye(2))

    def test_require_square_2d(self):
        # 1D case
        with pytest.raises(icontract.ViolationError, match="square 2D matrix"):
            ag_linalg.inv([1, 2, 3, 4])
        # Not square
        with pytest.raises(icontract.ViolationError, match="square 2D matrix"):
            ag_linalg.inv([[1, 2, 3], [4, 5, 6]])

    def test_postcondition_shape(self):
        a = np.eye(3)
        a_inv = ag_linalg.inv(a)
        assert a_inv.shape == a.shape

    def test_2x2_identity(self):
        a = np.eye(2)
        assert np.all(ag_linalg.inv(a) == a)

    def test_matches_upstream(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal((5, 5))
        a_inv_atom = ag_linalg.inv(a)
        a_inv_raw = np.linalg.inv(a)
        np.testing.assert_array_equal(a_inv_atom, a_inv_raw)

class TestDet:
    """Tests for the det atom."""

    def test_positive_basic(self):
        a = np.array([[1, 2], [3, 4]])
        assert abs(ag_linalg.det(a) - (-2.0)) < 1e-6

    def test_require_at_least_2d(self):
        with pytest.raises(icontract.ViolationError, match="(at least 2 dimensions|square)"):
            ag_linalg.det([1, 2, 3, 4])

    def test_require_square_last_two(self):
        with pytest.raises(icontract.ViolationError, match="square"):
            ag_linalg.det([[1, 2, 3], [4, 5, 6]])

    def test_postcondition_non_null(self):
        assert ag_linalg.det(np.eye(2)) is not None

    def test_3x3_zero_det(self):
        a = np.ones((3, 3))
        assert abs(ag_linalg.det(a)) < 1e-15

    def test_matches_upstream(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal((4, 4))
        det_atom = ag_linalg.det(a)
        det_raw = np.linalg.det(a)
        assert np.allclose(det_atom, det_raw)

class TestNorm:
    """Tests for the norm atom."""

    def test_positive_basic(self):
        v = [3.0, 4.0]
        assert ag_linalg.norm(v) == 5.0

    def test_require_non_null(self):
        with pytest.raises(icontract.ViolationError, match="must not be None"):
            ag_linalg.norm(None)

    def test_postcondition_non_negative(self):
        assert ag_linalg.norm([-1, -2]) >= 0

    def test_frobenius_norm(self):
        m = [[1, 2], [3, 4]]
        # sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(1+4+9+16) = sqrt(30)
        assert abs(ag_linalg.norm(m) - np.sqrt(30)) < 1e-6

    def test_matches_upstream(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal((3, 4))
        n_atom = ag_linalg.norm(a, ord='fro')
        n_raw = np.linalg.norm(a, ord='fro')
        assert np.allclose(n_atom, n_raw)
