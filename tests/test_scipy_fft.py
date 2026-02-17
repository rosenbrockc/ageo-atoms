"""Tests for ageoa.scipy.fft atoms (dct/idct)."""

import numpy as np
import pytest
import icontract
import scipy.fft

from ageoa.scipy import fft as ag_fft


class TestDct:
    """Tests for the dct atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        res = ag_fft.dct(x)
        expected = scipy.fft.dct(x)
        assert np.allclose(res, expected)

    def test_positive_type3(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        res = ag_fft.dct(x, type=3)
        expected = scipy.fft.dct(x, type=3)
        assert np.allclose(res, expected)

    # -- Category 2: Precondition violations --
    def test_require_non_null(self):
        with pytest.raises(icontract.ViolationError, match="not be None"):
            ag_fft.dct(None)

    def test_require_non_empty(self):
        with pytest.raises(icontract.ViolationError, match="not be empty"):
            ag_fft.dct([])

    # -- Category 3: Postcondition verification --
    def test_postcondition_shape(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        res = ag_fft.dct(x)
        assert res.shape == x.shape

    def test_postcondition_real(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        res = ag_fft.dct(x)
        assert np.isrealobj(res)

    # -- Category 4: Edge cases --
    def test_single_element(self):
        x = np.array([5.0])
        res = ag_fft.dct(x)
        expected = scipy.fft.dct(x)
        assert np.allclose(res, expected)

    def test_ortho_norm(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        res = ag_fft.dct(x, norm="ortho")
        expected = scipy.fft.dct(x, norm="ortho")
        assert np.allclose(res, expected)

    # -- Category 5: Upstream parity --
    def test_matches_upstream(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(16)
        res_atom = ag_fft.dct(x)
        res_raw = scipy.fft.dct(x)
        np.testing.assert_array_almost_equal(res_atom, res_raw)


class TestIdct:
    """Tests for the idct atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        spectrum = scipy.fft.dct(x)
        res = ag_fft.idct(spectrum)
        assert np.allclose(res, x)

    def test_positive_type3(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        spectrum = scipy.fft.dct(x, type=3)
        res = ag_fft.idct(spectrum, type=3)
        assert np.allclose(res, x)

    # -- Category 2: Precondition violations --
    def test_require_non_null(self):
        with pytest.raises(icontract.ViolationError, match="not be None"):
            ag_fft.idct(None)

    def test_require_non_empty(self):
        with pytest.raises(icontract.ViolationError, match="not be empty"):
            ag_fft.idct([])

    # -- Category 3: Postcondition verification --
    def test_postcondition_shape(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        res = ag_fft.idct(x)
        assert res.shape == x.shape

    def test_postcondition_real(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        res = ag_fft.idct(x)
        assert np.isrealobj(res)

    # -- Category 4: Edge cases --
    def test_round_trip(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        res = ag_fft.idct(ag_fft.dct(x))
        assert np.allclose(res, x)

    # -- Category 5: Upstream parity --
    def test_matches_upstream(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(16)
        res_atom = ag_fft.idct(x)
        res_raw = scipy.fft.idct(x)
        np.testing.assert_array_almost_equal(res_atom, res_raw)
