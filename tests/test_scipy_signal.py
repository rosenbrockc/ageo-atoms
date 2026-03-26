"""Tests for ageoa.scipy.signal atoms."""

import numpy as np
import pytest
import icontract
import scipy.signal

from ageoa.scipy import signal as ag_signal


class TestButter:
    """Tests for the butter atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        b, a = ag_signal.butter(4, 0.2)
        assert b.ndim == 1
        assert a.ndim == 1

    # -- Category 2: Precondition violations --
    def test_require_positive_order(self):
        with pytest.raises(icontract.ViolationError, match="positive integer"):
            ag_signal.butter(0, 0.2)

    def test_require_valid_freq(self):
        with pytest.raises(icontract.ViolationError, match="Critical frequency"):
            ag_signal.butter(4, 0.0)

    def test_require_freq_below_nyquist(self):
        with pytest.raises(icontract.ViolationError, match="Critical frequency"):
            ag_signal.butter(4, 500.0, fs=1000.0)

    # -- Category 3: Postcondition verification --
    def test_postcondition_tuple(self):
        result = ag_signal.butter(4, 0.2)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_postcondition_1d(self):
        b, a = ag_signal.butter(4, 0.2)
        assert b.ndim == 1
        assert a.ndim == 1

    # -- Category 4: Edge cases --
    def test_order_one(self):
        b, a = ag_signal.butter(1, 0.5)
        assert b.ndim == 1

    def test_highpass(self):
        b, a = ag_signal.butter(3, 0.3, btype="high")
        assert isinstance(b, np.ndarray)

    # -- Category 5: Upstream parity --
    def test_matches_upstream(self):
        b_atom, a_atom = ag_signal.butter(4, 0.2)
        b_raw, a_raw = scipy.signal.butter(4, 0.2)
        np.testing.assert_array_almost_equal(b_atom, b_raw)
        np.testing.assert_array_almost_equal(a_atom, a_raw)


class TestCheby1:
    """Tests for the cheby1 atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        b, a = ag_signal.cheby1(4, 1.0, 0.2)
        assert b.ndim == 1

    # -- Category 2: Precondition violations --
    def test_require_positive_order(self):
        with pytest.raises(icontract.ViolationError, match="positive integer"):
            ag_signal.cheby1(0, 1.0, 0.2)

    def test_require_positive_ripple(self):
        with pytest.raises(icontract.ViolationError, match="ripple"):
            ag_signal.cheby1(4, -1.0, 0.2)

    def test_require_valid_freq(self):
        with pytest.raises(icontract.ViolationError, match="Critical frequency"):
            ag_signal.cheby1(4, 1.0, 0.0)

    # -- Category 3: Postcondition verification --
    def test_postcondition_tuple(self):
        result = ag_signal.cheby1(4, 1.0, 0.2)
        assert isinstance(result, tuple)
        assert len(result) == 2

    # -- Category 4: Edge cases --
    def test_order_one(self):
        b, a = ag_signal.cheby1(1, 0.5, 0.3)
        assert b.ndim == 1

    # -- Category 5: Upstream parity --
    def test_matches_upstream(self):
        b_atom, a_atom = ag_signal.cheby1(4, 1.0, 0.2)
        b_raw, a_raw = scipy.signal.cheby1(4, 1.0, 0.2)
        np.testing.assert_array_almost_equal(b_atom, b_raw)
        np.testing.assert_array_almost_equal(a_atom, a_raw)


class TestCheby2:
    """Tests for the cheby2 atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        b, a = ag_signal.cheby2(4, 40.0, 0.2)
        assert b.ndim == 1

    # -- Category 2: Precondition violations --
    def test_require_positive_order(self):
        with pytest.raises(icontract.ViolationError, match="positive integer"):
            ag_signal.cheby2(0, 40.0, 0.2)

    def test_require_positive_attenuation(self):
        with pytest.raises(icontract.ViolationError, match="attenuation"):
            ag_signal.cheby2(4, -40.0, 0.2)

    def test_require_valid_freq(self):
        with pytest.raises(icontract.ViolationError, match="Critical frequency"):
            ag_signal.cheby2(4, 40.0, 0.0)

    # -- Category 3: Postcondition verification --
    def test_postcondition_tuple(self):
        result = ag_signal.cheby2(4, 40.0, 0.2)
        assert isinstance(result, tuple)
        assert len(result) == 2

    # -- Category 4: Edge cases --
    def test_order_one(self):
        b, a = ag_signal.cheby2(1, 20.0, 0.3)
        assert b.ndim == 1

    # -- Category 5: Upstream parity --
    def test_matches_upstream(self):
        b_atom, a_atom = ag_signal.cheby2(4, 40.0, 0.2)
        b_raw, a_raw = scipy.signal.cheby2(4, 40.0, 0.2)
        np.testing.assert_array_almost_equal(b_atom, b_raw)
        np.testing.assert_array_almost_equal(a_atom, a_raw)


class TestFirwin:
    """Tests for the firwin atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        h = ag_signal.firwin(51, 0.3)
        assert h.shape == (51,)

    # -- Category 2: Precondition violations --
    def test_require_positive_numtaps(self):
        with pytest.raises(icontract.ViolationError, match="positive integer"):
            ag_signal.firwin(0, 0.3)

    # -- Category 3: Postcondition verification --
    def test_postcondition_shape(self):
        h = ag_signal.firwin(101, 0.25)
        assert h.shape == (101,)

    def test_postcondition_real(self):
        h = ag_signal.firwin(51, 0.3)
        assert np.isrealobj(h)

    # -- Category 4: Edge cases --
    def test_numtaps_one(self):
        h = ag_signal.firwin(1, 0.5)
        assert h.shape == (1,)

    # -- Category 5: Upstream parity --
    def test_matches_upstream(self):
        h_atom = ag_signal.firwin(51, 0.3)
        h_raw = scipy.signal.firwin(51, 0.3)
        np.testing.assert_array_equal(h_atom, h_raw)


class TestSosfilt:
    """Tests for the sosfilt atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        sos = scipy.signal.butter(4, 0.2, output="sos")
        x = np.random.rand(100)
        res = ag_signal.sosfilt(sos, x)
        assert res.shape == x.shape

    # -- Category 2: Precondition violations --
    def test_require_sos_shape(self):
        with pytest.raises(icontract.ViolationError, match="n_sections, 6"):
            ag_signal.sosfilt(np.array([[1, 2, 3]]), np.array([1.0, 2.0]))

    def test_require_nonempty_signal(self):
        sos = scipy.signal.butter(2, 0.3, output="sos")
        with pytest.raises(icontract.ViolationError, match="not be empty"):
            ag_signal.sosfilt(sos, [])

    # -- Category 3: Postcondition verification --
    def test_postcondition_shape(self):
        sos = scipy.signal.butter(4, 0.2, output="sos")
        x = np.random.rand(50)
        res = ag_signal.sosfilt(sos, x)
        assert res.shape == x.shape

    # -- Category 4: Edge cases --
    def test_single_sample(self):
        sos = scipy.signal.butter(2, 0.3, output="sos")
        x = np.array([1.0])
        res = ag_signal.sosfilt(sos, x)
        assert res.shape == (1,)

    # -- Category 5: Upstream parity --
    def test_matches_upstream(self):
        sos = scipy.signal.butter(4, 0.2, output="sos")
        x = np.random.default_rng(42).standard_normal(100)
        res_atom = ag_signal.sosfilt(sos, x)
        res_raw = scipy.signal.sosfilt(sos, x)
        np.testing.assert_array_almost_equal(res_atom, res_raw)


class TestLfilter:
    """Tests for the lfilter atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        b, a = scipy.signal.butter(4, 0.2)
        x = np.random.rand(100)
        res = ag_signal.lfilter(b, a, x)
        assert res.shape == x.shape

    # -- Category 2: Precondition violations --
    def test_require_b_1d(self):
        with pytest.raises(icontract.ViolationError, match="1D"):
            ag_signal.lfilter(np.ones((2, 2)), [1], [1.0, 2.0])

    def test_require_a_1d(self):
        with pytest.raises(icontract.ViolationError, match="1D"):
            ag_signal.lfilter([1], np.ones((2, 2)), [1.0, 2.0])

    def test_require_a0_nonzero(self):
        with pytest.raises(icontract.ViolationError, match="a\\[0\\]"):
            ag_signal.lfilter([1.0], [0.0, 1.0], [1.0, 2.0])

    def test_require_nonempty_signal(self):
        with pytest.raises(icontract.ViolationError, match="not be empty"):
            ag_signal.lfilter([1.0], [1.0], [])

    # -- Category 3: Postcondition verification --
    def test_postcondition_shape(self):
        b, a = scipy.signal.butter(3, 0.25)
        x = np.random.rand(50)
        res = ag_signal.lfilter(b, a, x)
        assert res.shape == x.shape

    # -- Category 4: Edge cases --
    def test_fir_filter(self):
        b = [1.0, 0.5, 0.25]
        a = [1.0]
        x = np.random.rand(20)
        res = ag_signal.lfilter(b, a, x)
        assert res.shape == x.shape

    # -- Category 5: Upstream parity --
    def test_matches_upstream(self):
        b, a = scipy.signal.butter(4, 0.2)
        x = np.random.default_rng(42).standard_normal(100)
        res_atom = ag_signal.lfilter(b, a, x)
        res_raw = scipy.signal.lfilter(b, a, x)
        np.testing.assert_array_almost_equal(res_atom, res_raw)


class TestFreqz:
    """Tests for the freqz atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        b, a = scipy.signal.butter(4, 0.2)
        w, h = ag_signal.freqz(b, a)
        assert w.shape[0] == 512
        assert h.shape[0] == 512

    # -- Category 2: Precondition violations --
    def test_require_b_nonempty(self):
        with pytest.raises(icontract.ViolationError, match="non-empty"):
            ag_signal.freqz([])

    # -- Category 3: Postcondition verification --
    def test_postcondition_tuple(self):
        result = ag_signal.freqz([1.0, 0.5])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_postcondition_length(self):
        w, h = ag_signal.freqz([1.0, 0.5], worN=256)
        assert w.shape[0] == 256

    # -- Category 4: Edge cases --
    def test_fir_only(self):
        w, h = ag_signal.freqz([1.0, -1.0])
        assert w.shape[0] == 512

    # -- Category 5: Upstream parity --
    def test_matches_upstream(self):
        b, a = scipy.signal.butter(4, 0.2)
        fs = 2 * np.pi
        w_atom, h_atom = ag_signal.freqz(b, a, worN=128, fs=fs)
        w_raw, h_raw = scipy.signal.freqz(b, a, worN=128, fs=fs)
        np.testing.assert_array_almost_equal(w_atom, w_raw)
        np.testing.assert_array_almost_equal(h_atom, h_raw)
