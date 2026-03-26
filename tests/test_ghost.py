"""Tests for the Ghost Witness system."""

import pytest

from ageoa.ghost.abstract import AbstractSignal, AbstractBeatPool
from ageoa.ghost.registry import REGISTRY, register_atom, get_witness, list_registered
from ageoa.ghost.simulator import simulate_graph, SimNode, SimResult, PlanError
from ageoa.ghost.witnesses import (
    witness_fft,
    witness_ifft,
    witness_rfft,
    witness_irfft,
    witness_dct,
    witness_idct,
    witness_butter,
    witness_cheby1,
    witness_cheby2,
    witness_firwin,
    witness_lfilter,
    witness_sosfilt,
    witness_peak_detect,
    witness_freqz,
    witness_sqi_update,
    witness_graph_laplacian,
    witness_graph_fourier_transform,
    witness_heat_kernel_diffusion,
    AbstractFilterCoefficients,
    AbstractGraphMeta,
)


# ---------------------------------------------------------------------------
# Phase 1: AbstractSignal
# ---------------------------------------------------------------------------

class TestAbstractSignal:
    def test_create(self):
        sig = AbstractSignal(shape=(1024,), dtype="float64", sampling_rate=44100.0)
        assert sig.shape == (1024,)
        assert sig.domain == "time"

    def test_duration(self):
        sig = AbstractSignal(shape=(44100,), dtype="float64", sampling_rate=44100.0)
        assert sig.duration == pytest.approx(1.0)

    def test_duration_non_time(self):
        sig = AbstractSignal(shape=(1024,), dtype="complex128", sampling_rate=44100.0, domain="freq")
        assert sig.duration == 0.0

    def test_nyquist(self):
        sig = AbstractSignal(shape=(1024,), dtype="float64", sampling_rate=1000.0)
        assert sig.nyquist == 500.0

    def test_assert_compatible_ok(self):
        a = AbstractSignal(shape=(100,), dtype="float64", sampling_rate=1000.0)
        b = AbstractSignal(shape=(100,), dtype="float64", sampling_rate=1000.0)
        a.assert_compatible(b)  # should not raise

    def test_assert_compatible_fs_mismatch(self):
        a = AbstractSignal(shape=(100,), dtype="float64", sampling_rate=1000.0)
        b = AbstractSignal(shape=(100,), dtype="float64", sampling_rate=2000.0)
        with pytest.raises(ValueError, match="Sampling rate mismatch"):
            a.assert_compatible(b)

    def test_assert_compatible_shape_mismatch(self):
        a = AbstractSignal(shape=(100,), dtype="float64", sampling_rate=1000.0)
        b = AbstractSignal(shape=(200,), dtype="float64", sampling_rate=1000.0)
        with pytest.raises(ValueError, match="Shape mismatch"):
            a.assert_compatible(b)

    def test_assert_domain(self):
        sig = AbstractSignal(shape=(100,), dtype="float64", sampling_rate=1000.0, domain="time")
        sig.assert_domain("time")  # should not raise

    def test_assert_domain_mismatch(self):
        sig = AbstractSignal(shape=(100,), dtype="float64", sampling_rate=1000.0, domain="freq")
        with pytest.raises(ValueError, match="Domain mismatch"):
            sig.assert_domain("time")


class TestAbstractBeatPool:
    def test_create_empty(self):
        pool = AbstractBeatPool()
        assert pool.size == 0
        assert pool.is_calibrated is False

    def test_accumulate_below_threshold(self):
        pool = AbstractBeatPool(calibration_threshold=50)
        pool2 = pool.accumulate(20)
        assert pool2.size == 20
        assert pool2.is_calibrated is False

    def test_accumulate_above_threshold(self):
        pool = AbstractBeatPool(calibration_threshold=50)
        pool2 = pool.accumulate(60)
        assert pool2.size == 60
        assert pool2.is_calibrated is True

    def test_accumulate_chained(self):
        pool = AbstractBeatPool(calibration_threshold=30)
        pool = pool.accumulate(10)
        pool = pool.accumulate(10)
        pool = pool.accumulate(10)
        assert pool.size == 30
        assert pool.is_calibrated is True


# ---------------------------------------------------------------------------
# Phase 2: Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_register_and_lookup(self):
        def my_witness(x: int) -> int:
            return x

        @register_atom(witness=my_witness)
        def my_heavy(x):
            return x * 2

        assert "my_heavy" in REGISTRY
        assert REGISTRY["my_heavy"]["witness"] is my_witness
        assert REGISTRY["my_heavy"]["impl"] is my_heavy

        # Cleanup
        del REGISTRY["my_heavy"]

    def test_get_witness(self):
        def w(x: int) -> int:
            return x

        @register_atom(witness=w)
        def registered_func(x):
            return x

        assert get_witness("registered_func") is w

        # Cleanup
        del REGISTRY["registered_func"]

    def test_get_witness_missing(self):
        with pytest.raises(KeyError, match="No ghost witness"):
            get_witness("nonexistent_function_xyz")

    def test_list_registered(self):
        def w(x: int) -> int:
            return x

        @register_atom(witness=w)
        def list_test_func(x):
            return x

        assert "list_test_func" in list_registered()

        # Cleanup
        del REGISTRY["list_test_func"]


# ---------------------------------------------------------------------------
# Phase 3: Concrete witnesses — FFT family
# ---------------------------------------------------------------------------

class TestWitnessFft:
    def _time_sig(self, n=1024):
        return AbstractSignal(shape=(n,), dtype="float64", sampling_rate=44100.0, domain="time")

    def _freq_sig(self, n=1024):
        return AbstractSignal(shape=(n,), dtype="complex128", sampling_rate=44100.0, domain="freq")

    def test_fft_positive(self):
        out = witness_fft(self._time_sig())
        assert out.shape == (1024,)
        assert out.dtype == "complex128"
        assert out.domain == "freq"

    def test_fft_wrong_domain(self):
        with pytest.raises(ValueError, match="Domain mismatch"):
            witness_fft(self._freq_sig())

    def test_ifft_positive(self):
        out = witness_ifft(self._freq_sig())
        assert out.shape == (1024,)
        assert out.domain == "time"

    def test_ifft_wrong_domain(self):
        with pytest.raises(ValueError, match="Domain mismatch"):
            witness_ifft(self._time_sig())

    def test_fft_ifft_round_trip_shape(self):
        sig = self._time_sig(512)
        freq = witness_fft(sig)
        time_back = witness_ifft(freq)
        assert time_back.shape == sig.shape
        assert time_back.domain == "time"

    def test_rfft_shape(self):
        sig = self._time_sig(1024)
        out = witness_rfft(sig)
        assert out.shape == (513,)  # 1024//2 + 1
        assert out.dtype == "complex128"
        assert out.domain == "freq"

    def test_irfft_shape(self):
        freq = AbstractSignal(shape=(513,), dtype="complex128", sampling_rate=44100.0, domain="freq")
        out = witness_irfft(freq)
        assert out.shape == (1024,)  # 2*(513-1)
        assert out.dtype == "float64"
        assert out.domain == "time"


class TestWitnessDct:
    def _time_sig(self, n=256):
        return AbstractSignal(shape=(n,), dtype="float64", sampling_rate=8000.0, domain="time")

    def test_dct_positive(self):
        out = witness_dct(self._time_sig())
        assert out.shape == (256,)
        assert out.dtype == "float64"
        assert out.domain == "freq"

    def test_dct_complex_input_rejected(self):
        sig = AbstractSignal(shape=(256,), dtype="complex128", sampling_rate=8000.0, domain="time")
        with pytest.raises(ValueError, match="real-valued"):
            witness_dct(sig)

    def test_idct_positive(self):
        freq = AbstractSignal(shape=(256,), dtype="float64", sampling_rate=8000.0, domain="freq")
        out = witness_idct(freq)
        assert out.domain == "time"

    def test_dct_idct_round_trip_shape(self):
        sig = self._time_sig(128)
        freq = witness_dct(sig)
        back = witness_idct(freq)
        assert back.shape == sig.shape


# ---------------------------------------------------------------------------
# Phase 3: Concrete witnesses — Filters
# ---------------------------------------------------------------------------

class TestWitnessFilters:
    def test_butter_positive(self):
        coeff = witness_butter(order=4, wn=100.0, fs=1000.0)
        assert coeff.order == 4
        assert coeff.is_stable is True

    def test_butter_bad_order(self):
        with pytest.raises(ValueError, match="positive"):
            witness_butter(order=0, wn=100.0, fs=1000.0)

    def test_butter_freq_above_nyquist(self):
        with pytest.raises(ValueError, match="must be in"):
            witness_butter(order=4, wn=600.0, fs=1000.0)

    def test_cheby1_positive(self):
        coeff = witness_cheby1(order=4, rp=1.0, wn=100.0, fs=1000.0)
        assert coeff.is_stable is True

    def test_cheby1_bad_ripple(self):
        with pytest.raises(ValueError, match="ripple"):
            witness_cheby1(order=4, rp=-1.0, wn=100.0, fs=1000.0)

    def test_cheby2_positive(self):
        coeff = witness_cheby2(order=4, rs=40.0, wn=100.0, fs=1000.0)
        assert coeff.is_stable is True

    def test_firwin_positive(self):
        coeff = witness_firwin(numtaps=51, fs=1000.0)
        assert coeff.format == "fir"
        assert coeff.is_stable is True

    def test_firwin_bad_numtaps(self):
        with pytest.raises(ValueError, match="positive"):
            witness_firwin(numtaps=0, fs=1000.0)

    def test_lfilter_positive(self):
        coeff = AbstractFilterCoefficients(order=4, btype="low", is_stable=True)
        sig = AbstractSignal(shape=(1000,), dtype="float64", sampling_rate=1000.0, domain="time")
        out = witness_lfilter(coeff, sig)
        assert out.shape == sig.shape
        assert out.domain == "time"

    def test_lfilter_unstable_rejected(self):
        coeff = AbstractFilterCoefficients(order=4, btype="low", is_stable=False)
        sig = AbstractSignal(shape=(1000,), dtype="float64", sampling_rate=1000.0, domain="time")
        with pytest.raises(ValueError, match="unstable"):
            witness_lfilter(coeff, sig)

    def test_lfilter_wrong_domain(self):
        coeff = AbstractFilterCoefficients(order=4, btype="low", is_stable=True)
        sig = AbstractSignal(shape=(1000,), dtype="complex128", sampling_rate=1000.0, domain="freq")
        with pytest.raises(ValueError, match="Domain mismatch"):
            witness_lfilter(coeff, sig)

    def test_sosfilt_positive(self):
        coeff = AbstractFilterCoefficients(order=4, btype="low", is_stable=True)
        sig = AbstractSignal(shape=(500,), dtype="float64", sampling_rate=1000.0, domain="time")
        out = witness_sosfilt(coeff, sig)
        assert out.shape == sig.shape

    def test_peak_detect_positive(self):
        sig = AbstractSignal(shape=(1000,), dtype="float64", sampling_rate=1000.0, domain="time")
        out = witness_peak_detect(sig)
        assert out.dtype == "int64"
        assert out.domain == "index"

    def test_peak_detect_wrong_domain(self):
        sig = AbstractSignal(shape=(1000,), dtype="complex128", sampling_rate=1000.0, domain="freq")
        with pytest.raises(ValueError, match="Domain mismatch"):
            witness_peak_detect(sig)

    def test_freqz_positive(self):
        coeff = AbstractFilterCoefficients(order=4, btype="low", is_stable=True)
        out = witness_freqz(coeff, n_freqs=256)
        assert out.shape == (256,)
        assert out.domain == "freq"


# ---------------------------------------------------------------------------
# Phase 3: Concrete witnesses — GSP
# ---------------------------------------------------------------------------

class TestWitnessGsp:
    def _graph(self, n=10):
        return AbstractGraphMeta(n_nodes=n, is_symmetric=True)

    def _sig(self, n=10):
        return AbstractSignal(shape=(n,), dtype="float64", sampling_rate=1.0, domain="time")

    def test_graph_laplacian_positive(self):
        out = witness_graph_laplacian(self._graph())
        assert out.n_nodes == 10
        assert out.is_symmetric is True

    def test_graph_laplacian_asymmetric(self):
        g = AbstractGraphMeta(n_nodes=10, is_symmetric=False)
        with pytest.raises(ValueError, match="symmetric"):
            witness_graph_laplacian(g)

    def test_gft_positive(self):
        out = witness_graph_fourier_transform(self._graph(), self._sig())
        assert out.shape == (10,)
        assert out.domain == "freq"

    def test_gft_size_mismatch(self):
        g = self._graph(10)
        sig = self._sig(5)
        with pytest.raises(ValueError, match="must equal graph size"):
            witness_graph_fourier_transform(g, sig)

    def test_heat_diffusion_positive(self):
        out = witness_heat_kernel_diffusion(self._graph(), self._sig(), t=1.0)
        assert out.shape == (10,)

    def test_heat_diffusion_negative_t(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            witness_heat_kernel_diffusion(self._graph(), self._sig(), t=-1.0)

    def test_heat_diffusion_size_mismatch(self):
        g = self._graph(10)
        sig = self._sig(5)
        with pytest.raises(ValueError, match="must equal graph size"):
            witness_heat_kernel_diffusion(g, sig, t=1.0)


# ---------------------------------------------------------------------------
# Phase 3: SQI accumulator witness
# ---------------------------------------------------------------------------

class TestWitnessSqi:
    def test_sqi_update_positive(self):
        pool = AbstractBeatPool(calibration_threshold=50)
        sig = AbstractSignal(shape=(1000,), dtype="float64", sampling_rate=1000.0, domain="time")
        pool2 = witness_sqi_update(pool, sig)
        assert pool2.size == 10
        assert pool2.is_calibrated is False

    def test_sqi_update_reaches_calibration(self):
        pool = AbstractBeatPool(size=45, calibration_threshold=50)
        sig = AbstractSignal(shape=(1000,), dtype="float64", sampling_rate=1000.0, domain="time")
        pool2 = witness_sqi_update(pool, sig)
        assert pool2.size == 55
        assert pool2.is_calibrated is True

    def test_sqi_update_wrong_domain(self):
        pool = AbstractBeatPool()
        sig = AbstractSignal(shape=(1000,), dtype="complex128", sampling_rate=1000.0, domain="freq")
        with pytest.raises(ValueError, match="Domain mismatch"):
            witness_sqi_update(pool, sig)


# ---------------------------------------------------------------------------
# Phase 4: Simulator
# ---------------------------------------------------------------------------

class TestSimulator:
    """Tests for simulate_graph using witness_overrides (no global registry pollution)."""

    def test_simple_fft_pipeline(self):
        """Window -> FFT -> IFFT produces a valid time-domain output."""
        nodes = [
            SimNode(
                name="FFT",
                function_name="fft",
                inputs={"sig": "input_signal"},
                output_name="spectrum",
            ),
            SimNode(
                name="IFFT",
                function_name="ifft",
                inputs={"sig": "spectrum"},
                output_name="reconstructed",
            ),
        ]
        initial = {
            "input_signal": AbstractSignal(
                shape=(1024,), dtype="float64", sampling_rate=44100.0, domain="time"
            ),
        }
        result = simulate_graph(
            nodes, initial,
            witness_overrides={"fft": witness_fft, "ifft": witness_ifft},
        )
        assert result.node_count == 2
        assert result.trace == ["FFT", "IFFT"]
        assert result.final_state["reconstructed"].domain == "time"
        assert result.final_state["reconstructed"].shape == (1024,)

    def test_domain_error_caught(self):
        """Applying FFT to a frequency-domain signal raises PlanError."""
        nodes = [
            SimNode(
                name="Bad FFT",
                function_name="fft",
                inputs={"sig": "freq_signal"},
                output_name="output",
            ),
        ]
        initial = {
            "freq_signal": AbstractSignal(
                shape=(512,), dtype="complex128", sampling_rate=44100.0, domain="freq"
            ),
        }
        with pytest.raises(PlanError, match="Domain mismatch") as exc_info:
            simulate_graph(
                nodes, initial,
                witness_overrides={"fft": witness_fft},
            )
        assert exc_info.value.node_name == "Bad FFT"
        assert exc_info.value.function_name == "fft"

    def test_missing_input_key(self):
        """Referencing a state key that doesn't exist raises PlanError."""
        nodes = [
            SimNode(
                name="FFT",
                function_name="fft",
                inputs={"sig": "nonexistent_key"},
                output_name="output",
            ),
        ]
        with pytest.raises(PlanError, match="nonexistent_key"):
            simulate_graph(
                nodes, {},
                witness_overrides={"fft": witness_fft},
            )

    def test_missing_witness(self):
        """Referencing an unregistered function raises PlanError."""
        nodes = [
            SimNode(
                name="Unknown",
                function_name="totally_unknown_func",
                inputs={},
                output_name="output",
            ),
        ]
        with pytest.raises(PlanError, match="No ghost witness"):
            simulate_graph(nodes, {})

    def test_filter_pipeline(self):
        """Design -> Apply filter pipeline passes simulation."""
        nodes = [
            SimNode(
                name="Design Butterworth",
                function_name="butter",
                inputs={},
                kwargs={"order": 4, "wn": 100.0, "fs": 1000.0},
                output_name="coefficients",
            ),
            SimNode(
                name="Apply Filter",
                function_name="lfilter",
                inputs={"coefficients": "coefficients", "sig": "raw_signal"},
                output_name="filtered",
            ),
        ]
        initial = {
            "raw_signal": AbstractSignal(
                shape=(5000,), dtype="float64", sampling_rate=1000.0, domain="time"
            ),
        }
        result = simulate_graph(
            nodes, initial,
            witness_overrides={"butter": witness_butter, "lfilter": witness_lfilter},
        )
        assert result.final_state["filtered"].shape == (5000,)
        assert result.final_state["filtered"].domain == "time"

    def test_filter_wrong_domain_caught(self):
        """Applying a filter to frequency-domain data raises PlanError."""
        nodes = [
            SimNode(
                name="Design",
                function_name="butter",
                inputs={},
                kwargs={"order": 4, "wn": 100.0, "fs": 1000.0},
                output_name="coefficients",
            ),
            SimNode(
                name="Bad Apply",
                function_name="lfilter",
                inputs={"coefficients": "coefficients", "sig": "freq_data"},
                output_name="output",
            ),
        ]
        initial = {
            "freq_data": AbstractSignal(
                shape=(512,), dtype="complex128", sampling_rate=1000.0, domain="freq"
            ),
        }
        with pytest.raises(PlanError, match="Domain mismatch"):
            simulate_graph(
                nodes, initial,
                witness_overrides={"butter": witness_butter, "lfilter": witness_lfilter},
            )

    def test_sqi_accumulation_pipeline(self):
        """SQI accumulator eventually reaches calibration after enough windows."""
        overrides = {"sqi_update": witness_sqi_update}
        pool = AbstractBeatPool(calibration_threshold=30)
        sig = AbstractSignal(shape=(1000,), dtype="float64", sampling_rate=1000.0, domain="time")

        # Simulate 4 windows of SQI update
        nodes = []
        for i in range(4):
            in_pool = "pool" if i == 0 else f"pool_{i}"
            out_pool = f"pool_{i + 1}"
            nodes.append(SimNode(
                name=f"SQI Update {i}",
                function_name="sqi_update",
                inputs={"pool": in_pool, "new_beats": "signal"},
                output_name=out_pool,
            ))

        initial = {"pool": pool, "signal": sig}
        result = simulate_graph(nodes, initial, witness_overrides=overrides)

        final_pool = result.final_state["pool_4"]
        assert final_pool.size == 40  # 4 * 10
        assert final_pool.is_calibrated is True

    def test_gsp_pipeline(self):
        """Graph Laplacian -> GFT -> Heat Diffusion passes simulation."""
        overrides = {
            "graph_laplacian": witness_graph_laplacian,
            "gft": witness_graph_fourier_transform,
            "heat_diffusion": witness_heat_kernel_diffusion,
        }
        nodes = [
            SimNode(
                name="Compute Laplacian",
                function_name="graph_laplacian",
                inputs={"graph": "adjacency"},
                output_name="laplacian",
            ),
            SimNode(
                name="GFT",
                function_name="gft",
                inputs={"graph": "laplacian", "sig": "graph_signal"},
                output_name="spectrum",
            ),
            SimNode(
                name="Heat Diffusion",
                function_name="heat_diffusion",
                inputs={"graph": "laplacian", "sig": "graph_signal"},
                kwargs={"t": 1.0},
                output_name="smoothed",
            ),
        ]
        initial = {
            "adjacency": AbstractGraphMeta(n_nodes=50, is_symmetric=True),
            "graph_signal": AbstractSignal(
                shape=(50,), dtype="float64", sampling_rate=1.0, domain="time"
            ),
        }
        result = simulate_graph(nodes, initial, witness_overrides=overrides)
        assert result.final_state["smoothed"].shape == (50,)
        assert result.trace == ["Compute Laplacian", "GFT", "Heat Diffusion"]

    def test_kwargs_forwarded(self):
        """Extra kwargs on SimNode are forwarded to the witness."""
        nodes = [
            SimNode(
                name="Heat",
                function_name="heat_diffusion",
                inputs={"graph": "L", "sig": "x"},
                kwargs={"t": 5.0},
                output_name="smoothed",
            ),
        ]
        initial = {
            "L": AbstractGraphMeta(n_nodes=10, is_symmetric=True),
            "x": AbstractSignal(shape=(10,), dtype="float64", sampling_rate=1.0, domain="time"),
        }
        result = simulate_graph(
            nodes, initial,
            witness_overrides={"heat_diffusion": witness_heat_kernel_diffusion},
        )
        assert result.final_state["smoothed"].shape == (10,)
