"""Concrete Ghost Witnesses for DSP atoms.

Each witness function mirrors the *contract* of its heavy counterpart but
operates only on ``AbstractSignal`` metadata.  The witness:

1. Checks domain / shape / dtype preconditions (raises ``ValueError`` on
   violation — the simulator catches this as a ``PlanError``).
2. Propagates metadata to describe the output signal.

Witnesses are registered against the heavy atoms via ``@register_atom``
in this module's ``register_all()`` function, which must be called once
at import time (handled by the ``ghost`` package ``__init__``).
"""

from __future__ import annotations

from typing import Tuple

from ageoa.ghost.abstract import AbstractSignal, AbstractBeatPool
from ageoa.ghost.registry import register_atom


# ---------------------------------------------------------------------------
# FFT family
# ---------------------------------------------------------------------------

def witness_fft(sig: AbstractSignal) -> AbstractSignal:
    """Ghost witness for numpy.fft.fft.

    Preconditions:
        - Input must be in time domain.
    Postconditions:
        - Output shape matches input shape.
        - Output dtype is complex128.
        - Output domain is 'freq'.
    """
    sig.assert_domain("time")
    return AbstractSignal(
        shape=sig.shape,
        dtype="complex128",
        sampling_rate=sig.sampling_rate,
        domain="freq",
        units=sig.units,
    )


def witness_ifft(sig: AbstractSignal) -> AbstractSignal:
    """Ghost witness for numpy.fft.ifft.

    Preconditions:
        - Input must be in frequency domain.
    Postconditions:
        - Output shape matches input shape.
        - Output dtype is complex128.
        - Output domain is 'time'.
    """
    sig.assert_domain("freq")
    return AbstractSignal(
        shape=sig.shape,
        dtype="complex128",
        sampling_rate=sig.sampling_rate,
        domain="time",
        units=sig.units,
    )


def witness_rfft(sig: AbstractSignal) -> AbstractSignal:
    """Ghost witness for numpy.fft.rfft.

    Preconditions:
        - Input must be in time domain.
        - Input dtype must be real.
    Postconditions:
        - Output length is n//2 + 1.
        - Output dtype is complex128.
        - Output domain is 'freq'.
    """
    sig.assert_domain("time")
    if "complex" in sig.dtype:
        raise ValueError("rfft requires real-valued input, got dtype={sig.dtype}")
    n = sig.shape[0] if len(sig.shape) > 0 else 0
    out_len = n // 2 + 1
    return AbstractSignal(
        shape=(out_len,) + sig.shape[1:],
        dtype="complex128",
        sampling_rate=sig.sampling_rate,
        domain="freq",
        units=sig.units,
    )


def witness_irfft(sig: AbstractSignal) -> AbstractSignal:
    """Ghost witness for numpy.fft.irfft.

    Preconditions:
        - Input must be in frequency domain.
    Postconditions:
        - Output length is 2 * (input_length - 1).
        - Output dtype is float64.
        - Output domain is 'time'.
    """
    sig.assert_domain("freq")
    n = sig.shape[0] if len(sig.shape) > 0 else 0
    out_len = 2 * (n - 1)
    return AbstractSignal(
        shape=(out_len,) + sig.shape[1:],
        dtype="float64",
        sampling_rate=sig.sampling_rate,
        domain="time",
        units=sig.units,
    )


# ---------------------------------------------------------------------------
# DCT family
# ---------------------------------------------------------------------------

def witness_dct(sig: AbstractSignal) -> AbstractSignal:
    """Ghost witness for scipy.fft.dct.

    Preconditions:
        - Input must be in time domain.
        - Input dtype must be real.
    Postconditions:
        - Output shape matches input shape.
        - Output dtype is float64 (DCT is real-to-real).
        - Output domain is 'freq'.
    """
    sig.assert_domain("time")
    if "complex" in sig.dtype:
        raise ValueError(f"DCT requires real-valued input, got dtype={sig.dtype}")
    return AbstractSignal(
        shape=sig.shape,
        dtype="float64",
        sampling_rate=sig.sampling_rate,
        domain="freq",
        units=sig.units,
    )


def witness_idct(sig: AbstractSignal) -> AbstractSignal:
    """Ghost witness for scipy.fft.idct.

    Preconditions:
        - Input must be in frequency domain.
    Postconditions:
        - Output shape matches input shape.
        - Output dtype is float64.
        - Output domain is 'time'.
    """
    sig.assert_domain("freq")
    return AbstractSignal(
        shape=sig.shape,
        dtype="float64",
        sampling_rate=sig.sampling_rate,
        domain="time",
        units=sig.units,
    )


# ---------------------------------------------------------------------------
# Filter design witnesses
# ---------------------------------------------------------------------------

class AbstractFilterCoefficients:
    """Lightweight metadata for filter design output (b, a) or (sos)."""

    def __init__(
        self,
        order: int,
        btype: str,
        format: str = "ba",
        is_stable: bool = True,
    ) -> None:
        self.order = order
        self.btype = btype
        self.format = format
        self.is_stable = is_stable

    def assert_stable(self) -> None:
        if not self.is_stable:
            raise ValueError("Filter is unstable (poles outside unit circle)")


def witness_butter(
    order: int,
    wn: float,
    fs: float,
    btype: str = "low",
) -> AbstractFilterCoefficients:
    """Ghost witness for scipy.signal.butter.

    Preconditions:
        - Order must be positive.
        - Critical frequency must be below Nyquist.
    Postconditions:
        - Returns filter coefficients metadata.
        - Butterworth filters are always stable.
    """
    if order <= 0:
        raise ValueError(f"Filter order must be positive, got {order}")
    nyquist = fs / 2.0
    if wn <= 0 or wn >= nyquist:
        raise ValueError(
            f"Critical frequency {wn} must be in (0, {nyquist}) for fs={fs}"
        )
    return AbstractFilterCoefficients(
        order=order, btype=btype, format="ba", is_stable=True,
    )


def witness_cheby1(
    order: int,
    rp: float,
    wn: float,
    fs: float,
    btype: str = "low",
) -> AbstractFilterCoefficients:
    """Ghost witness for scipy.signal.cheby1.

    Preconditions:
        - Order must be positive.
        - Ripple must be positive.
        - Critical frequency must be below Nyquist.
    """
    if order <= 0:
        raise ValueError(f"Filter order must be positive, got {order}")
    if rp <= 0:
        raise ValueError(f"Passband ripple must be positive, got {rp}")
    nyquist = fs / 2.0
    if wn <= 0 or wn >= nyquist:
        raise ValueError(
            f"Critical frequency {wn} must be in (0, {nyquist}) for fs={fs}"
        )
    return AbstractFilterCoefficients(
        order=order, btype=btype, format="ba", is_stable=True,
    )


def witness_cheby2(
    order: int,
    rs: float,
    wn: float,
    fs: float,
    btype: str = "low",
) -> AbstractFilterCoefficients:
    """Ghost witness for scipy.signal.cheby2.

    Preconditions:
        - Order must be positive.
        - Stopband attenuation must be positive.
        - Critical frequency must be below Nyquist.
    """
    if order <= 0:
        raise ValueError(f"Filter order must be positive, got {order}")
    if rs <= 0:
        raise ValueError(f"Stopband attenuation must be positive, got {rs}")
    nyquist = fs / 2.0
    if wn <= 0 or wn >= nyquist:
        raise ValueError(
            f"Critical frequency {wn} must be in (0, {nyquist}) for fs={fs}"
        )
    return AbstractFilterCoefficients(
        order=order, btype=btype, format="ba", is_stable=True,
    )


def witness_firwin(numtaps: int, fs: float) -> AbstractFilterCoefficients:
    """Ghost witness for scipy.signal.firwin.

    Preconditions:
        - numtaps must be positive.
    Postconditions:
        - FIR filters are trivially stable (no feedback).
    """
    if numtaps <= 0:
        raise ValueError(f"numtaps must be positive, got {numtaps}")
    return AbstractFilterCoefficients(
        order=numtaps - 1, btype="low", format="fir", is_stable=True,
    )


# ---------------------------------------------------------------------------
# Filter application witnesses
# ---------------------------------------------------------------------------

def witness_lfilter(
    coefficients: AbstractFilterCoefficients,
    sig: AbstractSignal,
) -> AbstractSignal:
    """Ghost witness for scipy.signal.lfilter.

    Preconditions:
        - Filter must be stable.
        - Signal must be in time domain.
    Postconditions:
        - Output shape matches input shape.
        - Output dtype matches input dtype.
        - Output domain is 'time'.
    """
    coefficients.assert_stable()
    sig.assert_domain("time")
    return AbstractSignal(
        shape=sig.shape,
        dtype=sig.dtype,
        sampling_rate=sig.sampling_rate,
        domain="time",
        units=sig.units,
    )


def witness_sosfilt(
    coefficients: AbstractFilterCoefficients,
    sig: AbstractSignal,
) -> AbstractSignal:
    """Ghost witness for scipy.signal.sosfilt.

    Preconditions:
        - Signal must be in time domain.
    Postconditions:
        - Output shape matches input shape.
        - SOS format is stable by construction.
    """
    sig.assert_domain("time")
    return AbstractSignal(
        shape=sig.shape,
        dtype=sig.dtype,
        sampling_rate=sig.sampling_rate,
        domain="time",
        units=sig.units,
    )


# ---------------------------------------------------------------------------
# Analysis witnesses
# ---------------------------------------------------------------------------

def witness_peak_detect(sig: AbstractSignal) -> AbstractSignal:
    """Ghost witness for peak detection.

    Preconditions:
        - Input must be in time domain.
    Postconditions:
        - Output is a list of integer indices.
        - Shape is (0,) — dynamic length, unknown until runtime.
    """
    sig.assert_domain("time")
    return AbstractSignal(
        shape=(0,),
        dtype="int64",
        sampling_rate=sig.sampling_rate,
        domain="index",
        units="index",
    )


def witness_freqz(
    coefficients: AbstractFilterCoefficients,
    n_freqs: int = 512,
) -> AbstractSignal:
    """Ghost witness for scipy.signal.freqz.

    Postconditions:
        - Output is a frequency response of length n_freqs.
        - Domain is 'freq'.
    """
    if n_freqs <= 0:
        raise ValueError(f"n_freqs must be positive, got {n_freqs}")
    return AbstractSignal(
        shape=(n_freqs,),
        dtype="complex128",
        sampling_rate=1.0,  # freqz returns normalized frequency
        domain="freq",
        units="magnitude",
    )


# ---------------------------------------------------------------------------
# Accumulator / state witnesses
# ---------------------------------------------------------------------------

def witness_sqi_update(
    pool: AbstractBeatPool,
    new_beats: AbstractSignal,
) -> AbstractBeatPool:
    """Ghost witness for SQI (Signal Quality Index) accumulation.

    Simulates beat accumulation without processing waveforms.  Uses a
    heuristic estimate of ~10 beats per window.

    Preconditions:
        - new_beats must be in time domain.
    Postconditions:
        - Pool size increases.
        - Calibration flag is set once threshold is reached.
    """
    new_beats.assert_domain("time")
    return pool.accumulate(new_beat_count=10)


# ---------------------------------------------------------------------------
# Graph Signal Processing witnesses
# ---------------------------------------------------------------------------

class AbstractGraphMeta:
    """Lightweight metadata for a graph (Laplacian / adjacency)."""

    def __init__(self, n_nodes: int, is_symmetric: bool = True) -> None:
        self.n_nodes = n_nodes
        self.is_symmetric = is_symmetric

    def assert_square(self) -> None:
        pass  # always square by construction

    def assert_symmetric(self) -> None:
        if not self.is_symmetric:
            raise ValueError("Graph matrix must be symmetric")


def witness_graph_laplacian(graph: AbstractGraphMeta) -> AbstractGraphMeta:
    """Ghost witness for graph_laplacian.

    Preconditions:
        - Input must be symmetric.
    Postconditions:
        - Output is a symmetric PSD matrix of same size.
    """
    graph.assert_symmetric()
    return AbstractGraphMeta(n_nodes=graph.n_nodes, is_symmetric=True)


def witness_graph_fourier_transform(
    graph: AbstractGraphMeta,
    sig: AbstractSignal,
) -> AbstractSignal:
    """Ghost witness for graph_fourier_transform.

    Preconditions:
        - Signal length must equal graph node count.
    Postconditions:
        - Output is GFT coefficients of same length.
        - Domain switches to 'freq'.
    """
    if len(sig.shape) == 0 or sig.shape[0] != graph.n_nodes:
        raise ValueError(
            f"Signal length {sig.shape[0] if sig.shape else 0} "
            f"must equal graph size {graph.n_nodes}"
        )
    return AbstractSignal(
        shape=sig.shape,
        dtype="float64",
        sampling_rate=sig.sampling_rate,
        domain="freq",
        units="coefficient",
    )


def witness_heat_kernel_diffusion(
    graph: AbstractGraphMeta,
    sig: AbstractSignal,
    t: float,
) -> AbstractSignal:
    """Ghost witness for heat_kernel_diffusion.

    Preconditions:
        - t must be >= 0.
        - Signal length must equal graph node count.
    Postconditions:
        - Output shape matches input.
        - Output stays in the same domain.
        - Total variation is reduced (smoothing).
    """
    if t < 0:
        raise ValueError(f"Diffusion time must be >= 0, got {t}")
    if len(sig.shape) == 0 or sig.shape[0] != graph.n_nodes:
        raise ValueError(
            f"Signal length {sig.shape[0] if sig.shape else 0} "
            f"must equal graph size {graph.n_nodes}"
        )
    return AbstractSignal(
        shape=sig.shape,
        dtype=sig.dtype,
        sampling_rate=sig.sampling_rate,
        domain=sig.domain,
        units=sig.units,
    )
