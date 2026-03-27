from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_detect_signal_onsets_elgendi2013(signal: AbstractSignal,
    sampling_rate: AbstractScalar,
    peakwindow: AbstractScalar,
    beatwindow: AbstractScalar,
    beatoffset: AbstractScalar,
    mindelay: AbstractScalar,
) -> AbstractSignal:
    """Shape-and-type check for detect signal onsets elgendi2013. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_detectonsetevents(signal: AbstractSignal,
    sampling_rate: AbstractScalar,
    alpha: AbstractScalar,
    k: AbstractScalar,
    init_bpm: AbstractScalar,
    min_delay: AbstractScalar,
    max_BPM: AbstractScalar,
) -> AbstractSignal:
    """Ghost witness for detectonsetevents.

    Args:
        signal: Input signal metadata.
        sampling_rate: Sampling rate metadata.
        alpha: Algorithm coefficient metadata.
        k: Window/order parameter metadata.
        init_bpm: Initial tempo estimate metadata.
        min_delay: Minimum inter-onset delay metadata.
        max_BPM: Upper tempo bound metadata.

    Returns:
        Onset detection result signal metadata.
    """
    return AbstractSignal(
        shape=signal.shape,
        dtype="int64",
        sampling_rate=signal.sampling_rate,
        domain=signal.domain,
        units=signal.units,
    )
