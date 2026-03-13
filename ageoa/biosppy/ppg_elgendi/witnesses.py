from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_detect_signal_onsets_elgendi2013(signal: AbstractSignal,
    sampling_rate: AbstractScalar,
    peakwindow: AbstractScalar,
    beatwindow: AbstractScalar,
    beatoffset: AbstractScalar,
    mindelay: AbstractScalar,
) -> AbstractSignal:
    """Ghost witness for detect_signal_onsets_elgendi2013."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result
