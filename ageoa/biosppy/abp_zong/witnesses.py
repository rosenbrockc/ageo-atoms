from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_audio_onset_detection(signal: AbstractSignal,
    sampling_rate: AbstractScalar,
    sm_size: AbstractScalar,
    size: AbstractScalar,
    alpha: AbstractScalar,
    wrange: AbstractScalar,
    d1_th: AbstractScalar,
    d2_th: AbstractScalar,
) -> AbstractSignal:
    """Ghost witness for audio_onset_detection.

    Args:
        signal: Input audio signal metadata.
        sampling_rate: Sampling rate metadata.
        sm_size: Smoothing size metadata.
        size: Analysis window size metadata.
        alpha: Sensitivity parameter metadata.
        wrange: Search range metadata.
        d1_th: First-derivative threshold metadata.
        d2_th: Second-derivative threshold metadata.

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
