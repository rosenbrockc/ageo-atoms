"""Ghost witnesses for PPG Kavsaoglu onset detection atoms."""

from __future__ import annotations

try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractScalar
except ImportError:
    pass


def witness_detectonsetevents(
    signal: AbstractSignal,
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
