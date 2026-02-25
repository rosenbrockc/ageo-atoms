"""Auto-generated ghost witness functions for engzee_segmenter."""

from __future__ import annotations

try:
    from ageoa.ghost.abstract import AbstractArray, AbstractScalar
except ImportError:
    pass


def witness_engzee_signal_segmentation(
    signal: AbstractArray, sampling_rate: AbstractScalar, threshold: AbstractScalar
) -> AbstractArray:
    """Ghost witness for engzee_signal_segmentation.

    Args:
        signal: Abstract 1-D ECG signal.
        sampling_rate: Abstract positive scalar.
        threshold: Abstract scalar in (0, 1).

    Returns:
        Abstract 1-D array of detected R-peak indices.
    """
    return AbstractArray(
        shape=(signal.shape[0],),
        dtype="int64",
        min_val=0,
        max_val=signal.shape[0] - 1,
    )
