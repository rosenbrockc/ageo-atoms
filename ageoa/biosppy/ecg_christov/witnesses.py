"""Auto-generated ghost witness functions for christov_segmenter."""

from __future__ import annotations

try:
    from ageoa.ghost.abstract import AbstractArray, AbstractScalar
except ImportError:
    pass


def witness_christovqrsdetect(
    signal: AbstractArray, sampling_rate: AbstractScalar
) -> AbstractArray:
    """Ghost witness for christovqrsdetect.

    Propagates shape: a 1-D signal of length N produces a 1-D array of
    R-peak indices (length <= N).

    Args:
        signal: Abstract 1-D ECG signal.
        sampling_rate: Abstract positive scalar.

    Returns:
        Abstract 1-D array of detected R-peak indices.
    """
    return AbstractArray(
        shape=(signal.shape[0],),
        dtype="int64",
        min_val=0,
        max_val=signal.shape[0] - 1,
    )
