from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_engzee_signal_segmentation(signal: AbstractSignal,
    sampling_rate: AbstractScalar,
    threshold: AbstractScalar,
) -> AbstractArray:
    """Ghost witness for engzee_signal_segmentation.

Args:
    signal: Abstract 1-D electrocardiogram (ECG) signal.
    sampling_rate: Abstract positive scalar.
    threshold: Abstract scalar in (0, 1).

Returns:
    Abstract 1-D array of detected R-peak indices."""
    return AbstractArray(
        shape=(signal.shape[0],),
        dtype="int64",
        min_val=0,
        max_val=signal.shape[0] - 1,
    )
