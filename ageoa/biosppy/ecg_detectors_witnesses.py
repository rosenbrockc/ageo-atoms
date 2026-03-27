from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_thresholdbasedsignalsegmentation(signal: AbstractSignal,
    sampling_rate: AbstractScalar,
    Pth: AbstractScalar,
) -> AbstractSignal:
    """Shape-and-type check for threshold based signal segmentation. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_asi_signal_segmenter(signal: AbstractSignal, sampling_rate: AbstractSignal, Pth: AbstractSignal) -> AbstractSignal:
    """Shape-and-type check for asi signal segmenter. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_christovqrsdetect(signal: AbstractSignal,
    sampling_rate: AbstractScalar,
) -> AbstractArray:
    """Ghost witness for christovqrsdetect.

Propagates shape: a 1-D signal of length N produces a 1-D array of
R-peak indices (length <= N).

Args:
    signal: Abstract 1-D electrocardiogram (ECG) signal.
    sampling_rate: Abstract positive scalar.

Returns:
    Abstract 1-D array of detected R-peak indices."""
    return AbstractArray(
        shape=(signal.shape[0],),
        dtype="int64",
        min_val=0,
        max_val=signal.shape[0] - 1,
    )

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_christov_qrs_segmenter(signal: AbstractSignal, sampling_rate: AbstractSignal) -> AbstractSignal:
    """Shape-and-type check for christov qrs segmenter. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result

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

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_engzee_qrs_segmentation(signal: AbstractSignal, sampling_rate: AbstractSignal, threshold: AbstractSignal) -> AbstractSignal:
    """Shape-and-type check for engzee qrs segmentation. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_gamboa_segmentation(signal: AbstractSignal,
    sampling_rate: AbstractScalar,
    tol: AbstractScalar,
) -> AbstractSignal:
    """Shape-and-type check for gamboa segmentation. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_gamboa_segmenter(signal: AbstractSignal, sampling_rate: AbstractSignal, tol: AbstractSignal) -> AbstractSignal:
    """Shape-and-type check for gamboa segmenter. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_hamilton_segmentation(signal: AbstractSignal,
    sampling_rate: AbstractScalar,
) -> AbstractSignal:
    """Shape-and-type check for hamilton segmentation. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_hamilton_segmenter(signal: AbstractSignal, sampling_rate: AbstractSignal) -> AbstractSignal:
    """Shape-and-type check for hamilton segmenter. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result
