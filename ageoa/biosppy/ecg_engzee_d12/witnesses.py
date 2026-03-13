from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal
import networkx as nx  # type: ignore


def witness_engzee_qrs_segmentation(signal: AbstractSignal, sampling_rate: AbstractSignal, threshold: AbstractSignal) -> AbstractSignal:
    """Ghost witness for engzee_qrs_segmentation."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result
