from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_posterior_randmodel(pri: AbstractArray, G: AbstractArray, data: AbstractArray) -> AbstractArray:
    """Ghost witness for Posterior Randmodel."""
    result = AbstractArray(
        shape=pri.shape,
        dtype="float64",
    )
    return result

def witness_posterior_randmodel_weighted(pri: AbstractArray, G: AbstractArray, data: AbstractArray, w: AbstractArray) -> AbstractArray:
    """Ghost witness for Posterior Randmodel."""
    result = AbstractArray(
        shape=pri.shape,
        dtype="float64",
    )
    return result
