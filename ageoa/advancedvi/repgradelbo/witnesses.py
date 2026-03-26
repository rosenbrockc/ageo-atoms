from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_gradient_oracle_evaluation(rng_in: AbstractArray, obj: AbstractArray, adtype: AbstractArray, out_in: AbstractArray, state_in: AbstractArray, params: AbstractArray, restructure: AbstractArray) -> tuple[AbstractArray, AbstractArray, AbstractArray, AbstractArray]:
    """Shape-and-type check for gradient oracle evaluation. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=rng_in.shape,
        dtype="float64",
    )
    return result
