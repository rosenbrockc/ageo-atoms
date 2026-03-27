from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_update_state_estimate(prior_state: AbstractArray, *args, **kwargs) -> AbstractArray:
    result = AbstractArray(
        shape=prior_state.shape,
        dtype="float64",)

    return result
