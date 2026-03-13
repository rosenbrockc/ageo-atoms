from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_initializefilter(family, event_shape, *args, **kwargs):
    """Ghost witness for prior init: InitializeFilter."""
    return AbstractDistribution(
        family=family,
        event_shape=event_shape,)


def witness_predictstep(current_state: AbstractArray, model_params: AbstractArray, dt: AbstractArray) -> AbstractArray:
    """Ghost witness for PredictStep."""
    result = AbstractArray(
        shape=current_state.shape,
        dtype="float64",)

    return result

def witness_updatestep(prior: AbstractDistribution, likelihood: AbstractDistribution, data_shape: tuple[int, ...]) -> AbstractDistribution:
    """Ghost witness for posterior update: UpdateStep."""
    prior.assert_conjugate_to(likelihood)
    return AbstractDistribution(
        family=prior.family,
        event_shape=prior.event_shape,
        batch_shape=prior.batch_shape,
        support_lower=prior.support_lower,
        support_upper=prior.support_upper,
        is_discrete=prior.is_discrete,
    )

def witness_querystance(current_state: AbstractArray) -> AbstractArray:
    """Ghost witness for QueryStance."""
    result = AbstractArray(
        shape=current_state.shape,
        dtype="float64",)

    return result
