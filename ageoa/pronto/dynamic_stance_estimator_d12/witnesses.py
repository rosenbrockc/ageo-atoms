from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal


def witness_stancestateinit(config: AbstractArray) -> AbstractArray:
    """Shape-and-type check for stance state init. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=config.shape,
        dtype="float64",)
    
    return result

def witness_stanceestimation(prior: AbstractDistribution, likelihood: AbstractDistribution, data_shape: tuple[int, ...]) -> AbstractDistribution:
    """Shape-and-type check for posterior update: stance estimation. Returns output metadata without running the real computation."""
    prior.assert_conjugate_to(likelihood)
    return AbstractDistribution(
        family=prior.family,
        event_shape=prior.event_shape,
        batch_shape=prior.batch_shape,
        support_lower=prior.support_lower,
        support_upper=prior.support_upper,
        is_discrete=prior.is_discrete,)
    