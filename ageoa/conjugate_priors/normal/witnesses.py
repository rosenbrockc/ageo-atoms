from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_normal_gamma_posterior_update(prior: AbstractDistribution, ss: AbstractArray) -> AbstractDistribution:
    """Shape-and-type check for closed-form conjugate update: normal gamma posterior update. Returns output metadata without running the real computation."""
    # Closed-form update: no sampling trace or RNG threading required.
    return AbstractDistribution(
        family=prior.family,
        event_shape=prior.event_shape,
        batch_shape=prior.batch_shape,
        support_lower=prior.support_lower,
        support_upper=prior.support_upper,
        is_discrete=prior.is_discrete,
    )
