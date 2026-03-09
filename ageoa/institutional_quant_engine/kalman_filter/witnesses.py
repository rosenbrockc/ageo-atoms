"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations

import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore

try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
    from ageoa.ghost.abstract import AbstractDistribution
except ImportError:
    pass

def witness_kalmanfilterinit(process_variance: AbstractArray, measurement_variance: AbstractArray, estimated_measurement_variance: AbstractArray, state: AbstractArray) -> tuple[AbstractArray, AbstractArray]:
    """Ghost witness for KalmanFilterInit."""
    result = AbstractArray(
        shape=process_variance.shape,
        dtype="float64",
    )
    return result, state

def witness_kalmanmeasurementupdate(prior: AbstractDistribution, likelihood: AbstractDistribution, data_shape: tuple[int, ...]) -> AbstractDistribution:
    """Ghost witness for posterior update: KalmanMeasurementUpdate."""
    prior.assert_conjugate_to(likelihood)
    return AbstractDistribution(
        family=prior.family,
        event_shape=prior.event_shape,
        batch_shape=prior.batch_shape,
        support_lower=prior.support_lower,
        support_upper=prior.support_upper,
        is_discrete=prior.is_discrete,
    )
