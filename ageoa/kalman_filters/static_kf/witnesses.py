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

def witness_initializelineargaussianstatemodel(event_shape: tuple[int, ...], family: str = "normal") -> AbstractDistribution:
    """Ghost witness for prior init: InitializeLinearGaussianStateModel."""
    return AbstractDistribution(
        family=family,
        event_shape=event_shape,
    )

def witness_predictlatentstate(state_model: AbstractArray) -> AbstractArray:
    """Ghost witness for PredictLatentState."""
    result = AbstractArray(
        shape=state_model.shape,
        dtype="float64",
    )
    return result

def witness_updatewithmeasurement(prior: AbstractDistribution, likelihood: AbstractDistribution, data_shape: tuple[int, ...]) -> AbstractDistribution:
    """Ghost witness for posterior update: UpdateWithMeasurement."""
    prior.assert_conjugate_to(likelihood)
    return AbstractDistribution(
        family=prior.family,
        event_shape=prior.event_shape,
        batch_shape=prior.batch_shape,
        support_lower=prior.support_lower,
        support_upper=prior.support_upper,
        is_discrete=prior.is_discrete,
    )

def witness_exposelatentmean(current_state_model: AbstractArray) -> AbstractArray:
    """Ghost witness for ExposeLatentMean."""
    result = AbstractArray(
        shape=current_state_model.shape,
        dtype="float64",
    )
    return result

def witness_exposecovariance(current_state_model: AbstractArray) -> AbstractArray:
    """Ghost witness for ExposeCovariance."""
    result = AbstractArray(
        shape=current_state_model.shape,
        dtype="float64",
    )
    return result
