"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from typing import Any, Callable, TypeVar, cast
from ageoa.ghost.registry import register_atom as _register_atom  # type: ignore[import-untyped]

import ctypes
import ctypes.util
from pathlib import Path

F = TypeVar("F", bound=Callable[..., Any])


def register_atom(witness: object) -> Callable[[F], F]:
    return cast(Callable[[F], F], _register_atom(witness))


# Witness functions should be imported from the generated witnesses module
witness_initializelineargaussianstatemodel: object = object()
witness_predictlatentstate: object = object()
witness_updatewithmeasurement: object = object()
witness_exposelatentmean: object = object()
witness_exposecovariance: object = object()

@register_atom(witness_initializelineargaussianstatemodel)
@icontract.require(lambda initial_state: isinstance(initial_state, (float, int, np.number)), "initial_state must be numeric")
@icontract.require(lambda initial_covariance: isinstance(initial_covariance, (float, int, np.number)), "initial_covariance must be numeric")
@icontract.require(lambda transition_matrix: isinstance(transition_matrix, (float, int, np.number)), "transition_matrix must be numeric")
@icontract.require(lambda process_noise: isinstance(process_noise, (float, int, np.number)), "process_noise must be numeric")
@icontract.require(lambda observation_matrix: isinstance(observation_matrix, (float, int, np.number)), "observation_matrix must be numeric")
@icontract.require(lambda measurement_noise: isinstance(measurement_noise, (float, int, np.number)), "measurement_noise must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "InitializeLinearGaussianStateModel output must not be None")
def initializelineargaussianstatemodel(initial_state: object, initial_covariance: object, transition_matrix: object, process_noise: object, observation_matrix: object, measurement_noise: object) -> object:
    """Create the immutable Kalman state-space model with latent mean and covariance plus fixed system/noise matrices.

    Args:
        initial_state: Dimension n
        initial_covariance: Shape n x n; symmetric positive semi-definite
        transition_matrix: Shape n x n
        process_noise: Shape n x n; symmetric positive semi-definite
        observation_matrix: Shape m x n
        measurement_noise: Shape m x m; symmetric positive semi-definite

    Returns:
        Immutable object; no hidden mutation
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_predictlatentstate)
@icontract.require(lambda state_model: state_model is not None, "state_model cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "PredictLatentState output must not be None")
def predictlatentstate(state_model: object) -> object:
    """Apply the Kalman predict transition kernel to propagate latent mean/covariance forward in time.

    Args:
        state_model: Immutable prior/posterior from previous step

    Returns:
        New object with updated x and P only
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_updatewithmeasurement)
@icontract.require(lambda measurement: isinstance(measurement, (float, int, np.number)), "measurement must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "UpdateWithMeasurement output must not be None")
def updatewithmeasurement(predicted_state_model: object, measurement: object) -> object:
    """Apply the Kalman update kernel to incorporate a measurement and produce posterior latent mean/covariance.

    Args:
        predicted_state_model: Output of predict kernel
        measurement: Dimension m; compatible with H and R

    Returns:
        New object; analytical Bayesian posterior update
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_exposelatentmean)
@icontract.require(lambda current_state_model: current_state_model is not None, "current_state_model cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "ExposeLatentMean output must not be None")
def exposelatentmean(current_state_model: object) -> object:
    """Read out the current latent state mean estimate from immutable filter state.

    Args:
        current_state_model: Can be initialized, predicted, or updated state

    Returns:
        Dimension n
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_exposecovariance)
@icontract.require(lambda current_state_model: current_state_model is not None, "current_state_model cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "ExposeCovariance output must not be None")
def exposecovariance(current_state_model: object) -> object:
    """Read out the current latent covariance estimate from immutable filter state.

    Args:
        current_state_model: Can be initialized, predicted, or updated state

    Returns:
        Shape n x n; symmetric positive semi-definite
    """
    raise NotImplementedError("Wire to original implementation")


def initializelineargaussianstatemodel_ffi(initial_state: object, initial_covariance: object, transition_matrix: object, process_noise: object, observation_matrix: object, measurement_noise: object) -> object:
    raise NotImplementedError("FFI bridge not wired")


def predictlatentstate_ffi(state_model: object) -> object:
    raise NotImplementedError("FFI bridge not wired")


def updatewithmeasurement_ffi(predicted_state_model: object, measurement: object) -> object:
    raise NotImplementedError("FFI bridge not wired")


def exposelatentmean_ffi(current_state_model: object) -> object:
    raise NotImplementedError("FFI bridge not wired")


def exposecovariance_ffi(current_state_model: object) -> object:
    raise NotImplementedError("FFI bridge not wired")