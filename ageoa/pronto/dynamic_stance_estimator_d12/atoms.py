"""Auto-generated atom wrappers and FFI bindings following the ageoa pattern."""

from __future__ import annotations

from typing import Any

import ctypes
import ctypes.util
from pathlib import Path

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk
import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]


# Witness functions should be imported from the generated witnesses module
StanceState = Any
witness_stancestateinit: Any = None
witness_stanceestimation: Any = None


@register_atom(witness_stancestateinit)  # type: ignore[untyped-decorator]
@icontract.require(lambda config: config is not None, "config cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "StanceStateInit output must not be None")
def stancestateinit(config: dict[str, Any]) -> StanceState:
    """Bootstraps the internal state containers for the dynamic stance estimator — allocates covariance matrices (P, Q, R), latent mean/variance buffers, and any persistent bookkeeping needed before the first estimation step.

    Args:
        config: Must contain at minimum noise hyperparameters and dimensionality spec

    Returns:
        Initialized StanceState object
    """
    raise NotImplementedError("Wire to original implementation")


@register_atom(witness_stanceestimation)  # type: ignore[untyped-decorator]
@icontract.require(lambda observation: isinstance(observation, (float, int, np.number)), "observation must be numeric")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "StanceEstimation all outputs must not be None")
def stanceestimation(stance_state: StanceState, observation: np.ndarray[Any, np.dtype[Any]]) -> tuple[StanceState, np.ndarray[Any, np.dtype[Any]]]:
    """Core dynamic-stance estimation pass: consumes the current stance state and a new observation vector, runs the estimation kernel (predict + update), and returns an updated immutable stance state together with the estimated stance output.

    Args:
        stance_state: Output of StanceStateInit or a prior StanceEstimation call; treated as immutable
        observation: Raw sensor / feature vector aligned to the state dimensionality

    Returns:
        stance_state_out: New object; never mutates stance_state_in
        stance_estimate: Posterior mean of the stance vector at the current time step
    """
    raise NotImplementedError("Wire to original implementation")


def stancestateinit_ffi(config: Any) -> Any:
    """FFI bridge to C++ implementation of StanceStateInit."""
    _lib = ctypes.CDLL("./stancestateinit.so")
    _func_name = 'stancestateinit'
    _func = _lib[_func_name]
    _func.restype = ctypes.c_void_p
    return _func(config)


def stanceestimation_ffi(stance_state: Any, observation: Any) -> Any:
    """FFI bridge to C++ implementation of StanceEstimation."""
    _lib = ctypes.CDLL("./stanceestimation.so")
    _func_name = 'stanceestimation'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(stance_state, observation)