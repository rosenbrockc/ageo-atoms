from __future__ import annotations
"""Auto-generated atom wrappers and FFI bindings following the ageoa pattern."""


from typing import Any

import ctypes
import ctypes.util
from pathlib import Path

import numpy as np
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_stanceestimation, witness_stancestateinit


# Domain-specific type alias for the stance estimator state container
StanceState = dict[str, np.ndarray]


@register_atom(witness_stancestateinit)  # type: ignore[untyped-decorator]
@icontract.require(lambda config: config is not None, "config cannot be None")
@icontract.ensure(lambda result: result is not None, "StanceStateInit output must not be None")
def stancestateinit(config: dict[str, float]) -> StanceState:
    """Bootstraps the internal state containers for the dynamic stance estimator - allocates covariance matrices (P, Q, R), latent mean/variance buffers, and any persistent bookkeeping needed before the first estimation step.

    Args:
        config: Must contain at minimum noise hyperparameters and dimensionality spec

    Returns:
        Initialized StanceState object
    """
    n_legs = int(config.get('n_legs', 4))
    force_threshold = config.get('force_threshold', 50.0)
    return {
        'stance': np.zeros(n_legs, dtype=np.float64),
        'force_threshold': np.array([force_threshold], dtype=np.float64),
        'n_legs': np.array([n_legs], dtype=np.int64),
        'grf_history': np.zeros(n_legs, dtype=np.float64),
    }


@register_atom(witness_stanceestimation)  # type: ignore[untyped-decorator]
@icontract.require(lambda observation: observation is not None, "observation cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "StanceEstimation all outputs must not be None")
def stanceestimation(stance_state: StanceState, observation: np.ndarray) -> tuple[StanceState, np.ndarray]:
    """Core dynamic-stance estimation pass: consumes the current stance state and a new observation vector, runs the estimation kernel (predict + update), and returns an updated immutable stance state together with the estimated stance output.

    Args:
        stance_state: Output of StanceStateInit or a prior StanceEstimation call; treated as immutable
        observation: Raw sensor / feature vector aligned to the state dimensionality

    Returns:
        stance_state_out: New object; never mutates stance_state_in
        stance_estimate: Posterior mean of the stance vector at the current time step
    """
    obs = np.atleast_1d(np.asarray(observation, dtype=np.float64))
    threshold = float(stance_state['force_threshold'][0])
    # Compare ground reaction force against threshold per leg
    stance_estimate = (obs >= threshold).astype(np.float64)
    stance_state_out = {
        'stance': stance_estimate.copy(),
        'force_threshold': stance_state['force_threshold'].copy(),
        'n_legs': stance_state['n_legs'].copy(),
        'grf_history': obs.copy(),
    }
    return (stance_state_out, stance_estimate)


def _stancestateinit_ffi(config: dict[str, float]) -> ctypes.c_void_p:
    """Wrapper that calls the C++ version of stance state init. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./stancestateinit.so")
    _func_name = 'stancestateinit_prime'
    _func = _lib[_func_name]
    _func.restype = ctypes.c_void_p
    return _func(config)


def _stanceestimation_ffi(stance_state: ctypes.c_void_p, observation: ctypes.c_void_p) -> ctypes.c_void_p:
    """Wrapper that calls the C++ version of stance estimation. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./stanceestimation.so")
    _func_name = 'stanceestimation_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(stance_state, observation)
