from __future__ import annotations
from typing import Any
ModelParamsSpec: Any = Any
StateModelSpec: Any = Any
"""Auto-generated atom wrappers following the ageoa pattern."""

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_initializefilter, witness_predictstep, witness_querystance, witness_updatestep

import ctypes
import ctypes.util
from pathlib import Path
# StateModelSpec already defined as type alias above


@register_atom(witness_initializefilter)
@icontract.require(lambda initial_x: initial_x is not None, "initial_x cannot be None")
@icontract.require(lambda initial_P: initial_P is not None, "initial_P cannot be None")
@icontract.require(lambda A: A is not None, "A cannot be None")
@icontract.require(lambda H: H is not None, "H cannot be None")
@icontract.require(lambda Q: Q is not None, "Q cannot be None")
@icontract.require(lambda R: R is not None, "R cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "InitializeFilter all outputs must not be None")
def initializefilter(initial_x: np.ndarray, initial_P: np.ndarray, A: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[StateModelSpec, ModelParamsSpec]:
    """Initializes the state-space model with prior state, covariance, and static model parameters.

    Args:
        initial_x: Shape [2, 1]
        initial_P: Shape [2, 2]
        A: State transition matrix, shape [2, 2]
        H: Measurement matrix, shape [1, 2]
        Q: Process noise covariance, shape [2, 2]
        R: Measurement noise covariance, shape [1, 1]

    Returns:
        initial_state: Contains initial x and P
        model_params: Contains A, H, Q, R
    """
    initial_state = {'x': initial_x.copy(), 'P': initial_P.copy()}
    model_params = {'A': A.copy(), 'H': H.copy(), 'Q': Q.copy(), 'R': R.copy()}
    return (initial_state, model_params)

@register_atom(witness_predictstep)
@icontract.require(lambda dt: isinstance(dt, (float, int, np.number)), "dt must be numeric")
@icontract.ensure(lambda result: result is not None, "PredictStep output must not be None")
def predictstep(current_state: StateModelSpec, model_params: ModelParamsSpec, dt: float) -> StateModelSpec:
    """Predicts the next state and covariance based on the state transition model (prediction step of Kalman filter).

    Args:
        current_state: Current state (x, P)
        model_params: Contains A, Q
        dt: Time step

    Returns:
        Predicted state (x_prior, P_prior)
    """
    A = model_params['A']
    Q = model_params['Q']
    x = current_state['x']
    P = current_state['P']
    x_prior = A @ x
    P_prior = A @ P @ A.T + Q
    return {'x': x_prior, 'P': P_prior}

@register_atom(witness_updatestep)
@icontract.require(lambda predicted_state: predicted_state is not None, "predicted_state cannot be None")
@icontract.require(lambda model_params: model_params is not None, "model_params cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.ensure(lambda result: result is not None, "UpdateStep output must not be None")
def updatestep(predicted_state: StateModelSpec, model_params: ModelParamsSpec, z: np.ndarray) -> StateModelSpec:
    """Updates the state and covariance based on a new measurement (update step of Kalman filter).

    Args:
        predicted_state: Predicted state (x_prior, P_prior)
        model_params: Contains H, R
        z: Measurement

    Returns:
        Updated state (x_posterior, P_posterior)
    """
    H = model_params['H']
    R = model_params['R']
    x = predicted_state['x']
    P = predicted_state['P']
    innovation = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x_post = x + K @ innovation
    n = x.shape[0]
    P_post = (np.eye(n) - K @ H) @ P
    return {'x': x_post, 'P': P_post}

@register_atom(witness_querystance)
@icontract.require(lambda current_state: current_state is not None, "current_state cannot be None")
@icontract.ensure(lambda result: result is not None, "QueryStance output must not be None")
def querystance(current_state: StateModelSpec) -> float:
    """Extracts the stance (position) from the state vector.

    Args:
        current_state: Current state (x, P)

    Returns:
        The estimated position
    """
    return float(current_state['x'][0])


"""Auto-generated FFI bindings for cpp implementations."""


def _initializefilter_ffi(initial_x, initial_P, A, H, Q, R):
    """Wrapper that calls the C++ version of initialize filter. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./initializefilter.so")
    _func_name = 'initializefilter_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(initial_x, initial_P, A, H, Q, R)

def _predictstep_ffi(current_state, model_params, dt):
    """Wrapper that calls the C++ version of predict step. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./predictstep.so")
    _func_name = 'predictstep_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(current_state, model_params, dt)

def _updatestep_ffi(predicted_state, model_params, z):
    """Wrapper that calls the C++ version of update step. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./updatestep.so")
    _func_name = 'updatestep_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(predicted_state, model_params, z)

def _querystance_ffi(current_state):
    """Wrapper that calls the C++ version of query stance. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./querystance.so")
    _func_name = 'querystance_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(current_state)