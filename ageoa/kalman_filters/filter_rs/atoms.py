from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_evaluatemeasurementoracle, witness_initializekalmanstatemodel, witness_predictlatentstateandcovariance, witness_predictlatentstatesteadystate, witness_updateposteriorstateandcovariance, witness_updateposteriorstatesteadystate

import ctypes
import ctypes.util
from pathlib import Path


@register_atom(witness_initializekalmanstatemodel)
@icontract.require(lambda init_config: init_config is not None, "init_config cannot be None")
@icontract.ensure(lambda result: result is not None, "InitializeKalmanStateModel output must not be None")
def initializekalmanstatemodel(init_config: object) -> object:
    """Create the initial read-only state for a Kalman filter — a method that estimates hidden variables from noisy measurements.

    Args:
        init_config: starting estimate, uncertainty, and noise model matrices

    Returns:
        read-only state object
    """
    x = np.asarray(init_config.get('x', init_config.get('initial_state', [0.0])), dtype=float)
    P = np.asarray(init_config.get('P', init_config.get('initial_covariance', np.eye(len(x)))), dtype=float)
    return {'x': x, 'P': P}

@register_atom(witness_predictlatentstateandcovariance)  # type: ignore[untyped-decorator]
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.require(lambda u: u is not None, "u cannot be None")
@icontract.require(lambda B: B is not None, "B cannot be None")
@icontract.require(lambda F: F is not None, "F cannot be None")
@icontract.require(lambda Q: Q is not None, "Q cannot be None")
@icontract.ensure(lambda result: result is not None, "PredictLatentStateAndCovariance output must not be None")
def predictlatentstateandcovariance(state_in: object, u: object, B: object, F: object, Q: object) -> object:
    """Kalman predict kernel for full-covariance filtering; propagates latent mean and covariance forward in time and returns a new state object.

    Args:
        state_in: immutable input state
        u: dimension compatible with B
        B: if provided, shape compatible with u and x
        F: square, shape compatible with x
        Q: symmetric positive semidefinite

    Returns:
        new immutable object; x_pred and P_pred updated
    """
    x = np.asarray(state_in['x'], dtype=float)
    P = np.asarray(state_in['P'], dtype=float)
    F_mat = np.asarray(F, dtype=float)
    B_mat = np.asarray(B, dtype=float)
    Q_mat = np.asarray(Q, dtype=float)
    u_arr = np.asarray(u, dtype=float)
    x_pred = F_mat @ x + B_mat @ u_arr
    P_pred = F_mat @ P @ F_mat.T + Q_mat
    return {'x': x_pred, 'P': P_pred}

@register_atom(witness_predictlatentstatesteadystate)  # type: ignore[untyped-decorator]
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.require(lambda u: u is not None, "u cannot be None")
@icontract.require(lambda B: B is not None, "B cannot be None")
@icontract.ensure(lambda result: result is not None, "PredictLatentStateSteadyState output must not be None")
def predictlatentstatesteadystate(state_in: object, u: object, B: object) -> object:
    """Steady-state predict kernel variant where covariance/gain are treated as fixed and only the latent mean transition is applied.

    Args:
        state_in: fixed covariance/gain assumed precomputed
        u: dimension compatible with B
        B: if provided, shape compatible with u and x

    Returns:
        new immutable object
    """
    x = np.asarray(state_in['x'], dtype=float)
    P = np.asarray(state_in['P'], dtype=float)
    B_mat = np.asarray(B, dtype=float)
    u_arr = np.asarray(u, dtype=float)
    x_pred = x + B_mat @ u_arr
    return {'x': x_pred, 'P': P}

@register_atom(witness_evaluatemeasurementoracle)  # type: ignore[untyped-decorator]
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.require(lambda H: H is not None, "H cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "EvaluateMeasurementOracle all outputs must not be None")
def evaluatemeasurementoracle(x: object, z: object, H: object) -> tuple[object, object]:
    """Pure observation oracle that maps latent state to predicted measurement and innovation residual; performs no persistent state writes.

    Args:
        x: shape compatible with H
        z: shape compatible with Hx
        H: maps latent space to measurement space

    Returns:
        z_pred: z_pred = Hx
        innovation: innovation = z - z_pred
    """
    x_arr = np.asarray(x, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    H_mat = np.asarray(H, dtype=float)
    z_pred = H_mat @ x_arr
    innovation = z_arr - z_pred
    return (z_pred, innovation)

@register_atom(witness_updateposteriorstateandcovariance)  # type: ignore[untyped-decorator]
@icontract.require(lambda predicted_state: predicted_state is not None, "predicted_state cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.require(lambda R: R is not None, "R cannot be None")
@icontract.require(lambda H: H is not None, "H cannot be None")
@icontract.require(lambda innovation: innovation is not None, "innovation cannot be None")
@icontract.ensure(lambda result: result is not None, "UpdatePosteriorStateAndCovariance output must not be None")
def updateposteriorstateandcovariance(predicted_state: object, z: object, R: object, H: object, innovation: object) -> object:
    """Kalman update kernel for full-covariance filtering; fuses measurement with predicted state and returns a new posterior state object.

    Args:
        predicted_state: immutable predicted state
        z: shape compatible with H
        R: symmetric positive semidefinite
        H: shape compatible with x_pred and z
        innovation: if omitted, recomputed internally from z and Hx_pred

    Returns:
        new immutable object; x_post and P_post updated
    """
    x = np.asarray(predicted_state['x'], dtype=float)
    P = np.asarray(predicted_state['P'], dtype=float)
    H_mat = np.asarray(H, dtype=float)
    R_mat = np.asarray(R, dtype=float)
    inn = np.asarray(innovation, dtype=float)
    S = H_mat @ P @ H_mat.T + R_mat
    K = P @ H_mat.T @ np.linalg.inv(S)
    x_post = x + K @ inn
    n = len(x)
    P_post = (np.eye(n) - K @ H_mat) @ P
    return {'x': x_post, 'P': P_post}

@register_atom(witness_updateposteriorstatesteadystate)  # type: ignore[untyped-decorator]
@icontract.require(lambda predicted_state_steady: predicted_state_steady is not None, "predicted_state_steady cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.require(lambda innovation: innovation is not None, "innovation cannot be None")
@icontract.ensure(lambda result: result is not None, "UpdatePosteriorStateSteadyState output must not be None")
def updateposteriorstatesteadystate(predicted_state_steady: object, z: object, innovation: object) -> object:
    """Steady-state update kernel variant using fixed gain/covariance assumptions; returns a new posterior latent state.

    Args:
        predicted_state_steady: immutable predicted steady-state object
        z: shape compatible with H
        innovation: if omitted, recomputed internally

    Returns:
        new immutable object
    """
    x = np.asarray(predicted_state_steady['x'], dtype=float)
    P = np.asarray(predicted_state_steady['P'], dtype=float)
    inn = np.asarray(innovation, dtype=float)
    # Steady-state: use precomputed gain if available, otherwise just add innovation
    K = predicted_state_steady.get('K', None)
    if K is not None:
        K_mat = np.asarray(K, dtype=float)
        x_post = x + K_mat @ inn
    else:
        x_post = x + inn
    return {'x': x_post, 'P': P}


"""Auto-generated FFI bindings for rust implementations."""

# from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path


def _initializekalmanstatemodel_ffi(init_config: object) -> object:
    """Wrapper that calls the Rust version of initialize kalman state model. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = "initializekalmanstatemodel"
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(init_config)

def _predictlatentstateandcovariance_ffi(state_in: object, u: object, B: object, F: object, Q: object) -> object:
    """Wrapper that calls the Rust version of predict latent state and covariance. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = "predictlatentstateandcovariance"
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(state_in, u, B, F, Q)

def _predictlatentstatesteadystate_ffi(state_in: object, u: object, B: object) -> object:
    """Wrapper that calls the Rust version of predict latent state steady state. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = "predictlatentstatesteadystate"
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(state_in, u, B)

def _evaluatemeasurementoracle_ffi(x: object, z: object, H: object) -> object:
    """Wrapper that calls the Rust version of evaluate measurement oracle. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = "evaluatemeasurementoracle"
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(x, z, H)

def _updateposteriorstateandcovariance_ffi(predicted_state: object, z: object, R: object, H: object, innovation: object) -> object:
    """Wrapper that calls the Rust version of update posterior state and covariance. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = "updateposteriorstateandcovariance"
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(predicted_state, z, R, H, innovation)

def _updateposteriorstatesteadystate_ffi(predicted_state_steady: object, z: object, innovation: object) -> object:
    """Wrapper that calls the Rust version of update posterior state steady state. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = "updateposteriorstatesteadystate"
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(predicted_state_steady, z, innovation)
