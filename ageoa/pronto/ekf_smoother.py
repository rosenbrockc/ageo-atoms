from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypedDict

import icontract
import numpy as np

from ageoa.ghost.registry import register_atom

from .ekf_smoother_witnesses import witness_stateestimatorinit


class StateEstimatorState(TypedDict):
    x: np.ndarray
    P: np.ndarray
    A: np.ndarray
    Q: np.ndarray
    H: np.ndarray
    R: np.ndarray
    x_history: list[np.ndarray]
    P_history: list[np.ndarray]
    x_pred_history: list[np.ndarray]
    P_pred_history: list[np.ndarray]


@register_atom(witness_stateestimatorinit)
@icontract.require(lambda: True, "no preconditions for zero-parameter initializer")
@icontract.ensure(lambda result: result is not None, "StateEstimatorInit output must not be None")
def stateestimatorinit() -> StateEstimatorState:
    """Bootstraps a default EKF smoother state estimate bundle.

    Returns:
        Initialized filter state with finite state vectors and positive
        semi-definite covariance and noise matrices.
    """
    n = 6
    dt = 0.01

    x = np.zeros(n, dtype=np.float64)
    P = np.eye(n, dtype=np.float64) * 1e2
    A = np.eye(n, dtype=np.float64)
    A[0, 3] = dt
    A[1, 4] = dt
    A[2, 5] = dt
    Q = np.eye(n, dtype=np.float64) * 1e-3
    H = np.eye(n, dtype=np.float64)
    R = np.eye(n, dtype=np.float64) * 1e-1

    return {
        "x": x,
        "P": P,
        "A": A,
        "Q": Q,
        "H": H,
        "R": R,
        "x_history": [],
        "P_history": [],
        "x_pred_history": [],
        "P_pred_history": [],
    }


def _stateestimatorinit_ffi() -> ctypes.c_void_p:
    lib_path = Path(__file__).with_name("stateestimatorinit.so")
    _lib = ctypes.CDLL(str(lib_path))
    _func = _lib["stateestimatorinit_prime"]
    _func.restype = ctypes.c_void_p
    return _func()
