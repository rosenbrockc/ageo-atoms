from __future__ import annotations
from typing import Any
StateModelSpec: Any = Any

"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_stateestimatorinit

import ctypes
import ctypes.util
from pathlib import Path

@register_atom(witness_stateestimatorinit)
@icontract.require(lambda: True, "no preconditions for zero-parameter initializer")
@icontract.ensure(lambda result: result is not None, "StateEstimatorInit output must not be None")
def stateestimatorinit() -> StateModelSpec:
    """Bootstraps the StateEstimator instance: allocates internal containers, sets default hyperparameters, and establishes the initial state model ready for predict/update cycles.


    Returns:
        All matrices must be positive-semi-definite where applicable; initial state vector must be finite.
    """
    # Default 6-DOF state: [x, y, z, vx, vy, vz]
    n = 6
    x = np.zeros(n, dtype=np.float64)
    P = np.eye(n, dtype=np.float64) * 1e2
    A = np.eye(n, dtype=np.float64)
    # Position driven by velocity
    dt = 0.01
    A[0, 3] = dt
    A[1, 4] = dt
    A[2, 5] = dt
    Q = np.eye(n, dtype=np.float64) * 1e-3
    H = np.eye(n, dtype=np.float64)
    R = np.eye(n, dtype=np.float64) * 1e-1
    return {
        'x': x, 'P': P, 'A': A, 'Q': Q, 'H': H, 'R': R,
        'x_history': [], 'P_history': [], 'x_pred_history': [], 'P_pred_history': [],
    }


"""Auto-generated FFI bindings for cpp implementations."""


import ctypes
import ctypes.util
from pathlib import Path
# StateModelSpec already defined as type alias above
def _stateestimatorinit_ffi() -> ctypes.c_void_p:
    _func_name = 'stateestimatorinit_prime'
    _lib = ctypes.CDLL("./stateestimatorinit.so")
    _func_name = 'stateestimatorinit_prime'
    _func = _lib[_func_name]
    _func.restype = ctypes.c_void_p
    return _func()