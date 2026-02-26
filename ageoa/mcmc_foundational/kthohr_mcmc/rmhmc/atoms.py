"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

from typing import Callable
import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_buildrmhmctransitionkernel)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda target_log_kernel: target_log_kernel is not None, "target_log_kernel cannot be None")
@icontract.require(lambda tensor_fn: tensor_fn is not None, "tensor_fn cannot be None")
@icontract.ensure(lambda result: result is not None, "BuildRMHMCTransitionKernel output must not be None")
def buildrmhmctransitionkernel(target_log_kernel: Callable[[np.ndarray], float], tensor_fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Constructs a pure Riemannian Manifold HMC transition kernel from a target log-density oracle and metric/tensor oracle. The produced kernel is expected to thread immutable sampler state explicitly (e.g., position, momentum, mass/metric tensor, and PRNGKey) as state_in -> state_out.

    Args:
        target_log_kernel: Pure oracle-style log-density/log-kernel evaluator; no persistent state mutation.
        tensor_fn: Pure oracle-style metric/tensor evaluator compatible with RMHMC geometry.

    Returns:
        Pure MCMC transition function that consumes explicit state (including PRNGKey) and returns a new state object.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for cpp implementations."""

from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path


def _buildrmhmctransitionkernel_ffi(target_log_kernel: Callable[[np.ndarray], float], tensor_fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """FFI bridge to C++ implementation of BuildRMHMCTransitionKernel."""
    _lib = ctypes.CDLL("./buildrmhmctransitionkernel.so")
    _func_name = "buildrmhmctransitionkernel"
    _func = getattr(_lib, _func_name)
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(target_log_kernel, tensor_fn)