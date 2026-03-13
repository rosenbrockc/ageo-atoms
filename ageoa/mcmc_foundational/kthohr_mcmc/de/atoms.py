from __future__ import annotations
from typing import Callable
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

import ctypes
import ctypes.util
from pathlib import Path


from .witnesses import witness_build_de_transition_kernel

@register_atom(witness_build_de_transition_kernel)
@icontract.require(lambda target_log_kernel: target_log_kernel is not None, "target_log_kernel cannot be None")
@icontract.ensure(lambda result: result is not None, "build_de_transition_kernel output must not be None")
def build_de_transition_kernel(target_log_kernel: Callable[[np.ndarray], float]) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Creates a pure Differential Evolution transition kernel from the provided target log-density oracle.

Args:
    target_log_kernel: Stateless log-density oracle; no persistent state mutation.

Returns:
    Pure transition function; any stochastic state (e.g., random number generator (RNG)/PRNGKey) must be explicit input/output."""
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for cpp implementations."""


import ctypes
import ctypes.util
from pathlib import Path

def _build_de_transition_kernel_ffi(target_log_kernel: Callable[[np.ndarray], float]) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Wrapper that calls the C++ version of build de transition kernel. Passes arguments through and returns the result."""
    _func_name = 'build_de_transition_kernel'
    _func_name = 'build_de_transition_kernel'
    _func = ctypes.CDLL(None)[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(target_log_kernel)