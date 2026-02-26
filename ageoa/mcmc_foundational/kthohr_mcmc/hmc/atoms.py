"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

import ctypes
import ctypes.util
from pathlib import Path
from typing import Callable

# Witness functions should be imported from the generated witnesses module
witness_buildhmckernelfromlogdensityoracle: object = object()

@register_atom(witness_buildhmckernelfromlogdensityoracle)  # type: ignore[untyped-decorator]
@icontract.require(lambda target_log_kernel: callable(target_log_kernel), "target_log_kernel must be callable")
@icontract.ensure(lambda result: result is not None, "BuildHMCKernelFromLogDensityOracle output must not be None")
def buildhmckernelfromlogdensityoracle(target_log_kernel: Callable[[np.ndarray], float]) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Creates a pure Hamiltonian Monte Carlo transition kernel from a provided target log-density oracle, with stochasticity and chain state threaded explicitly.

    Args:
        target_log_kernel: Stateless oracle; no persistent writes; deterministic for fixed input.

    Returns:
        Pure transition; consumes and returns new PRNGKey; HMCState is immutable state_in->state_out and may include position, momentum, mass_matrix, and trace diagnostics.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for cpp implementations."""

"""Auto-generated FFI bindings for cpp implementations."""

def _buildhmckernelfromlogdensityoracle_ffi(target_log_kernel: ctypes.c_void_p) -> ctypes.c_void_p:
    return target_log_kernel