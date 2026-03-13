from __future__ import annotations
from typing import Callable
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom

import ctypes
import ctypes.util
from pathlib import Path


from .witnesses import witness_initializenutsstate, witness_runnutstransitions

@register_atom(witness_initializenutsstate)
@icontract.require(lambda initial_positions: isinstance(initial_positions, (float, int, np.number)), "initial_positions must be numeric")
@icontract.require(lambda target_accept_p: isinstance(target_accept_p, (float, int, np.number)), "target_accept_p must be numeric")
@icontract.ensure(lambda result: all(r is not None for r in result), "InitializeNUTSState all outputs must not be None")
def initializenutsstate(target: Callable[[np.ndarray], float], initial_positions: np.ndarray, target_accept_p: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build immutable NUTS state from the target log-density, initial position, acceptance target, and explicit RNG key state.

    Args:
        target: Pure/stateless likelihood or log-density evaluator
        initial_positions: Valid support of target distribution
        target_accept_p: 0 < target_accept_p < 1
        seed: Used to derive deterministic PRNGKey/RNG state

    Returns:
        nuts_state: Immutable state object; no hidden mutation
        rng_key: Explicit stochastic state threaded across calls
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_runnutstransitions)
@icontract.require(lambda nuts_state_in: isinstance(nuts_state_in, np.ndarray), "nuts_state_in must be np.ndarray")
@icontract.require(lambda rng_key_in: rng_key_in is not None, "rng_key_in cannot be None")
@icontract.require(lambda n_collect: n_collect is not None, "n_collect cannot be None")
@icontract.require(lambda n_discard: n_discard is not None, "n_discard cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "RunNUTSTransitions all outputs must not be None")
def runnutstransitions(nuts_state_in: np.ndarray, rng_key_in: np.ndarray, n_collect: int, n_discard: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply NUTS transition kernels for warmup/discard and collection while returning new immutable chain state, diagnostics trace, and split RNG key.

    Args:
        nuts_state_in: Input state is immutable
        rng_key_in: Must be consumed/split and returned as new key
        n_collect: >= 0
        n_discard: >= 0

    Returns:
        samples: Posterior draws after discard phase
        trace_out: Per-step diagnostics/history
        nuts_state_out: New immutable state object
        rng_key_out: New key after stochastic transitions
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for rust implementations."""

# duplicate future import removed

import ctypes
import ctypes.util
from pathlib import Path


def _initializenutsstate_ffi(target: Callable[[np.ndarray], float], initial_positions: np.ndarray, target_accept_p: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """FFI bridge to Rust implementation of InitializeNUTSState."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'initializenutsstate'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(target, initial_positions, target_accept_p, seed)

def _runnutstransitions_ffi(nuts_state_in: np.ndarray, rng_key_in: np.ndarray, n_collect: int, n_discard: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """FFI bridge to Rust implementation of RunNUTSTransitions."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'runnutstransitions'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(nuts_state_in, rng_key_in, n_collect, n_discard)