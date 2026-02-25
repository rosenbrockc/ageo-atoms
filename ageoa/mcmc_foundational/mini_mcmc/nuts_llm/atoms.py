"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom

import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module

@register_atom(lambda *args, **kwargs: None)
@icontract.require(lambda initial_positions: isinstance(initial_positions, (float, int, np.number)), "initial_positions must be numeric")
@icontract.require(lambda target_accept_p: isinstance(target_accept_p, (float, int, np.number)), "target_accept_p must be numeric")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "InitializeNUTSState all outputs must not be None")
def initializenutsstate(target: object, initial_positions: object, target_accept_p: float, seed: int) -> tuple[object, object]:
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

@register_atom(lambda *args, **kwargs: None)
@icontract.require(lambda nuts_state_in: nuts_state_in is not None, "nuts_state_in cannot be None")
@icontract.require(lambda rng_key_in: rng_key_in is not None, "rng_key_in cannot be None")
@icontract.require(lambda n_collect: n_collect is not None, "n_collect cannot be None")
@icontract.require(lambda n_discard: n_discard is not None, "n_discard cannot be None")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "RunNUTSTransitions all outputs must not be None")
def runnutstransitions(nuts_state_in: object, rng_key_in: object, n_collect: int, n_discard: int) -> tuple[object, object, object, object]:
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


def initializenutsstate_ffi(target: object, initial_positions: object, target_accept_p: float, seed: int) -> object:
    """FFI bridge to Rust implementation of InitializeNUTSState."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'initializenutsstate'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(target, initial_positions, target_accept_p, seed)

def runnutstransitions_ffi(nuts_state_in: object, rng_key_in: object, n_collect: int, n_discard: int) -> object:
    """FFI bridge to Rust implementation of RunNUTSTransitions."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'runnutstransitions'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(nuts_state_in, rng_key_in, n_collect, n_discard)