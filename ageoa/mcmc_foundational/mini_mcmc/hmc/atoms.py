"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

import ctypes
import ctypes.util
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

# Witness functions should be imported from the generated witnesses module
witness_initializesamplerandrngstate: object = object()
witness_applyhmcacceptrejectkernel: object = object()
witness_collectposteriorsamples: object = object()
def initializesamplerandrngstate(target_log_prob_fn: Any, initial_positions: Any, step_size: float, n_leapfrog: int, seed: int) -> tuple[Any, Any, Any]:
F = TypeVar("F", bound=Callable[..., Any])


def _typed_register_atom(witness: object) -> Callable[[F], F]:
    return cast(Callable[[F], F], register_atom(witness))

@register_atom(witness_initializesamplerandrngstate)
@icontract.require(lambda step_size: isinstance(step_size, (float, int, np.number)), "step_size must be numeric")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "InitializeSamplerAndRNGState all outputs must not be None")
def initializesamplerandrngstate(target_log_prob_fn, initial_positions, step_size: float, n_leapfrog: int, seed: int):
    """Construct immutable HMC sampler state and initialize explicit RNG/PRNGKey state.

    Args:
        target_log_prob_fn: pure oracle; no persistent mutation
        initial_positions: finite numeric values
        step_size: > 0
@_typed_register_atom(witness_integratehamiltonianproposal)
        seed: used to create RNG state / PRNGKey

    Returns:
def integratehamiltonianproposal(state_in: Any, log_prob_oracle: Any) -> Any:
        rng_state: threaded explicitly as input/output
        log_prob_oracle: stateless oracle
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_integratehamiltonianproposal)
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.require(lambda log_prob_oracle: log_prob_oracle is not None, "log_prob_oracle cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "IntegrateHamiltonianProposal output must not be None")
def integratehamiltonianproposal(state_in, log_prob_oracle):
    """Generate deterministic HMC proposal by leapfrog integration over position and momenta using mass matrix and log-probability gradients.
@_typed_register_atom(witness_applyhmcacceptrejectkernel)
    Args:
        state_in: immutable input state
        log_prob_oracle: pure oracle evaluation

def applyhmcacceptrejectkernel(state_in: Any, proposal_state: Any, rng_state_in: Any) -> tuple[Any, Any, Any]:
        new object; no in-place updates
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_applyhmcacceptrejectkernel)
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.require(lambda proposal_state: proposal_state is not None, "proposal_state cannot be None")
@icontract.require(lambda rng_state_in: rng_state_in is not None, "rng_state_in cannot be None")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "ApplyHMCAcceptRejectKernel all outputs must not be None")
def applyhmcacceptrejectkernel(state_in, proposal_state, rng_state_in):
    """Perform one HMC transition kernel step: consume current state, evaluate proposal, and return accepted/rejected next state with updated gradient.

    Args:
        state_in: immutable chain state
@_typed_register_atom(witness_collectposteriorsamples)
        rng_state_in: must be explicitly consumed/split

    Returns:
        state_out: new immutable state
        rng_state_out: advanced key/state
def collectposteriorsamples(state_in: Any, rng_state_in: Any, n_collect: int, n_discard: int) -> tuple[Any, Any, Any]:
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_collectposteriorsamples)
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.require(lambda rng_state_in: rng_state_in is not None, "rng_state_in cannot be None")
@icontract.require(lambda n_collect: n_collect is not None, "n_collect cannot be None")
@icontract.require(lambda n_discard: n_discard is not None, "n_discard cannot be None")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "CollectPosteriorSamples all outputs must not be None")
def collectposteriorsamples(state_in, rng_state_in, n_collect: int, n_discard: int):
    """Run burn-in and collection loops, repeatedly applying the HMC kernel and accumulating sample trace.

    Args:
        state_in: immutable chain state at loop start
        rng_state_in: explicit stochastic state flow
        n_collect: >= 0
        n_discard: >= 0

    Returns:
        samples_trace: immutable aggregated trace
        final_state: latest immutable chain state
        final_rng_state: latest explicit stochastic state
    """
    raise NotImplementedError("Wire to original implementation")

def initializesamplerandrngstate_ffi(target_log_prob_fn: Any, initial_positions: Any, step_size: Any, n_leapfrog: Any, seed: Any) -> Any:
"""Auto-generated FFI bindings for rust implementations."""

# from __future__ import annotations
    _func_name = "initializesamplerandrngstate"
import ctypes
import ctypes.util
from pathlib import Path


def integratehamiltonianproposal_ffi(state_in: Any, log_prob_oracle: Any) -> Any:
    """FFI bridge to Rust implementation of InitializeSamplerAndRNGState."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = "integratehamiltonianproposal"
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(target_log_prob_fn, initial_positions, step_size, n_leapfrog, seed)

def applyhmcacceptrejectkernel_ffi(state_in: Any, proposal_state: Any, rng_state_in: Any) -> Any:
    """FFI bridge to Rust implementation of IntegrateHamiltonianProposal."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = "applyhmcacceptrejectkernel"
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(state_in, log_prob_oracle)

def collectposteriorsamples_ffi(state_in: Any, rng_state_in: Any, n_collect: Any, n_discard: Any) -> Any:
    """FFI bridge to Rust implementation of ApplyHMCAcceptRejectKernel."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = "collectposteriorsamples"
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(state_in, proposal_state, rng_state_in)

def collectposteriorsamples_ffi(state_in, rng_state_in, n_collect, n_discard):
    """FFI bridge to Rust implementation of CollectPosteriorSamples."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'collectposteriorsamples'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(state_in, rng_state_in, n_collect, n_discard)