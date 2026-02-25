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

@register_atom(lambda *args, **kwargs: True)
@icontract.require(lambda step_size: isinstance(step_size, (float, int, np.number)), "step_size must be numeric")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "InitializeHMCKernelState all outputs must not be None")
def initializehmckernelstate(target: object, initial_positions: object, step_size: float, n_leapfrog: int) -> tuple[object, object]:
    """Construct immutable HMC kernel/state specification from target log-density, initial latent position, and integrator hyperparameters. Includes explicit latent state, cached log-probability/gradient slots, and mass-matrix assumptions.

    Args:
        target: pure log_prob/gradient-capable density oracle
        initial_positions: finite numeric latent state
        step_size: step_size > 0
        n_leapfrog: n_leapfrog >= 1

    Returns:
        kernel_spec: contains step_size, n_leapfrog, mass_matrix (explicit, immutable)
        chain_state_0: contains position, logp_current, gradient, momenta placeholder, trace init
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(lambda *args, **kwargs: True)
@icontract.require(lambda seed: seed is not None, "seed cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "InitializeSamplerRNG output must not be None")
def initializesamplerrng(seed: int) -> object:
    """Initialize explicit stochastic state for pure functional sampling. RNG state is threaded across all transitions and never mutated in place.

    Args:
        seed: deterministic reproducibility seed

    Returns:
        immutable key to be split per transition
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(lambda *args, **kwargs: True)
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.require(lambda kernel_spec: kernel_spec is not None, "kernel_spec cannot be None")
@icontract.require(lambda prng_key_in: prng_key_in is not None, "prng_key_in cannot be None")
@icontract.require(lambda logp_oracle: logp_oracle is not None, "logp_oracle cannot be None")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "HamiltonianTransitionKernel all outputs must not be None")
def hamiltoniantransitionkernel(state_in: object, kernel_spec: object, prng_key_in: object, logp_oracle: object) -> tuple[object, object, dict[str, object]]:
    """Perform one pure HMC transition: generate/consume momenta, run leapfrog integrator proposal, evaluate acceptance, and return a brand-new chain state plus updated RNG key.

    Args:
        state_in: includes position, logp_current, gradient, trace
        kernel_spec: includes step_size, n_leapfrog, mass_matrix
        prng_key_in: must be explicitly provided and split
        logp_oracle: pure likelihood/log_prob evaluation

    Returns:
        state_out: new immutable state with updated position/logp_current/gradient/momenta/trace
        prng_key_out: new key after stochastic draws
        transition_stats: contains accept/reject info and energy diagnostics
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(lambda *args, **kwargs: True)
@icontract.require(lambda n_collect: n_collect is not None, "n_collect cannot be None")
@icontract.require(lambda n_discard: n_discard is not None, "n_discard cannot be None")
@icontract.require(lambda chain_state_0: chain_state_0 is not None, "chain_state_0 cannot be None")
@icontract.require(lambda kernel_spec: kernel_spec is not None, "kernel_spec cannot be None")
@icontract.require(lambda prng_key_state: prng_key_state is not None, "prng_key_state cannot be None")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "CollectPosteriorChain all outputs must not be None")
def collectposteriorchain(n_collect: int, n_discard: int, chain_state_0: object, kernel_spec: object, prng_key_state: object) -> tuple[object, object, object, object]:
    """Drive warmup/discard and collection loops by repeatedly applying the transition kernel; optionally emit progress while preserving pure state threading.

    Args:
        n_collect: n_collect >= 1
        n_discard: n_discard >= 0
        chain_state_0: initial immutable chain state
        kernel_spec: transition hyperparameters
        prng_key_state: explicit RNG flow through all iterations

    Returns:
        samples: collected posterior positions
        final_state: immutable terminal state
        final_prng_key: terminal RNG state
        chain_trace: acceptance and trajectory diagnostics
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for rust implementations."""

# duplicate future import removed

import ctypes
import ctypes.util
from pathlib import Path


def initializehmckernelstate_ffi(target: object, initial_positions: object, step_size: object, n_leapfrog: object) -> object:
    """FFI bridge to Rust implementation of InitializeHMCKernelState."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'initializehmckernelstate'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(target, initial_positions, step_size, n_leapfrog)

def initializesamplerrng_ffi(seed: object) -> object:
    """FFI bridge to Rust implementation of InitializeSamplerRNG."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'initializesamplerrng'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(seed)

def hamiltoniantransitionkernel_ffi(state_in: object, kernel_spec: object, prng_key_in: object, logp_oracle: object) -> object:
    """FFI bridge to Rust implementation of HamiltonianTransitionKernel."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'hamiltoniantransitionkernel'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(state_in, kernel_spec, prng_key_in, logp_oracle)

def collectposteriorchain_ffi(n_collect: object, n_discard: object, chain_state_0: object, kernel_spec: object, prng_key_state: object) -> object:
    """FFI bridge to Rust implementation of CollectPosteriorChain."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'collectposteriorchain'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(n_collect, n_discard, chain_state_0, kernel_spec, prng_key_state)