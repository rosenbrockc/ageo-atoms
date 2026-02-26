"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

from typing import Callable
import numpy as np

import icontract
from ageoa.ghost.registry import register_atom

import ctypes
import ctypes.util
from pathlib import Path


from .witnesses import (
    witness_initializehmckernelstate,
    witness_initializesamplerrng,
    witness_hamiltoniantransitionkernel,
    witness_collectposteriorchain,
)

@register_atom(witness_initializehmckernelstate)
@icontract.require(lambda step_size: isinstance(step_size, (float, int, np.number)), "step_size must be numeric")
@icontract.ensure(lambda result: all(r is not None for r in result), "InitializeHMCKernelState all outputs must not be None")
def initializehmckernelstate(target: Callable[[np.ndarray], float], initial_positions: np.ndarray, step_size: float, n_leapfrog: int) -> tuple[np.ndarray, np.ndarray]:
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

@register_atom(witness_initializesamplerrng)
@icontract.require(lambda seed: isinstance(seed, int), "seed must be an int")
@icontract.ensure(lambda result: result is not None, "InitializeSamplerRNG output must not be None")
def initializesamplerrng(seed: int) -> np.ndarray:
    """Initialize explicit stochastic state for pure functional sampling. RNG state is threaded across all transitions and never mutated in place.

    Args:
        seed: deterministic reproducibility seed

    Returns:
        immutable key to be split per transition
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_hamiltoniantransitionkernel)
@icontract.require(lambda state_in: isinstance(state_in, np.ndarray), "state_in must be np.ndarray")
@icontract.require(lambda kernel_spec: kernel_spec is not None, "kernel_spec cannot be None")
@icontract.require(lambda prng_key_in: prng_key_in is not None, "prng_key_in cannot be None")
@icontract.require(lambda logp_oracle: logp_oracle is not None, "logp_oracle cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "HamiltonianTransitionKernel all outputs must not be None")
def hamiltoniantransitionkernel(state_in: np.ndarray, kernel_spec: np.ndarray, prng_key_in: np.ndarray, logp_oracle: Callable[[np.ndarray], float]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
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

@register_atom(witness_collectposteriorchain)
@icontract.require(lambda n_collect: isinstance(n_collect, int), "n_collect must be an int")
@icontract.require(lambda n_discard: n_discard is not None, "n_discard cannot be None")
@icontract.require(lambda chain_state_0: chain_state_0 is not None, "chain_state_0 cannot be None")
@icontract.require(lambda kernel_spec: kernel_spec is not None, "kernel_spec cannot be None")
@icontract.require(lambda prng_key_state: prng_key_state is not None, "prng_key_state cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "CollectPosteriorChain all outputs must not be None")
def collectposteriorchain(n_collect: int, n_discard: int, chain_state_0: np.ndarray, kernel_spec: np.ndarray, prng_key_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _initializehmckernelstate_ffi(target: Callable[[np.ndarray], float], initial_positions: np.ndarray, step_size: float, n_leapfrog: int) -> tuple[np.ndarray, np.ndarray]:
    """FFI bridge to Rust implementation of InitializeHMCKernelState."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'initializehmckernelstate'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(target, initial_positions, step_size, n_leapfrog)

def _initializesamplerrng_ffi(seed: int) -> np.ndarray:
    """FFI bridge to Rust implementation of InitializeSamplerRNG."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'initializesamplerrng'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(seed)

def _hamiltoniantransitionkernel_ffi(state_in: np.ndarray, kernel_spec: np.ndarray, prng_key_in: np.ndarray, logp_oracle: Callable[[np.ndarray], float]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """FFI bridge to Rust implementation of HamiltonianTransitionKernel."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'hamiltoniantransitionkernel'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(state_in, kernel_spec, prng_key_in, logp_oracle)

def _collectposteriorchain_ffi(n_collect: int, n_discard: int, chain_state_0: np.ndarray, kernel_spec: np.ndarray, prng_key_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """FFI bridge to Rust implementation of CollectPosteriorChain."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'collectposteriorchain'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(n_collect, n_discard, chain_state_0, kernel_spec, prng_key_state)