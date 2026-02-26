"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

from typing import Callable
import numpy as np

import icontract
from ageoa.ghost.registry import register_atom

import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_initialize_sampler)
@icontract.require(lambda target: isinstance(target, (float, int, np.number)), "target must be numeric")
@icontract.require(lambda target_accept_p: isinstance(target_accept_p, (float, int, np.number)), "target_accept_p must be numeric")
@icontract.ensure(lambda result: result is not None, "initialize_sampler output must not be None")
def initialize_sampler(target: Callable[[np.ndarray], float], initial_positions: np.ndarray, target_accept_p: float, seed: int) -> np.ndarray:
    """Sets up the initial state for the NUTS sampler, including the PRNG seed, target log-probability function, initial positions, and tuning parameters.

    Args:
        target: The log-probability density function of the target distribution. Must be a pure function (oracle).
        initial_positions: Starting points for the MCMC chains.
        target_accept_p: The target acceptance probability for step size adaptation.
        seed: Seed for the pseudo-random number generator.

    Returns:
        An immutable data structure containing the complete initial state of the sampler, including the PRNGKey, step size, and initial positions.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_run_mcmc_sampler)
@icontract.require(lambda sampler_state_in: sampler_state_in is not None, "sampler_state_in cannot be None")
@icontract.require(lambda n_collect: n_collect is not None, "n_collect cannot be None")
@icontract.require(lambda n_discard: n_discard is not None, "n_discard cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "run_mcmc_sampler all outputs must not be None")
def run_mcmc_sampler(sampler_state_in: np.ndarray, n_collect: int, n_discard: int) -> tuple[np.ndarray, np.ndarray]:
    """Executes the NUTS MCMC algorithm for a given number of warm-up and collection steps, producing posterior samples.

    Args:
        sampler_state_in: The current state of the NUTS sampler.
        n_collect: Number of posterior samples to generate and collect.
        n_discard: Number of warm-up (burn-in) samples to discard before collection.

    Returns:
        posterior_samples: Collected samples from the posterior distribution.
        final_sampler_state: The final state of the sampler after the run, including the updated PRNGKey.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for rust implementations."""

from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path


def _initialize_sampler_ffi(target, initial_positions, target_accept_p, seed):
    """FFI bridge to Rust implementation of initialize_sampler."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'initialize_sampler'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(target, initial_positions, target_accept_p, seed)

def _run_mcmc_sampler_ffi(sampler_state_in, n_collect, n_discard):
    """FFI bridge to Rust implementation of run_mcmc_sampler."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'run_mcmc_sampler'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(sampler_state_in, n_collect, n_discard)
