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

@register_atom(witness_initializesampler)
@icontract.require(lambda target_accept_p: isinstance(target_accept_p, (float, int, np.number)), "target_accept_p must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "InitializeSampler output must not be None")
def initializesampler(target: Callable, initial_positions: ArrayLike, target_accept_p: float, seed: int) -> SamplerState:
    """Initializes the NUTS sampler state, including the random seed, initial positions, and target acceptance probability.

    Args:
        target: A function that computes the log-probability (and its gradient) of the target distribution.
        initial_positions: The starting point(s) for the MCMC chain(s).
        target_accept_p: The target acceptance probability for adapting the step size, typically between 0 and 1.
        seed: Seed for the pseudo-random number generator.

    Returns:
        An immutable object containing the initial position, PRNG key, and other sampler parameters.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_runsampler)
@icontract.require(lambda initial_sampler_state: initial_sampler_state is not None, "initial_sampler_state cannot be None")
@icontract.require(lambda n_collect: n_collect is not None, "n_collect cannot be None")
@icontract.require(lambda n_discard: n_discard is not None, "n_discard cannot be None")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "RunSampler all outputs must not be None")
def runsampler(initial_sampler_state: SamplerState, n_collect: int, n_discard: int) -> tuple[ArrayLike, SamplerState]:
    """Runs the NUTS MCMC sampler by repeatedly applying its transition kernel to generate a chain of samples, after discarding a specified number of burn-in steps.

    Args:
        initial_sampler_state: The initial state from which to start the sampling process.
        n_collect: The number of samples to collect and return.
        n_discard: The number of initial (burn-in) samples to discard.

    Returns:
        collected_samples: The collected samples from the posterior distribution.
        final_sampler_state: The final state of the sampler after the run.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for rust implementations."""

from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path


def initializesampler_ffi(target, initial_positions, target_accept_p, seed):
    """FFI bridge to Rust implementation of InitializeSampler."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'initializesampler'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(target, initial_positions, target_accept_p, seed)

def runsampler_ffi(initial_sampler_state, n_collect, n_discard):
    """FFI bridge to Rust implementation of RunSampler."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'runsampler'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(initial_sampler_state, n_collect, n_discard)
