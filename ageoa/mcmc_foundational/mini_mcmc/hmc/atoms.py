"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations
from typing import Any, Callable
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


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_initializehmcstate)  # type: ignore[name-defined, untyped-decorator]
@icontract.require(lambda initial_positions: isinstance(initial_positions, (float, int, np.number)), "initial_positions must be numeric")
@icontract.require(lambda step_size: isinstance(step_size, (float, int, np.number)), "step_size must be numeric")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "InitializeHMCState all outputs must not be None")
def initializehmcstate(target: Callable[..., Any], initial_positions: Any, step_size: float, n_leapfrog: int, seed: int) -> tuple[Any, object]:
    """Construct immutable HMC state and static kernel parameters, including explicit RNG state (PRNGKey/seed-derived state).

    Args:
        target: stateless density/gradient evaluator
        initial_positions: shape fixed for chain state
        step_size: > 0
        n_leapfrog: >= 1
        seed: optional; when provided initializes RNG state

    Returns:
        hmc_state_0: contains positions, logp_current, gradient, rng_state, trace
        kernel_static: contains target, step_size, n_leapfrog, mass_matrix (implicit or explicit)
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_leapfrogproposalkernel)  # type: ignore[name-defined, untyped-decorator]
@icontract.require(lambda proposal_state_in: proposal_state_in is not None, "proposal_state_in cannot be None")
@icontract.require(lambda kernel_static: kernel_static is not None, "kernel_static cannot be None")
@icontract.require(lambda log_prob_oracle: log_prob_oracle is not None, "log_prob_oracle cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "LeapfrogProposalKernel output must not be None")
def leapfrogproposalkernel(proposal_state_in: Any, kernel_static: object, log_prob_oracle: Callable[..., Any]) -> Any:
    """Pure Hamiltonian proposal transition: consumes current position/momenta and returns proposed position/momenta plus refreshed log-probability gradient.

    Args:
        proposal_state_in: contains pos, momenta, gradient, logp
        kernel_static: uses step_size and n_leapfrog
        log_prob_oracle: pure evaluator for logp/gradient

    Returns:
        new pos, new momenta, updated logp, updated gradient
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_metropolishmctransition)  # type: ignore[name-defined, untyped-decorator]
@icontract.require(lambda chain_state_in: chain_state_in is not None, "chain_state_in cannot be None")
@icontract.require(lambda kernel_static: kernel_static is not None, "kernel_static cannot be None")
@icontract.require(lambda proposal_state_out: proposal_state_out is not None, "proposal_state_out cannot be None")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "MetropolisHMCTransition all outputs must not be None")
def metropolishmctransition(chain_state_in: Any, kernel_static: object, proposal_state_out: Any) -> tuple[Any, object]:
    """Single pure HMC kernel step: samples fresh momenta from RNG state, invokes leapfrog proposal, performs accept/reject, and returns new immutable chain state.

    Args:
        chain_state_in: contains positions, logp_current, gradient, rng_state, trace
        kernel_static: contains target/integrator parameters
        proposal_state_out: from leapfrog proposal

    Returns:
        chain_state_out: updated positions/logp_current/gradient/rng_state
        transition_stats: accept flag, acceptance prob, Hamiltonian delta
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_runsamplingloop)  # type: ignore[name-defined, untyped-decorator]
@icontract.require(lambda hmc_state_in: hmc_state_in is not None, "hmc_state_in cannot be None")
@icontract.require(lambda n_collect: n_collect is not None, "n_collect cannot be None")
@icontract.require(lambda n_discard: n_discard is not None, "n_discard cannot be None")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "RunSamplingLoop all outputs must not be None")
def runsamplingloop(hmc_state_in: Any, n_collect: int, n_discard: int) -> tuple[Any, object, Any]:
    """Drive warmup/discard and collection iterations by repeatedly applying the HMC transition kernel; produces trace and collected samples.

    Args:
        hmc_state_in: immutable state threaded across iterations
        n_collect: >= 0
        n_discard: >= 0

    Returns:
        samples: length n_collect
        trace: diagnostics over all iterations
        hmc_state_out: final positions/logp_current/gradient/rng_state
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for rust implementations."""

from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path


def initializehmcstate_ffi(target: Any, initial_positions: Any, step_size: Any, n_leapfrog: Any, seed: Any) -> Any:
    """FFI bridge to Rust implementation of InitializeHMCState."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'initializehmcstate'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(target, initial_positions, step_size, n_leapfrog, seed)

def leapfrogproposalkernel_ffi(proposal_state_in: Any, kernel_static: Any, log_prob_oracle: Any) -> Any:
    """FFI bridge to Rust implementation of LeapfrogProposalKernel."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'leapfrogproposalkernel'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(proposal_state_in, kernel_static, log_prob_oracle)

def metropolishmctransition_ffi(chain_state_in: Any, kernel_static: Any, proposal_state_out: Any) -> Any:
    """FFI bridge to Rust implementation of MetropolisHMCTransition."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'metropolishmctransition'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(chain_state_in, kernel_static, proposal_state_out)

def runsamplingloop_ffi(hmc_state_in: Any, n_collect: Any, n_discard: Any) -> Any:
    """FFI bridge to Rust implementation of RunSamplingLoop."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'runsamplingloop'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(hmc_state_in, n_collect, n_discard)