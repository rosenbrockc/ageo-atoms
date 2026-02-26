"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations
from typing import Callable
import numpy as np

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

import ctypes
import ctypes.util
from pathlib import Path


from .witnesses import (
    witness_initializehmcstate,
    witness_leapfrogproposalkernel,
    witness_metropolishmctransition,
    witness_runsamplingloop,
)

@register_atom(witness_initializehmcstate)
@icontract.require(lambda initial_positions: isinstance(initial_positions, (float, int, np.number)), "initial_positions must be numeric")
@icontract.require(lambda step_size: isinstance(step_size, (float, int, np.number)), "step_size must be numeric")
@icontract.ensure(lambda result: all(r is not None for r in result), "InitializeHMCState all outputs must not be None")
def initializehmcstate(target: Callable[[np.ndarray], float], initial_positions: np.ndarray, step_size: float, n_leapfrog: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
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

@register_atom(witness_leapfrogproposalkernel)
@icontract.require(lambda proposal_state_in: proposal_state_in is not None, "proposal_state_in cannot be None")
@icontract.require(lambda kernel_static: kernel_static is not None, "kernel_static cannot be None")
@icontract.require(lambda log_prob_oracle: log_prob_oracle is not None, "log_prob_oracle cannot be None")
@icontract.ensure(lambda result: result is not None, "LeapfrogProposalKernel output must not be None")
def leapfrogproposalkernel(proposal_state_in: np.ndarray, kernel_static: np.ndarray, log_prob_oracle: Callable[[np.ndarray], float]) -> np.ndarray:
    """Pure Hamiltonian proposal transition: consumes current position/momenta and returns proposed position/momenta plus refreshed log-probability gradient.

    Args:
        proposal_state_in: contains pos, momenta, gradient, logp
        kernel_static: uses step_size and n_leapfrog
        log_prob_oracle: pure evaluator for logp/gradient

    Returns:
        new pos, new momenta, updated logp, updated gradient
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_metropolishmctransition)
@icontract.require(lambda chain_state_in: chain_state_in is not None, "chain_state_in cannot be None")
@icontract.require(lambda kernel_static: kernel_static is not None, "kernel_static cannot be None")
@icontract.require(lambda proposal_state_out: proposal_state_out is not None, "proposal_state_out cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "MetropolisHMCTransition all outputs must not be None")
def metropolishmctransition(chain_state_in: np.ndarray, kernel_static: np.ndarray, proposal_state_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

@register_atom(witness_runsamplingloop)
@icontract.require(lambda hmc_state_in: hmc_state_in is not None, "hmc_state_in cannot be None")
@icontract.require(lambda n_collect: n_collect is not None, "n_collect cannot be None")
@icontract.require(lambda n_discard: n_discard is not None, "n_discard cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "RunSamplingLoop all outputs must not be None")
def runsamplingloop(hmc_state_in: np.ndarray, n_collect: int, n_discard: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _initializehmcstate_ffi(target: Callable[[np.ndarray], float], initial_positions: np.ndarray, step_size: float, n_leapfrog: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """FFI bridge to Rust implementation of InitializeHMCState."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'initializehmcstate'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(target, initial_positions, step_size, n_leapfrog, seed)

def _leapfrogproposalkernel_ffi(proposal_state_in: np.ndarray, kernel_static: np.ndarray, log_prob_oracle: Callable[[np.ndarray], float]) -> np.ndarray:
    """FFI bridge to Rust implementation of LeapfrogProposalKernel."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'leapfrogproposalkernel'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(proposal_state_in, kernel_static, log_prob_oracle)

def _metropolishmctransition_ffi(chain_state_in: np.ndarray, kernel_static: np.ndarray, proposal_state_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FFI bridge to Rust implementation of MetropolisHMCTransition."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'metropolishmctransition'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(chain_state_in, kernel_static, proposal_state_out)

def _runsamplingloop_ffi(hmc_state_in: np.ndarray, n_collect: int, n_discard: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FFI bridge to Rust implementation of RunSamplingLoop."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'runsamplingloop'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(hmc_state_in, n_collect, n_discard)