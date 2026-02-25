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


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_metropolishastingstransitionkernel)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda temper_val: isinstance(temper_val, (float, int, np.number)), "temper_val must be numeric")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "MetropolisHastingsTransitionKernel all outputs must not be None")
def metropolishastingstransitionkernel(temper_val: float, target_log_kernel: object, rng_key_in: object) -> tuple[object, object]:
    """Runs one pure Metropolis-Hastings transition: builds proposal-related terms, consumes oracle log-kernel evaluations, computes acceptance, and returns a new sample/state object.

    Args:
        temper_val: Finite positive temperature/scaling value.
        target_log_kernel: Pure function handle used for target log-kernel evaluation.
        rng_key_in: Thread explicitly for purity when stochastic proposal/acceptance draws are used.

    Returns:
        mh_step_state_out: Immutable result containing next sample and acceptance-related values.
        rng_key_out: New key/state after any random draws.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_targetlogkerneloracle)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda temper_val: isinstance(temper_val, (float, int, np.number)), "temper_val must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "TargetLogKernelOracle output must not be None")
def targetlogkerneloracle(state_candidate: object, temper_val: float) -> float:
    """Stateless oracle that evaluates target log-kernel values for candidate/current states used by MH acceptance logic.

    Args:
        state_candidate: Candidate (or current) state point to score.
        temper_val: Same temperature/scaling context used by the MH kernel.

    Returns:
        Finite log-density/log-kernel value.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for cpp implementations."""

# from __future__ import annotations  # already imported at top of file

import ctypes
import ctypes.util
from pathlib import Path


def metropolishastingstransitionkernel_ffi(temper_val: object, target_log_kernel: object, rng_key_in: object) -> object:
    """FFI bridge to C++ implementation of MetropolisHastingsTransitionKernel."""
    _lib = ctypes.CDLL("./metropolishastingstransitionkernel.so")
    _func_name = 'metropolishastingstransitionkernel'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(temper_val, target_log_kernel, rng_key_in)

def targetlogkerneloracle_ffi(state_candidate: object, temper_val: object) -> object:
    """FFI bridge to C++ implementation of TargetLogKernelOracle."""
    _lib = ctypes.CDLL("./targetlogkerneloracle.so")
    _func_name = 'targetlogkerneloracle'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(state_candidate, temper_val)