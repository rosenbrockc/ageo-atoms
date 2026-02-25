"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
# from ageoa.ghost.registry import register_atom

import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module

# @register_atom(witness_constructrandomwalkmetropoliskernel)
@icontract.require(lambda target_log_kernel: isinstance(target_log_kernel, (float, int, np.number)), "target_log_kernel must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "ConstructRandomWalkMetropolisKernel output must not be None")
def constructrandomwalkmetropoliskernel(target_log_kernel: object) -> object:
    """Builds a pure Random-Walk Metropolis-Hastings transition kernel from a target log-density oracle, with explicit immutable state and RNG threading.

    Args:
        target_log_kernel: Stateless/pure log-density oracle; no persistent state mutation.

    Returns:
        Pure MCMC kernel; chain_state is immutable (e.g., latent sample and cached log_prob), PRNGKey must be explicitly split/threaded.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for cpp implementations."""

from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path
from typing import cast


def constructrandomwalkmetropoliskernel_ffi(target_log_kernel: ctypes.c_void_p) -> ctypes.c_void_p:
    """FFI bridge to C++ implementation of ConstructRandomWalkMetropolisKernel."""
    _lib = ctypes.CDLL("./constructrandomwalkmetropoliskernel.so")
    _func_name = 'constructrandomwalkmetropoliskernel'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    return cast(ctypes.c_void_p, _func(target_log_kernel))
    return _func(target_log_kernel)