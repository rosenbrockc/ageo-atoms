"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

from typing import Callable
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

@register_atom(witness_mala_proposal_adjustment)
@icontract.require(lambda step_size: isinstance(step_size, (float, int, np.number)), "step_size must be numeric")
@icontract.ensure(lambda result: result is not None, "mala_proposal_adjustment output must not be None")
def mala_proposal_adjustment(step_size: float, vals_bound: np.ndarray, mala_mean_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Calculates the adjustment term for a Metropolis-Adjusted Langevin Algorithm (MALA) proposal. This typically involves the gradient of the log-posterior (via mala_mean_fn), which guides the proposal distribution.

    Args:
        step_size: Controls the magnitude of the Langevin dynamics step.
        vals_bound: Boundary conditions or constraints on the proposal values.
        mala_mean_fn: A function (oracle) that computes the mean of the proposal distribution, typically based on the gradient of the target log-probability.

    Returns:
        The calculated adjustment to be used in the MALA proposal.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for cpp implementations."""

from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path


def _mala_proposal_adjustment_ffi(step_size, vals_bound, mala_mean_fn):
    """FFI bridge to C++ implementation of mala_proposal_adjustment."""
    _lib = ctypes.CDLL("./mala_proposal_adjustment.so")
    _func_name = atom.method_names[0] if atom.method_names else 'mala_proposal_adjustment'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(step_size, vals_bound, mala_mean_fn)
