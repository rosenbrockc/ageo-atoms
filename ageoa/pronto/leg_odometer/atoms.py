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

@register_atom(witness_velocitystatereadout)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "VelocityStateReadout all outputs must not be None")
def velocitystatereadout(state_in: object) -> tuple[object, object]:
    """Reads immutable velocity state-space components (body-frame velocity and its covariance) and returns the current velocity estimate.

    Args:
        state_in: Immutable snapshot; includes latent velocity mean xd_b_ and covariance vel_cov_.

    Returns:
        velocity: Derived from xd_b_.
        velocity_covariance: Directly from vel_cov_; no mutation.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_posequeryaccessors)  # type: ignore[untyped-decorator,name-defined]
@icontract.ensure(lambda result: result is not None, "PoseQueryAccessors output must not be None")
def posequeryaccessors() -> object:
    """Provides stateless pose-related query endpoints and no-op/placeholder call sites with no declared state reads or writes.


    Returns:
        No persistent state mutation in provided summaries.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for cpp implementations."""

# from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path


def _velocitystatereadout_ffi(state_in: object) -> object:
    """FFI bridge to C++ implementation of VelocityStateReadout."""
    _lib = ctypes.CDLL("./velocitystatereadout.so")
    _func_name = 'velocitystatereadout'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(state_in)

def _posequeryaccessors_ffi() -> object:
    """FFI bridge to C++ implementation of PoseQueryAccessors."""
    _lib = ctypes.CDLL("./posequeryaccessors.so")
    _func_name = 'posequeryaccessors'
    _func = _lib[_func_name]
    _func.restype = ctypes.c_void_p
    return _func()