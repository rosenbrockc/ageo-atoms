from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_posequeryaccessors, witness_velocitystatereadout

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
@icontract.require(lambda: True, "no preconditions for zero-parameter initializer")
@icontract.ensure(lambda result: result is not None, "PoseQueryAccessors output must not be None")
def posequeryaccessors() -> object:
    """Provides stateless pose-related query endpoints and no-op/placeholder call sites with no declared state reads or writes.

    Returns:
        Pose query accessor object with no persistent state mutation.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for cpp implementations."""


import ctypes
import ctypes.util
from pathlib import Path


def _velocitystatereadout_ffi(state_in: object) -> object:
    """Wrapper that calls the C++ version of velocity state readout. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./velocitystatereadout.so")
    _func_name = 'velocitystatereadout_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(state_in)

def _posequeryaccessors_ffi() -> object:
    """Wrapper that calls the C++ version of pose query accessors. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./posequeryaccessors.so")
    _func_name = 'posequeryaccessors_prime'
    _func = _lib[_func_name]
    _func.restype = ctypes.c_void_p
    return _func()