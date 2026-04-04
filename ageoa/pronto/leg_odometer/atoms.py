from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""

from collections.abc import Mapping
from typing import Protocol

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_posequeryaccessors, witness_velocitystatereadout

import ctypes
import ctypes.util
from pathlib import Path


class _VelocityStateLike(Protocol):
    xd_b: np.ndarray
    vel_cov: np.ndarray


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_velocitystatereadout)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "VelocityStateReadout all outputs must not be None")
def velocitystatereadout(state_in: Mapping[str, np.ndarray] | _VelocityStateLike) -> tuple[np.ndarray, np.ndarray]:
    """Reads immutable velocity state-space components (body-frame velocity and its covariance) and returns the current velocity estimate.

    Args:
        state_in: Immutable snapshot; includes latent velocity mean xd_b_ and covariance vel_cov_.

    Returns:
        velocity: Derived from xd_b_.
        velocity_covariance: Directly from vel_cov_; no mutation.
    """
    if isinstance(state_in, Mapping):
        velocity = state_in.get('xd_b', state_in.get('velocity', np.zeros(3)))
        velocity_covariance = state_in.get('vel_cov', state_in.get('velocity_covariance', np.eye(3)))
    else:
        velocity = getattr(state_in, 'xd_b', np.zeros(3))
        velocity_covariance = getattr(state_in, 'vel_cov', np.eye(3))
    return (np.asarray(velocity, dtype=np.float64), np.asarray(velocity_covariance, dtype=np.float64))

@register_atom(witness_posequeryaccessors)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda: True, "no preconditions for zero-parameter initializer")
@icontract.ensure(lambda result: result is not None, "PoseQueryAccessors output must not be None")
def posequeryaccessors() -> dict[str, np.ndarray]:
    """Provides stateless pose-related query endpoints and no-op/placeholder call sites with no declared state reads or writes.

    Returns:
        Pose query accessor object with no persistent state mutation.
    """
    # Stateless accessor: return a dict with identity pose fields
    return {
        'position': np.zeros(3, dtype=np.float64),
        'orientation': np.eye(3, dtype=np.float64),
    }


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
