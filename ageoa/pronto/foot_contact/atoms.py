from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""

from collections.abc import Hashable, Mapping
from typing import Protocol

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_foot_sensing_state_update, witness_mode_snapshot_readout

import ctypes
import ctypes.util
from pathlib import Path


class _ModeStateLike(Protocol):
    mode: Hashable | None
    previous_mode: Hashable | None


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_foot_sensing_state_update)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda foot_sensing_state_in: foot_sensing_state_in is not None, "foot_sensing_state_in cannot be None")
@icontract.require(lambda foot_sensing_command: foot_sensing_command is not None, "foot_sensing_command cannot be None")
@icontract.ensure(lambda result: result is not None, "Foot Sensing State Update output must not be None")
def foot_sensing_state_update(foot_sensing_state_in: dict[str, bool], foot_sensing_command: dict[str, bool]) -> dict[str, bool]:
    """Applies a pure state transition for left/right foot sensing flags, returning a new sensing state object.

    Args:
        foot_sensing_state_in: Immutable input state snapshot.
        foot_sensing_command: Desired sensing values for this transition.

    Returns:
        New state object; no in-place mutation.
    """
    # Pure state transition: merge command into state
    new_state = dict(foot_sensing_state_in)
    new_state.update(foot_sensing_command)
    return new_state

@register_atom(witness_mode_snapshot_readout)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda mode_state_in: mode_state_in is not None, "mode_state_in cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "Mode Snapshot Readout all outputs must not be None")
def mode_snapshot_readout(
    mode_state_in: Mapping[str, Hashable | None] | _ModeStateLike,
) -> tuple[Hashable | None, Hashable | None]:
    """Reads current and previous mode from immutable classifier state and exposes them as explicit outputs.

    Args:
        mode_state_in: Read-only snapshot.

    Returns:
        mode: Current classifier mode.
        previous_mode: Previous classifier mode.
    """
    if isinstance(mode_state_in, Mapping):
        mode = mode_state_in.get('mode', None)
        previous_mode = mode_state_in.get('previous_mode', None)
    else:
        mode = getattr(mode_state_in, 'mode', mode_state_in)
        previous_mode = getattr(mode_state_in, 'previous_mode', mode_state_in)
    return (mode, previous_mode)


"""Auto-generated FFI bindings for cpp implementations."""

# removed duplicate future import

import ctypes
import ctypes.util
from pathlib import Path


def _foot_sensing_state_update_ffi(foot_sensing_state_in: ctypes.c_void_p, foot_sensing_command: ctypes.c_void_p) -> ctypes.c_void_p:
    """Wrapper that calls the C++ version of foot sensing state update. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./foot_sensing_state_update.so")
    _func_name = 'foot_sensing_state_update_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    _result: ctypes.c_void_p = _func(foot_sensing_state_in, foot_sensing_command)
    return _result

def _mode_snapshot_readout_ffi(mode_state_in: ctypes.c_void_p) -> ctypes.c_void_p:
    """Wrapper that calls the C++ version of mode snapshot readout. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./mode_snapshot_readout.so")
    _func_name = 'mode_snapshot_readout_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _result: ctypes.c_void_p = _func(mode_state_in)
    return _result
    return _func(mode_state_in)
