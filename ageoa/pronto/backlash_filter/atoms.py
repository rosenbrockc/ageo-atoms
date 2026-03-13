from __future__ import annotations
from typing import Any
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_initializebacklashfilterstate, witness_updatealphaparameter, witness_updatecrossingtimemaximum

import ctypes
import ctypes.util
from pathlib import Path

BacklashFilterState = np.ndarray
def witness_initializebacklashfilterstate(*args, **kwargs): pass
def witness_updatealphaparameter(*args, **kwargs): pass
def witness_updatecrossingtimemaximum(*args, **kwargs): pass

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_initializebacklashfilterstate)
@icontract.require(lambda: True, "no preconditions")
@icontract.ensure(lambda result: result is not None, "InitializeBacklashFilterState output must not be None")
def initializebacklashfilterstate() -> BacklashFilterState:
    """Create the initial immutable state object for the filter parameters.


@register_atom(witness_updatealphaparameter)  # type: ignore[untyped-decorator]
        Initialized with constructor/default values.

Returns:
    BacklashFilterState: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_updatealphaparameter)
@icontract.require(lambda alpha_in: isinstance(alpha_in, (float, int, np.number)), "alpha_in must be numeric")
@icontract.ensure(lambda result: result is not None, "UpdateAlphaParameter output must not be None")
def updatealphaparameter(state_in: BacklashFilterState, alpha_in: float) -> BacklashFilterState:
    """Produce a new filter state with an updated alpha parameter.

    Args:
        state_in: Immutable input state.
        alpha_in: Finite scalar.

@register_atom(witness_updatecrossingtimemaximum)  # type: ignore[untyped-decorator]
        Same as state_in, except alpha_ = alpha_in.

    Returns:
        BacklashFilterState: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_updatecrossingtimemaximum)
@icontract.require(lambda t_crossing_max_in: isinstance(t_crossing_max_in, (float, int, np.number)), "t_crossing_max_in must be numeric")
@icontract.ensure(lambda result: result is not None, "UpdateCrossingTimeMaximum output must not be None")
def updatecrossingtimemaximum(state_in: BacklashFilterState, t_crossing_max_in: float) -> BacklashFilterState:
    """Produce a new filter state with an updated maximum crossing time.

    Args:
        state_in: Immutable input state.
        t_crossing_max_in: Finite scalar, typically non-negative.

    Returns:
        Same as state_in, except t_crossing_max_ = t_crossing_max_in.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for cpp implementations."""

def _initializebacklashfilterstate_ffi() -> np.ndarray:
    """FFI bridge to C++ implementation of InitializeBacklashFilterState."""
    raise NotImplementedError("Wire to original implementation")

def _updatealphaparameter_ffi(state_in: np.ndarray, alpha_in: float) -> np.ndarray:
    """FFI bridge to C++ implementation of UpdateAlphaParameter."""
    raise NotImplementedError("Wire to original implementation")

def _updatecrossingtimemaximum_ffi(state_in: np.ndarray, t_crossing_max_in: float) -> np.ndarray:
    """FFI bridge to C++ implementation of UpdateCrossingTimeMaximum."""
    raise NotImplementedError("Wire to original implementation")