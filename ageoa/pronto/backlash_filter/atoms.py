from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""
from typing import Any


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_initializebacklashfilterstate, witness_updatealphaparameter, witness_updatecrossingtimemaximum

import ctypes
import ctypes.util
from pathlib import Path

BacklashFilterState = np.ndarray

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_initializebacklashfilterstate)
@icontract.ensure(lambda result: result is not None, "InitializeBacklashFilterState output must not be None")
def initializebacklashfilterstate() -> BacklashFilterState:
    """Create the initial immutable state object for the filter parameters.

    Initialized with constructor/default values.

    Returns:
        BacklashFilterState: The initial filter state.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_updatealphaparameter)
@icontract.ensure(lambda result: result is not None, "UpdateAlphaParameter output must not be None")
@icontract.require(lambda state_in: isinstance(state_in, np.ndarray), "state_in must be an ndarray")
@icontract.require(lambda alpha_in: isinstance(alpha_in, (float, int, np.number)), "alpha_in must be numeric")
def updatealphaparameter(state_in: BacklashFilterState, alpha_in: float) -> BacklashFilterState:
    """Produce a new filter state with an updated alpha parameter.

    Args:
        state_in: Immutable input state.
        alpha_in: Finite scalar.

    Returns:
        BacklashFilterState: Same as state_in, except alpha_ = alpha_in.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_updatecrossingtimemaximum)
@icontract.ensure(lambda result: result is not None, "UpdateCrossingTimeMaximum output must not be None")
@icontract.require(lambda state_in: isinstance(state_in, np.ndarray), "state_in must be an ndarray")
@icontract.require(lambda t_crossing_max_in: isinstance(t_crossing_max_in, (float, int, np.number)), "t_crossing_max_in must be numeric")
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
    """Wrapper that calls the C++ version of initialize backlash filter state. Passes arguments through and returns the result."""
    raise NotImplementedError("Wire to original implementation")

def _updatealphaparameter_ffi(state_in: np.ndarray, alpha_in: float) -> np.ndarray:
    """Wrapper that calls the C++ version of update alpha parameter. Passes arguments through and returns the result."""
    raise NotImplementedError("Wire to original implementation")

def _updatecrossingtimemaximum_ffi(state_in: np.ndarray, t_crossing_max_in: float) -> np.ndarray:
    """Wrapper that calls the C++ version of update crossing time maximum. Passes arguments through and returns the result."""
    raise NotImplementedError("Wire to original implementation")