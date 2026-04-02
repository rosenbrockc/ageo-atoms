from __future__ import annotations
"""Backlash filter state wrappers for the Pronto family."""


import numpy as np
from numpy.typing import NDArray

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import (
    witness_initializebacklashfilterstate,
    witness_updatealphaparameter,
    witness_updatecrossingtimemaximum,
)

BacklashFilterState = NDArray[np.float64]
_STATE_SHAPE = (4,)
_DEFAULT_STATE = np.array([0.5, 1.0, 0.0, 0.0], dtype=np.float64)


def _is_backlash_filter_state(value: object) -> bool:
    """Return whether ``value`` matches the local backlash filter state layout."""
    if not isinstance(value, np.ndarray):
        return False
    return value.ndim == 1 and value.shape == _STATE_SHAPE and np.isfinite(value).all()


def _is_finite_scalar(value: object) -> bool:
    """Return whether ``value`` is a finite numeric scalar."""
    return isinstance(value, (float, int, np.number)) and np.isfinite(value)

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_initializebacklashfilterstate)
@icontract.require(lambda: True, "no preconditions for zero-parameter initializer")
@icontract.ensure(lambda result: result is not None, "InitializeBacklashFilterState output must not be None")
def initializebacklashfilterstate() -> BacklashFilterState:
    """Create the initial immutable state object for the filter parameters.

    Initialized with constructor/default values.

    Returns:
        BacklashFilterState: The initial filter state.
    """
    # State layout: [alpha_, t_crossing_max_, last_output_, last_crossing_time_]
    return _DEFAULT_STATE.copy()

@register_atom(witness_updatealphaparameter)
@icontract.ensure(lambda result: _is_backlash_filter_state(result), "updated state must preserve the local backlash filter state layout")
@icontract.require(lambda state_in: _is_backlash_filter_state(state_in), "state_in must be a finite 1D backlash filter state with four entries")
@icontract.require(lambda alpha_in: _is_finite_scalar(alpha_in), "alpha_in must be a finite numeric scalar")
def updatealphaparameter(state_in: BacklashFilterState, alpha_in: float) -> BacklashFilterState:
    """Produce a new filter state with an updated alpha parameter.

    Args:
        state_in: Immutable input state.
        alpha_in: Finite scalar.

    Returns:
        BacklashFilterState: Same as state_in, except alpha_ = alpha_in.
    """
    out = np.asarray(state_in, dtype=np.float64).copy()
    out[0] = float(alpha_in)
    return out

@register_atom(witness_updatecrossingtimemaximum)
@icontract.ensure(lambda result: _is_backlash_filter_state(result), "updated state must preserve the local backlash filter state layout")
@icontract.require(lambda state_in: _is_backlash_filter_state(state_in), "state_in must be a finite 1D backlash filter state with four entries")
@icontract.require(lambda t_crossing_max_in: _is_finite_scalar(t_crossing_max_in), "t_crossing_max_in must be a finite numeric scalar")
def updatecrossingtimemaximum(state_in: BacklashFilterState, t_crossing_max_in: float) -> BacklashFilterState:
    """Produce a new filter state with an updated maximum crossing time.

    Args:
        state_in: Immutable input state.
        t_crossing_max_in: Finite scalar, typically non-negative.

    Returns:
        Same as state_in, except t_crossing_max_ = t_crossing_max_in.
    """
    out = np.asarray(state_in, dtype=np.float64).copy()
    out[1] = float(t_crossing_max_in)
    return out
