"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom

from ageoa.hftbacktest.witnesses import witness_initialize_glft_state, witness_update_glft_coefficients, witness_evaluate_spread_conditions  # type: ignore
@register_atom(witness_initialize_glft_state)
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "initialize_glft_state all outputs must not be None")
def initialize_glft_state() -> tuple[float, float]:
    """Initializes the state for the GLFT model coefficients.


    Returns:
        initial_c1: Represents the initial value of the state variable 'c1'.
        initial_c2: Represents the initial value of the state variable 'c2'.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_update_glft_coefficients)
@icontract.require(lambda last_c1: isinstance(last_c1, (float, int, np.number)), "last_c1 must be numeric")
@icontract.require(lambda last_c2: isinstance(last_c2, (float, int, np.number)), "last_c2 must be numeric")
@icontract.require(lambda xi: isinstance(xi, (float, int, np.number)), "xi must be numeric")
@icontract.require(lambda gamma: isinstance(gamma, (float, int, np.number)), "gamma must be numeric")
@icontract.require(lambda delta: isinstance(delta, (float, int, np.number)), "delta must be numeric")
@icontract.require(lambda A: isinstance(A, (float, int, np.number)), "A must be numeric")
@icontract.require(lambda k: isinstance(k, (float, int, np.number)), "k must be numeric")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "update_glft_coefficients all outputs must not be None")
def update_glft_coefficients(last_c1: float, last_c2: float, xi: float, gamma: float, delta: float, A: float, k: float) -> tuple[float, float]:
    """Updates the GLFT model coefficients based on market parameters and the previous state. This embodies the core state transition.

    Args:
        last_c1: The previous state of the 'c1' coefficient.
        last_c2: The previous state of the 'c2' coefficient.
        xi: Input data.
        gamma: Input data.
        delta: Input data.
        A: Input data.
        k: Input data.

    Returns:
        next_c1: The updated state of the 'c1' coefficient.
        next_c2: The updated state of the 'c2' coefficient.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_evaluate_spread_conditions)
@icontract.require(lambda c1: isinstance(c1, (float, int, np.number)), "c1 must be numeric")
@icontract.require(lambda c2: isinstance(c2, (float, int, np.number)), "c2 must be numeric")
@icontract.require(lambda delta: isinstance(delta, (float, int, np.number)), "delta must be numeric")
@icontract.require(lambda volatility: isinstance(volatility, (float, int, np.number)), "volatility must be numeric")
@icontract.require(lambda adj1: isinstance(adj1, (float, int, np.number)), "adj1 must be numeric")
@icontract.require(lambda threshold: isinstance(threshold, (float, int, np.number)), "threshold must be numeric")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "evaluate_spread_conditions all outputs must not be None")
def evaluate_spread_conditions(c1: float, c2: float, delta: float, volatility: float, adj1: float, threshold: float) -> tuple[float, bool]:
    """Computes the half-spread from the current state and checks if it meets a validity condition against the c1 coefficient.

    Args:
        c1: Current value of the 'c1' coefficient.
        c2: Current value of the 'c2' coefficient.
        delta: Input data.
        volatility: Input data.
        adj1: Input data.
        threshold: Input data.

    Returns:
        half_spread: Result data.
        is_valid_ratio: Result data.
    """
    raise NotImplementedError("Wire to original implementation")