from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_pinlikelihoodevaluator


@register_atom(witness_pinlikelihoodevaluator)
@icontract.require(lambda params: isinstance(params, dict), "params must be a dict")
@icontract.require(lambda B: isinstance(B, (float, int, np.number, np.ndarray)), "B must be numeric or an array")
@icontract.require(lambda S: isinstance(S, (float, int, np.number, np.ndarray)), "S must be numeric or an array")
@icontract.ensure(lambda result: isinstance(result, (float, int, np.number, np.floating)), "pinlikelihoodevaluator must return a numeric scalar")
def pinlikelihoodevaluator(params: dict[str, float], B: float | np.ndarray, S: float | np.ndarray) -> float:
    """Evaluates the log-likelihood (or likelihood) of observed data given model parameters and sufficient statistics B and S. This is a stateless, pure oracle computation with no side effects or persistent state.

    Args:
        params: must be valid parameter configuration for the Probability of Informed Trading (PIN) model
        B: buy order counts or volumes; non-negative
        S: sell order counts or volumes; non-negative

    Returns:
        real-valued likelihood score; -inf indicates zero likelihood"""
    raise NotImplementedError("Wire to original implementation")
