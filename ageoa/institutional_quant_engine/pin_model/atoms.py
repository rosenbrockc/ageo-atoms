from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_pinlikelihoodevaluation

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_pinlikelihoodevaluation)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda params: params is not None, "params cannot be None")
@icontract.require(lambda B: B is not None, "B cannot be None")
@icontract.require(lambda S: S is not None, "S cannot be None")
@icontract.ensure(lambda result: result is not None, "PinLikelihoodEvaluation output must not be None")
def pinlikelihoodevaluation(params: object, B: object, S: object) -> object:
    """Computes the likelihood of observed inputs under the provided parameterization.

    Args:
        params: Input data.
        B: Input data.
        S: Input data.

    Returns:
        Result data.
    """
    # PIN likelihood: params = [alpha, delta, mu, epsilon]
    alpha, delta, mu, epsilon = np.array(params, dtype=float)
    B = np.asarray(B, dtype=float)
    S = np.asarray(S, dtype=float)
    # Expected buys and sells under the model
    expected_B = alpha * delta * mu + epsilon
    expected_S = alpha * (1 - delta) * mu + epsilon
    # Sum of squared errors
    return float(np.sum((B - expected_B) ** 2 + (S - expected_S) ** 2))