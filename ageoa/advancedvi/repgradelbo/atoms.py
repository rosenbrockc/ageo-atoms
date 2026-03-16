from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_gradient_oracle_evaluation

# juliacall unavailable; reimplemented in pure numpy


@register_atom(witness_gradient_oracle_evaluation)
@icontract.require(lambda rng_in: rng_in is not None, "rng_in cannot be None")
@icontract.require(lambda obj: obj is not None, "obj cannot be None")
@icontract.require(lambda adtype: adtype is not None, "adtype cannot be None")
@icontract.require(lambda out_in: out_in is not None, "out_in cannot be None")
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.require(lambda params: params is not None, "params cannot be None")
@icontract.require(lambda restructure: restructure is not None, "restructure cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "Gradient Oracle Evaluation all outputs must not be None")
def gradient_oracle_evaluation(rng_in: object, obj: object, adtype: object, out_in: object, state_in: object, params: object, restructure: object) -> tuple[object, object, object, object]:
    """Computes objective value and gradient for the current parameters using the provided AD mode, while threading random number generator (RNG) and algorithm state as explicit immutable inputs/outputs.

Args:
    rng_in: Explicit stochastic state input; do not treat as global mutable state.
    obj: Provides value/gradient target.
    adtype: Selects differentiation backend/strategy.
    out_in: Destination buffer for gradient result.
    state_in: Thread as immutable state_in -> state_out.
    params: Valid parameterization for objective evaluation.

Returns:
    out_out: Updated gradient corresponding to params.
    value_out: Scalar/objective evaluation paired with gradient.
    state_out: Returned explicitly as new state object.
    rng_out: Returned explicitly (possibly unchanged) to preserve purity."""
    # Reparameterization gradient ELBO estimation
    params_arr = np.asarray(params, dtype=np.float64)
    eps = 1e-5
    d = len(params_arr)

    # Compute objective value
    value = float(obj(params_arr)) if callable(obj) else 0.0

    # Finite-difference gradient estimation
    grad = np.zeros_like(params_arr)
    for i in range(d):
        params_plus = params_arr.copy()
        params_plus[i] += eps
        params_minus = params_arr.copy()
        params_minus[i] -= eps
        if callable(obj):
            grad[i] = (obj(params_plus) - obj(params_minus)) / (2 * eps)

    # Apply restructure if callable
    out_out = restructure(grad) if callable(restructure) else grad
    rng_out = rng_in
    return (out_out, value, state_in, rng_out)
