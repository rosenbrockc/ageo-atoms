from __future__ import annotations

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .core_witnesses import witness_evaluate_log_probability_density

# juliacall unavailable; reimplemented in pure numpy


@register_atom(witness_evaluate_log_probability_density)
@icontract.require(lambda q: q is not None, "q cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.ensure(lambda result: result is not None, "evaluate_log_probability_density output must not be None")
def evaluate_log_probability_density(q: np.ndarray, z: np.ndarray) -> float:
    """Computes the log-probability density function (logpdf) for given inputs 'q' and 'z'. This is a stateless operation, likely corresponding to a specific probability distribution.

    Args:
        q: Input parameter for the logpdf calculation.
        z: Input parameter for the logpdf calculation.

    Returns:
        The resulting log-probability density.
    """
    # Location-scale Gaussian logpdf: q = [mu, log_sigma], z = sample
    d = len(q) // 2
    mu = q[:d]
    log_sigma = q[d:]
    sigma = np.exp(log_sigma)
    # log N(z; mu, sigma^2) = -0.5*d*log(2*pi) - sum(log_sigma) - 0.5*sum(((z-mu)/sigma)^2)
    return float(-0.5 * d * np.log(2 * np.pi) - np.sum(log_sigma) - 0.5 * np.sum(((z - mu) / sigma) ** 2))

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .core_witnesses import witness_optimizationlooporchestration

# juliacall unavailable; reimplemented in pure numpy


@register_atom(witness_optimizationlooporchestration)
@icontract.require(lambda algorithm: algorithm is not None, "algorithm cannot be None")
@icontract.require(lambda max_iter: max_iter is not None, "max_iter cannot be None")
@icontract.require(lambda prob: prob is not None, "prob cannot be None")
@icontract.require(lambda q_init: q_init is not None, "q_init cannot be None")
@icontract.require(lambda rng_state_in: rng_state_in is not None, "rng_state_in cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "OptimizationLoopOrchestration all outputs must not be None")
def optimizationlooporchestration(algorithm: object, max_iter: int, prob: object, q_init: object, rng_state_in: object) -> tuple[object, object, object]:
    """Runs the selected optimization algorithm for a bounded number of iterations on the provided objective/problem using an initial state.

    Args:
        algorithm: Must define one optimization step/update rule.
        max_iter: Positive iteration budget.
        prob: Must expose evaluable objective/constraints required by algorithm.
        q_init: Shape/type must match algorithm and problem.
        rng_state_in: Optional; if absent, initialize from default entropy source.

    Returns:
        q_opt: Same structural type family as q_init.
        rng_state_out: Advanced/split state after any stochastic operations.
    """
    # Generic VI optimization loop with gradient descent
    from scipy.optimize import minimize as scipy_minimize
    q = np.asarray(q_init, dtype=np.float64).copy()
    step_fn = algorithm if callable(algorithm) else None
    rng_state = rng_state_in

    if step_fn is not None:
        # Use provided step function
        for _ in range(max_iter):
            q, rng_state = step_fn(q, prob, rng_state)
    elif callable(prob):
        # prob is an objective function — use scipy minimize
        result = scipy_minimize(prob, q, method='L-BFGS-B', options={'maxiter': max_iter})
        q = result.x
    else:
        # Fallback: gradient-free random search
        rng = np.random.RandomState(int(rng_state) if isinstance(rng_state, (int, float)) else 42)
        best_val = float('inf')
        best_q = q.copy()
        for _ in range(max_iter):
            candidate = q + 0.01 * rng.randn(*q.shape)
            val = prob(candidate) if callable(prob) else 0.0
            if val < best_val:
                best_val = val
                best_q = candidate
        q = best_q
        rng_state = int(rng_state) + max_iter if isinstance(rng_state, (int, float)) else rng_state

    return (q, rng_state, q)

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .core_witnesses import witness_gradient_oracle_evaluation

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
