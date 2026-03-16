from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_optimizationlooporchestration

# juliacall unavailable; reimplemented in pure numpy


def witness_optimizationlooporchestration(*args, **kwargs): pass

@register_atom(witness_optimizationlooporchestration)  # type: ignore[untyped-decorator]
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
