"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

from juliacall import Main as jl  # type: ignore[import-untyped]


witness_optimizationlooporchestration: object = object()

@register_atom(witness_optimizationlooporchestration)  # type: ignore[untyped-decorator]
@icontract.require(lambda algorithm: algorithm is not None, "algorithm cannot be None")
@icontract.require(lambda max_iter: max_iter is not None, "max_iter cannot be None")
@icontract.require(lambda prob: prob is not None, "prob cannot be None")
@icontract.require(lambda q_init: q_init is not None, "q_init cannot be None")
@icontract.require(lambda rng_state_in: rng_state_in is not None, "rng_state_in cannot be None")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "OptimizationLoopOrchestration all outputs must not be None")
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
        opt_trace: May include objective values and convergence metadata.
        rng_state_out: Advanced/split state after any stochastic operations.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for julia implementations."""

from __future__ import annotations

from juliacall import Main as jl


def optimizationlooporchestration_ffi(algorithm: object, max_iter: int, prob: object, q_init: object, rng_state_in: object) -> object:
    """FFI bridge to Julia implementation of OptimizationLoopOrchestration."""
    return jl.eval("optimizationlooporchestration(algorithm, max_iter, prob, q_init, rng_state_in)")