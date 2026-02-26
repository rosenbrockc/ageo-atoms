"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom

from juliacall import Main as jl


# Witness functions should be imported from the generated witnesses module

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
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for julia implementations."""

from __future__ import annotations

from juliacall import Main as jl


def _evaluate_log_probability_density_ffi(q, z):
    """FFI bridge to Julia implementation of evaluate_log_probability_density."""
    return jl.eval("evaluate_log_probability_density(q, z)")
