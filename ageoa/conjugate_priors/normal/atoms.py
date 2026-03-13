from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_normal_gamma_posterior_update

from juliacall import Main as jl  # type: ignore[import-untyped]


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_normal_gamma_posterior_update)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda prior: prior is not None, "prior cannot be None")
@icontract.require(lambda ss: ss is not None, "ss cannot be None")
@icontract.ensure(lambda result: result is not None, "normal_gamma_posterior_update output must not be None")
def normal_gamma_posterior_update(prior: object, ss: object) -> object:
    """Computes a closed-form Normal-Gamma posterior from a Normal-Gamma prior and sufficient statistics as a pure, immutable conjugate update.

    Args:
        prior: Valid Normal-Gamma parameters (e.g., positive precision/shape/scale terms).
        ss: Contains the moments/count terms required by the conjugate posterior formula.

    Returns:
        Returned as a new immutable object; input prior is not mutated.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for julia implementations."""

# from __future__ import annotations
from juliacall import Main as jl  # type: ignore[import-untyped]
from juliacall import Main as jl

def _normal_gamma_posterior_update_ffi(prior: object, ss: object) -> object:
    """Wrapper that calls the Julia version of normal gamma posterior update. Passes arguments through and returns the result."""
    return jl.eval("normal_gamma_posterior_update(prior, ss)")
