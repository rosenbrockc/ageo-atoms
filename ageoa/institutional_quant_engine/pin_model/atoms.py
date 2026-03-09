from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import *  # type: ignore[import-untyped]

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
    raise NotImplementedError("Wire to original implementation")
