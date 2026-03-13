from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_computeoptimaltrajectory

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_computeoptimaltrajectory)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda total_shares: total_shares is not None, "total_shares cannot be None")
@icontract.require(lambda days: days is not None, "days cannot be None")
@icontract.require(lambda risk_aversion: risk_aversion is not None, "risk_aversion cannot be None")
@icontract.ensure(lambda result: result is not None, "ComputeOptimalTrajectory output must not be None")
def computeoptimaltrajectory(total_shares: float, days: int, risk_aversion: float) -> object:
    """Computes the optimal share-execution trajectory over the planning horizon given risk preference.

    Args:
        total_shares: Typically non-negative.
        days: Typically positive.
        risk_aversion: Risk preference parameter.

    Returns:
        Represents the optimized schedule for the provided horizon.
    """
    raise NotImplementedError("Wire to original implementation")