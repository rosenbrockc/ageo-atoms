from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import *  # type: ignore[import-untyped]

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_sample_hawkes_event_trajectory)  # type: ignore[untyped-decorator, name-defined]
@icontract.require(lambda mu: isinstance(mu, (float, int, np.number)), "mu must be numeric")
@icontract.require(lambda alpha: isinstance(alpha, (float, int, np.number)), "alpha must be numeric")
@icontract.require(lambda beta: isinstance(beta, (float, int, np.number)), "beta must be numeric")
@icontract.require(lambda T: isinstance(T, (float, int, np.number)), "T must be numeric")
@icontract.ensure(lambda result: result is not None, "sample_hawkes_event_trajectory output must not be None")
def sample_hawkes_event_trajectory(mu: float, alpha: float, beta: float, T: float) -> np.ndarray:  # type: ignore[type-arg]
    """Simulates a Hawkes point-process realization over a finite horizon using the provided base intensity and excitation/decay parameters.

    Args:
        mu: mu >= 0
        alpha: alpha >= 0
        beta: beta > 0
        T: T > 0

    Returns:
        sorted ascending, each t in [0, T]
    """
    raise NotImplementedError("Wire to original implementation")
