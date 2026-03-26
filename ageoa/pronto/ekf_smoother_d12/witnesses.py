from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal


def witness_stateestimatorinit() -> AbstractArray:
    """Shape-and-type check for state estimator init. Returns output metadata without running the real computation."""
    return AbstractArray(shape=(1,), dtype="float64")
