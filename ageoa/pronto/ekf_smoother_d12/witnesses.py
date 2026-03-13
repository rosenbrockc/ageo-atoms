from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal
import networkx as nx  # type: ignore


def witness_stateestimatorinit() -> AbstractArray:
    """Ghost witness for StateEstimatorInit."""
    return None