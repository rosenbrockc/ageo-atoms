from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_apccoreevaluation(x: AbstractArray) -> AbstractArray:
    """Ghost witness for ApcCoreEvaluation."""
    result = AbstractArray(
        shape=x.shape,
        dtype="float64",
    )
    return result
