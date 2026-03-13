from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_templatefeaturecomputation(hc: AbstractArray) -> AbstractArray:
    """Ghost witness for TemplateFeatureComputation."""
    result = AbstractArray(
        shape=hc.shape,
        dtype="float64",
    )
    return result
