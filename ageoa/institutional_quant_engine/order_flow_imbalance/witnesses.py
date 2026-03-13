from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_orderflowimbalanceevaluation(row: AbstractArray, prev_row: AbstractArray) -> AbstractArray:
    """Ghost witness for OrderFlowImbalanceEvaluation."""
    result = AbstractArray(
        shape=row.shape,
        dtype="float64",)
    
    return result