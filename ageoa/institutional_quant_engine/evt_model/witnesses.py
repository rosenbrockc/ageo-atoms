from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_fit_gpd_tail(returns, *args, **kwargs):
    result = AbstractArray(
        shape=returns.shape,
        dtype="float64",)
    
    return result