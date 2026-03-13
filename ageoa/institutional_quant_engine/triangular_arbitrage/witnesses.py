from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_detect_triangular_arbitrage(rates, *args, **kwargs):
    result = AbstractArray(
        shape=rates.shape,
        dtype="float64",)
    
    return result
