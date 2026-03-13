from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_propagate_supply_shock(adjacency, *args, **kwargs):
    result = AbstractArray(
        shape=adjacency.shape,
        dtype="float64",)
    
    return result