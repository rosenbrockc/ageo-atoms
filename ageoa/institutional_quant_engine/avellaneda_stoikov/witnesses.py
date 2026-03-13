from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_initializemarketmakerstate(s0: AbstractScalar, inventory: AbstractScalar) -> AbstractArray:
    """Ghost witness for InitializeMarketMakerState."""
    result = AbstractArray(
        shape=s0.shape,
        dtype="float64",)
    
    return result

def witness_computeinventoryadjustedquotes(state_model: AbstractArray) -> AbstractArray:
    """Ghost witness for ComputeInventoryAdjustedQuotes."""
    result = AbstractArray(
        shape=state_model.shape,
        dtype="float64",)
    
    return result