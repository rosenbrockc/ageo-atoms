from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal
import networkx as nx  # type: ignore


def witness_marketmakerstateinit(s0: AbstractArray, inventory: AbstractArray) -> AbstractArray:
    """Ghost witness for MarketMakerStateInit."""
    result = AbstractArray(
        shape=s0.shape,
        dtype="float64",)
    
    return result

def witness_optimalquotecalculation(gamma: AbstractArray, k: AbstractArray, q: AbstractArray, s: AbstractArray, sigma: AbstractArray) -> AbstractArray:
    """Ghost witness for OptimalQuoteCalculation."""
    result = AbstractArray(
        shape=gamma.shape,
        dtype="float64",)
    
    return result