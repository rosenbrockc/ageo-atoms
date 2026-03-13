from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal
import networkx as nx  # type: ignore


def witness_riskaversioninit(risk_aversion: AbstractArray) -> AbstractArray:
    """Ghost witness for RiskAversionInit."""
    result = AbstractArray(
        shape=risk_aversion.shape,
        dtype="float64",)
    
    return result

def witness_optimalexecutiontrajectory(risk_aversion: AbstractArray, total_shares: AbstractArray, days: AbstractArray) -> AbstractArray:
    """Ghost witness for OptimalExecutionTrajectory."""
    result = AbstractArray(
        shape=risk_aversion.shape,
        dtype="float64",)
    
    return result