from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal
import networkx as nx  # type: ignore


def witness_hrppipelinerun(data: AbstractArray) -> AbstractArray:
    """Ghost witness for HRPPipelineRun."""
    result = AbstractArray(
        shape=data.shape,
        dtype="float64",)
    
    return result