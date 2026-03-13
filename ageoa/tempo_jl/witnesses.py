from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal
def witness_graph_time_scale_management(data: AbstractArray, *args, **kwargs):
    """Witness for high_precision_duration."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_high_precision_duration(*args, **kwargs):
    """Skeleton for witness_high_precision_duration."""
    return AbstractArray(shape=(), dtype='float64')