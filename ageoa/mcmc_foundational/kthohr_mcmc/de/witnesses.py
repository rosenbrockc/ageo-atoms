from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_build_de_transition_kernel(target_log_kernel: AbstractArray, extra_arg: AbstractArray) -> AbstractArray:
    """Ghost witness for build_de_transition_kernel."""
    result = AbstractArray(
        shape=target_log_kernel.shape,
        dtype="float64",)
    
    return result