from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_buildrmhmctransitionkernel(target_log_kernel: AbstractArray, extra_arg: AbstractArray, tensor_fn: AbstractArray, initial_state: AbstractArray) -> AbstractArray:
    """Ghost witness for BuildRMHMCTransitionKernel."""
    result = AbstractArray(
        shape=target_log_kernel.shape,
        dtype="float64",)
    
    return result