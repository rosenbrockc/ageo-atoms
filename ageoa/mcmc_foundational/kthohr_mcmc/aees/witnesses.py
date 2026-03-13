from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_metropolishastingstransitionkernel(temper_val: AbstractScalar, target_log_kernel: AbstractArray, rng_key_in: AbstractArray) -> tuple[AbstractArray, AbstractArray]:
    """Ghost witness for MetropolisHastingsTransitionKernel."""
    mh_step_state_out = AbstractArray(shape=rng_key_in.shape, dtype="float64")
    rng_key_out = AbstractArray(shape=rng_key_in.shape, dtype="float64")
    return mh_step_state_out, rng_key_out

def witness_targetlogkerneloracle(state_candidate: AbstractArray, temper_val: AbstractArray) -> AbstractScalar:
    """Ghost witness for TargetLogKernelOracle."""
    return AbstractScalar(dtype="float64")
