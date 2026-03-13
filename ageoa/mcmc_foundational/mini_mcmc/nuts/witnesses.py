from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_nuts_recursive_tree_build(direction_val, step_size, log_slice_variable, initial_hmc_state, log_prob_oracle, integrator_fn, tree_depth, *args, **kwargs):
    """Ghost witness for nuts_recursive_tree_build."""
    return AbstractArray(
        shape=(1,),
        dtype="float64",)

def witness_run_mcmc_sampler(sampler_state_in: AbstractArray, n_collect: AbstractArray, n_discard: AbstractArray) -> AbstractArray:
    """Ghost witness for run_mcmc_sampler."""
    result = AbstractArray(
        shape=sampler_state_in.shape,
        dtype="float64",)
    return result
