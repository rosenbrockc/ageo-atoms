from __future__ import annotations
from typing import Tuple
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractMCMCTrace, AbstractRNGState, AbstractScalar, AbstractSignal

def witness_initializehmckernelstate(*args, **kwargs):
    """Ghost witness for InitializeHMCKernelState."""
    kernel_spec = AbstractArray(shape=("K",), dtype="float64")
    chain_state = AbstractMCMCTrace(n_samples=0, n_chains=1, param_dims=(1,), warmup_steps=0)
    return (chain_state, kernel_spec)


def witness_initializesamplerrng(*args, **kwargs):
    """Ghost witness for InitializeSamplerRNG."""
    return AbstractRNGState(seed=0, consumed=0, is_split=False)


def witness_hamiltoniantransitionkernel(*args, **kwargs):
    """Ghost witness for HamiltonianTransitionKernel."""
    state_out = AbstractMCMCTrace(
        n_samples=0,
        n_chains=1,
        param_dims=(1,),
        warmup_steps=0,
    )
    rng_out = AbstractRNGState(seed=0, consumed=1, is_split=True)
    stats = AbstractArray(shape=("3",), dtype="float64")
    return (state_out, rng_out, stats)


def witness_collectposteriorchain(*args, **kwargs):
    """Ghost witness for CollectPosteriorChain."""
    samples = AbstractArray(shape=("N", "D"), dtype="float64")
    final_state = AbstractMCMCTrace(
        n_samples=0,
        n_chains=1,
        param_dims=(1,),
        warmup_steps=0,
    )
    final_rng = AbstractRNGState(seed=0, consumed=0, is_split=False)
    trace = AbstractMCMCTrace(
        n_samples=0, n_chains=1, param_dims=(1,), warmup_steps=0,
    )
    return (samples, final_state, final_rng, trace)
