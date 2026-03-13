from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal, AbstractMCMCTrace, AbstractRNGState


def witness_continuousmultivariatesampler(trace: AbstractMCMCTrace, target: AbstractDistribution, rng: AbstractRNGState) -> tuple[AbstractMCMCTrace, AbstractRNGState]:
    """Ghost witness for MCMC sampler: ContinuousMultivariateSampler."""
    if trace.param_dims != target.event_shape:
        raise ValueError(
            f"param_dims {trace.param_dims} vs event_shape {target.event_shape}"
        )
    return trace.step(accepted=True), rng.advance(n_draws=1)

def witness_discreteeventsampler(trace: AbstractMCMCTrace, target: AbstractDistribution, rng: AbstractRNGState) -> tuple[AbstractMCMCTrace, AbstractRNGState]:
    """Ghost witness for MCMC sampler: DiscreteEventSampler."""
    if trace.param_dims != target.event_shape:
        raise ValueError(
            f"param_dims {trace.param_dims} vs event_shape {target.event_shape}"
        )
    return trace.step(accepted=True), rng.advance(n_draws=1)

def witness_combinatoricssampler(x: AbstractArray, axis: AbstractArray, a: AbstractArray, size: AbstractArray, replace: AbstractArray, p: AbstractArray) -> AbstractArray:
    """Ghost witness for CombinatoricsSampler."""
    result = AbstractArray(
        shape=x.shape,
        dtype="float64",
    )
    return result
