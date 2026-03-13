from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractMCMCTrace, AbstractRNGState, AbstractScalar, AbstractSignal

def witness_quantumproblemdefinition(family, event_shape, *args, **kwargs):
    """Ghost witness for prior init: QuantumProblemDefinition."""
    return AbstractDistribution(
        family=family,
        event_shape=event_shape,)


def witness_adiabaticquantumsampler(trace: AbstractMCMCTrace, target: AbstractDistribution, rng: AbstractRNGState) -> tuple[AbstractMCMCTrace, AbstractRNGState]:
    """Ghost witness for MCMC sampler: AdiabaticQuantumSampler."""
    if trace.param_dims != target.event_shape:
        raise ValueError(
            f"param_dims {trace.param_dims} vs event_shape {target.event_shape}"
        )
    return trace.step(accepted=True), rng.advance(n_draws=1)

def witness_solutionextraction(measurement_counts: AbstractArray, final_register: AbstractArray, num_solutions: AbstractArray) -> AbstractArray:
    """Ghost witness for SolutionExtraction."""
    result = AbstractArray(
        shape=measurement_counts.shape,
        dtype="float64",)

    return result
