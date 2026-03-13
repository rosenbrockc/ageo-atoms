from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractMCMCTrace, AbstractRNGState, AbstractScalar, AbstractSignal
import networkx as nx  # type: ignore


def witness_hawkesprocesssimulator(trace: AbstractMCMCTrace, target: AbstractDistribution, rng: AbstractRNGState) -> tuple[AbstractMCMCTrace, AbstractRNGState]:
    """Shape-and-type check for mcmc sampler: hawkes process simulator. Returns output metadata without running the real computation."""
    if trace.param_dims != target.event_shape:
        raise ValueError(
            f"param_dims {trace.param_dims} vs event_shape {target.event_shape}"
        )
        
    return trace.step(accepted=True), rng.advance(n_draws=1)