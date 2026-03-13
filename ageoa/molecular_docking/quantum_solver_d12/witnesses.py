from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractMCMCTrace, AbstractRNGState, AbstractScalar, AbstractSignal
import networkx as nx  # type: ignore


def witness_quantumsolverorchestrator(graph: AbstractArray, coordinates_layout: AbstractArray, num_sol: AbstractArray, display_info: AbstractArray) -> AbstractArray:
    """Ghost witness for QuantumSolverOrchestrator."""
    result = AbstractArray(
        shape=graph.shape,
        dtype="float64",)
    
    return result

def witness_interactionboundscomputer(register_coord: AbstractArray, graph: AbstractArray) -> AbstractArray:
    """Ghost witness for InteractionBoundsComputer."""
    result = AbstractArray(
        shape=register_coord.shape,
        dtype="float64",)
    
    return result

def witness_adiabaticpulseassembler(register: AbstractArray, parameters: AbstractArray) -> AbstractArray:
    """Ghost witness for AdiabaticPulseAssembler."""
    result = AbstractArray(
        shape=register.shape,
        dtype="float64",)
    
    return result

def witness_quantumcircuitsampler(trace: AbstractMCMCTrace, target: AbstractDistribution, rng: AbstractRNGState) -> tuple[AbstractMCMCTrace, AbstractRNGState]:
    """Ghost witness for MCMC sampler: QuantumCircuitSampler."""
    if trace.param_dims != target.event_shape:
        raise ValueError(
            f"param_dims {trace.param_dims} vs event_shape {target.event_shape}"
        )
        
    return trace.step(accepted=True), rng.advance(n_draws=1)

def witness_quantumsolutionextractor(count_dist: AbstractArray, register: AbstractArray, num_solutions: AbstractArray) -> AbstractArray:
    """Ghost witness for QuantumSolutionExtractor."""
    result = AbstractArray(
        shape=count_dist.shape,
        dtype="float64",)
    
    return result