from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_quantum_mwis_solver(data: AbstractArray) -> AbstractArray:
    """Shape-and-type check for quantum mwis solver. Returns output metadata without running the real computation."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_greedy_lattice_mapping(data: AbstractArray) -> AbstractArray:
    """Shape-and-type check for greedy lattice mapping. Returns output metadata without running the real computation."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)