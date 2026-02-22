"""Ghost witnesses."""

from ageoa.ghost.abstract import AbstractArray

def witness_quantum_mwis_solver(data: AbstractArray) -> AbstractArray:
    """Witness for quantum_mwis_solver."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_greedy_lattice_mapping(data: AbstractArray) -> AbstractArray:
    """Witness for greedy_lattice_mapping."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)
