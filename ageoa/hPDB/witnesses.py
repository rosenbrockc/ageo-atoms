from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal
import networkx as nx  # type: ignore


def witness_iterate_pdb_atoms(element: AbstractArray) -> AbstractArray:
    """Ghost witness for iterate_pdb_atoms."""
    result = AbstractArray(
        shape=element.shape,
        dtype="float64",
    )
    return result

def witness_iterate_pdb_residues(element: AbstractArray) -> AbstractArray:
    """Ghost witness for iterate_pdb_residues."""
    result = AbstractArray(
        shape=element.shape,
        dtype="float64",
    )
    return result
