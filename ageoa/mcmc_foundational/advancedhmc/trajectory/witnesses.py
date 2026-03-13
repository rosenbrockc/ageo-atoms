from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_buildnutstree(rng: AbstractArray, hamiltonian: AbstractArray, extra_arg: AbstractArray, start_state: AbstractArray, direction: AbstractScalar, tree_depth: AbstractScalar, initial_energy: AbstractScalar) -> AbstractArray:
    """Ghost witness for BuildNutsTree."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",)
    
    return result

def witness_nutstransitionkernel(rng: AbstractArray, hamiltonian: AbstractArray, initial_state: AbstractArray, trajectory_params: AbstractArray) -> AbstractArray:
    """Ghost witness for NutsTransitionKernel."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",)
    
    return result