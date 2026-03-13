from __future__ import annotations
from typing import Callable
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_buildnutstree, witness_nutstransitionkernel

from juliacall import Main as jl


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_buildnutstree)
@icontract.require(lambda initial_energy: isinstance(initial_energy, (float, int, np.number)), "initial_energy must be numeric")
@icontract.ensure(lambda result: result is not None, "BuildNutsTree output must not be None")
def buildnutstree(rng: np.ndarray, hamiltonian: Callable[[np.ndarray], float], start_state: np.ndarray, direction: int, tree_depth: int, initial_energy: float) -> np.ndarray:
    """Recursively builds a binary tree of states for a Hamiltonian Monte Carlo trajectory. It explores the trajectory in both forward and backward directions, doubling the number of states at each step, and terminates when the trajectory starts to turn back on itself (the No-U-Turn criterion).

    Args:
        rng: JAX-style random number generator key for stochastic operations.
        hamiltonian: An oracle that provides the energy and its gradient.
        start_state: The state at the beginning of the trajectory segment to be built.
        direction: The direction of integration (+1 for forward, -1 for backward).
        tree_depth: The current recursion depth of the tree construction.
        initial_energy: The energy of the initial state of the entire trajectory.

    Returns:
        A binary tree containing the states of the trajectory segment.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_nutstransitionkernel)
@icontract.require(lambda rng: rng is not None, "rng cannot be None")
@icontract.require(lambda hamiltonian: hamiltonian is not None, "hamiltonian cannot be None")
@icontract.require(lambda initial_state: initial_state is not None, "initial_state cannot be None")
@icontract.require(lambda trajectory_params: trajectory_params is not None, "trajectory_params cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "NutsTransitionKernel all outputs must not be None")
def nutstransitionkernel(rng: np.ndarray, hamiltonian: Callable[[np.ndarray], float], initial_state: np.ndarray, trajectory_params: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Orchestrates a single No-U-Turn Sampler (NUTS) transition. It initializes a trajectory, builds a proposal tree using the BuildNutsTree atom until a termination condition is met, samples a new state from the resulting tree, and uses a Metropolis-Hastings correction to ensure detailed balance. This produces the next state in the Markov chain.

    Args:
        rng: JAX-style random number generator key, split for each stochastic step.
        hamiltonian: An oracle that provides the energy and its gradient.
        initial_state: The current state of the Markov chain (e.g., position and momentum).
        trajectory_params: Parameters governing the trajectory, such as step size.

    Returns:
        next_state: The accepted next state in the Markov chain.
        transition_stats: Diagnostic information about the transition, like acceptance probability and tree depth.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for julia implementations."""


from juliacall import Main as jl


def _buildnutstree_ffi(rng, hamiltonian, start_state, direction, tree_depth, initial_energy):
    """Wrapper that calls the Julia version of build nuts tree. Passes arguments through and returns the result."""
    return jl.eval("buildnutstree(rng, hamiltonian, start_state, direction, tree_depth, initial_energy)")

def _nutstransitionkernel_ffi(rng, hamiltonian, initial_state, trajectory_params):
    """Wrapper that calls the Julia version of nuts transition kernel. Passes arguments through and returns the result."""
    return jl.eval("nutstransitionkernel(rng, hamiltonian, initial_state, trajectory_params)")