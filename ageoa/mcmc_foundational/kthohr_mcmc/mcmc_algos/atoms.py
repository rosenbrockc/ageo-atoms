from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_dispatch_mcmc_algorithm


@register_atom(witness_dispatch_mcmc_algorithm)
@icontract.require(lambda log_target_density: log_target_density.ndim >= 1, "log_target_density must have at least one dimension")
@icontract.require(lambda initial_state: initial_state.ndim >= 1, "initial_state must have at least one dimension")
@icontract.require(lambda log_target_density: log_target_density is not None, "log_target_density cannot be None")
@icontract.require(lambda log_target_density: isinstance(log_target_density, np.ndarray), "log_target_density must be np.ndarray")
@icontract.require(lambda initial_state: initial_state is not None, "initial_state cannot be None")
@icontract.require(lambda initial_state: isinstance(initial_state, np.ndarray), "initial_state must be np.ndarray")
@icontract.require(lambda n_draws: n_draws is not None, "n_draws cannot be None")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def dispatch_mcmc_algorithm(log_target_density: np.ndarray, initial_state: np.ndarray, n_draws: int) -> np.ndarray:
    """Routes to the chosen sampling algorithm for drawing random samples from a target distribution. Supports seven sampling methods including random-walk, gradient-based, and population-based approaches.

Args:
    log_target_density: Flattened evaluation of the log-target density at current chain positions
    initial_state: Initial parameter vector for the Markov chain, shape (n_params,)
    n_draws: Number of posterior samples to collect after warmup

Returns:
    Posterior samples array, shape (n_draws, n_params)"""
    raise NotImplementedError("Wire to original implementation")