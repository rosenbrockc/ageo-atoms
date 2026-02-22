"""Ghost witnesses."""

from ageoa.ghost.abstract import AbstractArray, AbstractScalar

def witness_functional_monte_carlo(data: AbstractArray) -> AbstractArray:
    """Witness for functional_monte_carlo."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_volatility_surface_modeling(data: AbstractArray) -> AbstractArray:
    """Witness for volatility_surface_modeling."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)


def witness_run_simulation(
    model,
    claim,
    seed: int,
    trials: int,
    anti: bool,
    simulator_name: str,
) -> AbstractScalar:
    """Witness for run_simulation.

    Validates that trials is positive and seed is non-negative,
    then returns an AbstractScalar representing the simulation result.
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return AbstractScalar(dtype="float64")


def witness_run_simulation_anti(
    model,
    claim,
    seed: int,
    trials: int,
    simulator_name: str,
) -> AbstractScalar:
    """Witness for run_simulation_anti.

    Validates that trials is positive and even, and seed is non-negative,
    then returns an AbstractScalar representing the antithetic simulation result.
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if trials % 2 != 0:
        raise ValueError(f"trials must be even for antithetic variates, got {trials}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return AbstractScalar(dtype="float64")


def witness_quick_sim_anti(
    model,
    claim,
    trials: int,
    simulator_name: str,
) -> AbstractScalar:
    """Witness for quick_sim_anti.

    Validates that trials is positive and even,
    then returns an AbstractScalar representing the simulation result.
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if trials % 2 != 0:
        raise ValueError(f"trials must be even for antithetic variates, got {trials}")
    return AbstractScalar(dtype="float64")
