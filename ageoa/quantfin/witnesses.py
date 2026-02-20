"""Ghost witnesses for QuantFin atoms."""

from __future__ import annotations

from typing import Any

from ageoa.ghost.abstract import AbstractScalar

def witness_run_simulation_anti(
    model: Any,
    claim: Any,
    seed: int,
    trials: int,
    simulator_name: str,
) -> AbstractScalar:
    """Ghost witness for Antithetic Variates Monte Carlo heuristic."""
    del model, claim, seed, simulator_name
    if trials <= 0:
        raise ValueError("trials must be positive")
    return AbstractScalar(dtype="float64")

def witness_quick_sim_anti(
    model: Any,
    claim: Any,
    trials: int,
    simulator_name: str,
) -> AbstractScalar:
    """Ghost witness for quick Antithetic Variates Monte Carlo heuristic."""
    del model, claim, simulator_name
    if trials <= 0:
        raise ValueError("trials must be positive")
    return AbstractScalar(dtype="float64")
