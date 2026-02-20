"""Ghost witnesses for QuantFin atoms."""

from ageoa.ghost.abstract import AbstractScalar

class AbstractMonteCarloState:
    """Lightweight metadata for Monte Carlo results."""
    def __init__(self, trials: int, is_antithetic: bool, variance: float):
        self.trials = trials
        self.is_antithetic = is_antithetic
        self.variance = variance

def witness_run_simulation_anti(
    model,
    claim,
    seed: int,
    trials: int,
    simulator
) -> AbstractMonteCarloState:
    """Ghost witness for Antithetic Variates Monte Carlo heuristic."""
    # Represent the statistical variance reduction of the output
    # By convention, antithetic variates approximately halve the variance.
    return AbstractMonteCarloState(
        trials=trials,
        is_antithetic=True,
        variance=0.5
    )

def witness_quick_sim_anti(
    model,
    claim,
    trials: int,
    simulator
) -> AbstractMonteCarloState:
    """Ghost witness for quick Antithetic Variates Monte Carlo heuristic."""
    return AbstractMonteCarloState(
        trials=trials,
        is_antithetic=True,
        variance=0.5
    )
