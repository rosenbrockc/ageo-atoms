import icontract
from typing import Dict

import numpy as np

from ageoa.ghost.registry import register_atom
from ageoa.quantfin.models import (
    ContingentClaim,
    DiscretizeModel,
    SeededMonteCarloSimulator,
)
from ageoa.quantfin.witnesses import witness_run_simulation_anti, witness_quick_sim_anti

SIMULATOR_REGISTRY: Dict[str, SeededMonteCarloSimulator] = {}


@icontract.require(lambda name: isinstance(name, str) and name.strip() != "", "name must be non-empty")
@icontract.require(lambda simulator: callable(simulator), "simulator must be callable")
def register_simulator(name: str, simulator: SeededMonteCarloSimulator) -> None:
    """Register a deterministic seeded simulator by name."""
    SIMULATOR_REGISTRY[name] = simulator


def list_simulators() -> list[str]:
    """Return sorted simulator registry keys."""
    return sorted(SIMULATOR_REGISTRY.keys())


@icontract.require(lambda simulator_name: simulator_name in SIMULATOR_REGISTRY, "simulator_name must reference a registered simulator")
def _resolve_simulator(simulator_name: str) -> SeededMonteCarloSimulator:
    return SIMULATOR_REGISTRY[simulator_name]


@icontract.require(lambda trials: trials > 0)
@icontract.require(lambda seed: seed >= 0, "seed must be non-negative")
@icontract.require(lambda simulator_name: isinstance(simulator_name, str) and simulator_name.strip() != "", "simulator_name must be non-empty")
@icontract.ensure(lambda result: isinstance(result, float), "result must be float")
@icontract.ensure(lambda result: np.isfinite(result), "result must be finite")
def run_simulation(
    model: DiscretizeModel,
    claim: ContingentClaim,
    seed: int,
    trials: int,
    anti: bool,
    simulator_name: str,
) -> float:
    """Run one Monte Carlo pass with an explicit seeded RNG boundary.

    <!-- conceptual_profile -->
    {
        "abstract_name": "Stochastic System Ensemble Simulator",
        "conceptual_transform": "Generates a large ensemble of potential future trajectories for a state-space model driven by a stochastic process, then aggregates the outcomes of a functional mapping applied to those trajectories. It provides a deterministic estimate of an expected value via repeated randomized sampling.",
        "abstract_inputs": [
            {
                "name": "model",
                "description": "An object defining the state transition dynamics and stochastic driving processes."
            },
            {
                "name": "claim",
                "description": "A functional mapping that transforms state trajectories into a scalar outcome."
            },
            {
                "name": "seed",
                "description": "An integer specifying the initial state of the pseudo-random number generator for reproducibility."
            },
            {
                "name": "trials",
                "description": "An integer specifying the number of independent trajectories to simulate."
            },
            {
                "name": "anti",
                "description": "A boolean indicating whether to use mirrored (antithetic) stochastic paths to reduce variance."
            },
            {
                "name": "simulator_name",
                "description": "A string identifier for the specific numerical implementation of the trajectory generator."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A scalar representing the expected value (average) across the ensemble of trials."
            }
        ],
        "algorithmic_properties": [
            "stochastic-simulation",
            "monte-carlo",
            "parallel-trials",
            "deterministic-seeded"
        ],
        "cross_disciplinary_applications": [
            "Estimating the probability of failure in a structural system under random environmental loads.",
            "Simulating particle diffusion in a complex porous medium in geology.",
            "Predicting the expected yield of a chemical process under uncertain temperature fluctuations."
        ]
    }
    <!-- /conceptual_profile -->
    """
    simulator = _resolve_simulator(simulator_name)
    rng = np.random.default_rng(seed)
    return float(simulator(model, claim, rng, trials, anti))

@register_atom(witness_run_simulation_anti)
@icontract.require(lambda trials: trials > 0)
@icontract.require(lambda trials: trials % 2 == 0, "Trials must be even for antithetic variates")
@icontract.require(lambda seed: seed >= 0, "seed must be non-negative")
@icontract.ensure(lambda result: isinstance(result, float))
@icontract.ensure(lambda result: np.isfinite(result), "result must be finite")
def run_simulation_anti(
    model: DiscretizeModel,
    claim: ContingentClaim,
    seed: int,
    trials: int,
    simulator_name: str,
) -> float:
    """Like 'run_simulation', but splits the trials in two and does antithetic variates.

    Applies antithetic variates to reduce Monte Carlo estimator variance by pairing
    each random trajectory with its mirror.

    <!-- conceptual_profile -->
    {
        "abstract_name": "Variance-Reduced Ensemble Simulator",
        "conceptual_transform": "Refines the stochastic ensemble simulation by generating pairs of perfectly negatively correlated (antithetic) trajectories. By ensuring that one path's stochastic deviation is the mirror of the other, it significantly reduces the statistical error of the aggregated estimate.",
        "abstract_inputs": [
            {
                "name": "model",
                "description": "The state-space transition model."
            },
            {
                "name": "claim",
                "description": "The functional outcome mapping."
            },
            {
                "name": "seed",
                "description": "RNG initialization state."
            },
            {
                "name": "trials",
                "description": "The total number of trajectories (must be even for pairing)."
            },
            {
                "name": "simulator_name",
                "description": "Implementation identifier."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A scalar representing the variance-reduced expected value estimate."
            }
        ],
        "algorithmic_properties": [
            "variance-reduction",
            "antithetic-variates",
            "paired-sampling",
            "monte-carlo"
        ],
        "cross_disciplinary_applications": [
            "Increasing the precision of radiation dose simulations in medical physics without increasing trial count.",
            "Reducing variance in path-dependent stochastic integral estimates.",
            "Improving the reliability of weather ensemble forecasts for extreme events."
        ]
    }
    <!-- /conceptual_profile -->
    """
    half_trials = trials // 2
    res_anti = run_simulation(model, claim, seed, half_trials, True, simulator_name)
    res_reg = run_simulation(model, claim, seed, half_trials, False, simulator_name)

    return float((res_anti + res_reg) / 2.0)

@register_atom(witness_quick_sim_anti)
@icontract.require(lambda trials: trials > 0)
@icontract.require(lambda trials: trials % 2 == 0, "Trials must be even for antithetic variates")
@icontract.ensure(lambda result: isinstance(result, float), "result must be float")
@icontract.ensure(lambda result: np.isfinite(result), "result must be finite")
def quick_sim_anti(
    model: DiscretizeModel,
    claim: ContingentClaim,
    trials: int,
    simulator_name: str,
) -> float:
    """Convenience wrapper for antithetic Monte Carlo with a default seed.

    <!-- conceptual_profile -->
    {
        "abstract_name": "Standardized Variance-Reduced Simulator",
        "conceptual_transform": "Executes a variance-reduced ensemble simulation using a fixed, standardized seed for the stochastic process. It provides a quick, consistent benchmark for the expected outcome of a stochastic system.",
        "abstract_inputs": [
            {
                "name": "model",
                "description": "The state-space transition model."
            },
            {
                "name": "claim",
                "description": "The functional outcome mapping."
            },
            {
                "name": "trials",
                "description": "The number of trajectories to simulate."
            },
            {
                "name": "simulator_name",
                "description": "Implementation identifier."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A scalar representing the benchmarked expected value."
            }
        ],
        "algorithmic_properties": [
            "benchmarking",
            "standardized-seed",
            "variance-reduction",
            "monte-carlo"
        ],
        "cross_disciplinary_applications": [
            "Rapid sensitivity analysis of a physical model to changes in its core parameters.",
            "Standardized testing of new stochastic simulation algorithms against a fixed baseline.",
            "Quick estimation of system robustness under a predefined 'worst-case' stochastic scenario."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return run_simulation_anti(model, claim, 500, trials, simulator_name)
