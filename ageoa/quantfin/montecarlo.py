import icontract
from typing import Callable
from ageoa.quantfin.models import DiscretizeModel, ContingentClaim
from ageoa.ghost.registry import register_atom
from ageoa.quantfin.witnesses import witness_run_simulation_anti, witness_quick_sim_anti

@icontract.require(lambda trials: trials > 0)
def run_simulation(
    model: DiscretizeModel,
    claim: ContingentClaim,
    seed: int,
    trials: int,
    anti: bool,
    simulator: Callable[[DiscretizeModel, ContingentClaim, int, int, bool], float]
) -> float:
    """Run a single MC simulation."""
    return simulator(model, claim, seed, trials, anti)

@register_atom(witness_run_simulation_anti)
@icontract.require(lambda trials: trials > 0)
@icontract.require(lambda trials: trials % 2 == 0, "Trials must be even for antithetic variates")
@icontract.ensure(lambda result: isinstance(result, float))
def run_simulation_anti(
    model: DiscretizeModel,
    claim: ContingentClaim,
    seed: int,
    trials: int,
    simulator: Callable[[DiscretizeModel, ContingentClaim, int, int, bool], float]
) -> float:
    """Like 'run_simulation', but splits the trials in two and does antithetic variates.
    
    Extracts the Antithetic Variates Monte Carlo heuristic from Haskell `quantfin` to
    reduce statistical variance.
    """
    half_trials = trials // 2
    res_anti = run_simulation(model, claim, seed, half_trials, True, simulator)
    res_reg  = run_simulation(model, claim, seed, half_trials, False, simulator)
    
    return float((res_anti + res_reg) / 2.0)

@register_atom(witness_quick_sim_anti)
@icontract.require(lambda trials: trials > 0)
@icontract.require(lambda trials: trials % 2 == 0, "Trials must be even for antithetic variates")
def quick_sim_anti(
    model: DiscretizeModel,
    claim: ContingentClaim,
    trials: int,
    simulator: Callable[[DiscretizeModel, ContingentClaim, int, int, bool], float]
) -> float:
    """'run_simulation_anti' with a default random number generator seed (e.g. 500)."""
    return run_simulation_anti(model, claim, 500, trials, simulator)
