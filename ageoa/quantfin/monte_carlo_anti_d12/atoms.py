from __future__ import annotations

"""Atom wrappers for Monte Carlo simulation with antithetic variates."""

import numpy as np
import icontract
from typing import Callable, List, Tuple

from ageoa.ghost.registry import register_atom
from .witnesses import (
    witness_avg,
    witness_evolve,
    witness_insertcf,
    witness_insertcflist,
    witness_maxstep,
    witness_process,
    witness_quicksim,
    witness_quicksimanti,
    witness_runmc,
    witness_runsim,
    witness_runsimulation,
    witness_runsimulationanti,
    witness_simulatestate,
)

import ctypes
import ctypes.util
from pathlib import Path


# ---------------------------------------------------------------------------
# runmc  — execute a Monte Carlo computation
# ---------------------------------------------------------------------------

@register_atom(witness_runmc)
@icontract.require(lambda initState: initState is not None, "initState must be provided")
@icontract.require(lambda randState: randState is not None, "randState must be provided")
@icontract.ensure(lambda result: result is not None, "runmc must produce a result")
def runmc(
    evalState: Callable,
    evalStateT: Callable,
    flip: Callable,
    initState: object,
    lift: Callable,
    mc: Callable,
    randState: object,
    sampleRVarTWith: Callable,
) -> float:
    """Run a Monte Carlo computation and return its final value.

    Evaluates the monadic Monte Carlo pipeline by sampling random
    variates through the state transformer, starting from the given
    initial and random-generator states.

    Args:
        evalState: Evaluate a stateful computation, discarding the state.
        evalStateT: Evaluate the inner state-transformer layer.
        flip: Flip argument order of a two-argument function.
        initState: Initial model state (observables and time).
        lift: Lift a computation into the monad-transformer stack.
        mc: The Monte Carlo computation to execute.
        randState: Initial random-generator state.
        sampleRVarTWith: Sample a random variable using a given lifter.

    Returns:
        Final numeric result of the Monte Carlo computation.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# runsimulation
# ---------------------------------------------------------------------------

@register_atom(witness_runsimulation)
@icontract.require(lambda trials: isinstance(trials, int) and trials > 0, "trials must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, float), "result must be a float")
def runsimulation(
    anti: bool,
    ccs: object,
    modl: object,
    run: Callable,
    runMC: Callable,
    seed: object,
    trials: int,
    undefined: object,
) -> float:
    """Run a full Monte Carlo simulation for a basket of contingent claims.

    Sets up the random state, constructs the simulation run, and calls
    runMC to compute the expected discounted payoff.

    Args:
        anti: Whether to use antithetic variates.
        ccs: Compiled contingent-claim basket.
        modl: Pricing model that implements the Discretize interface.
        run: Assembled simulation run (simulateState applied to args).
        runMC: Function that executes the Monte Carlo monad.
        seed: Initial random-number generator state.
        trials: Number of Monte Carlo trials.
        undefined: Placeholder initial observable (set during initialize).

    Returns:
        Estimated fair value as a float.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# runsimulationanti
# ---------------------------------------------------------------------------

@register_atom(witness_runsimulationanti)
@icontract.require(lambda trials: isinstance(trials, int) and trials > 0, "trials must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, float), "result must be a float")
def runsimulationanti(
    ccs: object,
    modl: object,
    runSim: Callable,
    seed: object,
    trials: int,
) -> float:
    """Run a Monte Carlo simulation using antithetic variates for variance reduction.

    Splits the trial count in half, runs once with normal variates and
    once with flipped variates, and averages the two results.

    Args:
        ccs: Compiled contingent-claim basket.
        modl: Pricing model that implements the Discretize interface.
        runSim: Partial application of runSimulation with anti flag.
        seed: Initial random-number generator state.
        trials: Total number of trials (split evenly between normal and flipped).

    Returns:
        Variance-reduced estimated fair value as a float.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# quicksim
# ---------------------------------------------------------------------------

@register_atom(witness_quicksim)
@icontract.require(lambda trials: isinstance(trials, int) and trials > 0, "trials must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, float), "result must be a float")
def quicksim(
    mdl: object,
    opts: object,
    pureMT: Callable,
    runSimulation: Callable,
    trials: int,
) -> float:
    """Run a quick Monte Carlo simulation with a default random seed.

    Convenience wrapper around runSimulation that uses pureMT(500) as
    the random state and disables antithetic variates.

    Args:
        mdl: Pricing model.
        opts: Contingent-claim basket.
        pureMT: Constructor for a Mersenne Twister generator from a seed.
        runSimulation: Full simulation runner.
        trials: Number of Monte Carlo trials.

    Returns:
        Estimated fair value as a float.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# quicksimanti
# ---------------------------------------------------------------------------

@register_atom(witness_quicksimanti)
@icontract.require(lambda trials: isinstance(trials, int) and trials > 0, "trials must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, float), "result must be a float")
def quicksimanti(
    mdl: object,
    opts: object,
    pureMT: Callable,
    runSimulationAnti: Callable,
    trials: int,
) -> float:
    """Run a quick Monte Carlo simulation with antithetic variates and a default seed.

    Convenience wrapper around runSimulationAnti that uses pureMT(500)
    as the random state.

    Args:
        mdl: Pricing model.
        opts: Contingent-claim basket.
        pureMT: Constructor for a Mersenne Twister generator from a seed.
        runSimulationAnti: Antithetic simulation runner.
        trials: Number of Monte Carlo trials.

    Returns:
        Variance-reduced estimated fair value as a float.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# evolve
# ---------------------------------------------------------------------------

@register_atom(witness_evolve)
@icontract.require(lambda mdl: mdl is not None, "mdl -- model must be provided")
@icontract.ensure(lambda result: result is not None, "evolve must produce a result")
def evolve(
    anti: bool,
    evolve: Callable,
    evolve_prime: Callable,
    get: Callable,
    maxStep: Callable,
    mdl: object,
    ms: float,
    t1: float,
    t2: float,
    timeDiff: Callable,
    timeOffset: Callable,
    unless: Callable,
) -> object:
    """Evolve the model state from current time to a target time.

    If the time gap exceeds the model's maximum step size, the
    evolution is broken into sub-steps.  Antithetic variates are
    applied when *anti* is True.

    Args:
        anti: Whether to flip random variates.
        evolve: Recursive reference for multi-step evolution.
        evolve_prime: Single-step evolution function (evolve').
        get: Retrieve current state from the monad.
        maxStep: Return the maximum allowed time step for the model.
        mdl: Pricing model.
        ms: Maximum step size returned by maxStep.
        t1: Current simulation time.
        t2: Target simulation time.
        timeDiff: Compute the difference between two times.
        timeOffset: Offset a time by a given amount.
        unless: Conditional guard (skip when times are equal).

    Returns:
        Updated monadic state after evolution.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# maxstep
# ---------------------------------------------------------------------------

@register_atom(witness_maxstep)
@icontract.ensure(lambda result: isinstance(result, float) and result > 0.0, "maxStep must be a positive float")
def maxstep() -> float:
    """Return the default maximum discretization time step.

    The default is 1/250, representing one trading day in a 250-day
    year.

    Returns:
        Maximum time step as a float (default 0.004).
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# simulatestate
# ---------------------------------------------------------------------------

@register_atom(witness_simulatestate)
@icontract.require(lambda trials: isinstance(trials, int) and trials > 0, "trials must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, float), "result must be a float")
def simulatestate(
    anti: bool,
    avg: Callable,
    ccb: list,
    modl: object,
    replicateM: Callable,
    singleTrial: Callable,
    trials: int,
) -> float:
    """Simulate the discounted payoff for a contingent-claim basket.

    Runs *trials* independent paths, each initializing the model,
    processing cash flows, and discounting.  The results are averaged.

    Args:
        anti: Whether to use antithetic variates.
        avg: Averaging function over trial results.
        ccb: List of contingent-claim processors (observation times and payoffs).
        modl: Pricing model.
        replicateM: Replicate a monadic action a given number of times.
        singleTrial: A single trial computation (initialize then process).
        trials: Number of independent trials to run.

    Returns:
        Average discounted payoff across all trials.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# runsim  — convenience wrapper
# ---------------------------------------------------------------------------

@register_atom(witness_runsim)
@icontract.require(lambda trials: isinstance(trials, int) and trials > 0, "trials must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, float), "result must be a float")
def runsim(
    ccs: object,
    div: Callable,
    modl: object,
    runSimulation: Callable,
    seed: object,
    trials: int,
    x: bool,
) -> float:
    """Run a simulation variant with a given antithetic flag.

    Helper used by runSimulationAnti to execute half the trials with
    a specific variate direction.

    Args:
        ccs: Compiled contingent-claim basket.
        div: Integer division function.
        modl: Pricing model.
        runSimulation: Full simulation runner.
        seed: Random-generator state.
        trials: Number of trials (will be halved internally).
        x: Antithetic flag for this half of the run.

    Returns:
        Estimated fair value for this half-run.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# process  — full cash-flow processing (variant 1: with remaining CFs)
# ---------------------------------------------------------------------------

@register_atom(witness_process)
@icontract.ensure(lambda result: isinstance(result, float), "result must be a float")
def process(
    allcfs: list,
    amt: float,
    anti: bool,
    c: object,
    ccs: list,
    cfList: list,
    cfs: list,
    cft: float,
    d: float,
    discCFs: float,
    discount: Callable,
    evolve: Callable,
    flip: Callable,
    foldl_prime: Callable,
    fst: Callable,
    gets: Callable,
    insert: Callable,
    insertCF: Callable,
    insertCFList: Callable,
    map: Callable,
    mf: list,
    modl: object,
    newCFs: list,
    obs: object,
    obsMap: dict,
    obsMap_prime: dict,
    process: Callable,
    t: float,
    xs: list,
) -> float:
    """Process cash flows by interleaving observation and payment times.

    When the next cash-flow time precedes the next observation time,
    the model evolves to that cash-flow time, discounts the payment,
    and accumulates.  Otherwise it evolves to the observation time,
    records the observable, generates new cash flows, and recurses.

    Args:
        allcfs: Full remaining cash-flow list.
        amt: Cash-flow amount at the current time.
        anti: Whether to use antithetic variates.
        c: Current contingent-claim processor.
        ccs: Remaining processors after the current one.
        cfList: Intermediate cash-flow list workspace.
        cfs: Remaining cash flows after the current one.
        cft: Time of the current cash flow.
        d: Discount factor at the current time.
        discCFs: Running sum of discounted cash flows.
        discount: Discounting function from the model.
        evolve: Model evolution function.
        flip: Argument-order flipper.
        foldl_prime: Strict left fold.
        fst: Extract the first element of a pair.
        gets: Extract a field from the monadic state.
        insert: Map insertion function.
        insertCF: Insert a single cash flow in sorted order.
        insertCFList: Insert a list of cash flows in sorted order.
        map: Map a function over a list.
        mf: Payout functions for the current processor.
        modl: Pricing model.
        newCFs: Newly generated cash flows from the current observation.
        obs: Current observable value.
        obsMap: Map of observation times to observable values.
        obsMap_prime: Updated observation map after insertion.
        process: Recursive reference to this function.
        t: Current observation time.
        xs: Workspace list for fold operations.

    Returns:
        Total discounted cash-flow value for this trial path.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# process  — variant 2: processors remain, no pending CFs
# ---------------------------------------------------------------------------

@register_atom(witness_process)
@icontract.ensure(lambda result: isinstance(result, float), "result must be a float")
def process(
    anti: bool,
    ccs: list,
    cfList: list,
    discCFs: float,
    evolve: Callable,
    flip: Callable,
    foldl_prime: Callable,
    fst: Callable,
    gets: Callable,
    insert: Callable,
    insertCF: Callable,
    insertCFList: Callable,
    map: Callable,
    mf: list,
    modl: object,
    newCFs: list,
    obs: object,
    obsMap: dict,
    obsMap_prime: dict,
    process: Callable,
    t: float,
    xs: list,
) -> float:
    """Process remaining claim processors when no cash flows are pending.

    Evolves to each observation time, records the observable, generates
    new cash flows via the payout functions, and recurses.

    Args:
        anti: Whether to use antithetic variates.
        ccs: Remaining processors.
        cfList: Intermediate cash-flow list workspace.
        discCFs: Running sum of discounted cash flows.
        evolve: Model evolution function.
        flip: Argument-order flipper.
        foldl_prime: Strict left fold.
        fst: Extract the first element of a pair.
        gets: Extract a field from the monadic state.
        insert: Map insertion function.
        insertCF: Insert a single cash flow in sorted order.
        insertCFList: Insert a list of cash flows in sorted order.
        map: Map a function over a list.
        mf: Payout functions for the current processor.
        modl: Pricing model.
        newCFs: Newly generated cash flows from the current observation.
        obs: Current observable value.
        obsMap: Map of observation times to observable values.
        obsMap_prime: Updated observation map after insertion.
        process: Recursive reference to this function.
        t: Current observation time.
        xs: Workspace list for fold operations.

    Returns:
        Total discounted cash-flow value for this trial path.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# process  — variant 3: only cash flows remain
# ---------------------------------------------------------------------------

@register_atom(witness_process)
@icontract.require(lambda discCFs: isinstance(discCFs, float), "discCFs must be a float")
@icontract.ensure(lambda result: isinstance(result, float), "result must be a float")
def process(
    anti: bool,
    cf: object,
    cfAmount: Callable,
    cfTime: Callable,
    cfs: list,
    d: float,
    discCFs: float,
    discount: Callable,
    evolve: Callable,
    modl: object,
    obsMap: dict,
    process: Callable,
) -> float:
    """Process remaining cash flows when no more processors exist.

    Evolves to each cash-flow time, discounts the amount, and
    accumulates into the running total.

    Args:
        anti: Whether to use antithetic variates.
        cf: Current cash-flow object.
        cfAmount: Extract the amount from a cash-flow object.
        cfTime: Extract the time from a cash-flow object.
        cfs: Remaining cash flows.
        d: Discount factor at the current cash-flow time.
        discCFs: Running sum of discounted cash flows.
        discount: Discounting function from the model.
        evolve: Model evolution function.
        modl: Pricing model.
        obsMap: Observation map (unused but passed for consistency).
        process: Recursive reference to this function.

    Returns:
        Total discounted cash-flow value for this trial path.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# process  — variant 4: base case
# ---------------------------------------------------------------------------

@register_atom(witness_process)
@icontract.require(lambda discCFs: isinstance(discCFs, float), "discCFs must be a float")
@icontract.ensure(lambda result: isinstance(result, float), "result must be a float")
def process(
    discCFs: float,
    return_val: Callable,
) -> float:
    """Return the accumulated discounted cash flows (base case).

    When no processors or cash flows remain, the accumulated total is
    returned as the trial result.

    Args:
        discCFs: Total accumulated discounted cash flows.
        return_val: Monadic return that wraps the value.

    Returns:
        Final discounted value for this trial.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# insertcf  — insert a cash flow in sorted order (recursive case)
# ---------------------------------------------------------------------------

@register_atom(witness_insertcf)
@icontract.ensure(lambda result: isinstance(result, list), "result must be a list")
def insertcf(
    amt: float,
    amt_prime: float,
    cfs: list,
    insertCF: Callable,
    otherwise: list,
    t: float,
    t_prime: float,
) -> list:
    """Insert a cash flow into a time-sorted list (recursive case).

    Compares the new cash-flow time *t* with the head of the list
    (*t_prime*).  If the new time is later, the head is kept and
    insertion recurses on the tail.

    Args:
        amt: Amount of the cash flow to insert.
        amt_prime: Amount of the head cash flow.
        cfs: Tail of the existing cash-flow list.
        insertCF: Recursive reference to this function.
        otherwise: Result list when the new CF goes before the head.
        t: Time of the cash flow to insert.
        t_prime: Time of the head cash flow.

    Returns:
        Cash-flow list with the new entry inserted in time order.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# insertcf  — base case (empty list)
# ---------------------------------------------------------------------------

@register_atom(witness_insertcf)
@icontract.ensure(lambda result: isinstance(result, list) and len(result) == 1, "result must be a single-element list")
def insertcf(
    cf: object,
) -> list:
    """Insert a cash flow into an empty list (base case).

    Args:
        cf: Cash-flow object to wrap in a singleton list.

    Returns:
        Single-element list containing the cash flow.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# avg
# ---------------------------------------------------------------------------

@register_atom(witness_avg)
@icontract.require(lambda trials: isinstance(trials, int) and trials > 0, "trials must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, float), "result must be a float")
def avg(
    fromIntegral: Callable,
    sum: Callable,
    trials: int,
    v: list,
) -> float:
    """Compute the arithmetic mean of trial results.

    Divides the sum of all trial values by the trial count.

    Args:
        fromIntegral: Convert an integer to a fractional type.
        sum: Sum function over a list of numbers.
        trials: Number of trials.
        v: List of per-trial result values.

    Returns:
        Arithmetic mean of the trial values.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# insertcflist  — insert multiple CFs (variant 1)
# ---------------------------------------------------------------------------

@register_atom(witness_insertcflist)
@icontract.ensure(lambda result: isinstance(result, list), "result must be a list")
def insertcflist(
    cfList: list,
    flip: Callable,
    foldl_prime: Callable,
    insertCF: Callable,
    xs: list,
) -> list:
    """Insert a list of cash flows into an existing sorted list.

    Uses a strict left fold to insert each new cash flow one at a
    time, maintaining sorted order.

    Args:
        cfList: Existing sorted cash-flow list.
        flip: Argument-order flipper.
        foldl_prime: Strict left fold.
        insertCF: Single cash-flow insertion function.
        xs: List of new cash flows to insert.

    Returns:
        Merged sorted cash-flow list.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# insertcflist  — variant 2
# ---------------------------------------------------------------------------

@register_atom(witness_insertcflist)
@icontract.require(lambda xs: isinstance(xs, list), "xs must be a list")
@icontract.ensure(lambda result: isinstance(result, list), "result must be a list")
def insertcflist(
    cfList: list,
    flip: Callable,
    foldl_prime: Callable,
    insertCF: Callable,
    xs: list,
) -> list:
    """Insert a list of cash flows into an existing sorted list (alternate witness).

    Identical logic to the first variant; registered under the same
    witness for pattern-matching flexibility.

    Args:
        cfList: Existing sorted cash-flow list.
        flip: Argument-order flipper.
        foldl_prime: Strict left fold.
        insertCF: Single cash-flow insertion function.
        xs: List of new cash flows to insert.

    Returns:
        Merged sorted cash-flow list.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# FFI bindings (auto-generated, kept for reference)
# ---------------------------------------------------------------------------

def runmc_ffi(evalState, evalStateT, flip, initState, lift, mc, randState, sampleRVarTWith):
    """Wrapper that calls the Haskell version of runmc."""
    _lib = ctypes.CDLL("./runmc.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 8
    _func.restype = ctypes.c_void_p
    return _func(evalState, evalStateT, flip, initState, lift, mc, randState, sampleRVarTWith)

def runsimulation_ffi(anti, ccs, modl, run, runMC, seed, trials, undefined):
    """Wrapper that calls the Haskell version of runsimulation."""
    _lib = ctypes.CDLL("./runsimulation.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 8
    _func.restype = ctypes.c_void_p
    return _func(anti, ccs, modl, run, runMC, seed, trials, undefined)

def runsimulationanti_ffi(ccs, modl, runSim, seed, trials):
    """Call the Haskell version of run-simulation-anti. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./runsimulationanti.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 5
    _func.restype = ctypes.c_void_p
    return _func(ccs, modl, runSim, seed, trials)

def quicksim_ffi(mdl, opts, pureMT, runSimulation, trials):
    """Wrapper that calls the Haskell version of quicksim."""
    _lib = ctypes.CDLL("./quicksim.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 5
    _func.restype = ctypes.c_void_p
    return _func(mdl, opts, pureMT, runSimulation, trials)

def quicksimanti_ffi(mdl, opts, pureMT, runSimulationAnti, trials):
    """Wrapper that calls the Haskell version of quicksimanti."""
    _lib = ctypes.CDLL("./quicksimanti.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 5
    _func.restype = ctypes.c_void_p
    return _func(mdl, opts, pureMT, runSimulationAnti, trials)

def evolve_ffi(anti, evolve, evolve_prime, get, maxStep, mdl, ms, t1, t2, timeDiff, timeOffset, unless):
    """Wrapper that calls the Haskell version of evolve."""
    _lib = ctypes.CDLL("./evolve.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 12
    _func.restype = ctypes.c_void_p
    return _func(anti, evolve, evolve_prime, get, maxStep, mdl, ms, t1, t2, timeDiff, timeOffset, unless)

def maxstep_ffi():
    """Wrapper that calls the Haskell version of maxstep."""
    _lib = ctypes.CDLL("./maxstep.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.restype = ctypes.c_void_p
    return _func()

def simulatestate_ffi(anti, avg, ccb, modl, replicateM, singleTrial, trials):
    """Wrapper that calls the Haskell version of simulatestate."""
    _lib = ctypes.CDLL("./simulatestate.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 7
    _func.restype = ctypes.c_void_p
    return _func(anti, avg, ccb, modl, replicateM, singleTrial, trials)

def runsim_ffi(ccs, div, modl, runSimulation, seed, trials, x):
    """Wrapper that calls the Haskell version of runsim."""
    _lib = ctypes.CDLL("./runsim.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 7
    _func.restype = ctypes.c_void_p
    return _func(ccs, div, modl, runSimulation, seed, trials, x)

def process_full_ffi(allcfs, amt, anti, c, ccs, cfList, cfs, cft, d, discCFs, discount, evolve, flip, foldl_prime, fst, gets, insert, insertCF, insertCFList, map, mf, modl, newCFs, obs, obsMap, obsMap_prime, process, t, xs):
    """Wrapper that calls the Haskell version of process (full variant)."""
    _lib = ctypes.CDLL("./process.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 29
    _func.restype = ctypes.c_void_p
    return _func(allcfs, amt, anti, c, ccs, cfList, cfs, cft, d, discCFs, discount, evolve, flip, foldl_prime, fst, gets, insert, insertCF, insertCFList, map, mf, modl, newCFs, obs, obsMap, obsMap_prime, process, t, xs)

def process_no_cfs_ffi(anti, ccs, cfList, discCFs, evolve, flip, foldl_prime, fst, gets, insert, insertCF, insertCFList, map, mf, modl, newCFs, obs, obsMap, obsMap_prime, process, t, xs):
    """Wrapper that calls the Haskell version of process (no pending CFs)."""
    _lib = ctypes.CDLL("./process.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 22
    _func.restype = ctypes.c_void_p
    return _func(anti, ccs, cfList, discCFs, evolve, flip, foldl_prime, fst, gets, insert, insertCF, insertCFList, map, mf, modl, newCFs, obs, obsMap, obsMap_prime, process, t, xs)

def process_cfs_only_ffi(anti, cf, cfAmount, cfTime, cfs, d, discCFs, discount, evolve, modl, obsMap, process):
    """Wrapper that calls the Haskell version of process (CFs only)."""
    _lib = ctypes.CDLL("./process.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 12
    _func.restype = ctypes.c_void_p
    return _func(anti, cf, cfAmount, cfTime, cfs, d, discCFs, discount, evolve, modl, obsMap, process)

def process_base_ffi(discCFs, return_val):
    """Wrapper that calls the Haskell version of process (base case)."""
    _lib = ctypes.CDLL("./process.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 2
    _func.restype = ctypes.c_void_p
    return _func(discCFs, return_val)

def insertcf_ffi(amt, amt_prime, cfs, insertCF, otherwise, t, t_prime):
    """Wrapper that calls the Haskell version of insertcf (recursive)."""
    _lib = ctypes.CDLL("./insertcf.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 7
    _func.restype = ctypes.c_void_p
    return _func(amt, amt_prime, cfs, insertCF, otherwise, t, t_prime)

def insertcf_base_ffi(cf):
    """Wrapper that calls the Haskell version of insertcf (base case)."""
    _lib = ctypes.CDLL("./insertcf.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(cf)

def avg_ffi(fromIntegral, sum, trials, v):
    """Wrapper that calls the Haskell version of avg."""
    _lib = ctypes.CDLL("./avg.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 4
    _func.restype = ctypes.c_void_p
    return _func(fromIntegral, sum, trials, v)

def insertcflist_ffi(cfList, flip, foldl_prime, insertCF, xs):
    """Wrapper that calls the Haskell version of insertcflist."""
    _lib = ctypes.CDLL("./insertcflist.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 5
    _func.restype = ctypes.c_void_p
    return _func(cfList, flip, foldl_prime, insertCF, xs)
