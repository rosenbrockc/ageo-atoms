from __future__ import annotations
from typing import Any
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from typing import Any
from ageoa.ghost.registry import register_atom
from .witnesses import witness_avg, witness_evolve, witness_insertcf, witness_insertcflist, witness_maxstep, witness_process, witness_quicksim, witness_quicksimanti, witness_runmc, witness_runsim, witness_runsimulation, witness_runsimulationanti, witness_simulatestate

import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_runmc)
@icontract.require(lambda evalState: evalState is not None, "evalState cannot be None")
@icontract.require(lambda evalStateT: evalStateT is not None, "evalStateT cannot be None")
@icontract.require(lambda flip: flip is not None, "flip cannot be None")
@icontract.require(lambda initState: initState is not None, "initState cannot be None")
@icontract.require(lambda lift: lift is not None, "lift cannot be None")
@icontract.require(lambda mc: mc is not None, "mc cannot be None")
@icontract.require(lambda randState: randState is not None, "randState cannot be None")
@icontract.require(lambda sampleRVarTWith: sampleRVarTWith is not None, "sampleRVarTWith cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Runmc output must not be None")
def runmc(evalState: Any, evalStateT: Any, flip: Any, initState: Any, lift: Any, mc: Any, randState: Any, sampleRVarTWith: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_runsimulation)
@icontract.ensure(lambda result, **kwargs: result is not None, "Runsimulation output must not be None")
def runsimulation(anti: Any, ccs: Any, modl: Any, run: Any, runMC: Any, seed: Any, trials: Any, undefined: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_runsimulationanti)
@icontract.require(lambda ccs: ccs is not None, "ccs cannot be None")
@icontract.require(lambda modl: modl is not None, "modl cannot be None")
@icontract.require(lambda runSim: runSim is not None, "runSim cannot be None")
@icontract.require(lambda seed: seed is not None, "seed cannot be None")
@icontract.require(lambda trials: trials is not None, "trials cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Runsimulationanti output must not be None")
def runsimulationanti(ccs: Any, modl: Any, runSim: Any, seed: Any, trials: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_quicksim)
@icontract.ensure(lambda result, **kwargs: result is not None, "Quicksim output must not be None")
def quicksim(mdl: Any, opts: Any, pureMT: Any, runSimulation: Any, trials: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_quicksimanti)
@icontract.require(lambda mdl: mdl is not None, "mdl cannot be None")
@icontract.require(lambda opts: opts is not None, "opts cannot be None")
@icontract.require(lambda pureMT: pureMT is not None, "pureMT cannot be None")
@icontract.require(lambda runSimulationAnti: runSimulationAnti is not None, "runSimulationAnti cannot be None")
@icontract.require(lambda trials: trials is not None, "trials cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Quicksimanti output must not be None")
def quicksimanti(mdl: Any, opts: Any, pureMT: Any, runSimulationAnti: Any, trials: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_evolve)
@icontract.ensure(lambda result, **kwargs: result is not None, "Evolve output must not be None")
def evolve(anti: Any, evolve: Any, evolve_prime: Any, get: Any, maxStep: Any, mdl: Any, ms: Any, t1: Any, t2: Any, timeDiff: Any, timeOffset: Any, unless: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_maxstep)
@icontract.ensure(lambda result, **kwargs: result is not None, "Maxstep output must not be None")
def maxstep() -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_simulatestate)
@icontract.require(lambda anti: anti is not None, "anti cannot be None")
@icontract.require(lambda avg: avg is not None, "avg cannot be None")
@icontract.require(lambda ccb: ccb is not None, "ccb cannot be None")
@icontract.require(lambda modl: modl is not None, "modl cannot be None")
@icontract.require(lambda replicateM: replicateM is not None, "replicateM cannot be None")
@icontract.require(lambda singleTrial: singleTrial is not None, "singleTrial cannot be None")
@icontract.require(lambda trials: trials is not None, "trials cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Simulatestate output must not be None")
def simulatestate(anti: Any, avg: Any, ccb: Any, modl: Any, replicateM: Any, singleTrial: Any, trials: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_runsim)
@icontract.ensure(lambda result, **kwargs: result is not None, "Runsim output must not be None")
def runsim(ccs: Any, div: Any, modl: Any, runSimulation: Any, seed: Any, trials: Any, x: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_process)
@icontract.ensure(lambda result, **kwargs: result is not None, "Process output must not be None")
def process(allcfs: Any, amt: Any, anti: Any, c: Any, ccs: Any, cfList: Any, cfs: Any, cft: Any, d: Any, discCFs: Any, discount: Any, evolve: Any, flip: Any, foldl_prime: Any, fst: Any, gets: Any, insert: Any, insertCF: Any, insertCFList: Any, map: Any, mf: Any, modl: Any, newCFs: Any, obs: Any, obsMap: Any, obsMap_prime: Any, process: Any, t: Any, xs: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_process)
@icontract.ensure(lambda result, **kwargs: result is not None, "Process output must not be None")
def process(anti: Any, ccs: Any, cfList: Any, discCFs: Any, evolve: Any, flip: Any, foldl_prime: Any, fst: Any, gets: Any, insert: Any, insertCF: Any, insertCFList: Any, map: Any, mf: Any, modl: Any, newCFs: Any, obs: Any, obsMap: Any, obsMap_prime: Any, process: Any, t: Any, xs: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_process)
@icontract.require(lambda anti: anti is not None, "anti cannot be None")
@icontract.require(lambda cf: cf is not None, "cf cannot be None")
@icontract.require(lambda cfAmount: cfAmount is not None, "cfAmount cannot be None")
@icontract.require(lambda cfTime: cfTime is not None, "cfTime cannot be None")
@icontract.require(lambda cfs: cfs is not None, "cfs cannot be None")
@icontract.require(lambda d: d is not None, "d cannot be None")
@icontract.require(lambda discCFs: discCFs is not None, "discCFs cannot be None")
@icontract.require(lambda discount: discount is not None, "discount cannot be None")
@icontract.require(lambda evolve: evolve is not None, "evolve cannot be None")
@icontract.require(lambda modl: modl is not None, "modl cannot be None")
@icontract.require(lambda obsMap: obsMap is not None, "obsMap cannot be None")
@icontract.require(lambda process: process is not None, "process cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Process output must not be None")
def process(anti: Any, cf: Any, cfAmount: Any, cfTime: Any, cfs: Any, d: Any, discCFs: Any, discount: Any, evolve: Any, modl: Any, obsMap: Any, process: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_process)
@icontract.require(lambda discCFs: discCFs is not None, "discCFs cannot be None")
@icontract.require(lambda return_val: return_val is not None, "return cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Process output must not be None")
def process(discCFs: Any, return_val: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_insertcf)
@icontract.ensure(lambda result, **kwargs: result is not None, "Insertcf output must not be None")
def insertcf(amt: Any, amt_prime: Any, cfs: Any, insertCF: Any, otherwise: Any, t: Any, t_prime: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_insertcf)
@icontract.require(lambda cf: cf is not None, "cf cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Insertcf output must not be None")
def insertcf(cf: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_avg)
@icontract.require(lambda fromIntegral: fromIntegral is not None, "fromIntegral cannot be None")
@icontract.require(lambda sum: sum is not None, "sum cannot be None")
@icontract.require(lambda trials: trials is not None, "trials cannot be None")
@icontract.require(lambda v: v is not None, "v cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Avg output must not be None")
def avg(fromIntegral: Any, sum: Any, trials: Any, v: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_insertcflist)
@icontract.ensure(lambda result, **kwargs: result is not None, "Insertcflist output must not be None")
def insertcflist(cfList: Any, flip: Any, foldl_prime: Any, insertCF: Any, xs: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_insertcflist)
@icontract.require(lambda cfList: cfList is not None, "cfList cannot be None")
@icontract.require(lambda flip: flip is not None, "flip cannot be None")
@icontract.require(lambda foldl_prime: foldl_prime is not None, "foldl_prime cannot be None")
@icontract.require(lambda insertCF: insertCF is not None, "insertCF cannot be None")
@icontract.require(lambda xs: xs is not None, "xs cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Insertcflist output must not be None")
def insertcflist(cfList: Any, flip: Any, foldl_prime: Any, insertCF: Any, xs: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for haskell implementations."""


import ctypes
import ctypes.util
from pathlib import Path


def runmc_ffi(evalState, evalStateT, flip, initState, lift, mc, randState, sampleRVarTWith):
    """Wrapper that calls the Haskell version of runmc. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./runmc.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(evalState, evalStateT, flip, initState, lift, mc, randState, sampleRVarTWith)

def runsimulation_ffi(anti, ccs, modl, run, runMC, seed, trials, undefined):
    """Wrapper that calls the Haskell version of runsimulation. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./runsimulation.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(anti, ccs, modl, run, runMC, seed, trials, undefined)

def runsimulationanti_ffi(ccs, modl, runSim, seed, trials):
    """Wrapper that calls the Haskell version of runsimulationanti. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./runsimulationanti.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(ccs, modl, runSim, seed, trials)

def quicksim_ffi(mdl, opts, pureMT, runSimulation, trials):
    """Wrapper that calls the Haskell version of quicksim. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./quicksim.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(mdl, opts, pureMT, runSimulation, trials)

def quicksimanti_ffi(mdl, opts, pureMT, runSimulationAnti, trials):
    """Wrapper that calls the Haskell version of quicksimanti. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./quicksimanti.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(mdl, opts, pureMT, runSimulationAnti, trials)

def evolve_ffi(anti, evolve, evolve_prime, get, maxStep, mdl, ms, t1, t2, timeDiff, timeOffset, unless):
    """Wrapper that calls the Haskell version of evolve. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./evolve.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(anti, evolve, evolve_prime, get, maxStep, mdl, ms, t1, t2, timeDiff, timeOffset, unless)

def maxstep_ffi():
    """Wrapper that calls the Haskell version of maxstep. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./maxstep.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.restype = ctypes.c_void_p
    return _func()

def simulatestate_ffi(anti, avg, ccb, modl, replicateM, singleTrial, trials):
    """Wrapper that calls the Haskell version of simulatestate. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./simulatestate.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(anti, avg, ccb, modl, replicateM, singleTrial, trials)

def runsim_ffi(ccs, div, modl, runSimulation, seed, trials, x):
    """Wrapper that calls the Haskell version of runsim. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./runsim.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(ccs, div, modl, runSimulation, seed, trials, x)

def process_ffi(allcfs, amt, anti, c, ccs, cfList, cfs, cft, d, discCFs, discount, evolve, flip, foldl_prime, fst, gets, insert, insertCF, insertCFList, map, mf, modl, newCFs, obs, obsMap, obsMap_prime, process, t, xs):
    """Wrapper that calls the Haskell version of process. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./process.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(allcfs, amt, anti, c, ccs, cfList, cfs, cft, d, discCFs, discount, evolve, flip, foldl_prime, fst, gets, insert, insertCF, insertCFList, map, mf, modl, newCFs, obs, obsMap, obsMap_prime, process, t, xs)

def process_ffi(anti, ccs, cfList, discCFs, evolve, flip, foldl_prime, fst, gets, insert, insertCF, insertCFList, map, mf, modl, newCFs, obs, obsMap, obsMap_prime, process, t, xs):
    """Wrapper that calls the Haskell version of process. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./process.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(anti, ccs, cfList, discCFs, evolve, flip, foldl_prime, fst, gets, insert, insertCF, insertCFList, map, mf, modl, newCFs, obs, obsMap, obsMap_prime, process, t, xs)

def process_ffi(anti, cf, cfAmount, cfTime, cfs, d, discCFs, discount, evolve, modl, obsMap, process):
    """Wrapper that calls the Haskell version of process. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./process.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(anti, cf, cfAmount, cfTime, cfs, d, discCFs, discount, evolve, modl, obsMap, process)

def process_ffi(discCFs, return_val):
    """Wrapper that calls the Haskell version of process. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./process.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(discCFs, return_val)

def insertcf_ffi(amt, amt_prime, cfs, insertCF, otherwise, t, t_prime):
    """Wrapper that calls the Haskell version of insertcf. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./insertcf.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(amt, amt_prime, cfs, insertCF, otherwise, t, t_prime)

def insertcf_ffi(cf):
    """Wrapper that calls the Haskell version of insertcf. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./insertcf.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(cf)

def avg_ffi(fromIntegral, sum, trials, v):
    """Wrapper that calls the Haskell version of avg. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./avg.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(fromIntegral, sum, trials, v)

def insertcflist_ffi(cfList, flip, foldl_prime, insertCF, xs):
    """Wrapper that calls the Haskell version of insertcflist. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./insertcflist.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(cfList, flip, foldl_prime, insertCF, xs)

def insertcflist_ffi(cfList, flip, foldl_prime, insertCF, xs):
    """Wrapper that calls the Haskell version of insertcflist. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./insertcflist.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(cfList, flip, foldl_prime, insertCF, xs)
