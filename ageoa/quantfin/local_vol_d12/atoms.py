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
from ageoa.ghost.registry import register_atom
from .witnesses import witness_allfort, witness_localvol, witness_var, witness_vol

import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_var)
@icontract.require(lambda s: s is not None, "s cannot be None")
@icontract.require(lambda t: t is not None, "t cannot be None")
@icontract.require(lambda t_prime: t_prime is not None, "t_prime cannot be None")
@icontract.require(lambda v: v is not None, "v cannot be None")
@icontract.require(lambda vs: vs is not None, "vs cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Var output must not be None")
def var(s: Any, t: Any, t_prime: Any, v: Any, vs: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_localvol)
@icontract.ensure(lambda result, **kwargs: result is not None, "Localvol output must not be None")
def localvol(dwdt: Any, k: Any, otherwise: Any, rcurve: Any, s0: Any, solution: Any, sqrt: Any, t: Any, v: Any, w: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_vol)
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Vol output must not be None")
def vol(x: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_vol)
@icontract.ensure(lambda result, **kwargs: result is not None, "Vol output must not be None")
def vol(interpolatedVs: Any, mats: Any, mats_prime: Any, quotes: Any, strike: Any, sts: Any, t: Any, tInterp: Any, timeFromZero: Any, vInterp: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_allfort)
@icontract.require(lambda map: map is not None, "map cannot be None")
@icontract.require(lambda quotes: quotes is not None, "quotes cannot be None")
@icontract.require(lambda sts: sts is not None, "sts cannot be None")
@icontract.require(lambda t_prime: t_prime is not None, "t_prime cannot be None")
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Allfort output must not be None")
def allfort(map: Any, quotes: Any, sts: Any, t_prime: Any, x: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for haskell implementations."""


import ctypes
import ctypes.util
from pathlib import Path


def var_ffi(s, t, t_prime, v, vs):
    """Wrapper that calls the Haskell version of var. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./var.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(s, t, t_prime, v, vs)

def localvol_ffi(dwdt, k, otherwise, rcurve, s0, solution, sqrt, t, v, w):
    """Wrapper that calls the Haskell version of localvol. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./localvol.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(dwdt, k, otherwise, rcurve, s0, solution, sqrt, t, v, w)

def vol_ffi(x):
    """Wrapper that calls the Haskell version of vol. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./vol.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(x)

def vol_ffi(interpolatedVs, mats, mats_prime, quotes, strike, sts, t, tInterp, timeFromZero, vInterp):
    """Wrapper that calls the Haskell version of vol. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./vol.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(interpolatedVs, mats, mats_prime, quotes, strike, sts, t, tInterp, timeFromZero, vInterp)

def allfort_ffi(map, quotes, sts, t_prime, x):
    """Wrapper that calls the Haskell version of allfort. Passes arguments through and returns the result."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./allfort.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(map, quotes, sts, t_prime, x)
