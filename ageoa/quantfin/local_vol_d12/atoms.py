from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import *

import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_var)
@icontract.require(lambda s: s is not None, "s cannot be None")
@icontract.require(lambda t: t is not None, "t cannot be None")
@icontract.require(lambda t': t' is not None, "t' cannot be None")
@icontract.require(lambda v: v is not None, "v cannot be None")
@icontract.require(lambda vs: vs is not None, "vs cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Var output must not be None")
def var(s: Any, t: Any, t': Any, v: Any, vs: Any) -> Any:
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
def vol(interpolatedVs: Any, mats: Any, mats': Any, quotes: Any, strike: Any, sts: Any, t: Any, tInterp: Any, timeFromZero: Any, vInterp: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_allfort)
@icontract.require(lambda map: map is not None, "map cannot be None")
@icontract.require(lambda quotes: quotes is not None, "quotes cannot be None")
@icontract.require(lambda sts: sts is not None, "sts cannot be None")
@icontract.require(lambda t': t' is not None, "t' cannot be None")
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Allfort output must not be None")
def allfort(map: Any, quotes: Any, sts: Any, t': Any, x: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for haskell implementations."""


import ctypes
import ctypes.util
from pathlib import Path


def var_ffi(s, t, t', v, vs):
    """FFI bridge to Haskell implementation of Var."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./var.so")
    _func_name = 'var'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(s, t, t', v, vs)

def localvol_ffi(dwdt, k, otherwise, rcurve, s0, solution, sqrt, t, v, w):
    """FFI bridge to Haskell implementation of Localvol."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./localvol.so")
    _func_name = 'localVol'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(dwdt, k, otherwise, rcurve, s0, solution, sqrt, t, v, w)

def vol_ffi(x):
    """FFI bridge to Haskell implementation of Vol."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./vol.so")
    _func_name = 'vol'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(x)

def vol_ffi(interpolatedVs, mats, mats', quotes, strike, sts, t, tInterp, timeFromZero, vInterp):
    """FFI bridge to Haskell implementation of Vol."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./vol.so")
    _func_name = 'vol'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(interpolatedVs, mats, mats', quotes, strike, sts, t, tInterp, timeFromZero, vInterp)

def allfort_ffi(map, quotes, sts, t', x):
    """FFI bridge to Haskell implementation of Allfort."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./allfort.so")
    _func_name = 'allForT'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(map, quotes, sts, t', x)
