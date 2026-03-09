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

@register_atom(witness_randomword32)
@icontract.require(lambda c: c is not None, "c cannot be None")
@icontract.require(lambda state: state is not None, "state cannot be None")
@icontract.require(lambda state': state' is not None, "state' cannot be None")
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.require(lambda xor: xor is not None, "xor cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Randomword32 output must not be None")
def randomword32(c: Any, state: Any, state': Any, x: Any, xor: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_randomint)
@icontract.require(lambda fromIntegral: fromIntegral is not None, "fromIntegral cannot be None")
@icontract.require(lambda g: g is not None, "g cannot be None")
@icontract.require(lambda g': g' is not None, "g' cannot be None")
@icontract.require(lambda i: i is not None, "i cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Randomint output must not be None")
def randomint(fromIntegral: Any, g: Any, g': Any, i: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_randomword64)
@icontract.ensure(lambda result, **kwargs: result is not None, "Randomword64 output must not be None")
def randomword64(buildWord64'': Any, x: Any, x'': Any, y1: Any, y2: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_randomdouble)
@icontract.require(lambda div: div is not None, "div cannot be None")
@icontract.require(lambda fromIntegral: fromIntegral is not None, "fromIntegral cannot be None")
@icontract.require(lambda val: val is not None, "val cannot be None")
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.require(lambda x': x' is not None, "x' cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Randomdouble output must not be None")
def randomdouble(div: Any, fromIntegral: Any, val: Any, x: Any, x': Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_randomint64)
@icontract.require(lambda fromIntegral: fromIntegral is not None, "fromIntegral cannot be None")
@icontract.require(lambda g: g is not None, "g cannot be None")
@icontract.require(lambda g': g' is not None, "g' cannot be None")
@icontract.require(lambda i: i is not None, "i cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Randomint64 output must not be None")
def randomint64(fromIntegral: Any, g: Any, g': Any, i: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_addmod64)
@icontract.require(lambda a: a is not None, "a cannot be None")
@icontract.require(lambda b: b is not None, "b cannot be None")
@icontract.require(lambda m: m is not None, "m cannot be None")
@icontract.require(lambda mod: mod is not None, "mod cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Addmod64 output must not be None")
def addmod64(a: Any, b: Any, m: Any, mod: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_mulmod64)
@icontract.require(lambda a: a is not None, "a cannot be None")
@icontract.require(lambda b: b is not None, "b cannot be None")
@icontract.require(lambda f: f is not None, "f cannot be None")
@icontract.require(lambda m: m is not None, "m cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Mulmod64 output must not be None")
def mulmod64(a: Any, b: Any, f: Any, m: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_powmod64)
@icontract.require(lambda a: a is not None, "a cannot be None")
@icontract.require(lambda e: e is not None, "e cannot be None")
@icontract.require(lambda f: f is not None, "f cannot be None")
@icontract.require(lambda m: m is not None, "m cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Powmod64 output must not be None")
def powmod64(a: Any, e: Any, f: Any, m: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_skip)
@icontract.require(lambda d: d is not None, "d cannot be None")
@icontract.require(lambda st: st is not None, "st cannot be None")
@icontract.require(lambda st': st' is not None, "st' cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Skip output must not be None")
def skip(d: Any, st: Any, st': Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_next)
@icontract.require(lambda fromIntegral: fromIntegral is not None, "fromIntegral cannot be None")
@icontract.require(lambda g: g is not None, "g cannot be None")
@icontract.require(lambda g': g' is not None, "g' cannot be None")
@icontract.require(lambda w: w is not None, "w cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Next output must not be None")
def next(fromIntegral: Any, g: Any, g': Any, w: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_split)
@icontract.require(lambda g: g is not None, "g cannot be None")
@icontract.require(lambda skip: skip is not None, "skip cannot be None")
@icontract.require(lambda skipConst: skipConst is not None, "skipConst cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Split output must not be None")
def split(g: Any, skip: Any, skipConst: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_f)
@icontract.ensure(lambda result, **kwargs: result is not None, "F output must not be None")
def f(a': Any, a1: Any, b': Any, b1: Any, f: Any, otherwise: Any, r: Any, r': Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_f)
@icontract.require(lambda acc: acc is not None, "acc cannot be None")
@icontract.require(lambda acc': acc' is not None, "acc' cannot be None")
@icontract.require(lambda e': e' is not None, "e' cannot be None")
@icontract.require(lambda e1: e1 is not None, "e1 cannot be None")
@icontract.require(lambda f: f is not None, "f cannot be None")
@icontract.require(lambda otherwise: otherwise is not None, "otherwise cannot be None")
@icontract.require(lambda sqr: sqr is not None, "sqr cannot be None")
@icontract.require(lambda sqr': sqr' is not None, "sqr' cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "F output must not be None")
def f(acc: Any, acc': Any, e': Any, e1: Any, f: Any, otherwise: Any, sqr: Any, sqr': Any) -> Any:
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for haskell implementations."""


import ctypes
import ctypes.util
from pathlib import Path


def randomword32_ffi(c, state, state', x, xor):
    """FFI bridge to Haskell implementation of Randomword32."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./randomword32.so")
    _func_name = 'randomWord32'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(c, state, state', x, xor)

def randomint_ffi(fromIntegral, g, g', i):
    """FFI bridge to Haskell implementation of Randomint."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./randomint.so")
    _func_name = 'randomInt'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(fromIntegral, g, g', i)

def randomword64_ffi(buildWord64'', x, x'', y1, y2):
    """FFI bridge to Haskell implementation of Randomword64."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./randomword64.so")
    _func_name = 'randomWord64'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(buildWord64'', x, x'', y1, y2)

def randomdouble_ffi(div, fromIntegral, val, x, x'):
    """FFI bridge to Haskell implementation of Randomdouble."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./randomdouble.so")
    _func_name = 'randomDouble'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(div, fromIntegral, val, x, x')

def randomint64_ffi(fromIntegral, g, g', i):
    """FFI bridge to Haskell implementation of Randomint64."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./randomint64.so")
    _func_name = 'randomInt64'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(fromIntegral, g, g', i)

def addmod64_ffi(a, b, m, mod):
    """FFI bridge to Haskell implementation of Addmod64."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./addmod64.so")
    _func_name = 'addMod64'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(a, b, m, mod)

def mulmod64_ffi(a, b, f, m):
    """FFI bridge to Haskell implementation of Mulmod64."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./mulmod64.so")
    _func_name = 'mulMod64'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(a, b, f, m)

def powmod64_ffi(a, e, f, m):
    """FFI bridge to Haskell implementation of Powmod64."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./powmod64.so")
    _func_name = 'powMod64'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(a, e, f, m)

def skip_ffi(d, st, st'):
    """FFI bridge to Haskell implementation of Skip."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./skip.so")
    _func_name = 'skip'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(d, st, st')

def next_ffi(fromIntegral, g, g', w):
    """FFI bridge to Haskell implementation of Next."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./next.so")
    _func_name = 'next'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(fromIntegral, g, g', w)

def split_ffi(g, skip, skipConst):
    """FFI bridge to Haskell implementation of Split."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./split.so")
    _func_name = 'split'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(g, skip, skipConst)

def f_ffi(a', a1, b', b1, f, otherwise, r, r'):
    """FFI bridge to Haskell implementation of F."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./f.so")
    _func_name = 'f'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(a', a1, b', b1, f, otherwise, r, r')

def f_ffi(acc, acc', e', e1, f, otherwise, sqr, sqr'):
    """FFI bridge to Haskell implementation of F."""
    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()
    _lib = ctypes.CDLL("./f.so")
    _func_name = 'f'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(acc, acc', e', e1, f, otherwise, sqr, sqr')
