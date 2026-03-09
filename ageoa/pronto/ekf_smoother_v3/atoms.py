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
from .witnesses import *  # type: ignore[import-untyped]

import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module
from typing import Any
def witness_stateestimatorinit(*args, **kwargs): pass

@icontract.ensure(lambda result, **kwargs: result is not None, "StateEstimatorInit output must not be None")
def stateestimatorinit() -> StateModelSpec:
    """Bootstraps the StateEstimator instance: allocates internal containers, sets default hyperparameters, and establishes the initial state model ready for predict/update cycles.


    Returns:
        All matrices must be positive-semi-definite where applicable; initial state vector must be finite.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for cpp implementations."""


import ctypes
import ctypes.util
from pathlib import Path
def stateestimatorinit_ffi() -> ctypes.c_void_p:
def stateestimatorinit_ffi():
    _func_name = 'stateestimatorinit'
    _lib = ctypes.CDLL("./stateestimatorinit.so")
    _func_name = atom.method_names[0] if atom.method_names else 'stateestimatorinit'
    _func = _lib[_func_name]
    _func.restype = ctypes.c_void_p
    return _func()
