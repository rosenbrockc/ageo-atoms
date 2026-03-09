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

from juliacall import Main as jl  # type: ignore[import-untyped]


def witness_tt2tdb_offset(*args, **kwargs): pass  # Witness functions should be imported from the generated witnesses module

@register_atom(witness_tt2tdb_offset)  # type: ignore[untyped-decorator]
@icontract.require(lambda seconds: isinstance(seconds, (float, int, np.number)), "seconds must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "TT2TDB_Offset output must not be None")
def tt2tdb_offset(seconds: float | np.ndarray) -> float | np.ndarray:  # type: ignore[type-arg]
    """Computes the relativistic time scale offset between Terrestrial Time (TT) and Barycentric Dynamical Time (TDB) in seconds, using a sinusoidal approximation of the periodic correction term derived from Earth's orbital eccentricity.

    Args:
        seconds: Elapsed seconds from a reference epoch (typically J2000.0); unbounded real number

    Returns:
        TDB - TT offset in seconds; typically within ±0.002 s
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for julia implementations."""


from juliacall import Main as jl


def tt2tdb_offset_ffi(seconds: float) -> float:
    """FFI bridge to Julia implementation of TT2TDB_Offset."""
    return float(jl.eval("tt2tdb_offset(seconds)"))
