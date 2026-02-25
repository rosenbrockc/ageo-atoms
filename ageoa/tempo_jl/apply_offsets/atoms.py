"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom

from juliacall import Main as jl


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_show)
@icontract.require(lambda io: io is not None, "io cannot be None")
@icontract.require(lambda s: s is not None, "s cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Show output must not be None")
def show(io: Any, s: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness__zero_offset)
@icontract.require(lambda seconds: seconds is not None, "seconds cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, " Zero Offset output must not be None")
def _zero_offset(seconds: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_apply_offsets)
@icontract.require(lambda sec: sec is not None, "sec cannot be None")
@icontract.require(lambda ts1: ts1 is not None, "ts1 cannot be None")
@icontract.require(lambda ts2: ts2 is not None, "ts2 cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Apply Offsets output must not be None")
def apply_offsets(sec: Any, ts1: Any, ts2: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for julia implementations."""

from __future__ import annotations

from juliacall import Main as jl


def show_ffi(io, s):
    """FFI bridge to Julia implementation of Show."""
    return jl.eval("show(io, s)")

def _zero_offset_ffi(seconds):
    """FFI bridge to Julia implementation of  Zero Offset."""
    return jl.eval("_zero_offset(seconds)")

def apply_offsets_ffi(sec, ts1, ts2):
    """FFI bridge to Julia implementation of Apply Offsets."""
    return jl.eval("apply_offsets(sec, ts1, ts2)")
