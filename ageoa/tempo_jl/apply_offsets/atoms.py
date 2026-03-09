from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import *

from juliacall import Main as jl


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_show)
@icontract.require(lambda io: io is not None, "io cannot be None")
@icontract.require(lambda s: s is not None, "s cannot be None")
@icontract.ensure(lambda result: result is not None, "Show output must not be None")
def show(io: str, s: str) -> str:
    """Show.

    Args:
        io (str): Description.
        s (str): Description.

    Returns:
        str: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness__zero_offset)
@icontract.require(lambda seconds: seconds is not None, "seconds cannot be None")
@icontract.ensure(lambda result: result is not None, " Zero Offset output must not be None")
def _zero_offset(seconds: float) -> float:
    """Zero offset.

    Args:
        seconds (float): Description.

    Returns:
        float: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_apply_offsets)
@icontract.require(lambda sec: sec is not None, "sec cannot be None")
@icontract.require(lambda ts1: ts1 is not None, "ts1 cannot be None")
@icontract.require(lambda ts2: ts2 is not None, "ts2 cannot be None")
@icontract.ensure(lambda result: result is not None, "Apply Offsets output must not be None")
def apply_offsets(sec: float, ts1: float, ts2: float) -> float:
    """Apply offsets.

    Args:
        sec (float): Description.
        ts1 (float): Description.
        ts2 (float): Description.

    Returns:
        float: Description.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for julia implementations."""


from juliacall import Main as jl


def _show_ffi(io, s):
    """FFI bridge to Julia implementation of Show."""
    return jl.eval("show(io, s)")

def _zero_offset_ffi(seconds):
    """FFI bridge to Julia implementation of  Zero Offset."""
    return jl.eval("_zero_offset(seconds)")

def _apply_offsets_ffi(sec, ts1, ts2):
    """FFI bridge to Julia implementation of Apply Offsets."""
    return jl.eval("apply_offsets(sec, ts1, ts2)")
