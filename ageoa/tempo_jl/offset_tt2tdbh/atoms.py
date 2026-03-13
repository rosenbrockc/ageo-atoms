from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_offset_tt2tdbh


@register_atom(witness_offset_tt2tdbh)
@icontract.require(lambda seconds: seconds is not None, "seconds cannot be None")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def offset_tt2tdbh(seconds: float) -> float:
    """Returns the TT-to-TDBH time-scale offset using the higher-accuracy Harada-Fukushima series expansion.

    Args:
        seconds: TT epoch expressed in seconds since J2000.0

    Returns:
        TDBH - TT offset in seconds via Harada-Fukushima series, maximum error ~10 µs from 1600-2200
    """
    raise NotImplementedError("Wire to original implementation")