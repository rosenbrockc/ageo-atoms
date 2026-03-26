from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_offset_tt2tdb


@register_atom(witness_offset_tt2tdb)
@icontract.require(lambda seconds: seconds is not None, "seconds cannot be None")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def offset_tt2tdb(seconds: float) -> float:
    """Returns the small time correction (seconds) between two clock standards used in astronomy, caused by Earth's elliptical orbit.

    Args:
        seconds: time in seconds since the year-2000 reference epoch

    Returns:
        offset in seconds, accurate to ~40 microseconds over 1900-2100"""
    import math
    k = 1.657e-3
    eb = 1.671e-2
    m0 = 6.239996
    m1 = 1.99096871e-7
    g = m0 + m1 * seconds
    return k * math.sin(g + eb * math.sin(g))