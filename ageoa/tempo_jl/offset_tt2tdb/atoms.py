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
    """Returns the periodic TT-to-TDB time-scale offset (seconds) due to relativistic effects from Earth_primes elliptical orbit.

    Args:
        seconds: TT epoch expressed in seconds since J2000.0

    Returns:
        TDB - TT offset in seconds, accurate to ~40 microseconds over 1900-2100
    """
    raise NotImplementedError("Wire to original implementation")