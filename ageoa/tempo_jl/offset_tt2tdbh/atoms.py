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
    """Computes a high-accuracy time correction between an Earth-based clock and one at the solar system center of mass.

    Args:
        seconds: time in seconds since the year-2000 reference epoch

    Returns:
        offset in seconds, maximum error ~10 microseconds over 1600-2200
    """
    raise NotImplementedError("Wire to original implementation")