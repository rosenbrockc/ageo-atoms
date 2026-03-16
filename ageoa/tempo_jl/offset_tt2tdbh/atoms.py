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
    import math
    CENTURY2SEC = 86400.0 * 36525.0
    T = seconds / CENTURY2SEC
    return (0.001657 * math.sin(628.3076 * T + 6.2401)
            + 0.000022 * math.sin(575.3385 * T + 4.2970)
            + 0.000014 * math.sin(1256.6152 * T + 6.1969)
            + 0.000005 * math.sin(606.9777 * T + 4.0212)
            + 0.000005 * math.sin(52.9691 * T + 0.4444)
            + 0.000002 * math.sin(21.3299 * T + 5.5431)
            + 0.000010 * T * math.sin(628.3076 * T + 4.2490))