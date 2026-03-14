from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_gamboa_segmenter

  # type: ignore[import-untyped]
@register_atom(witness_gamboa_segmenter)
@icontract.require(lambda tol: isinstance(tol, (float, int, np.number)), "tol must be numeric")
@icontract.ensure(lambda result: result is not None, "gamboa_segmenter output must not be None")
def gamboa_segmenter(signal: np.ndarray, sampling_rate: float, tol: float) -> np.ndarray:
    """
    Args:
        signal: non-empty, finite values
        sampling_rate: must be > 0
        tol: must be > 0

    Returns:
        indices within [0, len(signal))
    """
    raise NotImplementedError("Wire to original implementation")
