"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_kalman_hedge_ratio


@register_atom(witness_kalman_hedge_ratio)
@icontract.require(lambda asset_a: asset_a.ndim >= 1, "asset_a must have at least one dimension")
@icontract.require(lambda asset_b: asset_b.ndim >= 1, "asset_b must have at least one dimension")
@icontract.require(lambda asset_a: asset_a is not None, "asset_a cannot be None")
@icontract.require(lambda asset_a: isinstance(asset_a, np.ndarray), "asset_a must be np.ndarray")
@icontract.require(lambda asset_b: asset_b is not None, "asset_b cannot be None")
@icontract.require(lambda asset_b: isinstance(asset_b, np.ndarray), "asset_b must be np.ndarray")
@icontract.require(lambda delta: delta is not None, "delta cannot be None")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def kalman_hedge_ratio(asset_a: np.ndarray, asset_b: np.ndarray, delta: float) -> np.ndarray:
    """Estimates a time-varying hedge ratio between two assets using a Kalman filter, tracking a changing relationship.

    Args:
        asset_a: Time-series of the base asset prices or returns
        asset_b: Time-series of the hedge asset prices or returns
        delta: State-transition noise ratio controlling how fast the hedge ratio can drift

    Returns:
        Array of time-varying Kalman-estimated hedge ratios, same length as inputs
    """
    raise NotImplementedError("Wire to original implementation")
