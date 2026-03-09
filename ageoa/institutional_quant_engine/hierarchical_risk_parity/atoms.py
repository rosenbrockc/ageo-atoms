from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_compute_hrp_weights


@register_atom(witness_compute_hrp_weights)
@icontract.require(lambda returns: returns.ndim >= 1, "returns must have at least one dimension")
@icontract.require(lambda returns: returns is not None, "returns cannot be None")
@icontract.require(lambda returns: isinstance(returns, np.ndarray), "returns must be np.ndarray")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def compute_hrp_weights(returns: np.ndarray) -> np.ndarray:
    """Computes Hierarchical Risk Parity portfolio weights by clustering assets and recursively bisecting risk along the dendrogram.

    Args:
        returns: 2D array of asset returns, shape (n_samples, n_assets)

    Returns:
        HRP portfolio weights, shape (n_assets,), summing to 1.0
    """
    raise NotImplementedError("Wire to original implementation")
