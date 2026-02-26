"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_detect_triangular_arbitrage


@register_atom(witness_detect_triangular_arbitrage)
@icontract.require(lambda rates: rates.ndim >= 1, "rates must have at least one dimension")
@icontract.require(lambda rates: rates is not None, "rates cannot be None")
@icontract.require(lambda rates: isinstance(rates, np.ndarray), "rates must be np.ndarray")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def detect_triangular_arbitrage(rates: np.ndarray) -> np.ndarray:
    """Detects triangular arbitrage opportunities in an FX rate matrix by searching for negative-weight cycles in the log-rate graph.

    Args:
        rates: Exchange rate matrix, shape (n_currencies, n_currencies), where rates[i,j] is the rate from currency i to j

    Returns:
        Array of cycle profit factors; values > 1.0 indicate arbitrage opportunities
    """
    raise NotImplementedError("Wire to original implementation")
