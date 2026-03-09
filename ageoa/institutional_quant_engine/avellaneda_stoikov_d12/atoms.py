from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import *  # type: ignore[import-untyped]

# Witness functions should be imported from the generated witnesses module
from typing import Any
@register_atom(witness_marketmakerstateinit)  # type: ignore[untyped-decorator]
def witness_optimalquotecalculation(*args, **kwargs): pass
@register_atom(witness_marketmakerstateinit)
@icontract.require(lambda s0: isinstance(s0, (float, int, np.number)), "s0 must be numeric")
@icontract.require(lambda inventory: isinstance(inventory, (float, int, np.number)), "inventory must be numeric")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "MarketMakerStateInit all outputs must not be None")
def marketmakerstateinit(s0: float, inventory: float) -> tuple[float, float, float, float, float]:
    """Bootstraps the market-maker's immutable parameter state from a supplied initial mid-price and inventory position, materialising the five scalar fields — risk-aversion (gamma), market-depth (k), inventory (q), mid-price (s), and volatility (sigma) — that all downstream computations consume as pure inputs.

    Args:
        s0: s0 > 0
        inventory: Input data.

    Returns:
        gamma: gamma > 0
        k: k > 0
        q: Result data.
        s: s > 0
        sigma: sigma > 0
    """
@register_atom(witness_optimalquotecalculation)  # type: ignore[untyped-decorator]

@register_atom(witness_optimalquotecalculation)
@icontract.require(lambda gamma: isinstance(gamma, (float, int, np.number)), "gamma must be numeric")
@icontract.require(lambda k: isinstance(k, (float, int, np.number)), "k must be numeric")
@icontract.require(lambda q: isinstance(q, (float, int, np.number)), "q must be numeric")
@icontract.require(lambda s: isinstance(s, (float, int, np.number)), "s must be numeric")
@icontract.require(lambda sigma: isinstance(sigma, (float, int, np.number)), "sigma must be numeric")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "OptimalQuoteCalculation all outputs must not be None")
def optimalquotecalculation(gamma: float, k: float, q: float, s: float, sigma: float) -> tuple[float, float, float, float]:
    """
    Args:
        gamma: gamma > 0
        k: k > 0
        q: Input data.
        s: s > 0
        sigma: sigma > 0

    Returns:
        bid_price: bid_price < s
        ask_price: ask_price > s
        reservation_price: Result data.
        optimal_spread: optimal_spread > 0
    """
    raise NotImplementedError("Wire to original implementation")
