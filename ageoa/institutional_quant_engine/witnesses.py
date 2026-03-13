from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_almgren_chriss_execution(data, *args, **kwargs):
    """Witness for almgren_chriss_execution."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_pin_informed_trading(data: AbstractArray) -> AbstractArray:
    """Witness for pin_informed_trading."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_limit_order_queue_estimator(data: AbstractArray) -> AbstractArray:
    """Witness for limit_order_queue_estimator."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_market_making_avellaneda(data, *args, **kwargs):
    """Skeleton for witness_market_making_avellaneda."""
    return AbstractArray(shape=data.shape, dtype='float64')