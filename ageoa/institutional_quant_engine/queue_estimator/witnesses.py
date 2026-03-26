from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_initializeorderstate(my_order_id: AbstractScalar, my_qty: AbstractScalar, orders_ahead: AbstractScalar, state: AbstractArray, *args, **kwargs) -> AbstractArray:
    """Shape-and-type check for prior init: initialize order state. Returns output metadata without running the real computation."""
    return AbstractArray(shape=(1,), dtype="float64")
    

def witness_updatequeueontrade(current_order_state: AbstractArray, trade_qty: AbstractArray, state: AbstractArray, *args, **kwargs) -> tuple[AbstractArray, AbstractArray]:
    """Shape-and-type check for update queue on trade. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=(1,),
        dtype="float64",)
    new_state = AbstractArray(shape=(1,), dtype="float64")
    return result, new_state