from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_initializeorderstate(my_order_id, my_qty, orders_ahead, state, *args, **kwargs):
    """Ghost witness for prior init: InitializeOrderState."""
    return AbstractArray(shape=(1,), dtype="float64")
    

def witness_updatequeueontrade(current_order_state, trade_qty, state, *args, **kwargs):
    """Ghost witness for UpdateQueueOnTrade."""
    result = AbstractArray(
        shape=(1,),
        dtype="float64",)
    new_state = AbstractArray(shape=(1,), dtype="float64")
    return result, new_state