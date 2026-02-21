"""Ghost witnesses."""\n\nfrom ageoa.ghost.abstract import AbstractArray\n\ndef witness_market_making_avellaneda(data: AbstractArray) -> AbstractArray:
    """Witness for market_making_avellaneda."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_almgren_chriss_execution(data: AbstractArray) -> AbstractArray:
    """Witness for almgren_chriss_execution."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_pin_informed_trading(data: AbstractArray) -> AbstractArray:
    """Witness for pin_informed_trading."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_limit_order_queue_estimator(data: AbstractArray) -> AbstractArray:
    """Witness for limit_order_queue_estimator."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

