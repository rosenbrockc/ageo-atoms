"""Ghost witnesses."""\n\nfrom ageoa.ghost.abstract import AbstractArray\n\ndef witness_functional_monte_carlo(data: AbstractArray) -> AbstractArray:
    """Witness for functional_monte_carlo."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_volatility_surface_modeling(data: AbstractArray) -> AbstractArray:
    """Witness for volatility_surface_modeling."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

