"""Ghost witnesses."""\n\nfrom ageoa.ghost.abstract import AbstractArray\n\ndef witness_rbis_state_estimation(data: AbstractArray) -> AbstractArray:
    """Witness for rbis_state_estimation."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

