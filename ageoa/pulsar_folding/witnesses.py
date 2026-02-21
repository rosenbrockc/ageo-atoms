"""Ghost witnesses."""\n\nfrom ageoa.ghost.abstract import AbstractArray\n\ndef witness_dm_can_brute_force(data: AbstractArray) -> AbstractArray:
    """Witness for dm_can_brute_force."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_spline_bandpass_correction(data: AbstractArray) -> AbstractArray:
    """Witness for spline_bandpass_correction."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

