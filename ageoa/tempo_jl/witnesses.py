"""Ghost witnesses."""\n\nfrom ageoa.ghost.abstract import AbstractArray\n\ndef witness_graph_time_scale_management(data: AbstractArray) -> AbstractArray:
    """Witness for graph_time_scale_management."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_high_precision_duration(data: AbstractArray) -> AbstractArray:
    """Witness for high_precision_duration."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

