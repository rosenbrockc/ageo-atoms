"""Ghost witnesses."""\n\nfrom ageoa.ghost.abstract import AbstractArray\n\ndef witness_n_joint_arm_solver(data: AbstractArray) -> AbstractArray:
    """Witness for n_joint_arm_solver."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_dijkstra_path_planning(data: AbstractArray) -> AbstractArray:
    """Witness for dijkstra_path_planning."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

