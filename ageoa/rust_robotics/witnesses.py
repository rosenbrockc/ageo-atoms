from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_n_joint_arm_solver(data: AbstractArray) -> AbstractArray:
    """Shape-and-type check for n joint arm solver. Returns output metadata without running the real computation."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_dijkstra_path_planning(data: AbstractArray) -> AbstractArray:
    """Shape-and-type check for dijkstra path planning. Returns output metadata without running the real computation."""
    return AbstractArray(shape=data.shape, dtype=data.dtype)


def witness_pure_pursuit(
    position_current: AbstractArray,
    position_target: AbstractArray,
    yaw_current: float,
    target_distance: float,
    wheelbase: float,
) -> AbstractArray:
    """Witness for pure_pursuit.

    Validates that target_distance and wheelbase are positive, then
    returns an AbstractArray with scalar shape representing the steering angle.
    """
    if target_distance <= 0:
        raise ValueError(f"target_distance must be positive, got {target_distance}")
    if wheelbase <= 0:
        raise ValueError(f"wheelbase must be positive, got {wheelbase}")
    return AbstractArray(shape=(), dtype="float64")


def witness_rk4(
    func: object,
    x0: AbstractArray,
    t0: float,
    tf: float,
) -> AbstractArray:
    """Witness for rk4.

    Validates that tf > t0 and returns an AbstractArray matching x0 shape.
    """
    if tf <= t0:
        raise ValueError(f"tf must be greater than t0, got t0={t0}, tf={tf}")
    return AbstractArray(shape=x0.shape, dtype=x0.dtype)
