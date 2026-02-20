"""Ghost witnesses for rust_robotics atoms."""

from __future__ import annotations

from typing import Any

from ageoa.ghost.abstract import AbstractArray, AbstractScalar


def witness_pure_pursuit(
    position_current: AbstractArray,
    position_target: AbstractArray,
    yaw_current: AbstractScalar,
    target_distance: AbstractScalar,
    wheelbase: AbstractScalar,
) -> AbstractScalar:
    """Pure pursuit outputs a scalar steering angle."""
    del yaw_current, target_distance, wheelbase
    if position_current.shape != (2,):
        raise ValueError("position_current must be a 2-vector in abstract simulation")
    if position_target.shape != (2,):
        raise ValueError("position_target must be a 2-vector in abstract simulation")
    return AbstractScalar(dtype="float64")


def witness_rk4(
    func: Any,
    x0: AbstractArray,
    t0: AbstractScalar,
    tf: AbstractScalar,
) -> AbstractArray:
    """RK4 preserves vector shape and emits float64 state."""
    del func, t0, tf
    if len(x0.shape) != 1:
        raise ValueError("x0 must be a 1D vector in abstract simulation")
    return AbstractArray(shape=x0.shape, dtype="float64")
