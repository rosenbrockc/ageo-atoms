"""Ghost witnesses for Pronto atoms."""

from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractScalar


class AbstractEKFState:
    """Lightweight metadata for EKF State tracking."""

    def __init__(self, vec_shape: tuple[int, ...], quat_shape: tuple[int, ...]) -> None:
        self.vec_shape = vec_shape
        self.quat_shape = quat_shape


def witness_ekf_update(
    state,
    gyro: AbstractArray,
    accel: AbstractArray,
    dt: AbstractScalar,
) -> AbstractEKFState:
    """Ghost witness for EKF predict step."""
    del dt
    if state.vec.shape != (21,):
        raise ValueError("State vec must be (21,)")
    if state.quat.shape != (4,):
        raise ValueError("State quat must be (4,)")
    if gyro.shape != (3,):
        raise ValueError("Gyro must be (3,)")
    if accel.shape != (3,):
        raise ValueError("Accel must be (3,)")

    return AbstractEKFState((21,), (4,))


def witness_contact_gating(
    classifier_ptr: AbstractScalar,
    utime: AbstractScalar,
    left_contact: bool,
    right_contact: bool,
    left_contact_strong: bool,
    right_contact_strong: bool,
) -> AbstractScalar:
    """Ghost witness for Contact Classifier update loop."""
    del (
        classifier_ptr,
        utime,
        left_contact,
        right_contact,
        left_contact_strong,
        right_contact_strong,
    )
    return AbstractScalar(dtype="int32")
