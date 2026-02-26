"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations



try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
except ImportError:
    pass

def witness_initializeyawlockstate() -> AbstractArray:
    """Ghost witness for InitializeYawLockState."""
    return AbstractArray(shape=("S",), dtype="float64")

def witness_configurecorrectionandyawslippolicy(state_in: AbstractArray, correction_period_in: AbstractArray, yaw_slip_detect_in: AbstractArray, yaw_slip_threshold_degrees_in: AbstractArray, yaw_slip_disable_period_in: AbstractArray) -> AbstractArray:
    """Ghost witness for ConfigureCorrectionAndYawSlipPolicy."""
    result = AbstractArray(
        shape=state_in.shape,
        dtype="float64",
    )
    return result

def witness_setrobotstandingstatus(state_in: AbstractArray, is_robot_standing_in: AbstractArray) -> AbstractArray:
    """Ghost witness for SetRobotStandingStatus."""
    result = AbstractArray(
        shape=state_in.shape,
        dtype="float64",
    )
    return result

def witness_readrobotstandingstatus(state_in: AbstractArray) -> AbstractArray:
    """Ghost witness for ReadRobotStandingStatus."""
    result = AbstractArray(
        shape=state_in.shape,
        dtype="float64",
    )
    return result

def witness_setjointposeandinitialangles(state_in: AbstractArray, joint_name_in: AbstractArray, joint_position_in: AbstractArray, joint_angles_init_in: AbstractArray) -> AbstractArray:
    """Ghost witness for SetJointPoseAndInitialAngles."""
    result = AbstractArray(
        shape=state_in.shape,
        dtype="float64",
    )
    return result

def witness_readinitialjointangles(state_in: AbstractArray) -> AbstractArray:
    """Ghost witness for ReadInitialJointAngles."""
    result = AbstractArray(
        shape=state_in.shape,
        dtype="float64",
    )
    return result

def witness_setstandinglinktargets(state_in: AbstractArray, left_standing_link_in: AbstractArray, right_standing_link_in: AbstractArray) -> AbstractArray:
    """Ghost witness for SetStandingLinkTargets."""
    result = AbstractArray(
        shape=state_in.shape,
        dtype="float64",
    )
    return result
