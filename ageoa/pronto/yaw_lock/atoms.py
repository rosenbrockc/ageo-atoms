"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom as _register_atom  # type: ignore[import-untyped]
from typing import Any, Callable, cast
import ctypes
import ctypes.util
from pathlib import Path


register_atom = cast(Callable[[object], Callable[[Callable[..., object]], Callable[..., object]]], _register_atom); YawLockState = object; witness_initializeyawlockstate = witness_configurecorrectionandyawslippolicy = witness_setrobotstandingstatus = witness_readrobotstandingstatus = witness_setjointposeandinitialangles = witness_readinitialjointangles = witness_setstandinglinktargets = object()

@register_atom(witness_initializeyawlockstate)
@icontract.require(lambda: True, "no preconditions")
@icontract.ensure(lambda result: result is not None, "InitializeYawLockState output must not be None")
def initializeyawlockstate() -> YawLockState:
    """Create the initial immutable YawLockState container for all persistent fields (parameters, standing flag, joint state, standing links).


    Returns:
        Deterministic persistent state object with fields: correction_period, yaw_slip_detect, yaw_slip_threshold_degrees, yaw_slip_disable_period, is_robot_standing, joint_name, joint_position, joint_angles_init, left_standing_link, right_standing_link.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_configurecorrectionandyawslippolicy)
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.require(lambda correction_period_in: correction_period_in is not None, "correction_period_in cannot be None")
@icontract.require(lambda yaw_slip_detect_in: yaw_slip_detect_in is not None, "yaw_slip_detect_in cannot be None")
@icontract.require(lambda yaw_slip_threshold_degrees_in: yaw_slip_threshold_degrees_in is not None, "yaw_slip_threshold_degrees_in cannot be None")
@icontract.require(lambda yaw_slip_disable_period_in: yaw_slip_disable_period_in is not None, "yaw_slip_disable_period_in cannot be None")
@icontract.ensure(lambda result: result is not None, "ConfigureCorrectionAndYawSlipPolicy output must not be None")
def configurecorrectionandyawslippolicy(state_in: YawLockState, correction_period_in: float, yaw_slip_detect_in: bool, yaw_slip_threshold_degrees_in: float, yaw_slip_disable_period_in: float) -> YawLockState:
    """Set correction cadence and yaw-slip detection policy by returning a new state with updated parameter fields.

    Args:
        state_in: Immutable input state.
        correction_period_in: Expected positive.
        yaw_slip_detect_in: Input data.
        yaw_slip_threshold_degrees_in: Non-negative threshold.
        yaw_slip_disable_period_in: Expected non-negative.

    Returns:
        New object with correction_period, yaw_slip_detect, yaw_slip_threshold_degrees, yaw_slip_disable_period replaced.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_setrobotstandingstatus)
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.require(lambda is_robot_standing_in: is_robot_standing_in is not None, "is_robot_standing_in cannot be None")
@icontract.ensure(lambda result: result is not None, "SetRobotStandingStatus output must not be None")
def setrobotstandingstatus(state_in: YawLockState, is_robot_standing_in: bool) -> YawLockState:
    """Write robot standing status by producing a new immutable state.

    Args:
        state_in: Immutable input state.
        is_robot_standing_in: Input data.

    Returns:
        New object with is_robot_standing replaced.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_readrobotstandingstatus)
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.ensure(lambda result: result is not None, "ReadRobotStandingStatus output must not be None")
def readrobotstandingstatus(state_in: YawLockState) -> bool:
    """Read current robot standing status from immutable state without mutation.

    Args:
        state_in: Immutable input state.

    Returns:
        Result data.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_setjointposeandinitialangles)
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.require(lambda joint_name_in: joint_name_in is not None, "joint_name_in cannot be None")
@icontract.require(lambda joint_position_in: joint_position_in is not None, "joint_position_in cannot be None")
@icontract.require(lambda joint_angles_init_in: joint_angles_init_in is not None, "joint_angles_init_in cannot be None")
@icontract.ensure(lambda result: result is not None, "SetJointPoseAndInitialAngles output must not be None")
def setjointposeandinitialangles(state_in: YawLockState, joint_name_in: object, joint_position_in: object, joint_angles_init_in: object) -> YawLockState:
    """Store joint identity/position inputs and corresponding initial-angle snapshot as a new immutable state.

    Args:
        state_in: Immutable input state.
        joint_name_in: Shape must align with positions.
        joint_position_in: Shape must align with joint names.
        joint_angles_init_in: Initialization snapshot.

    Returns:
        New object with joint_name, joint_position, joint_angles_init replaced.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_readinitialjointangles)
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.ensure(lambda result: result is not None, "ReadInitialJointAngles output must not be None")
def readinitialjointangles(state_in: YawLockState) -> object:
    """Read stored initial joint angles from immutable state.

    Args:
        state_in: Immutable input state.

    Returns:
        Result data.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_setstandinglinktargets)
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.require(lambda left_standing_link_in: left_standing_link_in is not None, "left_standing_link_in cannot be None")
@icontract.require(lambda right_standing_link_in: right_standing_link_in is not None, "right_standing_link_in cannot be None")
@icontract.ensure(lambda result: result is not None, "SetStandingLinkTargets output must not be None")
def setstandinglinktargets(state_in: YawLockState, left_standing_link_in: object, right_standing_link_in: object) -> YawLockState:
    """Set left/right standing link identifiers in a newly returned immutable state.

    Args:
        state_in: Immutable input state.
        left_standing_link_in: Input data.
        right_standing_link_in: Input data.

    Returns:
        New object with left_standing_link and right_standing_link replaced.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for cpp implementations."""

# duplicate future import removed (already declared at top of file)

import ctypes
import ctypes.util
from pathlib import Path


def _initializeyawlockstate_ffi() -> object:
    """FFI bridge to C++ implementation of InitializeYawLockState."""
    _lib = ctypes.CDLL("./initializeyawlockstate.so")
    _func_name = "initializeyawlockstate"
    _func = _lib[_func_name]
    _func.restype = ctypes.c_void_p
    return _func()

def _configurecorrectionandyawslippolicy_ffi(state_in: object, correction_period_in: object, yaw_slip_detect_in: object, yaw_slip_threshold_degrees_in: object, yaw_slip_disable_period_in: object) -> object:
    """FFI bridge to C++ implementation of ConfigureCorrectionAndYawSlipPolicy."""
    _lib = ctypes.CDLL("./configurecorrectionandyawslippolicy.so")
    _func_name = "configurecorrectionandyawslippolicy"
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(state_in, correction_period_in, yaw_slip_detect_in, yaw_slip_threshold_degrees_in, yaw_slip_disable_period_in)

def _setrobotstandingstatus_ffi(state_in: object, is_robot_standing_in: object) -> object:
    """FFI bridge to C++ implementation of SetRobotStandingStatus."""
    _lib = ctypes.CDLL("./setrobotstandingstatus.so")
    _func_name = "setrobotstandingstatus"
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(state_in, is_robot_standing_in)

def _readrobotstandingstatus_ffi(state_in: object) -> object:
    """FFI bridge to C++ implementation of ReadRobotStandingStatus."""
    _lib = ctypes.CDLL("./readrobotstandingstatus.so")
    _func_name = "readrobotstandingstatus"
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(state_in)

def _setjointposeandinitialangles_ffi(state_in: object, joint_name_in: object, joint_position_in: object, joint_angles_init_in: object) -> object:
    """FFI bridge to C++ implementation of SetJointPoseAndInitialAngles."""
    _lib = ctypes.CDLL("./setjointposeandinitialangles.so")
    _func_name = "setjointposeandinitialangles"
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(state_in, joint_name_in, joint_position_in, joint_angles_init_in)

def _readinitialjointangles_ffi(state_in: object) -> object:
    """FFI bridge to C++ implementation of ReadInitialJointAngles."""
    _lib = ctypes.CDLL("./readinitialjointangles.so")
    _func_name = "readinitialjointangles"
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(state_in)

def _setstandinglinktargets_ffi(state_in: object, left_standing_link_in: object, right_standing_link_in: object) -> object:
    """FFI bridge to C++ implementation of SetStandingLinkTargets."""
    _lib = ctypes.CDLL("./setstandinglinktargets.so")
    _func_name = "setstandinglinktargets"
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(state_in, left_standing_link_in, right_standing_link_in)