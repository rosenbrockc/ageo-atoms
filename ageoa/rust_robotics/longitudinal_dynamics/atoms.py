"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

import ctypes
import ctypes.util
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])
Model = object
@register_atom_typed(witness_initialize_model)

# Witness functions should be imported from the generated witnesses module
witness_initialize_model: object = object()
witness_compute_aerodynamic_force: object = object()
witness_compute_rolling_force: object = object()
witness_compute_gravity_grade_force: object = object()
witness_evaluate_dynamics_derivatives: object = object()
witness_linearize_dynamics: object = object()
witness_solve_control_for_target_derivative: object = object()
witness_deserialize_model_spec: object = object()
@register_atom(witness_initialize_model)
@icontract.require(lambda mass: isinstance(mass, (float, int, np.number)), "mass must be numeric")
@icontract.require(lambda area_frontal: isinstance(area_frontal, (float, int, np.number)), "area_frontal must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "initialize_model output must not be None")
def initialize_model(mass: float, area_frontal: float) -> Model:
@register_atom_typed(witness_compute_aerodynamic_force)

    Args:
        mass: mass > 0
        area_frontal: area_frontal > 0

    Returns:
        immutable value object for downstream pure calls
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_compute_aerodynamic_force)
@icontract.require(lambda velocity: isinstance(velocity, (float, int, np.number)), "velocity must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "compute_aerodynamic_force output must not be None")
@register_atom_typed(witness_compute_rolling_force)
    """Compute aerodynamic drag force from velocity.

    Args:
        velocity: real-valued; sign convention consistent with model

    Returns:
        force in model units
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_compute_rolling_force)
@icontract.require(lambda grade_angle: isinstance(grade_angle, (float, int, np.number)), "grade_angle must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "compute_rolling_force output must not be None")
@register_atom_typed(witness_compute_gravity_grade_force)
    """Compute rolling resistance force from grade angle.

    Args:
        grade_angle: angle in radians (or consistent internal unit)

    Returns:
        force in model units
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_compute_gravity_grade_force)
@icontract.require(lambda grade_angle: isinstance(grade_angle, (float, int, np.number)), "grade_angle must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "compute_gravity_grade_force output must not be None")
@register_atom_typed(witness_evaluate_dynamics_derivatives)
    """Compute gravity-induced force component along the road grade.

    Args:
        grade_angle: angle in radians (or consistent internal unit)

    Returns:
        force in model units
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_evaluate_dynamics_derivatives)
@icontract.require(lambda _t: isinstance(_t, (float, int, np.number)), "_t must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "evaluate_dynamics_derivatives output must not be None")
def evaluate_dynamics_derivatives(x: object, u: object, _t: float) -> object:
    """Compute state derivatives for the system dynamics.
@register_atom_typed(witness_linearize_dynamics)
    Args:
        x: shape must match model state dimension
        u: shape must match input dimension
        _t: time scalar; may be unused

    Returns:
        same state dimension as x
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_linearize_dynamics)
@icontract.require(lambda _t: isinstance(_t, (float, int, np.number)), "_t must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "linearize_dynamics output must not be None")
def linearize_dynamics(x: object, _u: object, _t: float) -> object:
    """Compute Jacobian of system dynamics with respect to state.
@register_atom_typed(witness_solve_control_for_target_derivative)
    Args:
        x: shape must match model state dimension
        _u: shape must match input dimension; may be unused
        _t: time scalar; may be unused

    Returns:
        state_dim x state_dim
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_solve_control_for_target_derivative)
@icontract.require(lambda _t: isinstance(_t, (float, int, np.number)), "_t must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "solve_control_for_target_derivative output must not be None")
def solve_control_for_target_derivative(x: object, x_dot_desired: object, _t: float) -> object:
    """Compute control input required to match desired state derivative.
@register_atom_typed(witness_deserialize_model_spec)
    Args:
        x: shape must match model state dimension
        x_dot_desired: same dimension as x
        _t: time scalar; may be unused

    Returns:
        shape must match input dimension
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_deserialize_model_spec)
@icontract.require(lambda filename: filename is not None, "filename cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "deserialize_model_spec output must not be None")
def deserialize_model_spec(filename: object) -> Model:
    """Load model parameters from file and construct model-ready data.

    Args:
        filename: must reference readable model config file

    Returns:
        fully initialized model instance
    """
class _AtomShim:
def initialize_model_ffi(mass: object, area_frontal: object) -> object:

atom = _AtomShim()


"""Auto-generated FFI bindings for rust implementations."""

from __future__ import annotations
# from __future__ import annotations
import ctypes
def compute_aerodynamic_force_ffi(velocity: object) -> object:
from pathlib import Path


def initialize_model_ffi(mass, area_frontal):
    """FFI bridge to Rust implementation of initialize_model."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'initialize_model'
    _func = _lib[_func_name]
def compute_rolling_force_ffi(grade_angle: object) -> object:
    _func.restype = ctypes.c_void_p
    return _func(mass, area_frontal)

def compute_aerodynamic_force_ffi(velocity):
    """FFI bridge to Rust implementation of compute_aerodynamic_force."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'compute_aerodynamic_force'
    _func = _lib[_func_name]
def compute_gravity_grade_force_ffi(grade_angle: object) -> object:
    _func.restype = ctypes.c_void_p
    return _func(velocity)

def compute_rolling_force_ffi(grade_angle):
    """FFI bridge to Rust implementation of compute_rolling_force."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'compute_rolling_force'
    _func = _lib[_func_name]
def evaluate_dynamics_derivatives_ffi(x: object, u: object, _t: object) -> object:
    _func.restype = ctypes.c_void_p
    return _func(grade_angle)

def compute_gravity_grade_force_ffi(grade_angle):
    """FFI bridge to Rust implementation of compute_gravity_grade_force."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'compute_gravity_grade_force'
    _func = _lib[_func_name]
def linearize_dynamics_ffi(x: object, _u: object, _t: object) -> object:
    _func.restype = ctypes.c_void_p
    return _func(grade_angle)

def evaluate_dynamics_derivatives_ffi(x, u, _t):
    """FFI bridge to Rust implementation of evaluate_dynamics_derivatives."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'evaluate_dynamics_derivatives'
    _func = _lib[_func_name]
def solve_control_for_target_derivative_ffi(x: object, x_dot_desired: object, _t: object) -> object:
    _func.restype = ctypes.c_void_p
    return _func(x, u, _t)

def linearize_dynamics_ffi(x, _u, _t):
    """FFI bridge to Rust implementation of linearize_dynamics."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'linearize_dynamics'
    _func = _lib[_func_name]
def deserialize_model_spec_ffi(filename: object) -> object:
    _func.restype = ctypes.c_void_p
    return _func(x, _u, _t)

def solve_control_for_target_derivative_ffi(x, x_dot_desired, _t):
    """FFI bridge to Rust implementation of solve_control_for_target_derivative."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'solve_control_for_target_derivative'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(x, x_dot_desired, _t)

def deserialize_model_spec_ffi(filename):
    """FFI bridge to Rust implementation of deserialize_model_spec."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'deserialize_model_spec'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(filename)