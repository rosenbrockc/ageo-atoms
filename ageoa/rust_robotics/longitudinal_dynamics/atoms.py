"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import icontract
from ageoa.ghost.registry import register_atom

from .witnesses import (
    witness_initialize_model,
    witness_compute_aerodynamic_force,
    witness_compute_rolling_force,
    witness_compute_gravity_grade_force,
    witness_evaluate_dynamics_derivatives,
    witness_linearize_dynamics,
    witness_solve_control_for_target_derivative,
    witness_deserialize_model_spec,
)


@register_atom(witness_initialize_model)
@icontract.require(lambda mass: isinstance(mass, (float, int, np.number)), "mass must be numeric")
@icontract.require(lambda area_frontal: isinstance(area_frontal, (float, int, np.number)), "area_frontal must be numeric")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def initialize_model(mass: float, area_frontal: float) -> object:
    """Create an immutable vehicle dynamics model from physical parameters.

    Args:
        mass: Vehicle mass, > 0.
        area_frontal: Frontal area, > 0.

    Returns:
        Immutable model value object for downstream pure calls.
    """
    raise NotImplementedError("Wire to original implementation")


@register_atom(witness_compute_aerodynamic_force)
@icontract.require(lambda velocity: isinstance(velocity, (float, int, np.number)), "velocity must be numeric")
@icontract.ensure(lambda result: isinstance(result, (float, int, np.number)), "result must be numeric")
def compute_aerodynamic_force(velocity: float) -> float:
    """Compute aerodynamic drag force from velocity.

    Args:
        velocity: Vehicle velocity; sign convention consistent with model.

    Returns:
        Drag force in model units.
    """
    raise NotImplementedError("Wire to original implementation")


@register_atom(witness_compute_rolling_force)
@icontract.require(lambda grade_angle: isinstance(grade_angle, (float, int, np.number)), "grade_angle must be numeric")
@icontract.ensure(lambda result: isinstance(result, (float, int, np.number)), "result must be numeric")
def compute_rolling_force(grade_angle: float) -> float:
    """Compute rolling resistance force from grade angle.

    Args:
        grade_angle: Road angle in radians.

    Returns:
        Rolling force in model units.
    """
    raise NotImplementedError("Wire to original implementation")


@register_atom(witness_compute_gravity_grade_force)
@icontract.require(lambda grade_angle: isinstance(grade_angle, (float, int, np.number)), "grade_angle must be numeric")
@icontract.ensure(lambda result: isinstance(result, (float, int, np.number)), "result must be numeric")
def compute_gravity_grade_force(grade_angle: float) -> float:
    """Compute gravity-induced force component along road grade.

    Args:
        grade_angle: Road angle in radians.

    Returns:
        Gravity force in model units.
    """
    raise NotImplementedError("Wire to original implementation")


@register_atom(witness_evaluate_dynamics_derivatives)
@icontract.require(lambda x: isinstance(x, np.ndarray), "x must be np.ndarray")
@icontract.require(lambda u: isinstance(u, np.ndarray), "u must be np.ndarray")
@icontract.require(lambda _t: isinstance(_t, (float, int, np.number)), "_t must be numeric")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
def evaluate_dynamics_derivatives(x: np.ndarray, u: np.ndarray, _t: float) -> np.ndarray:
    """Compute state derivatives for the system dynamics.

    Args:
        x: Current state vector.
        u: Control input vector.
        _t: Time scalar.

    Returns:
        State derivative vector, same dimension as x.
    """
    raise NotImplementedError("Wire to original implementation")


@register_atom(witness_linearize_dynamics)
@icontract.require(lambda x: isinstance(x, np.ndarray), "x must be np.ndarray")
@icontract.require(lambda _t: isinstance(_t, (float, int, np.number)), "_t must be numeric")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
def linearize_dynamics(x: np.ndarray, _u: np.ndarray, _t: float) -> np.ndarray:
    """Compute Jacobian of system dynamics with respect to state.

    Args:
        x: Current state vector.
        _u: Control input vector.
        _t: Time scalar.

    Returns:
        Jacobian matrix, state_dim x state_dim.
    """
    raise NotImplementedError("Wire to original implementation")


@register_atom(witness_solve_control_for_target_derivative)
@icontract.require(lambda x: isinstance(x, np.ndarray), "x must be np.ndarray")
@icontract.require(lambda x_dot_desired: isinstance(x_dot_desired, np.ndarray), "x_dot_desired must be np.ndarray")
@icontract.require(lambda _t: isinstance(_t, (float, int, np.number)), "_t must be numeric")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
def solve_control_for_target_derivative(x: np.ndarray, x_dot_desired: np.ndarray, _t: float) -> np.ndarray:
    """Compute control input to match desired state derivative.

    Args:
        x: Current state vector.
        x_dot_desired: Desired state derivative, same dimension as x.
        _t: Time scalar.

    Returns:
        Control input vector.
    """
    raise NotImplementedError("Wire to original implementation")


@register_atom(witness_deserialize_model_spec)
@icontract.require(lambda filename: isinstance(filename, str), "filename must be str")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def deserialize_model_spec(filename: str) -> object:
    """Load model parameters from file and construct model data.

    Args:
        filename: Path to readable model config file.

    Returns:
        Fully initialized model instance.
    """
    raise NotImplementedError("Wire to original implementation")
