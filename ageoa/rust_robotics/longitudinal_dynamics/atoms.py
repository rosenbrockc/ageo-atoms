from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import icontract
from typing import TypedDict
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


class VehicleModelSpec(TypedDict):
    mass: float
    area_frontal: float
    Cd: float
    rho: float
    Cr: float
    g: float


_MODEL: VehicleModelSpec = {
    'mass': 1500.0,
    'area_frontal': 2.2,
    'Cd': 0.3,
    'rho': 1.225,
    'Cr': 0.01,
    'g': 9.81,
}

@register_atom(witness_initialize_model)
@icontract.require(lambda mass: isinstance(mass, (float, int, np.number)), "mass must be numeric")
@icontract.require(lambda area_frontal: isinstance(area_frontal, (float, int, np.number)), "area_frontal must be numeric")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def initialize_model(mass: float, area_frontal: float) -> VehicleModelSpec:
    """Create an immutable vehicle dynamics model from physical parameters.

    Args:
        mass: Vehicle mass, > 0.
        area_frontal: Frontal area, > 0.

    Returns:
        Immutable model value object for downstream pure calls.
    """
    global _MODEL
    model: VehicleModelSpec = {
        'mass': float(mass),
        'area_frontal': float(area_frontal),
        'Cd': 0.3,
        'rho': 1.225,
        'Cr': 0.01,
        'g': 9.81,
    }
    _MODEL = model
    return model


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
    v = float(velocity)
    area = _MODEL['area_frontal']
    rho = _MODEL['rho']
    Cd = _MODEL['Cd']
    return float(0.5 * rho * Cd * area * v * v)


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
    grade = float(grade_angle)
    mass = _MODEL['mass']
    g = _MODEL['g']
    Cr = _MODEL['Cr']
    return float(mass * g * Cr * np.cos(grade))


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
    grade = float(grade_angle)
    mass = _MODEL['mass']
    g = _MODEL['g']
    return float(mass * g * np.sin(grade))


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
    # State: [position, velocity], Control: [F_traction, grade_angle]
    x_arr = np.asarray(x, dtype=float)
    u_arr = np.asarray(u, dtype=float)
    velocity = x_arr[1]
    F_traction = u_arr[0]
    grade = u_arr[1] if len(u_arr) > 1 else 0.0
    mass = _MODEL['mass']
    area = _MODEL['area_frontal']
    rho = _MODEL['rho']
    Cd = _MODEL['Cd']
    Cr = _MODEL['Cr']
    g = _MODEL['g']
    F_aero = 0.5 * rho * Cd * area * velocity**2
    F_roll = mass * g * Cr * np.cos(grade)
    F_grade = mass * g * np.sin(grade)
    accel = (F_traction - F_aero - F_roll - F_grade) / mass
    return np.array([velocity, accel])


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
    # Jacobian df/dx for state [position, velocity]
    x_arr = np.asarray(x, dtype=float)
    velocity = x_arr[1]
    mass = _MODEL['mass']
    area = _MODEL['area_frontal']
    rho = _MODEL['rho']
    Cd = _MODEL['Cd']
    # d(accel)/d(velocity) = -rho*Cd*area*v / mass
    A = np.zeros((2, 2))
    A[0, 1] = 1.0
    A[1, 1] = -rho * Cd * area * velocity / mass
    return A


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
    # Invert dynamics: given desired x_dot, find required control
    x_arr = np.asarray(x, dtype=float)
    x_dot_arr = np.asarray(x_dot_desired, dtype=float)
    velocity = x_arr[1]
    desired_accel = x_dot_arr[1]
    mass = _MODEL['mass']
    area = _MODEL['area_frontal']
    rho = _MODEL['rho']
    Cd = _MODEL['Cd']
    Cr = _MODEL['Cr']
    g = _MODEL['g']
    grade = 0.0
    F_aero = 0.5 * rho * Cd * area * velocity**2
    F_roll = mass * g * Cr * np.cos(grade)
    F_grade = mass * g * np.sin(grade)
    F_traction = mass * desired_accel + F_aero + F_roll + F_grade
    return np.array([F_traction, grade])


@register_atom(witness_deserialize_model_spec)
@icontract.require(lambda filename: isinstance(filename, str), "filename must be str")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def deserialize_model_spec(filename: str) -> VehicleModelSpec:
    """Load model parameters from file and construct model data.

    Args:
        filename: Path to readable model config file.

    Returns:
        Fully initialized model instance.
    """
    import json
    with open(filename) as f:
        data = json.load(f)
    return initialize_model(data.get('mass', 1500.0), data.get('area_frontal', 2.2))
