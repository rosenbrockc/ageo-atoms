from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_initialize_model(mass: AbstractArray, area_frontal: AbstractArray) -> AbstractArray:
    """Shape-and-type check for initialize model. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=mass.shape,
        dtype="float64",
    )
    return result

def witness_compute_aerodynamic_force(velocity: AbstractArray) -> AbstractArray:
    """Shape-and-type check for compute aerodynamic force. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=velocity.shape,
        dtype="float64",
    )
    return result

def witness_compute_rolling_force(grade_angle: AbstractArray) -> AbstractArray:
    """Shape-and-type check for compute rolling force. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=grade_angle.shape,
        dtype="float64",
    )
    return result

def witness_compute_gravity_grade_force(grade_angle: AbstractArray) -> AbstractArray:
    """Shape-and-type check for compute gravity grade force. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=grade_angle.shape,
        dtype="float64",
    )
    return result

def witness_evaluate_dynamics_derivatives(x: AbstractArray, u: AbstractArray, _t: AbstractArray) -> AbstractArray:
    """Shape-and-type check for evaluate dynamics derivatives. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=x.shape,
        dtype="float64",
    )
    return result

def witness_linearize_dynamics(x: AbstractArray, _u: AbstractArray, _t: AbstractArray) -> AbstractArray:
    """Shape-and-type check for linearize dynamics. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=x.shape,
        dtype="float64",
    )
    return result

def witness_solve_control_for_target_derivative(x: AbstractArray, x_dot_desired: AbstractArray, _t: AbstractArray) -> AbstractArray:
    """Shape-and-type check for solve control for target derivative. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=x.shape,
        dtype="float64",
    )
    return result

def witness_deserialize_model_spec(filename: AbstractArray) -> AbstractArray:
    """Shape-and-type check for deserialize model spec. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=filename.shape,
        dtype="float64",
    )
    return result
