"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations

import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore

try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
except ImportError:
    pass

def witness_initialize_model(mass: AbstractArray, area_frontal: AbstractArray) -> AbstractArray:
    """Ghost witness for initialize_model."""
    result = AbstractArray(
        shape=mass.shape,
        dtype="float64",
    )
    return result

def witness_compute_aerodynamic_force(velocity: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_aerodynamic_force."""
    result = AbstractArray(
        shape=velocity.shape,
        dtype="float64",
    )
    return result

def witness_compute_rolling_force(grade_angle: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_rolling_force."""
    result = AbstractArray(
        shape=grade_angle.shape,
        dtype="float64",
    )
    return result

def witness_compute_gravity_grade_force(grade_angle: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_gravity_grade_force."""
    result = AbstractArray(
        shape=grade_angle.shape,
        dtype="float64",
    )
    return result

def witness_evaluate_dynamics_derivatives(x: AbstractArray, u: AbstractArray, _t: AbstractArray) -> AbstractArray:
    """Ghost witness for evaluate_dynamics_derivatives."""
    result = AbstractArray(
        shape=x.shape,
        dtype="float64",
    )
    return result

def witness_linearize_dynamics(x: AbstractArray, _u: AbstractArray, _t: AbstractArray) -> AbstractArray:
    """Ghost witness for linearize_dynamics."""
    result = AbstractArray(
        shape=x.shape,
        dtype="float64",
    )
    return result

def witness_solve_control_for_target_derivative(x: AbstractArray, x_dot_desired: AbstractArray, _t: AbstractArray) -> AbstractArray:
    """Ghost witness for solve_control_for_target_derivative."""
    result = AbstractArray(
        shape=x.shape,
        dtype="float64",
    )
    return result

def witness_deserialize_model_spec(filename: AbstractArray) -> AbstractArray:
    """Ghost witness for deserialize_model_spec."""
    result = AbstractArray(
        shape=filename.shape,
        dtype="float64",
    )
    return result
