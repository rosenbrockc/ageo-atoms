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

def witness_constructgeometrymodel(length_front: AbstractArray, length_rear: AbstractArray) -> AbstractArray:
    """Ghost witness for ConstructGeometryModel."""
    result = AbstractArray(
        shape=length_front.shape,
        dtype="float64",
    )
    return result

def witness_loadmodelfromfile(filename: AbstractArray) -> AbstractArray:
    """Ghost witness for LoadModelFromFile."""
    result = AbstractArray(
        shape=filename.shape,
        dtype="float64",
    )
    return result

def witness_querygeometryparameters(model_spec: AbstractArray) -> AbstractArray:
    """Ghost witness for QueryGeometryParameters."""
    result = AbstractArray(
        shape=model_spec.shape,
        dtype="float64",
    )
    return result

def witness_computesideslipangle(model_spec: AbstractArray, road_wheel_angle: AbstractArray) -> AbstractArray:
    """Ghost witness for ComputeSideslipAngle."""
    result = AbstractArray(
        shape=model_spec.shape,
        dtype="float64",
    )
    return result

def witness_computelinearizedstatematrices(model_spec: AbstractArray, x: AbstractArray, u: AbstractArray) -> AbstractArray:
    """Ghost witness for ComputeLinearizedStateMatrices."""
    result = AbstractArray(
        shape=model_spec.shape,
        dtype="float64",
    )
    return result

def witness_evaluateandinvertdynamics(model_spec: AbstractArray, x: AbstractArray, u: AbstractArray, _t: AbstractArray, _x_dot: AbstractArray) -> AbstractArray:
    """Ghost witness for EvaluateAndInvertDynamics."""
    result = AbstractArray(
        shape=model_spec.shape,
        dtype="float64",
    )
    return result
