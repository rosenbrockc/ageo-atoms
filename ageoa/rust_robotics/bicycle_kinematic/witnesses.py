from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_constructgeometrymodel(length_front: AbstractArray, length_rear: AbstractArray) -> AbstractArray:
    """Shape-and-type check for construct geometry model. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=length_front.shape,
        dtype="float64",
    )
    return result

def witness_loadmodelfromfile(filename: AbstractArray) -> AbstractArray:
    """Shape-and-type check for load model from file. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=filename.shape,
        dtype="float64",
    )
    return result

def witness_querygeometryparameters(model_spec: AbstractArray) -> AbstractArray:
    """Shape-and-type check for query geometry parameters. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=model_spec.shape,
        dtype="float64",
    )
    return result

def witness_computesideslipangle(model_spec: AbstractArray, road_wheel_angle: AbstractArray) -> AbstractArray:
    """Shape-and-type check for compute sideslip angle. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=model_spec.shape,
        dtype="float64",
    )
    return result

def witness_computelinearizedstatematrices(model_spec: AbstractArray, x: AbstractArray, u: AbstractArray) -> AbstractArray:
    """Shape-and-type check for compute linearized state matrices. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=model_spec.shape,
        dtype="float64",
    )
    return result

def witness_evaluateandinvertdynamics(model_spec: AbstractArray, x: AbstractArray, u: AbstractArray, _t: AbstractArray, _x_dot: AbstractArray) -> AbstractArray:
    """Shape-and-type check for evaluate and invert dynamics. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=model_spec.shape,
        dtype="float64",
    )
    return result
