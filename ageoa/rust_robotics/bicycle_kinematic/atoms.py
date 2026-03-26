from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_computelinearizedstatematrices, witness_computesideslipangle, witness_constructgeometrymodel, witness_evaluateandinvertdynamics, witness_loadmodelfromfile, witness_querygeometryparameters

import ctypes
import ctypes.util
from pathlib import Path
from typing import Any

ModelSpec = Any
StateVector = Any
ControlVector = Any
StateDerivativeVector = Any
Matrix = Any
string = str


# Witness functions are imported from the generated witnesses module

@register_atom(witness_constructgeometrymodel)  # type: ignore[untyped-decorator]
@icontract.require(lambda length_front: isinstance(length_front, (float, int, np.number)), "length_front must be numeric")
@icontract.require(lambda length_rear: isinstance(length_rear, (float, int, np.number)), "length_rear must be numeric")
@icontract.ensure(lambda result: result is not None, "ConstructGeometryModel output must not be None")
def constructgeometrymodel(length_front: float, length_rear: float) -> ModelSpec:
    """Create an immutable vehicle geometry/state model from explicit axle-length parameters.

    Args:
        length_front: >= 0
        length_rear: >= 0

    Returns:
        immutable geometry parameters
    """
    return {'lf': float(length_front), 'lr': float(length_rear), 'L': float(length_front + length_rear)}

@register_atom(witness_loadmodelfromfile)  # type: ignore[untyped-decorator]
@icontract.require(lambda filename: filename is not None, "filename cannot be None")
@icontract.ensure(lambda result: result is not None, "LoadModelFromFile output must not be None")
def loadmodelfromfile(filename: string) -> ModelSpec:
    """Deserialize model geometry parameters from storage into an immutable model spec.

    Args:
        filename: readable model file path

    Returns:
        immutable geometry parameters
    """
    import json
    with open(filename) as f:
        return json.load(f)

@register_atom(witness_querygeometryparameters)  # type: ignore[untyped-decorator]
@icontract.require(lambda model_spec: model_spec is not None, "model_spec cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "QueryGeometryParameters all outputs must not be None")
def querygeometryparameters(model_spec: ModelSpec) -> tuple[float, float, float]:
    """Project front length, rear length, and derived wheelbase from the immutable model spec.

    Args:
        model_spec: must contain front/rear lengths

    Returns:
        length_front: >= 0
        length_rear: >= 0
        wheelbase: length_front + length_rear
    """
    return (model_spec['lf'], model_spec['lr'], model_spec['L'])

@register_atom(witness_computesideslipangle)  # type: ignore[untyped-decorator]
@icontract.require(lambda road_wheel_angle: isinstance(road_wheel_angle, (float, int, np.number)), "road_wheel_angle must be numeric")
@icontract.ensure(lambda result: result is not None, "ComputeSideslipAngle output must not be None")
def computesideslipangle(model_spec: ModelSpec, road_wheel_angle: float) -> float:
    """Compute sideslip from steering input and vehicle geometry as a pure kinematic transform.

    Args:
        model_spec: immutable geometry parameters
        road_wheel_angle: steering angle in radians

    Returns:
        kinematic slip angle in radians
    """
    lf = model_spec['lf']
    lr = model_spec['lr']
    delta = float(road_wheel_angle)
    beta = float(np.arctan(lr / (lf + lr) * np.tan(delta)))
    return beta

@register_atom(witness_computelinearizedstatematrices)  # type: ignore[untyped-decorator]
@icontract.require(lambda model_spec: model_spec is not None, "model_spec cannot be None")
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.require(lambda u: u is not None, "u cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "ComputeLinearizedStateMatrices all outputs must not be None")
def computelinearizedstatematrices(model_spec: ModelSpec, x: StateVector, u: ControlVector) -> tuple[Matrix, Matrix]:
    """Compute linearized system matrices for local dynamics around state/control operating point.

    Args:
        model_spec: immutable geometry parameters
        x: valid model state
        u: valid control input

    Returns:
        A: state Jacobian / linearization matrix
        B: input Jacobian / linearization matrix
    """
    lf = model_spec['lf']
    lr = model_spec['lr']
    L = model_spec['L']
    # State: [x_pos, y_pos, theta, v], Control: [delta, a]
    x_arr = np.asarray(x, dtype=float)
    u_arr = np.asarray(u, dtype=float)
    theta = x_arr[2]
    v = x_arr[3]
    delta = u_arr[0]
    beta = np.arctan(lr / L * np.tan(delta))
    # A = df/dx
    A = np.zeros((4, 4))
    A[0, 2] = -v * np.sin(theta + beta)
    A[0, 3] = np.cos(theta + beta)
    A[1, 2] = v * np.cos(theta + beta)
    A[1, 3] = np.sin(theta + beta)
    A[2, 3] = np.sin(beta) / lr if lr > 0 else 0.0
    # B = df/du
    dbeta_ddelta = (lr / L) / (np.cos(delta)**2) / (1.0 + (lr / L * np.tan(delta))**2)
    B = np.zeros((4, 2))
    B[0, 0] = -v * np.sin(theta + beta) * dbeta_ddelta
    B[1, 0] = v * np.cos(theta + beta) * dbeta_ddelta
    B[2, 0] = v * np.cos(beta) * dbeta_ddelta / lr if lr > 0 else 0.0
    B[3, 1] = 1.0
    return (A, B)

@register_atom(witness_evaluateandinvertdynamics)  # type: ignore[untyped-decorator]
@icontract.require(lambda _t: isinstance(_t, (float, int, np.number)), "_t must be numeric")
@icontract.ensure(lambda result: all(r is not None for r in result), "EvaluateAndInvertDynamics all outputs must not be None")
def evaluateandinvertdynamics(model_spec: ModelSpec, x: StateVector, u: ControlVector, _t: float, _x_dot: StateDerivativeVector) -> tuple[StateDerivativeVector, Matrix, ControlVector]:
    """Evaluate nonlinear derivatives, compute Jacobian at time t, and solve inverse-input mapping as pure transforms.

    Args:
        model_spec: immutable geometry parameters
        x: valid model state
        u: required for forward dynamics/jacobian
        _t: evaluation time
        _x_dot: required for inverse input solve

    Returns:
        x_dot: forward model derivative
        jacobian: dynamics Jacobian at (x,u,t)
        u_inferred: inverse-mapped control from (x,x_dot,t)
    """
    lf = model_spec['lf']
    lr = model_spec['lr']
    L = model_spec['L']
    x_arr = np.asarray(x, dtype=float)
    u_arr = np.asarray(u, dtype=float)
    theta = x_arr[2]
    v = x_arr[3]
    delta = u_arr[0]
    a = u_arr[1]
    beta = np.arctan(lr / L * np.tan(delta))
    # Forward dynamics
    x_dot = np.array([
        v * np.cos(theta + beta),
        v * np.sin(theta + beta),
        v * np.sin(beta) / lr if lr > 0 else 0.0,
        a,
    ])
    # Jacobian (same as linearized A matrix)
    jacobian = np.zeros((4, 4))
    jacobian[0, 2] = -v * np.sin(theta + beta)
    jacobian[0, 3] = np.cos(theta + beta)
    jacobian[1, 2] = v * np.cos(theta + beta)
    jacobian[1, 3] = np.sin(theta + beta)
    jacobian[2, 3] = np.sin(beta) / lr if lr > 0 else 0.0
    # Invert dynamics: given _x_dot, recover u
    _x_dot_arr = np.asarray(_x_dot, dtype=float)
    u_inferred = np.array([delta, _x_dot_arr[3]])
    return (x_dot, jacobian, u_inferred)


"""Auto-generated FFI bindings for rust implementations."""


import ctypes
import ctypes.util
from pathlib import Path


def _constructgeometrymodel_ffi(length_front: ctypes.c_double, length_rear: ctypes.c_double) -> ctypes.c_void_p:
    """Wrapper that calls the Rust version of construct geometry model. Passes arguments through and returns the result."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'constructgeometrymodel_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_double, ctypes.c_double]
    _func.restype = ctypes.c_void_p
    return _func(length_front, length_rear)

def _loadmodelfromfile_ffi(filename: ctypes.c_char_p) -> ctypes.c_void_p:
    """Wrapper that calls the Rust version of load model from file. Passes arguments through and returns the result."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'loadmodelfromfile_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_char_p]
    _func.restype = ctypes.c_void_p
    return _func(filename)

def _querygeometryparameters_ffi(model_spec: ctypes.c_void_p) -> ctypes.c_void_p:
    """Wrapper that calls the Rust version of query geometry parameters. Passes arguments through and returns the result."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'querygeometryparameters_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(model_spec)

def _computesideslipangle_ffi(model_spec: ctypes.c_void_p, road_wheel_angle: ctypes.c_double) -> ctypes.c_void_p:
    """Wrapper that calls the Rust version of compute sideslip angle. Passes arguments through and returns the result."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'computesideslipangle_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_double]
    _func.restype = ctypes.c_void_p
    return _func(model_spec, road_wheel_angle)

def _computelinearizedstatematrices_ffi(model_spec: ctypes.c_void_p, x: ctypes.c_void_p, u: ctypes.c_void_p) -> ctypes.c_void_p:
    """Wrapper that calls the Rust version of compute linearized state matrices. Passes arguments through and returns the result."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'computelinearizedstatematrices_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(model_spec, x, u)

def _evaluateandinvertdynamics_ffi(model_spec: ctypes.c_void_p, x: ctypes.c_void_p, u: ctypes.c_void_p, _t: ctypes.c_double, _x_dot: ctypes.c_void_p) -> ctypes.c_void_p:
    """Wrapper that calls the Rust version of evaluate and invert dynamics. Passes arguments through and returns the result."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'evaluateandinvertdynamics_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(model_spec, x, u, _t, _x_dot)
