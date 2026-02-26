"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

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

witness_constructgeometrymodel: Any = None
witness_loadmodelfromfile: Any = None
witness_querygeometryparameters: Any = None
witness_computesideslipangle: Any = None
witness_computelinearizedstatematrices: Any = None
witness_evaluateandinvertdynamics: Any = None
atom: Any = None

# Witness functions should be imported from the generated witnesses module

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
    raise NotImplementedError("Wire to original implementation")

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
    raise NotImplementedError("Wire to original implementation")

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
    raise NotImplementedError("Wire to original implementation")

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
    raise NotImplementedError("Wire to original implementation")

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
    raise NotImplementedError("Wire to original implementation")

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
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for rust implementations."""

from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path


def _constructgeometrymodel_ffi(length_front: Any, length_rear: Any) -> Any:
    """FFI bridge to Rust implementation of ConstructGeometryModel."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'constructgeometrymodel'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(length_front, length_rear)

def _loadmodelfromfile_ffi(filename: Any) -> Any:
    """FFI bridge to Rust implementation of LoadModelFromFile."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'loadmodelfromfile'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(filename)

def _querygeometryparameters_ffi(model_spec: Any) -> Any:
    """FFI bridge to Rust implementation of QueryGeometryParameters."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'querygeometryparameters'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(model_spec)

def _computesideslipangle_ffi(model_spec: Any, road_wheel_angle: Any) -> Any:
    """FFI bridge to Rust implementation of ComputeSideslipAngle."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'computesideslipangle'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(model_spec, road_wheel_angle)

def _computelinearizedstatematrices_ffi(model_spec: Any, x: Any, u: Any) -> Any:
    """FFI bridge to Rust implementation of ComputeLinearizedStateMatrices."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'computelinearizedstatematrices'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(model_spec, x, u)

def _evaluateandinvertdynamics_ffi(model_spec: Any, x: Any, u: Any, _t: Any, _x_dot: Any) -> Any:
    """FFI bridge to Rust implementation of EvaluateAndInvertDynamics."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'evaluateandinvertdynamics'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(model_spec, x, u, _t, _x_dot)