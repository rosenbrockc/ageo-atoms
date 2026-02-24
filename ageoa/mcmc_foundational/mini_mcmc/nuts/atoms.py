"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom

import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_find_reasonable_epsilon)
@icontract.require(lambda position: position is not None, "position cannot be None")
@icontract.require(lambda mom: mom is not None, "mom cannot be None")
@icontract.require(lambda gradient_target: gradient_target is not None, "gradient_target cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Find Reasonable Epsilon output must not be None")
def find_reasonable_epsilon(position: Any, mom: Any, gradient_target: Any) -> T:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_build_tree)
@icontract.ensure(lambda result, **kwargs: result is not None, "Build Tree output must not be None")
def build_tree(position: Any, mom: Any, grad: Any, logu: Any, v: Any, j: Any, epsilon: Any, gradient_target: Any, joint_0: Any, rng: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_all_real)
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "All Real output must not be None")
def all_real(x: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_stop_criterion)
@icontract.require(lambda position_minus: position_minus is not None, "position_minus cannot be None")
@icontract.require(lambda position_plus: position_plus is not None, "position_plus cannot be None")
@icontract.require(lambda mom_minus: mom_minus is not None, "mom_minus cannot be None")
@icontract.require(lambda mom_plus: mom_plus is not None, "mom_plus cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Stop Criterion output must not be None")
def stop_criterion(position_minus: Any, position_plus: Any, mom_minus: Any, mom_plus: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_leapfrog)
@icontract.ensure(lambda result, **kwargs: result is not None, "Leapfrog output must not be None")
def leapfrog(position: Any, mom: Any, grad: Any, epsilon: Any, gradient_target: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for rust implementations."""

from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path


def find_reasonable_epsilon_ffi(position, mom, gradient_target):
    """FFI bridge to Rust implementation of Find Reasonable Epsilon."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'find_reasonable_epsilon'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(position, mom, gradient_target)

def build_tree_ffi(position, mom, grad, logu, v, j, epsilon, gradient_target, joint_0, rng):
    """FFI bridge to Rust implementation of Build Tree."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'build_tree'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(position, mom, grad, logu, v, j, epsilon, gradient_target, joint_0, rng)

def all_real_ffi(x):
    """FFI bridge to Rust implementation of All Real."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'all_real'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(x)

def stop_criterion_ffi(position_minus, position_plus, mom_minus, mom_plus):
    """FFI bridge to Rust implementation of Stop Criterion."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'stop_criterion'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(position_minus, position_plus, mom_minus, mom_plus)

def leapfrog_ffi(position, mom, grad, epsilon, gradient_target):
    """FFI bridge to Rust implementation of Leapfrog."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = atom.method_names[0] if atom.method_names else 'leapfrog'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(position, mom, grad, epsilon, gradient_target)
