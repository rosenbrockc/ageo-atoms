import ctypes
import os
import pathlib
from typing import Callable

import numpy as np
import icontract

# Load the shared library
_lib_path = pathlib.Path(__file__).parent / "librust_robotics.dylib"
_lib = ctypes.CDLL(str(_lib_path))

# C signature for rk4_ffi:
# pub extern "C" fn rk4_ffi(func_ptr: OdeFuncC, x0_ptr: *const f64, dim: usize, t0: f64, tf: f64, out_ptr: *mut f64)
# OdeFuncC = extern "C" fn(x: *const f64, t: f64, dx: *mut f64, dim: usize)
ODE_CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t)

_lib.rk4_ffi.argtypes = [
    ODE_CALLBACK_TYPE,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
]
_lib.rk4_ffi.restype = None

@icontract.require(lambda x0: x0.ndim == 1, "x0 must be a 1D vector")
@icontract.require(lambda tf, t0: tf > t0, "tf must be strictly greater than t0")
@icontract.ensure(lambda result, x0: result.shape == x0.shape, "result must preserve x0 shape")
def rk4(
    func: Callable[[np.ndarray, float], np.ndarray],
    x0: np.ndarray,
    t0: float,
    tf: float,
) -> np.ndarray:
    """Solve an ODE using the Runge-Kutta 4th order (RK4) method.

    This atom wraps the rust_robotics FFI RK4 solver using ctypes.

    Args:
        func: Python callable of the form dx_dt = func(x, t), where x and
            dx_dt are 1D float NumPy arrays of the same size.
        x0: Initial state vector as a 1D float NumPy array.
        t0: Initial time.
        tf: Final time.

    Returns:
        The final state vector after one integration step (t0 -> tf).
    """
    dim = x0.shape[0]

    def _c_callback(x_ptr, t, dx_ptr, dim):
        # Convert C pointer to numpy array (read-only)
        x_array = np.ctypeslib.as_array(x_ptr, shape=(dim,))
        
        # Call the python function
        dx_array = func(x_array, t)
        
        # Copy the result into the out pointer
        out_dx_array = np.ctypeslib.as_array(dx_ptr, shape=(dim,))
        np.copyto(out_dx_array, dx_array.astype(np.float64, copy=False))

    callback = ODE_CALLBACK_TYPE(_c_callback)

    x0_c = x0.astype(np.float64, copy=True)
    out_c = np.empty_like(x0_c)

    _lib.rk4_ffi(
        callback,
        x0_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(dim),
        ctypes.c_double(t0),
        ctypes.c_double(tf),
        out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    return out_c
