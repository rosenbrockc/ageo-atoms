import ctypes
import pathlib

import numpy as np
import icontract
from pydantic import BaseModel, ConfigDict, Field

from ageoa.ghost.registry import register_atom
from .witnesses import witness_ekf_update, witness_contact_gating

_lib: ctypes.CDLL | None = None
_signatures_configured = False


def _get_lib() -> ctypes.CDLL:
    global _lib, _signatures_configured
    if _lib is None:
        lib_path = pathlib.Path(__file__).parent / "libpronto_ffi.dylib"
        try:
            _lib = ctypes.CDLL(str(lib_path))
        except OSError as exc:
            raise RuntimeError(
                f"Unable to load pronto shared library at '{lib_path}'"
            ) from exc

    if not _signatures_configured:
        _lib.ekf_update_state.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.POINTER(ctypes.c_int64),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_double,
        ]
        _lib.ekf_update_state.restype = None

        _lib.foot_contact_classifier_create.argtypes = []
        _lib.foot_contact_classifier_create.restype = ctypes.c_void_p

        _lib.foot_contact_classifier_destroy.argtypes = [ctypes.c_void_p]
        _lib.foot_contact_classifier_destroy.restype = None

        _lib.foot_contact_classifier_update.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.c_bool,
        ]
        _lib.foot_contact_classifier_update.restype = ctypes.c_int
        _signatures_configured = True

    return _lib

class EKFState(BaseModel):
    """Pydantic memory bridge to C++ Eigen EKF state."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    vec: np.ndarray = Field(..., description="21-element RBIS state vector")
    quat: np.ndarray = Field(..., description="4-element quaternion (w, x, y, z)")
    utime: int = Field(..., description="Microsecond timestamp")

@register_atom(witness_ekf_update)
@icontract.require(lambda state: state.vec.shape == (21,))
@icontract.require(lambda state: state.quat.shape == (4,))
@icontract.require(lambda gyro: gyro.shape == (3,))
@icontract.require(lambda accel: accel.shape == (3,))
@icontract.require(lambda dt: dt > 0.0, "dt must be strictly positive")
@icontract.ensure(lambda result: result.vec.shape == (21,))
@icontract.ensure(lambda result: result.quat.shape == (4,))
def ekf_update(state: EKFState, gyro: np.ndarray, accel: np.ndarray, dt: float) -> EKFState:
    """Propagate an extended Kalman filter state estimate through a nonlinear dynamics model.

    Provides a pure stateless Python CDG node tracking C++ memory safely via FFI.
    """
    vec_c = state.vec.astype(np.float64, copy=True)
    quat_c = state.quat.astype(np.float64, copy=True)
    utime_c = ctypes.c_int64(state.utime)
    
    g_c = gyro.astype(np.float64, copy=False)
    a_c = accel.astype(np.float64, copy=False)
    
    lib = _get_lib()
    lib.ekf_update_state(
        vec_c,
        quat_c,
        ctypes.byref(utime_c),
        g_c,
        a_c,
        ctypes.c_double(dt),
    )
    
    return EKFState(vec=vec_c, quat=quat_c, utime=utime_c.value)


@register_atom(witness_contact_gating)
@icontract.require(lambda classifier_ptr: classifier_ptr != 0, "classifier_ptr must be non-zero")
def contact_classifier_update(
    classifier_ptr: int,
    utime: int,
    left_contact: bool,
    right_contact: bool,
    left_contact_strong: bool,
    right_contact_strong: bool
) -> int:
    """Probabilistic Contact Gating heuristic.
    """
    lib = _get_lib()
    result = lib.foot_contact_classifier_update(
        ctypes.c_void_p(classifier_ptr),
        ctypes.c_int64(utime),
        ctypes.c_bool(left_contact),
        ctypes.c_bool(right_contact),
        ctypes.c_bool(left_contact_strong),
        ctypes.c_bool(right_contact_strong)
    )
    return result

def contact_classifier_create() -> int:
    """Instantiate a new C++ contact classifier."""
    ptr = _get_lib().foot_contact_classifier_create()
    return ptr or 0

def contact_classifier_destroy(ptr: int):
    """Free memory associated with the contact classifier."""
    if ptr:
        _get_lib().foot_contact_classifier_destroy(ctypes.c_void_p(ptr))
