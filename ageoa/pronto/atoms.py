import ctypes
import pathlib
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_ekf_update, witness_contact_gating

# Load library
_lib_path = pathlib.Path(__file__).parent / "libpronto_ffi.dylib"
_lib = ctypes.CDLL(str(_lib_path))

class EKFState(BaseModel):
    """Pydantic memory bridge to C++ Eigen EKF state."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    vec: np.ndarray = Field(..., description="21-element RBIS state vector")
    quat: np.ndarray = Field(..., description="4-element quaternion (w, x, y, z)")
    utime: int = Field(..., description="Microsecond timestamp")

_lib.ekf_update_state.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.POINTER(ctypes.c_int64),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_double
]
_lib.ekf_update_state.restype = None

@register_atom(witness_ekf_update)
@icontract.require(lambda state: state.vec.shape == (21,))
@icontract.require(lambda state: state.quat.shape == (4,))
@icontract.require(lambda gyro: gyro.shape == (3,))
@icontract.require(lambda accel: accel.shape == (3,))
@icontract.ensure(lambda result: result.vec.shape == (21,))
@icontract.ensure(lambda result: result.quat.shape == (4,))
def ekf_update(state: EKFState, gyro: np.ndarray, accel: np.ndarray, dt: float) -> EKFState:
    """Propagate an extended Kalman filter state estimate through a nonlinear dynamics model.

    Provides a pure stateless Python CDG node tracking C++ memory safely via FFI.

    <!-- conceptual_profile
    {
        "abstract_name": "Non-Linear Rigid-Body State Propagator",
        "conceptual_transform": "Propagates the high-dimensional state of a rigid body (position, velocity, orientation, and sensor biases) through a small time increment using angular velocity and linear acceleration inputs. It implements a non-linear motion model that maintains the integrity of orientation representations (quaternions).",
        "abstract_inputs": [
            {
                "name": "state",
                "description": "An object containing the current 21-element state vector and orientation quaternion."
            },
            {
                "name": "gyro",
                "description": "A 3-element vector representing instantaneous angular velocity."
            },
            {
                "name": "accel",
                "description": "A 3-element vector representing instantaneous linear acceleration."
            },
            {
                "name": "dt",
                "description": "A scalar representing the integration time step."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "An object containing the predicted future state and orientation."
            }
        ],
        "algorithmic_properties": [
            "non-linear-integration",
            "state-propagation",
            "rigid-body-kinematics",
            "quaternion-based"
        ],
        "cross_disciplinary_applications": [
            "Fusing inertial measurement streams with position fixes for mobile platform state estimation.",
            "Tracking the evolving state of an orbiting body from sparse range observations.",
            "Estimating pose in reference-denied environments using proprioceptive sensor fusion."
        ]
    }
    /conceptual_profile -->
    """
    vec_c = state.vec.astype(np.float64, copy=True)
    quat_c = state.quat.astype(np.float64, copy=True)
    utime_c = ctypes.c_int64(state.utime)
    
    g_c = gyro.astype(np.float64, copy=False)
    a_c = accel.astype(np.float64, copy=False)
    
    _lib.ekf_update_state(vec_c, quat_c, ctypes.byref(utime_c), g_c, a_c, ctypes.c_double(dt))
    
    return EKFState(vec=vec_c, quat=quat_c, utime=utime_c.value)


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
    ctypes.c_bool
]
_lib.foot_contact_classifier_update.restype = ctypes.c_int

@register_atom(witness_contact_gating)
def contact_classifier_update(
    classifier_ptr: int,
    utime: int,
    left_contact: bool,
    right_contact: bool,
    left_contact_strong: bool,
    right_contact_strong: bool
) -> int:
    """Probabilistic Contact Gating heuristic.

    <!-- conceptual_profile
    {
        "abstract_name": "Probabilistic Boundary Interaction Classifier",
        "conceptual_transform": "Updates the classification of discrete boundary interaction events based on a stream of temporal measurements and heuristic strength indicators. It maintains a probabilistic internal state to resolve whether a specific interface is in an active contact or released state.",
        "abstract_inputs": [
            {
                "name": "classifier_ptr",
                "description": "An integer handle to a persistent internal classifier state."
            },
            {
                "name": "utime",
                "description": "A temporal coordinate (timestamp)."
            },
            {
                "name": "left_contact",
                "description": "Boolean indicator of potential interface contact (left/primary)."
            },
            {
                "name": "right_contact",
                "description": "Boolean indicator of potential interface contact (right/secondary)."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "An integer representing the resolved classification state."
            }
        ],
        "algorithmic_properties": [
            "stateful-classification",
            "heuristic-fusion",
            "temporal-event-detection"
        ],
        "cross_disciplinary_applications": [
            "Detecting discrete contact events in a multi-limbed articulated mechanism.",
            "Classifying grasp states in an automated pick-and-place system.",
            "Identifying the engagement/disengagement states of a mechanical coupling."
        ]
    }
    /conceptual_profile -->
    """
    result = _lib.foot_contact_classifier_update(
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
    ptr = _lib.foot_contact_classifier_create()
    return ptr or 0

def contact_classifier_destroy(ptr: int):
    """Free memory associated with the contact classifier."""
    if ptr:
        _lib.foot_contact_classifier_destroy(ctypes.c_void_p(ptr))
