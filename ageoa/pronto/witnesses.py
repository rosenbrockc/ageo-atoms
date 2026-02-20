"""Ghost witnesses for Pronto atoms."""

from ageoa.ghost.abstract import AbstractSignal

class AbstractEKFState:
    """Lightweight metadata for EKF State tracking"""
    def __init__(self, vec_shape, quat_shape):
        self.vec_shape = vec_shape
        self.quat_shape = quat_shape

def witness_ekf_update(state, gyro, accel, dt):
    """Ghost witness for EKF predict step"""
    if state.vec.shape != (21,):
        raise ValueError("State vec must be (21,)")
    if state.quat.shape != (4,):
        raise ValueError("State quat must be (4,)")
    if gyro.shape != (3,):
        raise ValueError("Gyro must be (3,)")
    if accel.shape != (3,):
        raise ValueError("Accel must be (3,)")
        
    return AbstractEKFState((21,), (4,))

def witness_contact_gating(classifier_ptr, utime, left_contact, right_contact, left_contact_strong, right_contact_strong):
    """Ghost witness for Contact Classifier update loop"""
    return AbstractSignal(
        shape=(), 
        dtype="int32", 
        sampling_rate=0.0, 
        domain="mode", 
        units="mode"
    )
