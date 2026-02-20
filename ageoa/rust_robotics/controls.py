import ctypes
import pathlib

import icontract
from pydantic import BaseModel, Field

from ageoa.ghost.registry import register_atom
from ageoa.rust_robotics.witnesses import witness_pure_pursuit

class Point2D(BaseModel):
    """Pydantic BaseModel representing a 2D Point (equivalent to na::Point2)."""
    x: float = Field(..., description="x coordinate")
    y: float = Field(..., description="y coordinate")

class RecordPoint(BaseModel):
    """Pydantic BaseModel representing a 3D RecordPoint with time."""
    time: float = Field(..., description="Time of the record")
    x: float = Field(..., description="x coordinate")
    y: float = Field(..., description="y coordinate")
    z: float = Field(..., description="z coordinate")

_lib: ctypes.CDLL | None = None
_pure_pursuit_signature_configured = False


def _get_lib() -> ctypes.CDLL:
    global _lib, _pure_pursuit_signature_configured
    if _lib is None:
        lib_path = pathlib.Path(__file__).parent / "librust_robotics.dylib"
        try:
            _lib = ctypes.CDLL(str(lib_path))
        except OSError as exc:
            raise RuntimeError(
                f"Unable to load rust_robotics shared library at '{lib_path}'"
            ) from exc

    if not _pure_pursuit_signature_configured:
        _lib.pure_pursuit_ffi.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        _lib.pure_pursuit_ffi.restype = ctypes.c_double
        _pure_pursuit_signature_configured = True

    return _lib

@register_atom(witness_pure_pursuit)
@icontract.require(lambda position_current: position_current is not None, "position_current must be non-null")
@icontract.require(lambda position_target: position_target is not None, "position_target must be non-null")
@icontract.require(lambda target_distance: target_distance > 0, "target_distance must be strictly positive")
@icontract.require(lambda wheelbase: wheelbase > 0, "wheelbase must be strictly positive")
@icontract.ensure(lambda result: isinstance(result, float), "result must be a float representing the steering angle")
def pure_pursuit(
    position_current: Point2D,
    position_target: Point2D,
    yaw_current: float,
    target_distance: float,
    wheelbase: float,
) -> float:
    """Calculate the steering angle using the Pure Pursuit algorithm.

    This atom wraps the rust_robotics FFI pure_pursuit implementation.

    Args:
        position_current: Current 2D position (x, y) as a Pydantic Point2D.
        position_target: Target 2D position (x, y) as a Pydantic Point2D.
        yaw_current: Current yaw (heading) angle in radians.
        target_distance: Lookahead distance to the target in meters.
        wheelbase: The vehicle's wheelbase in meters.

    Returns:
        Steering angle in radians.
    """
    lib = _get_lib()
    return float(
        lib.pure_pursuit_ffi(
            float(position_current.x),
            float(position_current.y),
            float(position_target.x),
            float(position_target.y),
            float(yaw_current),
            float(target_distance),
            float(wheelbase),
        )
    )
