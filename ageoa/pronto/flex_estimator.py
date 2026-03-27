from __future__ import annotations

from typing import Any
Boolean: Any = Any

"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import icontract
from ageoa.ghost.registry import register_atom
from .flex_estimator_witnesses import witness_estimate_flex_deflection
# Boolean already defined as type alias above


@register_atom(witness_estimate_flex_deflection)
@icontract.require(lambda hip_positions: hip_positions.ndim >= 1, "hip_positions must have at least one dimension")
@icontract.require(lambda hip_efforts: hip_efforts.ndim >= 1, "hip_efforts must have at least one dimension")
@icontract.require(lambda stance_mask: stance_mask.ndim >= 1, "stance_mask must have at least one dimension")
@icontract.require(lambda hip_positions: hip_positions is not None, "hip_positions cannot be None")
@icontract.require(lambda hip_positions: isinstance(hip_positions, np.ndarray), "hip_positions must be np.ndarray")
@icontract.require(lambda hip_efforts: hip_efforts is not None, "hip_efforts cannot be None")
@icontract.require(lambda hip_efforts: isinstance(hip_efforts, np.ndarray), "hip_efforts must be np.ndarray")
@icontract.require(lambda stance_mask: stance_mask is not None, "stance_mask cannot be None")
@icontract.require(lambda stance_mask: isinstance(stance_mask, np.ndarray), "stance_mask must be np.ndarray")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def estimate_flex_deflection(hip_positions: np.ndarray, hip_efforts: np.ndarray, stance_mask: np.ndarray) -> np.ndarray:
    """Estimates hip flexure deflection for a quadruped robot using kinematic state and joint effort to correct leg odometry.

    Args:
        hip_positions: Hip joint positions for each leg, shape (4, 3)
        hip_efforts: Measured joint torques/efforts at each hip, shape (4,)
        stance_mask: Boolean stance state for each leg, shape (4,)

    Returns:
        Estimated flex deflection vector per leg, shape (4,)
    """
    # Estimate flex deflection using joint effort and stance mask
    # For stance legs, deflection is proportional to torque (compliance model)
    # For swing legs, deflection is zero
    compliance = 1e-3  # rad/Nm compliance constant
    deflection = np.zeros(hip_efforts.shape[0], dtype=np.float64)
    for i in range(hip_efforts.shape[0]):
        if stance_mask[i]:
            deflection[i] = compliance * hip_efforts[i]
    return deflection
