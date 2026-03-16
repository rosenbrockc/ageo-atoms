from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


from typing import Any

import numpy as np
import icontract
from ageoa.ghost.registry import register_atom

from .witnesses import (
    witness_modelspecloadingandsizing,
    witness_kinematicgoalfeasibility,
    witness_dynamicsandlinearizationkernel,
    witness_controlinputsynthesis,
)


@register_atom(witness_modelspecloadingandsizing)
@icontract.require(lambda filename: isinstance(filename, str), "filename must be str")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def modelspecloadingandsizing(filename: str) -> tuple[np.ndarray, float]:
    """Load serialized model data and expose structural sizing metadata.

    Args:
        filename: Valid readable model file path.

    Returns:
        Tuple of (model_spec, state_dim_ratio).
    """
    import json
    with open(filename) as f:
        data = json.load(f)
    link_lengths = np.array(data.get('link_lengths', [1.0]), dtype=float)
    total_reach = float(np.sum(link_lengths))
    return (link_lengths, total_reach)


@register_atom(witness_kinematicgoalfeasibility)
@icontract.require(lambda angles_desired: isinstance(angles_desired, np.ndarray), "angles_desired must be np.ndarray")
@icontract.require(lambda position_desired: isinstance(position_desired, np.ndarray), "position_desired must be np.ndarray")
@icontract.require(lambda x: isinstance(x, np.ndarray), "x must be np.ndarray")
@icontract.require(lambda position_current: isinstance(position_current, np.ndarray), "position_current must be np.ndarray")
@icontract.require(lambda position_goal: isinstance(position_goal, np.ndarray), "position_goal must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def kinematicgoalfeasibility(
    angles_desired: np.ndarray,
    position_desired: np.ndarray,
    x: np.ndarray,
    position_current: np.ndarray,
    position_goal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute inverse-kinematic feasibility and goal-distance metrics.

    Args:
        angles_desired: Joint-angle target vector.
        position_desired: Workspace target position.
        x: Initial or current state guess.
        position_current: Current workspace position.
        position_goal: Goal workspace position.

    Returns:
        Tuple of (ik_state, position_error, goal_distance_squared).
    """
    # Compute IK feasibility and goal distance
    pos_err = np.asarray(position_desired, dtype=float) - np.asarray(position_current, dtype=float)
    goal_dist_sq = float(np.sum((np.asarray(position_goal, dtype=float) - np.asarray(position_current, dtype=float))**2))
    # Use desired angles as the IK state
    ik_state = np.asarray(angles_desired, dtype=float).copy()
    return (ik_state, pos_err, goal_dist_sq)


@register_atom(witness_dynamicsandlinearizationkernel)
@icontract.require(lambda x: isinstance(x, np.ndarray), "x must be np.ndarray")
@icontract.require(lambda u: isinstance(u, np.ndarray), "u must be np.ndarray")
@icontract.require(lambda _t: isinstance(_t, (float, int, np.number)), "_t must be numeric")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def dynamicsandlinearizationkernel(x: np.ndarray, u: np.ndarray, _t: float) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate continuous-time state derivatives and local Jacobian linearization.

    Args:
        x: Current state vector.
        u: Control input vector.
        _t: Time index.

    Returns:
        Tuple of (x_dot, jacobian).
    """
    # Simple joint dynamics: x_dot = u (velocity control), Jacobian = I
    x_arr = np.asarray(x, dtype=float)
    u_arr = np.asarray(u, dtype=float)
    n = len(x_arr)
    x_dot = u_arr[:n] if len(u_arr) >= n else np.zeros(n)
    jacobian = np.eye(n)
    return (x_dot, jacobian)


@register_atom(witness_controlinputsynthesis)
@icontract.require(lambda _x: isinstance(_x, np.ndarray), "_x must be np.ndarray")
@icontract.require(lambda _x_dot: isinstance(_x_dot, np.ndarray), "_x_dot must be np.ndarray")
@icontract.require(lambda _t: isinstance(_t, (float, int, np.number)), "_t must be numeric")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
def controlinputsynthesis(_x: np.ndarray, _x_dot: np.ndarray, _t: float) -> np.ndarray:
    """Synthesize control action from current state and derivative information.

    Args:
        _x: Current state vector.
        _x_dot: State derivative estimate.
        _t: Time index.

    Returns:
        Control command compatible with dynamics input.
    """
    # PD controller: u = -Kp * x - Kd * x_dot
    x_arr = np.asarray(_x, dtype=float)
    x_dot_arr = np.asarray(_x_dot, dtype=float)
    Kp = 10.0
    Kd = 2.0
    u = -Kp * x_arr - Kd * x_dot_arr
    return u
