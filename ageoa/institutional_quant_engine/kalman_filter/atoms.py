from __future__ import annotations
"""Auto-generated stateful atom wrappers following the ageoa pattern."""


import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import *  # type: ignore[import-untyped]

from typing import Any

# Import the original class for __new__ instantiation
# from <source_module> import KalmanFilter
KalmanFilter: Any

# State model should be imported from the generated state_models module
# from <state_module> import KalmanState

# Witness functions should be imported from the generated witnesses module
witness_kalmanfilterinit: Any
witness_kalmanmeasurementupdate: Any
@register_atom(witness_kalmanfilterinit)
@icontract.require(lambda process_variance: isinstance(process_variance, (float, int, np.number)), "process_variance must be numeric")
@icontract.require(lambda measurement_variance: isinstance(measurement_variance, (float, int, np.number)), "measurement_variance must be numeric")
@icontract.require(lambda estimated_measurement_variance: isinstance(estimated_measurement_variance, (float, int, np.number)), "estimated_measurement_variance must be numeric")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "KalmanFilterInit all outputs must not be None")
def kalmanfilterinit(process_variance: float, measurement_variance: float, estimated_measurement_variance: float, state: KalmanState) -> tuple[tuple[float, float, float, float], KalmanState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Bootstraps the Kalman filter state by initialising the error covariance matrix P, process-noise covariance Q, measurement-noise covariance R, and the prior state estimate X. No computation is performed; all values are stored as immutable starting state for downstream kernels.

    Args:
        process_variance: > 0
        measurement_variance: > 0
        estimated_measurement_variance: > 0
        state: KalmanState object containing cross-window persistent state.

    Returns:
        tuple[tuple[P, Q, R, X], KalmanState]:
            The first element is the functional result, the second is the updated state.
    """
    obj = KalmanFilter.__new__(KalmanFilter)
    obj.X = state.X
    obj.P = state.P
    obj.Q = state.Q
    obj.R = state.R
    new_state = state.model_copy(update={
        "X": obj.X,
        "P": obj.P,
        "Q": obj.Q,
@register_atom(witness_kalmanmeasurementupdate)  # type: ignore[untyped-decorator]
        "R": obj.R,
    result = (obj.P, obj.Q, obj.R, obj.X)
    return result, new_state

@register_atom(witness_kalmanmeasurementupdate)
@icontract.require(lambda P: isinstance(P, (float, int, np.number)), "P must be numeric")
@icontract.require(lambda Q: isinstance(Q, (float, int, np.number)), "Q must be numeric")
@icontract.require(lambda R: isinstance(R, (float, int, np.number)), "R must be numeric")
@icontract.require(lambda X: isinstance(X, (float, int, np.number)), "X must be numeric")
@icontract.require(lambda measurement: isinstance(measurement, (float, int, np.number)), "measurement must be numeric")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "KalmanMeasurementUpdate all outputs must not be None")
def kalmanmeasurementupdate(P: float, Q: float, R: float, X: float, measurement: float, state: KalmanState) -> tuple[tuple[float, float], KalmanState]:
    """Stateless wrapper: Functional Core, Imperative Shell.

    Pure functional Kalman predict-and-update kernel. Consumes the current immutable state tuple (P, Q, R, X) together with a scalar measurement and produces a brand-new state tuple (P_new, X_new). Internally fuses the time-update (predict) step — advancing P by Q — with the measurement-update step that computes the Kalman gain, corrects the state estimate X, and contracts P by the complement of the gain. Returns new objects; the input state is never mutated.

    Args:
        P: > 0
        Q: > 0
        R: > 0
        X: scalar
        measurement: scalar
        state: KalmanState object containing cross-window persistent state.

    Returns:
        tuple[tuple[P_new, X_new], KalmanState]:
            The first element is the functional result, the second is the updated state.
    """
    obj = KalmanFilter.__new__(KalmanFilter)
    obj.X = state.X
    obj.P = state.P
    obj.Q = state.Q
    obj.R = state.R
    obj.update(P, Q, R, X, measurement)
    new_state = state.model_copy(update={
        "X": obj.X,
        "P": obj.P,
        "Q": obj.Q,
        "R": obj.R,
    })
    result = (obj.P_new, obj.X_new)
    return result, new_state
