from __future__ import annotations
from typing import Any
"""Auto-generated Pydantic state models for cross-window state."""



import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore

from pydantic import BaseModel, ConfigDict, Field

class KalmanState(BaseModel):
    """Immutable snapshot of a scalar Kalman filter at a single time-step. X is the current posterior state estimate; P is the strictly-positive error covariance that quantifies estimation uncertainty; Q is the process-noise covariance (time-invariant); R is the measurement-noise covariance (time-invariant). Invariants: P > 0, Q > 0, R > 0. KalmanFilterInit produces the initial KalmanState; KalmanMeasurementUpdate consumes one KalmanState plus a scalar observation and returns a NEW KalmanState - the input is never mutated. Q and R are carried in the state so that every transition kernel is fully self-contained and requires no hidden instance members. The cross-window mutable fields (X, P) are explicitly threaded through state_in → state_out to satisfy immutable state-threading requirements."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    X: float | None = Field(default=None)
    P: float | None = Field(default=None)
    Q: float | None = Field(default=None)
    R: float | None = Field(default=None)