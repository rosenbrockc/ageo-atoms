from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_initializekalmanstatemodel(init_config: AbstractSignal) -> AbstractSignal:
    """Shape-and-type check for initialize kalman state model. Returns output metadata without running the real computation."""
    return AbstractSignal()


def witness_predictlatentstateandcovariance(state_in: AbstractSignal, u: AbstractArray, B: AbstractArray, F: AbstractArray, Q: AbstractArray) -> AbstractSignal:
    """Shape-and-type check for predict latent state and covariance. Returns output metadata without running the real computation."""
    return AbstractSignal()


def witness_predictlatentstatesteadystate(state_in: AbstractSignal, u: AbstractArray, B: AbstractArray) -> AbstractSignal:
    """Shape-and-type check for predict latent state steady state. Returns output metadata without running the real computation."""
    return AbstractSignal()


def witness_evaluatemeasurementoracle(x: AbstractArray, z: AbstractArray, H: AbstractArray) -> tuple[AbstractArray, AbstractArray]:
    """Shape-and-type check for evaluate measurement oracle. Returns output metadata without running the real computation."""
    z_pred = AbstractArray(shape=z.shape, dtype="float64")
    innovation = AbstractArray(shape=z.shape, dtype="float64")
    return z_pred, innovation


def witness_updateposteriorstateandcovariance(predicted_state: AbstractSignal, z: AbstractArray, R: AbstractArray, H: AbstractArray, innovation: AbstractArray) -> AbstractSignal:
    """Shape-and-type check for update posterior state and covariance. Returns output metadata without running the real computation."""
    return AbstractSignal()


def witness_updateposteriorstatesteadystate(predicted_state_steady: AbstractSignal, z: AbstractArray, innovation: AbstractArray) -> AbstractSignal:
    """Shape-and-type check for update posterior state steady state. Returns output metadata without running the real computation."""
    return AbstractSignal()
