from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_initializekalmanstatemodel(init_config: AbstractSignal) -> AbstractSignal:
    """Ghost witness for InitializeKalmanStateModel."""
    return AbstractSignal()


def witness_predictlatentstateandcovariance(state_in: AbstractSignal, u: AbstractArray, B: AbstractArray, F: AbstractArray, Q: AbstractArray) -> AbstractSignal:
    """Ghost witness for PredictLatentStateAndCovariance."""
    return AbstractSignal()


def witness_predictlatentstatesteadystate(state_in: AbstractSignal, u: AbstractArray, B: AbstractArray) -> AbstractSignal:
    """Ghost witness for PredictLatentStateSteadyState."""
    return AbstractSignal()


def witness_evaluatemeasurementoracle(x: AbstractArray, z: AbstractArray, H: AbstractArray) -> tuple[AbstractArray, AbstractArray]:
    """Ghost witness for EvaluateMeasurementOracle."""
    z_pred = AbstractArray(shape=z.shape, dtype="float64")
    innovation = AbstractArray(shape=z.shape, dtype="float64")
    return z_pred, innovation


def witness_updateposteriorstateandcovariance(predicted_state: AbstractSignal, z: AbstractArray, R: AbstractArray, H: AbstractArray, innovation: AbstractArray) -> AbstractSignal:
    """Ghost witness for UpdatePosteriorStateAndCovariance."""
    return AbstractSignal()


def witness_updateposteriorstatesteadystate(predicted_state_steady: AbstractSignal, z: AbstractArray, innovation: AbstractArray) -> AbstractSignal:
    """Ghost witness for UpdatePosteriorStateSteadyState."""
    return AbstractSignal()
