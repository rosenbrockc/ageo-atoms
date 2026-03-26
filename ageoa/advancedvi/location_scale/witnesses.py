from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal

def witness_evaluate_log_probability_density(dist: AbstractDistribution, samples: AbstractArray) -> AbstractScalar:
    """Shape-and-type check for log-prob: evaluate log probability density. Returns output metadata without running the real computation."""
    n_event = len(dist.event_shape)
    if n_event > 0:
        sample_tail = samples.shape[-n_event:]
        if sample_tail != dist.event_shape:
            raise ValueError(
                f"Sample dims {sample_tail} vs event_shape {dist.event_shape}"
            )
    return AbstractScalar(dtype="float64", max_val=0.0)
