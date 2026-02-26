"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations



try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
    from ageoa.ghost.abstract import AbstractDistribution
except ImportError:
    pass

def witness_evaluate_log_probability_density(dist: AbstractDistribution, samples: AbstractArray) -> AbstractScalar:
    """Ghost witness for log-prob: evaluate_log_probability_density."""
    n_event = len(dist.event_shape)
    if n_event > 0:
        sample_tail = samples.shape[-n_event:]
        if sample_tail != dist.event_shape:
            raise ValueError(
                f"Sample dims {sample_tail} vs event_shape {dist.event_shape}"
            )
    return AbstractScalar(dtype="float64", max_val=0.0)
