"""Ghost Witness functions for Pulsar Folding pipeline."""

from __future__ import annotations
from ageoa.ghost.abstract import AbstractSignal, AbstractScalar

def witness_de_disperse(data: AbstractSignal, DM: float, fchan: float, width: float, tsamp: float) -> AbstractSignal:
    """Ghost witness for dedispersion.
    
    Postconditions:
        - Preserves 2D spectrogram shape (Time, Frequency).
    """
    data.assert_domain("time")
    return AbstractSignal(
        shape=data.shape,
        dtype="float64",
        sampling_rate=data.sampling_rate,
        domain="time",
        units="power",
    )

def witness_fold_signal(data: AbstractSignal, period: int) -> AbstractSignal:
    """Ghost witness for signal folding.
    
    Postconditions:
        - Reduces 2D (Time, Frequency) to 1D (Period,).
    """
    data.assert_domain("time")
    if len(data.shape) < 2:
        raise ValueError("Fold signal requires 2D input (Time, Frequency)")
    
    return AbstractSignal(
        shape=(period,),
        dtype="float64",
        sampling_rate=1.0 / (period * (1.0 / data.sampling_rate)),
        domain="time",
        units="normalized_power",
    )

def witness_snr(profile: AbstractSignal) -> AbstractScalar:
    """Ghost witness for SNR calculation."""
    return AbstractScalar(
        dtype="float64",
        min_val=0.0,
    )

def witness_delay_from_dm(DM: float, freq: float) -> AbstractScalar:
    """Ghost witness for DM delay calculation."""
    return AbstractScalar(dtype="float64", min_val=0.0)
