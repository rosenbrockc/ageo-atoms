"""Ghost Witness functions for Pulsar Folding pipeline."""

from __future__ import annotations

from ageoa.ghost.abstract import AbstractScalar, AbstractSignal


def witness_de_disperse(
    data: AbstractSignal,
    DM: float,
    fchan: float,
    width: float,
    tsamp: float,
) -> AbstractSignal:
    """Ghost witness for dedispersion preserving 2D spectrogram structure."""
    data.assert_domain("time")
    return AbstractSignal(
        shape=data.shape,
        dtype="float64",
        sampling_rate=data.sampling_rate,
        domain="time",
        units="power",
    )


def witness_fold_signal(data: AbstractSignal, period: int) -> AbstractSignal:
    """Ghost witness for signal folding into a 1D profile."""
    data.assert_domain("time")
    if len(data.shape) < 2:
        raise ValueError("Fold signal requires 2D input (Time, Frequency)")
    if period <= 0:
        raise ValueError("period must be positive")

    return AbstractSignal(
        shape=(period,),
        dtype="float64",
        sampling_rate=data.sampling_rate,
        domain="time",
        units="normalized_power",
    )


def witness_snr(arr: AbstractSignal) -> AbstractScalar:
    """Ghost witness for SNR calculation."""
    return AbstractScalar(dtype="float64", min_val=0.0)


def witness_delay_from_dm(DM: float, freq_emitted: float) -> AbstractScalar:
    """Ghost witness for DM delay calculation."""
    return AbstractScalar(dtype="float64", min_val=0.0)
